from __future__ import annotations

import csv
import json
import os
import statistics
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import yaml

from derail.derived.layout import atomic_write_json, atomic_write_jsonl, sha256_file
from derail.derived.schema import validate_schema
from derail.longhorizon.annotations import build_failure_annotation
from derail.longhorizon.cases import (
    CaseCandidate,
    FailureSource,
    case_yield,
    dedup_cases,
    instantiate_depths,
    stratum_counts,
    stratum_from_human_label,
)
from derail.longhorizon.continuation import ContinuationStats, LoopConfig, compute_continuation
from derail.longhorizon.taxonomy import FailureTaxonomy

REPORT_SCHEMA_VERSION = "post-error-continuation-report/0.1"
PRIMARY_GRID = "primary"
FAILURE_STATE = "failure"


@dataclass(frozen=True)
class PrecheckConfig:
    labels_csv: Path
    human_labels_dir: Path
    taxonomy_config: Path
    output_dir: Path
    depth_grids: Mapping[str, Tuple[int, ...]]
    loop: LoopConfig
    early_stop_window: int
    task_horizon_dir: Optional[Path] = None

    @classmethod
    def from_yaml(cls, path: Path, repo_root: Path) -> "PrecheckConfig":
        raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
        if not isinstance(raw, Mapping):
            raise ValueError("precheck config must be a YAML mapping")
        loop_raw = raw.get("loop_detection", {})
        grids = {PRIMARY_GRID: tuple(int(depth) for depth in raw["depth_grid"])}
        for name, grid in (raw.get("comparison_depth_grids") or {}).items():
            if str(name) == PRIMARY_GRID:
                raise ValueError("comparison grid may not be named %r" % PRIMARY_GRID)
            grids[str(name)] = tuple(int(depth) for depth in grid)
        for name, grid in grids.items():
            if list(grid) != sorted(set(grid)) or grid[0] != 0:
                raise ValueError("depth grid %s must be strictly increasing and start at 0" % name)
        horizon_dir = raw.get("task_horizon_dir")
        return cls(
            labels_csv=repo_root / raw["labels_csv"],
            human_labels_dir=repo_root / raw["human_labels_dir"],
            taxonomy_config=repo_root / raw["taxonomy_config"],
            output_dir=repo_root / raw["output_dir"],
            depth_grids=grids,
            loop=LoopConfig(
                min_repeats=int(loop_raw.get("min_repeats", 3)),
                click_cell_px=int(loop_raw.get("click_cell_px", 40)),
                exclude_kinds=tuple(loop_raw.get("exclude_kinds", ("wait",))),
            ),
            early_stop_window=int(raw.get("early_stop_window_steps", 5)),
            task_horizon_dir=(repo_root / horizon_dir) if horizon_dir else None,
        )


@dataclass(frozen=True)
class FailureRecord:
    trajectory_id: str
    model: str
    task_id: str
    task_category: str
    annotator: str
    paper_types: Tuple[str, ...]
    dropped_labels: Tuple[str, ...]
    primary_paper_type: str
    paper_category: str
    group: str
    reversibility: str
    identifiable_at_action_index: Optional[int]
    stats: ContinuationStats
    root_action: Mapping[str, Any]
    task_horizon_bucket: Optional[str]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "trajectory_id": self.trajectory_id,
            "model": self.model,
            "task_id": self.task_id,
            "task_category": self.task_category,
            "annotator": self.annotator,
            "paper_types": list(self.paper_types),
            "dropped_labels": list(self.dropped_labels),
            "primary_paper_type": self.primary_paper_type,
            "paper_category": self.paper_category,
            "group": self.group,
            "reversibility": self.reversibility,
            "identifiable_at_action_index": self.identifiable_at_action_index,
            "task_horizon_bucket": self.task_horizon_bucket,
            **self.stats.to_dict(),
        }


def read_failure_rows(labels_csv: Path) -> List[Dict[str, str]]:
    with Path(labels_csv).open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    return [row for row in rows if row.get("state") == FAILURE_STATE]


def load_task_horizons(task_dir: Optional[Path]) -> Dict[str, str]:
    if task_dir is None:
        return {}
    horizons: Dict[str, str] = {}
    for path in sorted(Path(task_dir).glob("*/*.rubrics.json")):
        for task in json.loads(path.read_text(encoding="utf-8")):
            if task.get("horizon"):
                horizons[str(task["id"])] = str(task["horizon"])
    return horizons


def _read_actions(trajectory_path: Path) -> List[Dict[str, Any]]:
    actions = []
    with Path(trajectory_path).open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                actions.append(json.loads(line)["action"])
    return actions


def build_failure_record(
    row: Mapping[str, str],
    taxonomy: FailureTaxonomy,
    config: PrecheckConfig,
    task_horizons: Mapping[str, str],
) -> FailureRecord:
    actions = _read_actions(Path(row["trajectory_path"]))
    root = int(row["root_cause_action_index"])
    annotation = json.loads(Path(row["annotation_path"]).read_text(encoding="utf-8"))
    task_config = json.loads(Path(row["task_config_path"]).read_text(encoding="utf-8"))
    labels, dropped = taxonomy.normalize(
        [label for label in row["error_types"].split("|") if label]
    )
    primary = taxonomy.primary_paper_type(labels)
    if primary is None:
        raise ValueError("failure %s has no frozen paper type" % row["trajectory_id"])
    stats = compute_continuation(
        actions, root, loop=config.loop, early_stop_window=config.early_stop_window
    )
    identifiable = annotation.get("identifiable_at_action_index")
    return FailureRecord(
        trajectory_id=str(row["trajectory_id"]),
        model=str(row["model"]),
        task_id=str(task_config["id"]),
        task_category=str(task_config.get("category", "")),
        annotator=str(row["annotator"]),
        paper_types=labels,
        dropped_labels=dropped,
        primary_paper_type=primary,
        paper_category=taxonomy.category_of(primary),
        group=taxonomy.group_of(primary),
        reversibility=str(annotation["reversibility"]),
        identifiable_at_action_index=int(identifiable) if identifiable is not None else None,
        stats=stats,
        root_action=actions[root],
        task_horizon_bucket=task_horizons.get(str(task_config["id"])),
    )


def _quantiles(values: Sequence[int]) -> Dict[str, float]:
    if not values:
        return {"n": 0}
    ordered = sorted(values)
    quartiles = (
        statistics.quantiles(ordered, n=4, method="inclusive")
        if len(ordered) > 1
        else [ordered[0]] * 3
    )
    return {
        "n": len(ordered),
        "min": ordered[0],
        "p25": quartiles[0],
        "median": statistics.median(ordered),
        "p75": quartiles[2],
        "max": ordered[-1],
        "mean": round(statistics.fmean(ordered), 2),
    }


def _summary(
    records: Sequence[FailureRecord], depth_grids: Mapping[str, Tuple[int, ...]]
) -> Dict[str, Any]:
    steps = [record.stats.post_error_steps for record in records]
    n = len(records)
    loops = sum(record.stats.loop_detected for record in records)
    explicit = sum(record.stats.terminated_explicitly for record in records)
    early = sum(record.stats.early_stop for record in records)
    per_grid = {}
    for name, grid in depth_grids.items():
        per_depth = {}
        for depth in grid:
            eligible = [record for record in records if record.stats.post_error_steps >= depth]
            loop_free = [
                record
                for record in eligible
                if not record.stats.loop_detected
                or (record.stats.loop_established_offset or 0) > depth
            ]
            per_depth[str(depth)] = {
                "candidates": len(eligible),
                "candidates_loop_free": len(loop_free),
                "share": round(len(eligible) / n, 3) if n else 0.0,
            }
        per_grid[name] = per_depth
    return {
        "n": n,
        "post_error_steps": _quantiles(steps),
        "loop_detected": loops,
        "loop_rate": round(loops / n, 3) if n else 0.0,
        "terminated_explicitly": explicit,
        "explicit_terminate_rate": round(explicit / n, 3) if n else 0.0,
        "early_stop": early,
        "early_stop_rate": round(early / n, 3) if n else 0.0,
        "depth_candidates": per_grid,
    }


def _candidates_for_grid(
    records: Sequence[FailureRecord], grid: Sequence[int], click_cell_px: int
) -> Tuple[List[CaseCandidate], Dict[str, int]]:
    candidates: List[CaseCandidate] = []
    skipped: Counter = Counter()
    for record in records:
        source = FailureSource(
            task_id=record.task_id,
            source_rollout_id=record.trajectory_id,
            source_agent=record.model,
            root_cause_action_index=record.stats.root_cause_action_index,
            root_action=record.root_action,
            paper_type=record.primary_paper_type,
            paper_category=record.paper_category,
            group=record.group,
            reversibility_stratum=stratum_from_human_label(record.reversibility),
            post_error_steps_available=record.stats.post_error_steps,
            task_horizon_bucket=record.task_horizon_bucket,
            provenance={
                "annotator": record.annotator,
                "stratum_source": "human_reversibility_label",
            },
        )
        created, missing = instantiate_depths(source, grid, click_cell_px=click_cell_px)
        candidates.extend(created)
        skipped.update(missing.values())
    return candidates, dict(skipped)


def _string_depths(table: Mapping[str, Mapping[int, int]]) -> Dict[str, Dict[str, int]]:
    return {
        key: {str(depth): count for depth, count in counts.items()} for key, counts in table.items()
    }


def build_report(
    records: Sequence[FailureRecord], config: PrecheckConfig, taxonomy: FailureTaxonomy
) -> Dict[str, Any]:
    by_model: Dict[str, List[FailureRecord]] = defaultdict(list)
    by_category: Dict[str, List[FailureRecord]] = defaultdict(list)
    by_model_category: Dict[str, Dict[str, List[FailureRecord]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for record in records:
        by_model[record.model].append(record)
        by_category[record.paper_category].append(record)
        by_model_category[record.model][record.paper_category].append(record)

    case_level = {}
    for name, grid in config.depth_grids.items():
        candidates, skipped = _candidates_for_grid(records, grid, config.loop.click_cell_px)
        kept, removed = dedup_cases(candidates)
        case_level[name] = {
            "depth_grid": list(grid),
            "candidates_before_dedup": len(candidates),
            "skipped_depth_instances": skipped,
            "dedup_removed": len(removed),
            "dedup_removed_rate": round(len(removed) / len(candidates), 3) if candidates else 0.0,
            "case_yield_by_paper_category": _string_depths(
                case_yield(kept, grid, by="paper_category")
            ),
            "case_yield_by_agent": _string_depths(case_yield(kept, grid, by="source_agent")),
            "stratum_counts": {
                stratum: _string_depths(per_type)
                for stratum, per_type in stratum_counts(kept, grid).items()
            },
        }

    label_counts = Counter(label for record in records for label in record.paper_types)
    return {
        "schema_version": REPORT_SCHEMA_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "taxonomy_version": taxonomy.taxonomy_version,
        "primary_type_rule": {
            "category_priority": list(taxonomy.category_priority),
            "status": "proposed_pending_researcher_confirmation",
        },
        "loop_heuristic": {
            "version": records[0].stats.loop_heuristic_version if records else None,
            "min_repeats": config.loop.min_repeats,
            "click_cell_px": config.loop.click_cell_px,
            "exclude_kinds": list(config.loop.exclude_kinds),
        },
        "early_stop_window_steps": config.early_stop_window,
        "depth_grids": {name: list(grid) for name, grid in config.depth_grids.items()},
        "overall": _summary(records, config.depth_grids),
        "by_agent": {
            model: _summary(items, config.depth_grids) for model, items in sorted(by_model.items())
        },
        "by_paper_category": {
            category: _summary(items, config.depth_grids)
            for category, items in sorted(by_category.items())
        },
        "by_agent_and_category": {
            model: {
                category: _summary(items, config.depth_grids)
                for category, items in sorted(inner.items())
            }
            for model, inner in sorted(by_model_category.items())
        },
        "multi_label_type_counts": dict(sorted(label_counts.items())),
        "primary_type_counts": dict(
            sorted(Counter(record.primary_paper_type for record in records).items())
        ),
        "reversibility_counts": dict(
            sorted(Counter(record.reversibility for record in records).items())
        ),
        "case_level": case_level,
    }


def _md_table(headers: Sequence[str], rows: Iterable[Sequence[Any]]) -> str:
    lines = ["| " + " | ".join(headers) + " |", "|" + "---|" * len(headers)]
    for row in rows:
        lines.append("| " + " | ".join(str(cell) for cell in row) + " |")
    return "\n".join(lines)


def _grid_order(names: Iterable[str]) -> List[str]:
    ordered = sorted(names)
    if PRIMARY_GRID in ordered:
        ordered.remove(PRIMARY_GRID)
        ordered.insert(0, PRIMARY_GRID)
    return ordered


def render_markdown(report: Mapping[str, Any]) -> str:
    grids = {name: report["depth_grids"][name] for name in _grid_order(report["depth_grids"])}
    parts = [
        "# Post-error continuation pre-check (Phase 0)",
        "",
        "Generated %s from the frozen label index (taxonomy %s). Primary paper type follows the "
        "category priority %s (proposed, pending confirmation). Loop heuristic: %s, "
        "min_repeats=%d, click cell %dpx. Early stop window: %d steps."
        % (
            report["created_at"],
            report["taxonomy_version"],
            report["primary_type_rule"]["category_priority"],
            report["loop_heuristic"]["version"],
            report["loop_heuristic"]["min_repeats"],
            report["loop_heuristic"]["click_cell_px"],
            report["early_stop_window_steps"],
        ),
        "",
        "## Overall",
        "",
    ]
    overall = report["overall"]
    q = overall["post_error_steps"]
    parts.append(
        _md_table(
            [
                "n",
                "post-error steps min/p25/median/p75/max",
                "mean",
                "loop rate",
                "explicit terminate rate",
                "early stop rate",
            ],
            [
                [
                    overall["n"],
                    "%s / %s / %s / %s / %s"
                    % (q["min"], q["p25"], q["median"], q["p75"], q["max"]),
                    q["mean"],
                    overall["loop_rate"],
                    overall["explicit_terminate_rate"],
                    overall["early_stop_rate"],
                ]
            ],
        )
    )
    for grid_name, grid in grids.items():
        parts += ["", "### Depth candidates, grid `%s` = %s" % (grid_name, grid), ""]
        rows = []
        for depth in grid:
            cell = overall["depth_candidates"][grid_name][str(depth)]
            rows.append([depth, cell["candidates"], cell["share"], cell["candidates_loop_free"]])
        parts.append(
            _md_table(
                [
                    "d",
                    "candidates (post_error_steps >= d)",
                    "share of failures",
                    "loop-free before d",
                ],
                rows,
            )
        )

    def _section(title: str, table: Mapping[str, Mapping[str, Any]]) -> None:
        parts.extend(["", "## %s" % title, ""])
        headers = ["group", "n", "median steps", "mean", "loop", "explicit term.", "early stop"]
        for grid_name, grid in grids.items():
            headers += ["%s d=%d" % (grid_name, depth) for depth in grid]
        rows = []
        for key, summary in table.items():
            row = [
                key,
                summary["n"],
                summary["post_error_steps"].get("median"),
                summary["post_error_steps"].get("mean"),
                summary["loop_rate"],
                summary["explicit_terminate_rate"],
                summary["early_stop_rate"],
            ]
            for grid_name, grid in grids.items():
                row += [
                    summary["depth_candidates"][grid_name][str(depth)]["candidates"]
                    for depth in grid
                ]
            rows.append(row)
        parts.append(_md_table(headers, rows))

    _section("By agent", report["by_agent"])
    _section("By paper category (primary type)", report["by_paper_category"])
    for model, inner in report["by_agent_and_category"].items():
        _section("Agent %s by category" % model, inner)

    parts += ["", "## Case-level view after dedup", ""]
    for grid_name in _grid_order(report["case_level"]):
        block = report["case_level"][grid_name]
        parts += [
            "### Grid `%s`" % grid_name,
            "",
            "- candidates before dedup: %d; skipped depth instances: %s"
            % (block["candidates_before_dedup"], block["skipped_depth_instances"]),
            "- dedup removed: %d (%.1f%%)"
            % (block["dedup_removed"], 100 * block["dedup_removed_rate"]),
            "",
        ]
        headers = ["paper category"] + ["d=%d" % depth for depth in block["depth_grid"]]
        rows = [
            [category] + [counts[str(depth)] for depth in block["depth_grid"]]
            for category, counts in block["case_yield_by_paper_category"].items()
        ]
        parts.append(_md_table(headers, rows))
        parts.append("")
        headers = ["stratum", "paper type"] + ["d=%d" % depth for depth in block["depth_grid"]]
        rows = []
        for stratum, per_type in block["stratum_counts"].items():
            for paper_type, counts in per_type.items():
                rows.append(
                    [stratum, paper_type] + [counts[str(depth)] for depth in block["depth_grid"]]
                )
        parts.append(_md_table(headers, rows))
        parts.append("")
    parts += [
        "## Reading guide",
        "",
        "- `candidates` counts failures whose post-error suffix is long enough for depth d; "
        "nothing is padded.",
        "- `loop-free before d` additionally requires that no action signature reaches "
        "`min_repeats` within the first d post-error steps.",
        "- Early stop and explicit termination are reported separately: budget exhaustion "
        "ends a trajectory without a terminate action.",
        "- The case-level tables apply the section 14A.4 dedup rule (same task, root within "
        "one step, same primary type, same root-action signature) across agents; "
        "`merged_from` keeps provenance.",
        "",
    ]
    return "\n".join(parts)


def run_precheck(
    config: PrecheckConfig,
    *,
    repo_root: Path,
    command: str = "",
    schema_root: Optional[Path] = None,
) -> Dict[str, Any]:
    schema_root = schema_root or Path(
        os.environ.get("DERAIL_REPO_ROOT", Path(__file__).resolve().parents[3])
    )

    taxonomy = FailureTaxonomy.from_yaml(config.taxonomy_config)
    rows = read_failure_rows(config.labels_csv)
    horizons = load_task_horizons(config.task_horizon_dir)
    records = [build_failure_record(row, taxonomy, config, horizons) for row in rows]
    report = build_report(records, config, taxonomy)
    output = config.output_dir
    output.mkdir(parents=True, exist_ok=True)
    atomic_write_jsonl(
        output / "failure_continuation_records.jsonl", [record.to_dict() for record in records]
    )
    annotations = [
        build_failure_annotation(
            annotation_version="mypcbench-primary-%s" % taxonomy.taxonomy_version,
            task_id=record.task_id,
            model_id=record.model,
            rollout_id=record.trajectory_id,
            raw_error_types=record.paper_types,
            reversibility=record.reversibility,
            stats=record.stats,
            taxonomy=taxonomy,
            identifiable_at_action_index=record.identifiable_at_action_index,
            provenance={"annotator": record.annotator, "source": "trajectory_labels.csv"},
        )
        for record in records
    ]
    for annotation in annotations:
        validate_schema(annotation, "failure_annotation.schema.json", schema_root)
    atomic_write_jsonl(output / "failure_annotations.jsonl", annotations)
    atomic_write_json(output / "post_error_continuation_report.json", report)
    (output / "post_error_continuation_report.md").write_text(
        render_markdown(report), encoding="utf-8"
    )
    manifest = {
        "schema_version": "stage-manifest/0.1",
        "stage": "phase0_post_error_precheck",
        "created_at": report["created_at"],
        "command": command,
        "inputs": {
            "labels_csv": {"uri": str(config.labels_csv), "sha256": sha256_file(config.labels_csv)},
            "taxonomy_config": {
                "uri": str(config.taxonomy_config),
                "sha256": sha256_file(config.taxonomy_config),
            },
        },
        "counts": {
            "failure_rows": len(rows),
            "records": len(records),
            "models": sorted({record.model for record in records}),
        },
        "outputs": [
            str(output / "failure_continuation_records.jsonl"),
            str(output / "failure_annotations.jsonl"),
            str(output / "post_error_continuation_report.json"),
            str(output / "post_error_continuation_report.md"),
        ],
        "human_review_status": "pending",
        "go_no_go": "researcher_decision_required",
    }
    atomic_write_json(output / "stage_manifest.json", manifest)
    return manifest
