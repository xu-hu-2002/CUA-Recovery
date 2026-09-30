#!/usr/bin/env python3
"""Export takeover prefixes at each error depth, one file per agent and depth.

For every human-labelled failure we take the *annotated* canonical trajectory --- the cleaning
proposal's ``drop_candidates`` before the root cause removed and the remaining actions
renumbered --- and emit the prefix ending ``depth`` actions after the root cause.  Depth 0 is the
root-cause action itself.  Drop candidates at or after the root cause are rejected (paper App. C
"Prefix repair" only repairs mistakes before t_r) and counted in the manifest.

Depth eligibility is ``derail.construction.cases.eligible_depths``, the rule shared with takeover
selection (paper App. C): a depth is skipped when ``root + depth`` runs past the end of the
trajectory or past the action at which the failure first becomes identifiable.

Output: ``<out-dir>/<agent>_<depth>.jsonl``, one case per line.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from derail.construction.cases import eligible_depths  # noqa: E402

CASE_SCHEMA_VERSION = "error-depth-slice/0.1"


def _load_jsonl(path: Path) -> List[Dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _index_canonical(builds_root: Path) -> Dict[str, Path]:
    """Map trajectory_id -> canonical trajectory.jsonl, across flat and per-VM build layouts."""
    found: Dict[str, Path] = {}
    for path in builds_root.glob("*_traj/**/canonical/*/trajectory.jsonl"):
        trajectory_id = path.parent.name
        if trajectory_id in found and found[trajectory_id] != path:
            raise SystemExit("trajectory %s appears in two builds" % trajectory_id)
        found[trajectory_id] = path
    return found


def _index_proposals(proposals_dir: Path) -> Dict[Tuple[str, str], Path]:
    """Map (trajectory_id, reviewer_id) -> cleaning proposal path."""
    found: Dict[Tuple[str, str], Path] = {}
    for path in proposals_dir.glob("*.json"):
        payload = json.loads(path.read_text(encoding="utf-8"))
        found[(payload["trajectory_id"], payload.get("reviewer_id", ""))] = path
    return found


def _pick_proposal(
    proposals: Dict[Tuple[str, str], Path], trajectory_id: str, annotator: str
) -> Optional[Path]:
    """The annotator who produced the primary record owns the prefix cleaning for it."""
    exact = proposals.get((trajectory_id, annotator))
    if exact is not None:
        return exact
    candidates = [path for (tid, _), path in proposals.items() if tid == trajectory_id]
    return candidates[0] if len(candidates) == 1 else None


def _remap(original_index: int, retained: Sequence[int]) -> Optional[int]:
    """Position of ``original_index`` in the retained sequence.

    When the index itself was dropped we fall back to the last retained action at or before it,
    which keeps ``root + depth`` conservative rather than letting a drop push the takeover point
    later than the annotator intended.  ``None`` means nothing survives at or before it.
    """
    position = -1
    for offset, value in enumerate(retained):
        if value <= original_index:
            position = offset
        else:
            break
    return position if position >= 0 else None


def build_cases(
    record: Dict[str, Any],
    steps: Sequence[Dict[str, Any]],
    drops: Sequence[int],
    depths: Sequence[int],
    require_error_explicit: bool = True,
) -> Tuple[List[Dict[str, Any]], str]:
    """Return (cases, skip_reason).  A non-empty reason means the trajectory yielded nothing."""
    identifiable = record.get("identifiable_at_action_index")
    original_root = int(record["root_cause_action_index"])
    dropped = {index for index in drops if index < original_root}
    rejected = sorted(index for index in drops if index >= original_root)
    retained_steps = [s for s in steps if s["action_index_global"] not in dropped]
    retained_indices = [s["action_index_global"] for s in retained_steps]
    if not retained_steps:
        return [], "every action dropped"

    root = _remap(int(record["root_cause_action_index"]), retained_indices)
    if root is None or retained_indices[root] != int(record["root_cause_action_index"]):
        return [], "root-cause action was dropped or is missing"
    limit_identifiable = None if identifiable is None else _remap(int(identifiable), retained_indices)
    available, skipped = eligible_depths(
        root,
        len(retained_steps) - 1,
        limit_identifiable,
        depths,
        require_error_explicit=require_error_explicit,
    )
    cases: List[Dict[str, Any]] = []
    for depth in available:
        end = root + depth
        prefix = retained_steps[: end + 1]
        cases.append(
            {
                "schema_version": CASE_SCHEMA_VERSION,
                "case_id": "%s-d%d" % (record["trajectory_id"], depth),
                "trajectory_id": record["trajectory_id"],
                "task_id": record.get("task_id"),
                "task_category": record.get("task_category"),
                "model": record.get("model"),
                "annotator": record.get("annotator"),
                "depth": depth,
                "root_cause_index": root,
                "identifiable_at_index": limit_identifiable,
                "takeover_after_index": end,
                "original_root_cause_action_index": record["root_cause_action_index"],
                "original_identifiable_at_action_index": identifiable,
                "dropped_action_indices": sorted(dropped),
                "rejected_post_root_drop_indices": rejected,
                "retained_action_count": len(retained_steps),
                "original_action_count": len(steps),
                "reversibility": record.get("reversibility"),
                "primary_paper_type": record.get("primary_paper_type"),
                "paper_types": record.get("paper_types"),
                "loop_detected": record.get("loop_detected"),
                "terminated_explicitly": record.get("terminated_explicitly"),
                "terminate_status": record.get("terminate_status"),
                "prefix_steps": prefix,
            }
        )
    if not cases:
        return [], "; ".join(sorted(set(skipped.values())))
    return cases, ""


def main(argv: Iterable[str] = ()) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    repo_default = Path(__file__).resolve().parents[1]
    parser.add_argument("--repo-root", type=Path, default=repo_default)
    parser.add_argument(
        "--records", default="data/synthesis/phase0_precheck/failure_continuation_records.jsonl"
    )
    parser.add_argument("--builds-root", default="artifacts/derail_builds")
    parser.add_argument(
        "--proposals-dir", default="artifacts/derail_builds/human_labels/cleaning_proposals"
    )
    parser.add_argument("--benchmark-config", default="configs/benchmark/derail_v1.yaml")
    parser.add_argument(
        "--require-error-explicit",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="default: depth_eligibility.require_error_explicit in the benchmark config",
    )
    parser.add_argument(
        "--out-dir",
        default="artifacts/error depth",
        help="repository-relative output directory (note the space, as requested)",
    )
    args = parser.parse_args(list(argv) if argv else None)

    import yaml

    repo = args.repo_root.resolve()
    benchmark = yaml.safe_load((repo / args.benchmark_config).read_text())
    depths = [int(d) for d in benchmark["depths"]]
    require_error_explicit = args.require_error_explicit
    if require_error_explicit is None:
        require_error_explicit = bool(
            (benchmark.get("depth_eligibility") or {}).get("require_error_explicit", True)
        )
    records = _load_jsonl(repo / args.records)
    canonical = _index_canonical(repo / args.builds_root)
    proposals = _index_proposals(repo / args.proposals_dir)

    buckets: Dict[Tuple[str, int], List[Dict[str, Any]]] = {}
    skipped: List[Dict[str, str]] = []
    drop_totals = Counter()
    rejected_totals = Counter()
    for record in records:
        trajectory_id = record["trajectory_id"]
        path = canonical.get(trajectory_id)
        if path is None:
            skipped.append({"trajectory_id": trajectory_id, "reason": "no canonical trajectory"})
            continue
        proposal_path = _pick_proposal(proposals, trajectory_id, str(record.get("annotator", "")))
        if proposal_path is None:
            skipped.append({"trajectory_id": trajectory_id, "reason": "no cleaning proposal"})
            continue
        proposal = json.loads(proposal_path.read_text(encoding="utf-8"))
        drops = [int(c["action_index_global"]) for c in proposal.get("drop_candidates") or []]
        root = int(record["root_cause_action_index"])
        drop_totals[trajectory_id] = sum(1 for index in drops if index < root)
        rejected_totals[trajectory_id] = sum(1 for index in drops if index >= root)
        cases, reason = build_cases(
            record, _load_jsonl(path), drops, depths, require_error_explicit
        )
        if reason:
            skipped.append({"trajectory_id": trajectory_id, "reason": reason})
        for case in cases:
            buckets.setdefault((str(record["model"]), case["depth"]), []).append(case)

    out_dir = repo / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    written: Dict[str, int] = {}
    models = sorted({str(r["model"]) for r in records})
    for model in models:
        for depth in depths:
            cases = buckets.get((model, depth), [])
            name = "%s_%d.jsonl" % (model, depth)
            with (out_dir / name).open("w", encoding="utf-8") as handle:
                for case in cases:
                    handle.write(json.dumps(case, ensure_ascii=False) + "\n")
            written[name] = len(cases)

    manifest = {
        "schema_version": CASE_SCHEMA_VERSION,
        "depths": depths,
        "source_records": args.records,
        "builds_root": args.builds_root,
        "proposals_dir": args.proposals_dir,
        "constraint": (
            "derail.construction.cases.eligible_depths: root + depth <= min(identifiable_at, "
            "last action), both renumbered after the cleaning proposal's pre-root "
            "drop_candidates are removed; unobserved identifiable_at %s"
            % ("excluded" if require_error_explicit else "kept")
        ),
        "trajectories_in": len(records),
        "trajectories_with_drops": sum(1 for v in drop_totals.values() if v),
        "actions_dropped": sum(drop_totals.values()),
        "post_root_drops_rejected": sum(rejected_totals.values()),
        "trajectories_with_post_root_drops_rejected": sum(
            1 for v in rejected_totals.values() if v
        ),
        "cases_written": written,
        "cases_total": sum(written.values()),
        "skipped": skipped,
    }
    (out_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({k: manifest[k] for k in
                      ("depths", "trajectories_in", "trajectories_with_drops", "actions_dropped",
                       "post_root_drops_rejected", "cases_total", "cases_written")}, ensure_ascii=False, indent=2))
    print("skipped: %d" % len(skipped))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
