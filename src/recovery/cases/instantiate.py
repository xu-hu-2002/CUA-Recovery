from __future__ import annotations

from recovery.cases.gold_action import attach_gold_actions
from recovery.construction.cases import eligible_depths

from dataclasses import replace
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from recovery.longhorizon.cases import (
    CaseCandidate,
    FailureSource,
    dedup_cases,
    instantiate_depths,
    reversibility_stratum,
)
from recovery.longhorizon.ontology import Ontology

CASE_VERSION = "recovery-case/1.0"
_HIGH_CONSEQUENCE = {"R2", "R3"}


def effects_within(
    trace: Mapping[str, Any], gold: Mapping[str, Any], start: int, end: int
) -> List[Dict[str, Any]]:
    class_by_table: Dict[str, str] = {}
    type_by_table: Dict[str, str] = {}
    for write in gold.get("writes_gold", ()):
        class_by_table[write["table"]] = write["reversibility_class"]
        type_by_table[write["table"]] = write["effect_type"]
    out = []
    for step in trace["steps"]:
        index = int(step["action_index"])
        if index < start or index > end:
            continue
        for row in step.get("delta", ()):
            table = "%s.%s" % (row["db"], row["tbl"])
            klass = class_by_table.get(table)
            if klass is None:
                klass = (
                    "R3"
                    if row["op"] == "INSERT"
                    and any(k in row["tbl"] for k in ("message", "mail", "order", "payment"))
                    else "R1"
                )
            out.append(
                {
                    "action_index": index,
                    "table": table,
                    "rowid": row["rowid"],
                    "op": row["op"],
                    "effect_type": type_by_table.get(table, row["op"].lower()),
                    "reversibility_class": klass,
                }
            )
    return out


def build_cases(
    trace: Mapping[str, Any],
    analysis: Mapping[str, Any],
    gold: Mapping[str, Any],
    repair: Mapping[str, Any],
    profile: Optional[Mapping[str, Any]],
    depth_grid: Sequence[int],
    ontology: Ontology,
    step_budget: Optional[int] = None,
    task_ir: Optional[Mapping[str, Any]] = None,
    test_eligibility: bool = True,
    require_error_explicit: bool = True,
) -> Tuple[List[Dict[str, Any]], Dict[int, str]]:
    root = analysis.get("root_cause_action_index")
    if (
        root is None
        or analysis.get("residual_code")
        or repair.get("status") == "PREFIX_UNREPAIRABLE"
    ):
        return [], {}
    steps = trace["steps"]
    last = max(int(s["action_index"]) for s in steps)
    post = last - int(root)
    root_step = next(s for s in steps if int(s["action_index"]) == int(root))
    root_action = {
        "type": root_step["action"]["type"],
        "raw": root_step["action"].get("raw"),
        "params": root_step.get("params", []),
    }
    node_horizon = ((profile or {}).get("provenance", {}).get("node_horizons", {}) or {}).get(
        analysis.get("root_cause_node_id") or "", {}
    )
    source = FailureSource(
        task_id=str(trace["task_id"]),
        source_rollout_id=str(trace["rollout_id"]),
        source_agent=str(trace["agent"]),
        root_cause_action_index=int(root),
        root_action=root_action,
        paper_type=str(analysis["paper_type"]),
        paper_category=str(analysis["paper_category"]),
        group=str(analysis["group"]),
        reversibility_stratum="reversible",
        post_error_steps_available=post,
        provenance={
            "analysis_version": analysis.get("analysis_version"),
            "repair_status": repair.get("status"),
        },
    )
    candidates, skipped = instantiate_depths(source, list(depth_grid))
    if test_eligibility:
        _, horizon_skips = eligible_depths(
            int(root),
            last,
            analysis.get("earliest_identifiable_action_index"),
            [c.error_depth for c in candidates],
            require_error_explicit=require_error_explicit,
        )
        skipped.update(horizon_skips)
        candidates = [c for c in candidates if c.error_depth not in horizon_skips]
    records = []
    for candidate in candidates:
        effects = effects_within(trace, gold, int(root), int(root) + candidate.error_depth)
        stratum, high = reversibility_stratum(effects, ontology)
        root_effects = [e for e in effects if e["action_index"] == int(root)]
        record = replace(
            candidate,
            reversibility_stratum=stratum,
            high_consequence=high,
            root_cause_reversibility_class=max(
                (e["reversibility_class"] for e in root_effects), default=None
            ),
            downstream_commit_class=max((e["reversibility_class"] for e in effects), default=None),
            verification_path="auto",
            takeover_step_budget=step_budget,
        ).to_record()
        record.update(
            {
                "schema_version": CASE_VERSION,
                "world_id": str(trace["world_id"]),
                "root_cause_node_id": analysis.get("root_cause_node_id"),
                "predicted_latent_horizon_semantic": node_horizon.get("latent_horizon_static"),
                "predicted_observability_class": node_horizon.get("observability_class"),
                "measured_semantic_horizon": analysis.get("semantic_horizon"),
                "measured_action_horizon": analysis.get("action_horizon"),
                "horizon_censored": bool(analysis.get("horizon_censored")),
                "prefix_modality": str(trace.get("modality", {}).get("primary", "gui")),
                "analysis_path": str(analysis.get("analysis_path", "auto")),
                "prefix_repair_ref": "%s:%s" % (repair.get("repair_version"), trace["rollout_id"]),
            }
        )
        records.append(record)
    attach_gold_actions(records, analysis, gold, task_ir)
    return records, skipped


def dedup_records(
    records: Sequence[Mapping[str, Any]],
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    by_id = {r["case_id"]: dict(r) for r in records}
    candidates = []
    for r in records:
        candidates.append(
            CaseCandidate(
                case_id=r["case_id"],
                task_id=r["task_id"],
                source_rollout_id=r["source_rollout_id"],
                source_agent=r["source_agent"],
                error_depth=r["error_depth"],
                root_cause_action_index=r["root_cause_action_index"],
                paper_type=r["paper_type"],
                paper_category=r["paper_category"],
                group=r["group"],
                reversibility_stratum=r["reversibility_stratum"],
                post_error_steps_available=r["post_error_steps_available"],
                root_action_signature=r["root_action_signature"],
                verification_path=r["verification_path"],
                high_consequence=r["high_consequence"],
                status=r["status"],
                merged_from=tuple(r.get("merged_from", ())),
            )
        )
    kept, removed = dedup_cases(candidates)
    kept_records = []
    for c in kept:
        record = by_id[c.case_id]
        record["merged_from"] = list(c.merged_from)
        kept_records.append(record)
    removed_records = []
    for c in removed:
        record = by_id[c.case_id]
        record["status"] = c.status
        removed_records.append(record)
    return kept_records, removed_records
