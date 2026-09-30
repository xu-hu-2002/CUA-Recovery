from __future__ import annotations

from typing import Any, Dict, Mapping, Optional, Sequence

from derail.longhorizon.continuation import ContinuationStats
from derail.longhorizon.taxonomy import FailureTaxonomy

ANNOTATION_SCHEMA_VERSION = "failure-annotation/0.1"


def build_failure_annotation(
    *,
    annotation_version: str,
    task_id: str,
    model_id: str,
    rollout_id: str,
    raw_error_types: Sequence[str],
    reversibility: str,
    stats: ContinuationStats,
    taxonomy: FailureTaxonomy,
    review_status: str = "single_annotator",
    identifiable_at_action_index: Optional[int] = None,
    horizon_censored: Optional[bool] = None,
    multi_root_cause: bool = False,
    milestone_hint: Optional[Mapping[str, Any]] = None,
    root_cause_in_lineage_hint: Optional[bool] = None,
    provenance: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    labels, dropped = taxonomy.normalize(raw_error_types)
    primary = taxonomy.primary_paper_type(labels)
    if primary is None:
        raise ValueError("failure %s carries no frozen paper type" % rollout_id)
    if identifiable_at_action_index is not None:
        action_horizon: Optional[int] = identifiable_at_action_index - stats.root_cause_action_index
        censored = False
    else:
        action_horizon = None
        censored = True if horizon_censored is None else horizon_censored
    return {
        "schema_version": ANNOTATION_SCHEMA_VERSION,
        "annotation_version": annotation_version,
        "task_id": task_id,
        "model_id": model_id,
        "rollout_id": rollout_id,
        "root_cause_action_index": stats.root_cause_action_index,
        "root_cause_milestone_id": None,
        "paper_type": primary,
        "paper_category": taxonomy.category_of(primary),
        "group": taxonomy.group_of(primary),
        "error_types": list(labels),
        "secondary_causes": [label for label in labels if label != primary],
        "multi_root_cause": multi_root_cause,
        "reversibility_stratum": reversibility,
        "root_cause_reversibility_class": None,
        "downstream_commit_class": None,
        "earliest_identifiable_action_index": identifiable_at_action_index,
        "earliest_identifiable_milestone_id": (
            milestone_hint.get("earliest_identifiable_milestone_id") if milestone_hint else None
        ),
        "action_horizon": action_horizon,
        "semantic_horizon": None,
        "horizon_censored": censored,
        "post_error_steps": stats.post_error_steps,
        "loop_detected": stats.loop_detected,
        "loop_established_offset": stats.loop_established_offset,
        "terminated_explicitly": stats.terminated_explicitly,
        "loop_heuristic_version": stats.loop_heuristic_version,
        "milestone_hint_used": milestone_hint is not None,
        "root_cause_in_lineage_hint": root_cause_in_lineage_hint,
        "review_status": review_status,
        "provenance": {
            "dropped_labels": list(dropped),
            "primary_type_rule": list(taxonomy.category_priority),
            **(dict(provenance) if provenance else {}),
        },
    }
