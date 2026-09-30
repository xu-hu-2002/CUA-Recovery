"""``analyze_failure(trace, gold, task_ir, config) -> failure-analysis/1.0``.

Paper §3 "Constructing erroneous states" and Alg. 1 ``AnalyzeFailure``: the recorded execution
is compared with the gold lineage to locate ``t_r``, the error type and ``t_e``.
``annotation_proposal`` turns the result into a record with the human annotation fields
(``derail.annotation.records.HumanAnnotation``) so annotators verify it in the annotation UI;
a failure whose root is unresolved or whose type confidence is below the configured threshold
(``residual_threshold`` in ``configs/synthesis/typing_rules_v1.yaml``, paper 0.7) is routed to
human annotation instead of verification.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Union

from derail.failure_analysis.detectors import (
    omission_candidates,
    parameter_candidates,
    pick_root_cause,
    state_candidates,
)
from derail.failure_analysis.horizon import earliest_identifiable, semantic_horizon
from derail.failure_analysis.typing_rules import TypingRules
from derail.longhorizon.taxonomy import FailureTaxonomy
from derail.world.volatile import VolatileColumns

ANALYSIS_VERSION = "failure-analysis-v1/1.0"
PROPOSAL_SCHEMA_VERSION = "auto-annotation-proposal/0.1"
ROUTE_VERIFY = "human_verify"
ROUTE_ANNOTATE = "human_annotate"


@dataclass(frozen=True)
class AnalysisConfig:
    rules: TypingRules
    volatile: Optional[VolatileColumns]

    @classmethod
    def from_repo(
        cls,
        repo_root: Union[str, Path],
        typing_rules: str = "configs/synthesis/typing_rules_v1.yaml",
        taxonomy: str = "configs/synthesis/failure_taxonomy_v0.1.yaml",
        volatile: str = "configs/synthesis/volatile_columns_v1.yaml",
    ) -> "AnalysisConfig":
        root = Path(repo_root)
        tax = FailureTaxonomy.from_yaml(root / taxonomy)
        return cls(
            rules=TypingRules.from_yaml(root / typing_rules, tax),
            volatile=VolatileColumns.from_yaml(root / volatile),
        )


def analyze_failure(
    trace: Mapping[str, Any],
    gold: Mapping[str, Any],
    task_ir: Mapping[str, Any],
    config: AnalysisConfig,
) -> Dict[str, Any]:
    state = state_candidates(trace, gold, config.volatile)
    parameter = parameter_candidates(trace, gold)
    omission = omission_candidates(trace, gold, task_ir)
    root = pick_root_cause(state + parameter + omission, config.rules.residual_threshold)
    outcome = trace.get("outcome", {})
    paper_type, category, group, confidence, alternatives = config.rules.classify(
        root.evidence_pattern if root else None, outcome
    )
    identifiable, ident_node, ident_source = (None, None, None)
    if root is not None:
        identifiable, ident_node, ident_source = earliest_identifiable(trace, gold, task_ir, root)
    last_index = max((int(s["action_index"]) for s in trace["steps"]), default=0)
    residual = None
    if root is None and paper_type is None:
        residual = "ROOT_CAUSE_RESIDUAL"
    elif root is not None and confidence < config.rules.residual_threshold:
        residual = "ROOT_CAUSE_RESIDUAL"
    root_index = root.action_index if root else None
    if root is None and paper_type is not None:
        # termination failures without a state or parameter trace: the root is the terminal action
        root_index = last_index
    return {
        "schema_version": "failure-analysis/1.0",
        "rollout_id": str(trace["rollout_id"]),
        "task_id": str(trace["task_id"]),
        "agent": str(trace["agent"]),
        "detectors": {
            "state": [c.to_dict() for c in state],
            "parameter": [c.to_dict() for c in parameter],
            "omission": [c.to_dict() for c in omission],
        },
        "root_cause_action_index": root_index,
        "root_cause_node_id": root.node_id if root else None,
        "root_cause_detector": root.detector if root else None,
        "paper_type": paper_type,
        "paper_category": category,
        "group": group,
        "type_confidence": confidence if paper_type else None,
        "type_alternatives": alternatives,
        "cross_type_labels": [],
        "earliest_identifiable_action_index": identifiable,
        "earliest_identifiable_node_id": ident_node,
        "earliest_identifiable_source": ident_source,
        "action_horizon": (identifiable - root.action_index)
        if (root and identifiable is not None)
        else None,
        "semantic_horizon": semantic_horizon(task_ir, root.node_id if root else None, ident_node),
        "horizon_censored": root is not None and identifiable is None,
        "analysis_path": "auto",
        "residual_code": residual,
        "multi_root_cause": False,
        "evidence_refs": [c.to_dict() for c in ([root] if root else [])],
        "analysis_version": ANALYSIS_VERSION,
        "provenance": {"root_detail": root.detail if root else None, "outcome": dict(outcome)},
    }


def annotation_proposal(
    record: Mapping[str, Any],
    *,
    trajectory_id: str,
    source_trajectory_sha256: str,
    annotator_id: str,
    taxonomy_version: str,
    confidence_threshold: float,
) -> Dict[str, Any]:
    """An automatic failure label in the human annotation layout, for human verification.

    The fields shared with ``HumanAnnotation.to_dict`` keep their names and meaning (0-based
    action indices, ``identifiable_at = root + horizon``); ``annotator_role`` is ``auto`` so the
    record never passes for a human label.  ``reversibility`` is left for the annotator.
    ``review_route`` is ``human_verify`` when the root is resolved and the type confidence
    reaches the threshold, else ``human_annotate`` (Alg. 1 drops such failures from the automatic
    path).
    """

    root = record.get("root_cause_action_index")
    confidence = record.get("type_confidence")
    resolved = root is not None and record.get("paper_type") is not None
    confident = resolved and confidence is not None and confidence >= confidence_threshold
    route = ROUTE_VERIFY if confident and not record.get("residual_code") else ROUTE_ANNOTATE
    identifiable = record.get("earliest_identifiable_action_index")
    horizon = identifiable - root if resolved and identifiable is not None else None
    evidence = "; ".join(
        "%s (%s)" % (ref.get("evidence_pattern"), ref.get("node_id") or "-")
        for ref in record.get("evidence_refs", ())
    )
    return {
        "schema_version": PROPOSAL_SCHEMA_VERSION,
        "annotation_id": "%s__%s" % (trajectory_id, annotator_id),
        "trajectory_id": trajectory_id,
        "source_trajectory_sha256": source_trajectory_sha256,
        "annotator_id": annotator_id,
        "annotator_role": "auto",
        "root_cause_action_index": root if resolved else None,
        "error_horizon_actions": horizon,
        "identifiable_at_action_index": identifiable if horizon is not None else None,
        "error_types": [record["paper_type"]] if resolved else [],
        "reversibility": None,
        "rationale": "automatic lineage comparison: %s" % (evidence or "no detector evidence"),
        "taxonomy_version": taxonomy_version,
        "type_confidence": confidence,
        "confidence_threshold": confidence_threshold,
        "review_route": route,
        "review_status": "pending",
        "failure_analysis": {
            "rollout_id": record.get("rollout_id"),
            "analysis_version": record.get("analysis_version"),
            "residual_code": record.get("residual_code"),
            "root_cause_node_id": record.get("root_cause_node_id"),
            "root_cause_detector": record.get("root_cause_detector"),
            "horizon_censored": record.get("horizon_censored"),
            "semantic_horizon": record.get("semantic_horizon"),
            "candidate_counts": {
                name: len(items) for name, items in (record.get("detectors") or {}).items()
            },
        },
    }
