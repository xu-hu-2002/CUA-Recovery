from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional, Tuple

from derail.longhorizon.dag import DagIndex
from derail.longhorizon.ontology import Ontology

RUBRIC_SCHEMA_VERSION = "rubric/0.2"
CREDIT_POLICY = "no_credit_if_prerequisite_value_is_wrong"


def derive_milestones(fragment: Mapping[str, Any], ontology: Ontology) -> Dict[str, Any]:
    dag = DagIndex.from_fragment(fragment)
    milestone_by_node: Dict[str, str] = {}
    milestones: List[Dict[str, Any]] = []
    for index, node_id in enumerate(dag.order, start=1):
        node = dag.nodes[node_id]
        milestone_id = "m%d" % index
        milestone_by_node[node_id] = milestone_id
        verifier = node.get("verifier") or {}
        observability = verifier.get("observability", "unobservable")
        if observability not in ontology.milestone_observability:
            raise ValueError("unknown observability %r on node %s" % (observability, node_id))
        default_weight = 2.0 if node.get("side_effects") or node["op"] == "decide" else 1.0
        milestones.append(
            {
                "milestone_id": milestone_id,
                "semantic_goal": node.get("semantic_goal", "%s in %s" % (node["op"], node["app"])),
                "source_nodes": [node_id],
                "depends_on": [milestone_by_node[parent] for parent in dag.incoming[node_id]],
                "predicate": verifier.get("predicate"),
                "observability": observability,
                "evidence": list(verifier.get("evidence", [])),
                "verifier": verifier.get("verifier_id"),
                "weight": float(node.get("weight", default_weight)),
                "critical": bool(node.get("critical", True)),
                "credit_policy": CREDIT_POLICY,
                "data_dependent": any(
                    edge.get("kind") in {"data_dependency", "state_dependency"}
                    for edge in dag.edges
                    if edge["to"]["node_id"] == node_id
                ),
            }
        )
    return {"schema_version": RUBRIC_SCHEMA_VERSION, "milestones": milestones}


@dataclass(frozen=True)
class MilestoneScore:
    milestone_completion: float
    dependency_consistent_completion: float
    first_failed_dependency: Optional[str]
    critical_error_count: int
    completion_curve: Tuple[float, ...]
    milestone_auc: float
    unscored_milestones: Tuple[str, ...]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "milestone_completion": self.milestone_completion,
            "dependency_consistent_completion": self.dependency_consistent_completion,
            "first_failed_dependency": self.first_failed_dependency,
            "critical_error_count": self.critical_error_count,
            "completion_curve": list(self.completion_curve),
            "milestone_auc": self.milestone_auc,
            "unscored_milestones": list(self.unscored_milestones),
        }


def score_milestones(
    rubric: Mapping[str, Any], results: Mapping[str, Optional[bool]]
) -> MilestoneScore:
    milestones = list(rubric["milestones"])
    weights = {item["milestone_id"]: float(item["weight"]) for item in milestones}
    total = sum(weights.values())
    if total <= 0:
        raise ValueError("rubric has no positive milestone weight")
    passed: Dict[str, bool] = {}
    consistent: Dict[str, bool] = {}
    unscored = []
    first_failed = None
    critical_errors = 0
    curve = []
    cumulative = 0.0
    for item in milestones:
        milestone_id = item["milestone_id"]
        outcome = results.get(milestone_id)
        if outcome is None or item["observability"] == "unobservable":
            unscored.append(milestone_id)
            outcome = False
        passed[milestone_id] = bool(outcome)
        consistent[milestone_id] = bool(outcome) and all(
            consistent.get(parent, False) for parent in item["depends_on"]
        )
        if not passed[milestone_id] and item["critical"]:
            critical_errors += 1
            if first_failed is None:
                first_failed = milestone_id
        cumulative += weights[milestone_id] if consistent[milestone_id] else 0.0
        curve.append(round(cumulative / total, 6))
    return MilestoneScore(
        milestone_completion=sum(weights[m] for m, ok in passed.items() if ok) / total,
        dependency_consistent_completion=(
            sum(weights[m] for m, ok in consistent.items() if ok) / total
        ),
        first_failed_dependency=first_failed,
        critical_error_count=critical_errors,
        completion_curve=tuple(curve),
        milestone_auc=(sum(curve) / len(curve)) if curve else 0.0,
        unscored_milestones=tuple(unscored),
    )


def earliest_identifiable_hint(
    rubric: Mapping[str, Any],
    results: Mapping[str, Optional[bool]],
    fragment: Mapping[str, Any],
) -> Dict[str, Any]:
    dag = DagIndex.from_fragment(fragment)
    for item in rubric["milestones"]:
        if not item["critical"] or item["observability"] == "unobservable":
            continue
        if results.get(item["milestone_id"]) is False:
            node_id = item["source_nodes"][0]
            lineage = sorted(dag.ancestors(node_id)) if item.get("data_dependent") else []
            return {
                "earliest_identifiable_milestone_id": item["milestone_id"],
                "milestone_node_id": node_id,
                "lineage_applicable": bool(item.get("data_dependent")),
                "lineage_node_ids": lineage,
                "search_policy": (
                    "search earlier for an identifiable observation; never search later; "
                    "cross-lineage errors need a full-trajectory search"
                ),
                "is_root_cause": False,
            }
    return {
        "earliest_identifiable_milestone_id": None,
        "milestone_node_id": None,
        "lineage_applicable": False,
        "lineage_node_ids": [],
        "search_policy": "no failed critical milestone; full-trajectory search",
        "is_root_cause": False,
    }
