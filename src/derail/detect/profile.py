from __future__ import annotations

import statistics
from typing import Any, Dict, List, Mapping, Optional, Sequence

from derail.detect.latent_static import NodeHorizon
from derail.ir.model import as_fragment, dag_index
from derail.longhorizon.carry import decorative_carry_violations
from derail.longhorizon.dag import DagIndex

PROFILE_VERSION = "latent-profile-static/1.0"
CLASSES = ("required_next", "required_later", "incidental_only", "verifier_only", "silent")


def bucket_for(horizon: Optional[int], observability_class: str, buckets: Sequence[str]) -> str:
    if horizon is None:
        return "verifier_only" if "verifier_only" in buckets else "none"
    for label in buckets:
        if label == "verifier_only":
            continue
        if label.startswith(">="):
            if horizon >= int(label[2:]):
                return label
        elif "-" in label:
            low, high = label.split("-", 1)
            if int(low) <= horizon <= int(high):
                return label
        elif label.isdigit() and horizon == int(label):
            return label
    return "none"


def _independent_component_ratio(dag: DagIndex) -> float:
    nodes = list(dag.order)
    if not nodes:
        return 0.0
    seen: set = set()
    largest = 0
    for start in nodes:
        if start in seen:
            continue
        stack, size = [start], 0
        while stack:
            current = stack.pop()
            if current in seen:
                continue
            seen.add(current)
            size += 1
            stack.extend(dag.outgoing.get(current, ()))
            stack.extend(dag.incoming.get(current, ()))
        largest = max(largest, size)
    return round(1 - largest / len(nodes), 4)


def build_profile(
    task_ir: Mapping[str, Any],
    horizons: Mapping[str, NodeHorizon],
    buckets: Sequence[str],
    target_buckets: Sequence[str],
    carry_threshold: int = 3,
    second_lineage_depth: int = 2,
) -> Dict[str, Any]:
    dag = dag_index(task_ir)
    order = list(dag.order)
    class_counts = {c: 0 for c in CLASSES}
    class_counts["cross_app"] = 0
    values: List[int] = []
    node_buckets: Dict[str, str] = {}
    for node_id in order:
        h = horizons[node_id]
        class_counts[h.observability_class] = class_counts.get(h.observability_class, 0) + 1
        if h.cross_app:
            class_counts["cross_app"] += 1
        if h.latent_horizon_static is not None:
            values.append(h.latent_horizon_static)
        node_buckets[node_id] = bucket_for(h.latent_horizon_static, h.observability_class, buckets)
    first_r3 = next(
        (i for i, n in enumerate(order) if dag.nodes[n].get("reversibility_class") == "R3"), None
    )
    verifier_only_before_r3 = (
        sum(1 for n in order[:first_r3] if horizons[n].observability_class == "verifier_only")
        if first_r3 is not None
        else None
    )
    depth = max(dag.longest_depth().values(), default=0)
    carry = decorative_carry_violations(
        dag, threshold=carry_threshold, min_lineage_depth=second_lineage_depth
    )
    cross_or_verifier = sum(
        1
        for n in order
        if horizons[n].cross_app or horizons[n].observability_class == "verifier_only"
    )
    in_target = sum(1 for n in order if node_buckets[n] in target_buckets)
    return {
        "schema_version": "latent-profile/1.0",
        "task_id": str(task_ir["task_id"]),
        "mode": "static",
        "node_count": len(order),
        "class_counts": class_counts,
        "semantic_horizon_max": max(values) if values else None,
        "semantic_horizon_median": float(statistics.median(values)) if values else None,
        "cross_app_or_verifier_only_ratio": round(cross_or_verifier / len(order), 4)
        if order
        else 0.0,
        "verifier_only_before_first_r3": verifier_only_before_r3,
        "bucket": max(
            set(node_buckets.values()), key=lambda b: (list(node_buckets.values()).count(b), b)
        )
        if node_buckets
        else "none",
        "node_buckets": node_buckets,
        "nodes_in_target_bucket": in_target,
        "structural": {
            "dependency_depth": int(depth),
            "cross_app_dependency_count": sum(
                1
                for e in as_fragment(task_ir)["edges"]
                if dag.nodes[str(e["from"]["node_id"])]["app"]
                != dag.nodes[str(e["to"]["node_id"])]["app"]
            ),
            "independent_component_ratio": _independent_component_ratio(dag),
            "decorative_carry_violations": len(carry),
        },
        "profile_version": PROFILE_VERSION,
        "provenance": {"node_horizons": {n: horizons[n].to_dict() for n in order}},
    }
