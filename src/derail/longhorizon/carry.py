from __future__ import annotations

from typing import Any, Dict, List

from derail.longhorizon.dag import DagIndex


def carry_distances(dag: DagIndex) -> Dict[str, int]:
    cache: Dict[str, Dict[str, int]] = {}
    result: Dict[str, int] = {}
    for edge in dag.value_edges():
        producer, consumer = dag.edge_endpoints(edge)
        if producer not in cache:
            cache[producer] = dag.longest_distances_from(producer)
        result[str(edge["edge_id"])] = cache[producer][consumer]
    return result


def decorative_carry_violations(
    dag: DagIndex, *, threshold: int = 3, min_lineage_depth: int = 2
) -> List[Dict[str, Any]]:
    if threshold < 1 or min_lineage_depth < 1:
        raise ValueError("threshold and min_lineage_depth must be positive")
    depth = dag.longest_depth()
    distances = carry_distances(dag)
    producers_by_consumer: Dict[str, List[str]] = {node_id: [] for node_id in dag.order}
    for edge in dag.value_edges():
        producer, consumer = dag.edge_endpoints(edge)
        producers_by_consumer[consumer].append(producer)

    violations: List[Dict[str, Any]] = []
    for edge in dag.value_edges():
        edge_id = str(edge["edge_id"])
        distance = distances[edge_id]
        if distance < threshold:
            continue
        producer, consumer = dag.edge_endpoints(edge)
        related = dag.ancestors(producer) | dag.descendants(producer) | {producer}
        justified = any(
            other not in related and depth[other] >= min_lineage_depth
            for other in producers_by_consumer[consumer]
        )
        if not justified:
            violations.append(
                {
                    "edge_id": edge_id,
                    "producer": producer,
                    "consumer": consumer,
                    "carry_distance": distance,
                    "reason": "no second lineage of depth >= %d feeds the consumer"
                    % min_lineage_depth,
                }
            )
    return violations
