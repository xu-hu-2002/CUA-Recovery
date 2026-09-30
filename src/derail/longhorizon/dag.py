"""Small immutable DAG index over a validated task fragment.

``derail.synthesis.graph`` validates fragments; this index exposes the derived structure
(adjacency, topological order, depths, ancestry) that several v0.2 metrics need, so each metric
module stays a pure function over the index instead of re-deriving the graph.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Any, Dict, FrozenSet, Iterable, List, Mapping, Optional, Sequence, Tuple

from derail.synthesis.graph import validate_task_fragment


def _endpoint(edge: Mapping[str, Any], side: str) -> Tuple[str, str]:
    return str(edge[side]["node_id"]), str(edge[side]["port_id"])


@dataclass(frozen=True)
class DagIndex:
    nodes: Mapping[str, Mapping[str, Any]]
    edges: Tuple[Mapping[str, Any], ...]
    outgoing: Mapping[str, Tuple[str, ...]]
    incoming: Mapping[str, Tuple[str, ...]]
    order: Tuple[str, ...]

    @classmethod
    def from_fragment(cls, fragment: Mapping[str, Any]) -> "DagIndex":
        validate_task_fragment(fragment)
        nodes = {str(node["node_id"]): node for node in fragment["nodes"]}
        outgoing: Dict[str, List[str]] = {node_id: [] for node_id in nodes}
        incoming: Dict[str, List[str]] = {node_id: [] for node_id in nodes}
        for edge in fragment["edges"]:
            source, _ = _endpoint(edge, "from")
            target, _ = _endpoint(edge, "to")
            outgoing[source].append(target)
            incoming[target].append(source)
        indegree = {node_id: len(set(sources)) for node_id, sources in incoming.items()}
        ready = deque(sorted(node_id for node_id, degree in indegree.items() if degree == 0))
        order: List[str] = []
        while ready:
            node_id = ready.popleft()
            order.append(node_id)
            for target in sorted(set(outgoing[node_id])):
                indegree[target] -= 1
                if indegree[target] == 0:
                    ready.append(target)
        return cls(
            nodes=nodes,
            edges=tuple(fragment["edges"]),
            outgoing={key: tuple(sorted(set(value))) for key, value in outgoing.items()},
            incoming={key: tuple(sorted(set(value))) for key, value in incoming.items()},
            order=tuple(order),
        )

    @property
    def sources(self) -> Tuple[str, ...]:
        return tuple(node_id for node_id in self.order if not self.incoming[node_id])

    def edge_endpoints(self, edge: Mapping[str, Any]) -> Tuple[str, str]:
        return _endpoint(edge, "from")[0], _endpoint(edge, "to")[0]

    def longest_depth(self) -> Dict[str, int]:
        """Longest path from any source, counted in nodes (sources have depth 1)."""

        depth = {node_id: 1 for node_id in self.order}
        for node_id in self.order:
            for target in self.outgoing[node_id]:
                depth[target] = max(depth[target], depth[node_id] + 1)
        return depth

    def shortest_depth(self) -> Dict[str, int]:
        """Shortest path from any source, counted in nodes (sources have depth 1)."""

        depth: Dict[str, Optional[int]] = {node_id: None for node_id in self.order}
        queue: deque = deque()
        for source in self.sources:
            depth[source] = 1
            queue.append(source)
        while queue:
            node_id = queue.popleft()
            current = depth[node_id]
            assert current is not None
            for target in self.outgoing[node_id]:
                if depth[target] is None or current + 1 < int(depth[target]):
                    depth[target] = current + 1
                    queue.append(target)
        return {node_id: int(value) for node_id, value in depth.items() if value is not None}

    def longest_distances_from(self, source: str) -> Dict[str, int]:
        """Longest path length in edges from ``source``; unreachable nodes are omitted."""

        distance = {node_id: -1 for node_id in self.order}
        distance[source] = 0
        for node_id in self.order:
            if distance[node_id] < 0:
                continue
            for target in self.outgoing[node_id]:
                distance[target] = max(distance[target], distance[node_id] + 1)
        return {node_id: value for node_id, value in distance.items() if value >= 0}

    def _closure(self, start: str, neighbours: Mapping[str, Sequence[str]]) -> FrozenSet[str]:
        seen = set()
        stack = list(neighbours[start])
        while stack:
            node_id = stack.pop()
            if node_id in seen:
                continue
            seen.add(node_id)
            stack.extend(neighbours[node_id])
        return frozenset(seen)

    def ancestors(self, node_id: str) -> FrozenSet[str]:
        return self._closure(node_id, self.incoming)

    def descendants(self, node_id: str) -> FrozenSet[str]:
        return self._closure(node_id, self.outgoing)

    def value_edges(self) -> Iterable[Mapping[str, Any]]:
        """Edges that carry a value or state (control edges are excluded)."""

        return (edge for edge in self.edges if edge.get("kind") != "control_dependency")
