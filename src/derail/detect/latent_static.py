"""Static latent horizon (execution doc v1.2 sections 2.2 and 6.2, fact-level per v1.1).

For a node ``i`` the *contamination set* ``C_i`` holds the facts an error at ``i`` would
corrupt: the facts ``i`` itself obtained (``reads(i)``: re-reading one of them later exposes a
contradiction with the carried value), the cells written by ``i`` and by every node whose
value derives from ``i`` along data edges.  The static latent horizon of ``i`` is the smallest
number of semantic steps to a downstream node ``j`` whose required reads intersect ``C_i``;
if none exists but a verifier depends on ``i``'s lineage the class is ``verifier_only``,
otherwise ``silent``.

Facts are compared at the granularity the IR gives: ``(table, column, entity)`` where entity
is a concrete row id, ``<table>:*`` (any row: intersects every entity of that table and
column) or ``derived:<node>:<name>`` (the row a node resolved; resolved to concrete ids when a
gold lineage is supplied, else treated like the derived token).

Entry points
------------
``static_latent_horizons(task_ir, gold_lineage=None) -> Dict[node_id, NodeHorizon]``
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, FrozenSet, List, Mapping, Optional, Set, Tuple

from derail.ir.model import dag_index

Fact = Tuple[str, str, str]  # (app-qualified table, column, entity)
_V_REF = re.compile(r'V\[\s*["\']([A-Za-z0-9_-]+)["\']\s*\]')


@dataclass
class NodeHorizon:
    node_id: str
    latent_horizon_static: Optional[int]
    observability_class: str  # required_next | required_later | verifier_only | silent
    static_class: str  # immediate | same_app_later | cross_app | verifier_only
    cross_app: bool
    visible_node_id: Optional[str]
    contamination: List[Fact] = field(default_factory=list)
    witness_facts: List[Fact] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "node_id": self.node_id,
            "latent_horizon_static": self.latent_horizon_static,
            "observability_class": self.observability_class,
            "static_class": self.static_class,
            "cross_app": self.cross_app,
            "visible_node_id": self.visible_node_id,
            "contamination_size": len(self.contamination),
            "witness_facts": [list(f) for f in self.witness_facts],
        }


def _entity_sets(
    gold_lineage: Optional[Mapping[str, Any]],
) -> Dict[Tuple[str, str, str, str], List[str]]:
    """``(node, table, column, entity_ref) -> concrete entity ids`` from ``resolved_reads``."""

    out: Dict[Tuple[str, str, str, str], List[str]] = {}
    for read in (gold_lineage or {}).get("resolved_reads", ()):
        if read.get("entity_set"):
            table = str(read["table"]).split(".", 1)[1]
            out[(read["node_id"], read["table"], read["column"], read["entity_ref"])] = [
                "%s:%s" % (table, e) for e in read["entity_set"]
            ]
    return out


def _facts(
    node: Mapping[str, Any], kind: str, resolved: Mapping[Tuple[str, str, str, str], List[str]]
) -> Set[Fact]:
    facts: Set[Fact] = set()
    for ref in node.get(kind, ()):
        table, column, entity = str(ref["table"]), str(ref["column"]), str(ref["entity_ref"])
        key = (str(node["node_id"]), table, column, entity)
        if key in resolved:
            facts.update((table, column, e) for e in resolved[key])
        else:
            facts.add((table, column, entity))
    return facts


def _path_length(dag: Any, source: str, target: str) -> Optional[int]:
    """Shortest number of edges from ``source`` to ``target`` (None when no path)."""

    frontier, seen, steps = [source], {source}, 0
    while frontier:
        steps += 1
        next_frontier = []
        for node in frontier:
            for nxt in dag.outgoing.get(node, ()):
                if nxt == target:
                    return steps
                if nxt not in seen:
                    seen.add(nxt)
                    next_frontier.append(nxt)
        frontier = next_frontier
    return None


def facts_intersect(a: Fact, b: Fact) -> bool:
    """Same table and column, and the entities are the same, or one of them is a wildcard."""

    if a[0] != b[0] or a[1] != b[1]:
        return False
    if a[2] == b[2]:
        return True
    return a[2].endswith(":*") or b[2].endswith(":*")


def _verifier_depends_on(
    node: Mapping[str, Any], lineage: FrozenSet[str], task_ir: Mapping[str, Any]
) -> bool:
    """A sql verifier on a lineage node, or a derived verifier naming a lineage node."""

    verifier = node.get("verifier")
    if not verifier:
        return False
    if str(node["node_id"]) in lineage:
        return True
    predicate = str(verifier.get("predicate", ""))
    if verifier.get("kind") == "derived":
        return any(ref in lineage for ref in _V_REF.findall(predicate))
    return False


def static_latent_horizons(
    task_ir: Mapping[str, Any], gold_lineage: Optional[Mapping[str, Any]] = None
) -> Dict[str, NodeHorizon]:
    dag = dag_index(task_ir)
    order = list(dag.order)
    position = {node_id: index for index, node_id in enumerate(order)}
    nodes = dag.nodes
    resolved = _entity_sets(gold_lineage)
    reads = {n: _facts(nodes[n], "reads", resolved) for n in order}
    writes = {n: _facts(nodes[n], "writes", resolved) for n in order}
    final = task_ir.get("final_verifier") or {}
    final_parts = final.get("parts", ()) if final.get("kind") == "all_of" else [final]
    final_refs = set()
    for part in final_parts:
        if part.get("kind") == "derived":
            final_refs |= set(_V_REF.findall(str(part.get("predicate", ""))))
    out: Dict[str, NodeHorizon] = {}
    for node_id in order:
        lineage: Set[str] = set(dag.descendants(node_id))
        lineage_all = frozenset(lineage | {node_id})
        contamination: Set[Fact] = set(reads[node_id]) | set(writes[node_id])
        for other in lineage:
            contamination |= writes[other]
        witness: Optional[Tuple[str, List[Fact]]] = None
        for other in order[position[node_id] + 1 :]:
            hits = [f for f in reads[other] if any(facts_intersect(f, c) for c in contamination)]
            if hits:
                witness = (other, hits)
                break
        app = str(nodes[node_id]["app"])
        if witness is not None:
            visible, hits = witness
            # Semantic steps = shortest dependency path when the witness depends on i; a
            # witness with no dependency path (an unrelated re-read) counts topological distance.
            distance = _path_length(dag, node_id, visible) or (
                position[visible] - position[node_id]
            )
            cross = str(nodes[visible]["app"]) != app
            adjacent = distance == 1
            out[node_id] = NodeHorizon(
                node_id=node_id,
                latent_horizon_static=distance,
                observability_class="required_next" if adjacent else "required_later",
                static_class="cross_app"
                if cross
                else ("immediate" if adjacent else "same_app_later"),
                cross_app=cross,
                visible_node_id=visible,
                contamination=sorted(contamination),
                witness_facts=hits,
            )
            continue
        verifier_only = bool(final) and (
            final.get("kind") != "derived" or bool(final_refs & lineage_all)
        )
        verifier_only = verifier_only or any(
            _verifier_depends_on(nodes[o], lineage_all, task_ir) for o in order[position[node_id] :]
        )
        out[node_id] = NodeHorizon(
            node_id=node_id,
            latent_horizon_static=None,
            observability_class="verifier_only" if verifier_only else "silent",
            static_class="verifier_only",
            cross_app=False,
            visible_node_id=None,
            contamination=sorted(contamination),
            witness_facts=[],
        )
    return out
