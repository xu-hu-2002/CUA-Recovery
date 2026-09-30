"""Load, validate and index ``task-ir/1.0`` records.

``task-ir/1.0`` keeps the v0.2 fragment structure for nodes and edges, so the v0.2 DAG
utilities are reused rather than re-implemented: :func:`as_fragment` returns the
``task-fragment/0.1`` view of a Task IR and :func:`dag_index` feeds it to
``derail.longhorizon.dag.DagIndex``.

Entry points
------------
``load_task_ir(path, repository)``  read + JSON-schema validate + structural validate
``validate_task_ir(task_ir, repository)``
``as_fragment(task_ir)``            fragment view (schema_version task-fragment/0.1)
``dag_index(task_ir)``              DagIndex over the fragment view
``parse_entity_ref(ref)``           ``("row", table, rowid)`` | ``("derived", node, name)`` |
                                    ``("literal",)``
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Mapping, Tuple, Union

from derail.derived.schema import validate_schema
from derail.longhorizon.dag import DagIndex
from derail.synthesis.graph import validate_task_fragment

TASK_IR_SCHEMA_VERSION = "task-ir/1.0"
FRAGMENT_SCHEMA_VERSION = "task-fragment/0.1"


class TaskIRError(ValueError):
    pass


def as_fragment(task_ir: Mapping[str, Any]) -> Dict[str, Any]:
    """The ``task-fragment/0.1`` view: same node and edge objects, fragment envelope."""

    return {
        "schema_version": FRAGMENT_SCHEMA_VERSION,
        "fragment_id": str(task_ir["task_id"]),
        "nodes": list(task_ir["nodes"]),
        "edges": list(task_ir["edges"]),
    }


def find_cycle(task_ir: Mapping[str, Any]) -> List[str]:
    """Edge ids forming one cycle (empty when the graph is acyclic)."""

    outgoing: Dict[str, List[Tuple[str, str]]] = {}
    for edge in task_ir.get("edges", ()):
        outgoing.setdefault(str(edge["from"]["node_id"]), []).append(
            (str(edge["to"]["node_id"]), str(edge.get("edge_id", "?")))
        )
    state: Dict[str, int] = {}
    path: List[Tuple[str, str]] = []

    def visit(node: str) -> List[str]:
        state[node] = 1
        for nxt, edge_id in outgoing.get(node, ()):
            if state.get(nxt) == 1:
                start = [i for i, (n, _) in enumerate(path) if n == nxt]
                edges = [e for _, e in path]
                return (edges[start[0] + 1 :] if start else edges) + [edge_id]
            if state.get(nxt, 0) == 0:
                path.append((nxt, edge_id))
                found = visit(nxt)
                path.pop()
                if found:
                    return found
        state[node] = 2
        return []

    for node in sorted({str(n["node_id"]) for n in task_ir.get("nodes", ())}):
        if state.get(node, 0) == 0:
            path.clear()
            found = visit(node)
            if found:
                return found
    return []


def validate_task_ir(task_ir: Mapping[str, Any], repository: Union[str, Path]) -> None:
    """JSON-schema validation, then the v0.2 structural checks (ports, bindings, acyclicity),
    then the v1.0 read/write consistency checks."""

    if task_ir.get("schema_version") != TASK_IR_SCHEMA_VERSION:
        raise TaskIRError(
            "expected %s, got %r" % (TASK_IR_SCHEMA_VERSION, task_ir.get("schema_version"))
        )
    validate_schema(dict(task_ir), "task_ir.schema.json", Path(repository))
    cycle = find_cycle(task_ir)
    if cycle:
        raise TaskIRError("NOT_A_DAG: edges %s form a cycle" % " -> ".join(cycle))
    validate_task_fragment(as_fragment(task_ir))
    node_ids = {str(node["node_id"]) for node in task_ir["nodes"]}
    for node in task_ir["nodes"]:
        node_id = str(node["node_id"])
        if node["reversibility_class"] == "R0" and node["side_effects"]:
            raise TaskIRError("%s is R0 but carries side effects" % node_id)
        if node["writes"] and node["reversibility_class"] == "R0":
            raise TaskIRError("%s declares writes but is R0" % node_id)
        for write in node["writes"]:
            kind = parse_entity_ref(write["entity_ref"])
            if kind[0] == "derived" and kind[1] not in node_ids:
                raise TaskIRError("%s write references unknown node %s" % (node_id, kind[1]))
        for read in node["reads"]:
            kind = parse_entity_ref(read["entity_ref"])
            if kind[0] == "derived" and kind[1] not in node_ids:
                raise TaskIRError("%s read references unknown node %s" % (node_id, kind[1]))


def load_task_ir(path: Union[str, Path], repository: Union[str, Path]) -> Dict[str, Any]:
    task_ir = json.loads(Path(path).read_text(encoding="utf-8"))
    validate_task_ir(task_ir, repository)
    return task_ir


def dag_index(task_ir: Mapping[str, Any]) -> DagIndex:
    return DagIndex.from_fragment(as_fragment(task_ir))


def parse_entity_ref(ref: str) -> Tuple[str, ...]:
    """Split an ``entity_ref``.

    ``"events:530"``        -> ``("row", "events", "530")``
    ``"derived:n2:event"``  -> ``("derived", "n2", "event")``
    ``"literal"``           -> ``("literal",)``
    """

    if ref == "literal":
        return ("literal",)
    if ref.startswith("derived:"):
        _, node_id, name = ref.split(":", 2)
        return ("derived", node_id, name)
    table, _, rowid = ref.partition(":")
    if not rowid:
        raise TaskIRError("malformed entity_ref %r" % ref)
    return ("row", table, rowid)
