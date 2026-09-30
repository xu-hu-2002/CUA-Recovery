"""Schema-derived typed compatibility between Task IR ports (execution doc v1.2 section 7.1).

An output port of IR ``A`` can feed an input port of IR ``B`` when

- the port types unify in the registry (``ValueTypeRegistry`` / v0.1 ``TypeSystem``), or
- both ports are grounded in the same ``(table, column)`` or in columns joined by a
  foreign-key path of the schema graph (generation relations only, D-015), or
- a registered deterministic converter maps one type to the other;

and the v0.2 pairwise checks hold (identity scope, time scope, cardinality, side-effect
conflicts).  Literal input ports are the graft targets: replacing a literal by an upstream
value is what turns two seed tasks into one long task.

``compat-edge/1.0`` records are emitted for every acceptable (output, input) pair.
"""

from __future__ import annotations

import re

from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from derail.longhorizon.types import ValueTypeRegistry
from derail.world.schema_graph import SchemaGraph

COMPAT_VERSION = "compat-edge/1.0"


_SQL_FROM = re.compile(r"\bFROM\s+([A-Za-z_][A-Za-z0-9_]*)", re.IGNORECASE)
_SQL_SELECT = re.compile(r"^\s*SELECT\s+(?:DISTINCT\s+)?(.*?)\s+FROM\b", re.IGNORECASE | re.DOTALL)
_BARE_COLUMN = re.compile(r"^(?:[A-Za-z_][A-Za-z0-9_]*\.)?([A-Za-z_][A-Za-z0-9_]*)$")


def _split_top_level(text: str) -> List[str]:
    items, depth, current = [], 0, []
    for char in text:
        if char == "(":
            depth += 1
        elif char == ")":
            depth -= 1
        if char == "," and depth == 0:
            items.append("".join(current))
            current = []
        else:
            current.append(char)
    items.append("".join(current))
    return [i.strip() for i in items if i.strip()]


def sql_scalar_cell(query: str, app: str) -> Optional[Tuple[str, str]]:
    """``(app.table, column)`` when the query projects exactly one bare column (optionally
    aliased) from one table; aggregates and expressions carry no cell."""

    select, source = _SQL_SELECT.search(query or ""), _SQL_FROM.search(query or "")
    if not select or not source:
        return None
    items = _split_top_level(select.group(1))
    if len(items) != 1:
        return None
    item = re.sub(r"\s+AS\s+[A-Za-z_][A-Za-z0-9_]*$", "", items[0], flags=re.IGNORECASE)
    match = _BARE_COLUMN.match(item.strip())
    if not match or item.strip() == "*":
        return None
    return "%s.%s" % (app, source.group(1)), match.group(1)


def sql_param_cell(query: str, app: str, param: str) -> Optional[Tuple[str, str]]:
    """``(app.table, column)`` of the column a query compares with ``:param``."""

    source = _SQL_FROM.search(query or "")
    match = re.search(
        r"(?:[A-Za-z_][A-Za-z0-9_]*\s*\(\s*)?"  # optional wrapping function, lower(col)
        r"(?:[A-Za-z_][A-Za-z0-9_]*\.)?([A-Za-z_][A-Za-z0-9_]*)\s*\)?\s*"
        r"(?:=|LIKE|IN\s*\()\s*(?:'%%'\s*\|\|\s*)?(?:lower\s*\(\s*)?:%s\b" % re.escape(param),
        query or "",
        re.IGNORECASE,
    )
    if not source or not match:
        return None
    return "%s.%s" % (app, source.group(1)), match.group(1)


def _grounding_cell(
    task_ir: Mapping[str, Any], node_id: str, name: str
) -> Optional[Tuple[str, str]]:
    """``(table, column)`` a produce refers to: the read/write entry whose entity_ref is
    derived:<node>:<name>, else a read of the node with a matching column name, else the single
    bare column its SQL derivation projects."""

    ref = "derived:%s:%s" % (node_id, name)
    node = next((n for n in task_ir["nodes"] if n["node_id"] == node_id), None)
    if node is None:
        return None
    for entry in list(node.get("reads", ())) + list(node.get("writes", ())):
        if entry.get("entity_ref") == ref:
            return str(entry["table"]), "id"
    for entry in node.get("reads", ()):
        if str(entry["column"]) == name:
            return str(entry["table"]), str(entry["column"])
    for produce in node.get("produces", ()):
        derivation = produce.get("derivation") or {}
        if str(produce.get("name")) == name and derivation.get("kind") == "sql":
            return sql_scalar_cell(
                str(derivation.get("query") or derivation.get("sql") or ""), str(node["app"])
            )
    return None


def _input_cell(node: Mapping[str, Any], port_id: str) -> Optional[Tuple[str, str]]:
    """``(table, column)`` a literal input is compared against: a read with the port's name,
    else the column a SQL derivation of the node compares with ``:<port_id>``."""

    for entry in node.get("reads", ()):
        if str(entry.get("column")) == port_id:
            return str(entry["table"]), str(entry["column"])
    for produce in node.get("produces", ()):
        derivation = produce.get("derivation") or {}
        if derivation.get("kind") == "sql":
            cell = sql_param_cell(
                str(derivation.get("query") or derivation.get("sql") or ""),
                str(node["app"]),
                port_id,
            )
            if cell:
                return cell
    return None


@dataclass(frozen=True)
class PortRef:
    task_id: str
    node_id: str
    port_id: str
    type: str
    cardinality: str
    cell: Optional[Tuple[str, str]]
    app: str


def output_ports(task_ir: Mapping[str, Any]) -> List[PortRef]:
    out = []
    for node in task_ir["nodes"]:
        for port in node.get("outputs", ()):
            out.append(
                PortRef(
                    str(task_ir["task_id"]),
                    str(node["node_id"]),
                    str(port["port_id"]),
                    str(port.get("type", "")),
                    str(port.get("cardinality", "one")),
                    _grounding_cell(task_ir, str(node["node_id"]), str(port["port_id"])),
                    str(node["app"]),
                )
            )
    return out


def literal_input_ports(
    task_ir: Mapping[str, Any], excluded_literals: Sequence[Any] = ()
) -> List[PortRef]:
    """Input ports that carry an instruction constant: the graft targets.  Persona identity
    literals (the signed-in user's name / e-mail) are excluded: they are not task inputs."""

    excluded = {str(v).strip().lower() for v in excluded_literals}
    out = []
    for node in task_ir["nodes"]:
        for port in node.get("inputs", ()):
            if "literal" not in port or str(port["literal"]).strip().lower() in excluded:
                continue
            out.append(
                PortRef(
                    str(task_ir["task_id"]),
                    str(node["node_id"]),
                    str(port["port_id"]),
                    str(port.get("type", "")),
                    str(port.get("cardinality", "one")),
                    _input_cell(node, str(port["port_id"])),
                    str(node["app"]),
                )
            )
    return out


@dataclass(frozen=True)
class CompatConfig:
    max_fk_hops: int = 2
    allow_cross_app: bool = True
    min_score: float = 0.5
    # Types whose name alone says nothing about meaning (a count is not a number of months):
    # a same-type match on them scores ``generic_same_type_score`` unless a cell or foreign
    # key path backs it (D-039).
    generic_types: Tuple[str, ...] = ()
    generic_same_type_score: float = 0.4
    # Literals that never are graft targets (persona identity, D-009).
    excluded_literals: Tuple[Any, ...] = ()
    # A port whose schema column cannot be identified is not eligible for composition
    # (paper appendix B, typed compatibility).
    require_cell: bool = True


def port_compatibility(
    source: PortRef,
    target: PortRef,
    registry: ValueTypeRegistry,
    graph: Optional[SchemaGraph],
    config: CompatConfig,
) -> Optional[Dict[str, Any]]:
    """Score and reason when ``source`` may feed ``target``; None when it may not."""

    if source.cardinality != target.cardinality and not (
        source.cardinality == "one" and target.cardinality == "optional"
    ):
        return None
    if config.require_cell and (source.cell is None or target.cell is None):
        return None
    reasons: List[str] = []
    score = 0.0
    generic = {t.lower() for t in config.generic_types}
    typed = False
    if source.type and source.type == target.type:
        reasons.append("same_type")
        typed = True
        score = config.generic_same_type_score if source.type.lower() in generic else 1.0
    elif source.type and target.type:
        resolved = registry.resolve(source.type)
        if resolved == registry.resolve(target.type):
            reasons.append("registry_unify")
            typed = True
            unified_name = str(resolved.canonical or "").lower()
            score = config.generic_same_type_score if unified_name in generic else 0.9
    if source.cell and target.cell and graph is not None:
        if source.cell == target.cell:
            # Same table and column: the strongest evidence, whatever the type names say.
            reasons.append("same_cell")
            score = max(score, 1.0)
        elif typed and graph.foreign_key_path(source.cell[0], target.cell[0], config.max_fk_hops):
            # A key path between the tables backs a type match (also a generic one); a path
            # alone (an id fed into a name port) is not evidence.
            reasons.append("foreign_key_path")
            score = max(score, 0.8)
    if not reasons:
        return None
    if source.app != target.app and not config.allow_cross_app:
        return None
    if score < config.min_score:
        return None
    return {
        "schema_version": COMPAT_VERSION,
        "from": {
            "task_id": source.task_id,
            "node_id": source.node_id,
            "port_id": source.port_id,
            "type": source.type,
        },
        "to": {
            "task_id": target.task_id,
            "node_id": target.node_id,
            "port_id": target.port_id,
            "type": target.type,
        },
        "cross_app": source.app != target.app,
        "reasons": reasons,
        "score": round(score, 3),
    }


def build_compat_index(
    irs: Sequence[Mapping[str, Any]],
    registry: ValueTypeRegistry,
    graph: Optional[SchemaGraph] = None,
    config: Optional[CompatConfig] = None,
) -> List[Dict[str, Any]]:
    """All acceptable (output of IR A -> literal input of IR B, A != B) edges."""

    config = config or CompatConfig()
    outputs = [(ir, p) for ir in irs for p in output_ports(ir)]
    inputs = [(ir, p) for ir in irs for p in literal_input_ports(ir, config.excluded_literals)]
    edges = []
    for _, source in outputs:
        for _, target in inputs:
            if source.task_id == target.task_id:
                continue
            edge = port_compatibility(source, target, registry, graph, config)
            if edge:
                edges.append(edge)
    return edges
