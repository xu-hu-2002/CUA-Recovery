from __future__ import annotations

import hashlib
import json
import re
import sqlite3
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

import yaml

from derail.ir.model import TaskIRError, validate_task_ir
from derail.ir.validate import validate_grounding
from derail.longhorizon.extraction import (
    ExtractionParseError,
    ExtractorConfig,
    LLMClient,
    LLMResponse,
    render_prompt,
)
from derail.longhorizon.ontology import Ontology
from derail.longhorizon.types import ValueTypeRegistry
from derail.longhorizon.world import AppAliases
from derail.synthesis.graph import SynthesisValidationError
from derail.world.facts import qualified_table
from derail.world.files import FileInventory
from derail.world.schema_graph import SchemaGraph

READ_ONLY_OPS = frozenset(
    {"retrieve", "resolve", "filter", "compare", "aggregate", "verify", "decide", "confirm"}
)
_DEFAULT_EFFECT = {
    "create": "create_record",
    "modify": "modify_record",
    "delete": "delete_record",
    "communicate": "send_message",
}
_NODE_SYNONYMS = {
    "operation": "op",
    "application": "app",
    "goal": "semantic_goal",
    "database": "app",
}
_OP_SYNONYMS = {
    "place_order": "create",
    "send": "communicate",
    "send_message": "communicate",
    "write": "create",
    "update": "modify",
    "edit": "modify",
    "remove": "delete",
    "cancel": "modify",
    "read": "retrieve",
    "lookup": "resolve",
    "search": "retrieve",
    "compute": "decide",
    "calculate": "aggregate",
    "check": "verify",
    "confirm_state": "confirm",
}
_KIND_SYNONYMS = {
    "derived": "expr",
    "expression": "expr",
    "query": "sql",
    "select": "sql",
    "constant": "literal",
    "read_file": "file",
    "insert": "create",
}
_PORT_SYNONYMS = {"name": "port_id", "port": "port_id", "value": "literal"}


@dataclass(frozen=True)
class V1ExtractorConfig:
    base: ExtractorConfig
    schema_graph_path: Path
    database_dir_env: str
    world_id: str
    reference_time: str
    app_databases: Mapping[str, Optional[str]]
    sample_rows_per_table: int
    max_columns_per_table: int
    max_cell_chars: int
    hide_tables: Tuple[str, ...]
    persona_literals: Tuple[Any, ...] = ()
    file_inventory_path: Optional[Path] = None

    @classmethod
    def from_yaml(cls, path: Path, repo_root: Path) -> "V1ExtractorConfig":
        base = ExtractorConfig.from_yaml(path, repo_root)
        raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
        v1 = raw.get("v1") or {}
        prompt = v1.get("prompt") or {}
        literals: List[Any] = []
        if v1.get("persona_file"):
            persona_path = repo_root / v1["persona_file"]
            if persona_path.is_file():
                persona = json.loads(persona_path.read_text(encoding="utf-8"))
                for dotted in v1.get("persona_literal_fields", ()):
                    value: Any = persona
                    for part in str(dotted).split("."):
                        value = value.get(part) if isinstance(value, Mapping) else None
                    if value is not None:
                        literals.append(value)
        return cls(
            base=base,
            schema_graph_path=repo_root / v1["schema_graph"],
            database_dir_env=str(v1["database_dir_env"]),
            world_id=str(v1["world_id"]),
            reference_time=str(v1["reference_time"]),
            app_databases=dict(v1.get("app_databases") or {}),
            sample_rows_per_table=int(prompt.get("sample_rows_per_table", 2)),
            max_columns_per_table=int(prompt.get("max_columns_per_table", 40)),
            max_cell_chars=int(prompt.get("max_cell_chars", 60)),
            hide_tables=tuple(prompt.get("hide_tables", ())),
            persona_literals=tuple(literals),
            file_inventory_path=(repo_root / v1["file_inventory"])
            if v1.get("file_inventory")
            else None,
        )

    def databases_for(self, app_ids: Sequence[str]) -> List[str]:
        out = []
        for app in app_ids:
            stem = self.app_databases.get(app)
            if stem and stem not in out:
                out.append(str(stem))
        return out


def sample_rows(
    db_path: Path, tables: Sequence[str], per_table: int, max_cell_chars: int
) -> Dict[str, List[Dict[str, Any]]]:
    conn = sqlite3.connect("file:%s?mode=ro" % db_path, uri=True)
    conn.row_factory = sqlite3.Row
    out: Dict[str, List[Dict[str, Any]]] = {}
    try:
        for table in tables:
            try:
                rows = conn.execute(
                    'SELECT * FROM "%s" ORDER BY rowid LIMIT ?' % table, (per_table,)
                ).fetchall()
            except sqlite3.OperationalError:
                rows = conn.execute('SELECT * FROM "%s" LIMIT ?' % table, (per_table,)).fetchall()
            cleaned = []
            for row in rows:
                record = {}
                for key in row.keys():
                    value = row[key]
                    if isinstance(value, (bytes, bytearray)):
                        value = "<blob %d bytes>" % len(value)
                    elif isinstance(value, str) and len(value) > max_cell_chars:
                        value = value[: max_cell_chars - 1] + "…"
                    record[key] = value
                cleaned.append(record)
            out[table] = cleaned
    finally:
        conn.close()
    return out


def render_schemas(
    schema_graph: SchemaGraph,
    databases: Sequence[str],
    samples: Mapping[str, Mapping[str, List[Dict[str, Any]]]],
    config: V1ExtractorConfig,
) -> str:
    lines: List[str] = []
    for db in databases:
        tables = [t for t in schema_graph.tables() if t.startswith(db + ".")]
        if not tables:
            lines.append("## %s: (no database; observed through files)" % db)
            continue
        lines.append("## %s" % db)
        for table in tables:
            name = table.split(".", 1)[1]
            if name in config.hide_tables:
                continue
            record = schema_graph.record_for(table)
            columns = record["columns"][: config.max_columns_per_table]
            cols = ", ".join(
                "%s %s%s" % (c["name"], c["type"] or "ANY", " PK" if c["primary_key"] else "")
                for c in columns
            )
            more = "" if len(record["columns"]) <= config.max_columns_per_table else ", …"
            lines.append("- %s (%d rows): %s%s" % (name, record["row_count"], cols, more))
            for relation in schema_graph.references(table):
                if relation["to_table"]:
                    lines.append(
                        "    %s -> %s.%s (%s)"
                        % (
                            relation["from_column"],
                            relation["to_table"],
                            relation["to_column"],
                            relation["kind"],
                        )
                    )
            for row in samples.get(db, {}).get(name, []):
                lines.append("    sample: %s" % json.dumps(row, ensure_ascii=False))
    return "\n".join(lines) or "(no schemas)"


def build_prompt_fields_v1(
    task: Mapping[str, Any],
    config: V1ExtractorConfig,
    schema_graph: SchemaGraph,
    databases: Sequence[str],
    samples: Mapping[str, Mapping[str, List[Dict[str, Any]]]],
    ontology: Ontology,
    registry: Optional[ValueTypeRegistry],
    inventory: Optional[FileInventory] = None,
) -> Dict[str, str]:
    rubrics = task.get("grading", {}).get("rubrics", [])
    return {
        "task_id": str(task["id"]),
        "category": str(task.get("category", "")),
        "apps": ", ".join(databases) or "(none)",
        "reference_time": config.reference_time,
        "instruction": str(task["instruction"]),
        "rubrics": "\n".join(
            "%d. (weight %s) %s" % (i, r.get("weight"), r.get("criterion"))
            for i, r in enumerate(rubrics, 1)
        )
        or "(none)",
        "schemas": render_schemas(schema_graph, databases, samples, config),
        "files": "\n".join(inventory.summary_lines()) if inventory else "(no file inventory)",
        "operations": ", ".join(sorted(ontology.operations)),
        "value_types": registry.vocabulary_text() if registry else "(no registry supplied)",
        "reversibility_classes": ", ".join(ontology.reversibility_classes),
        "effect_types": ", ".join(sorted(ontology.effect_types)),
        "commit_scopes": ", ".join(sorted(ontology.commit_scopes)),
    }


def parse_task_ir_reply(text: str) -> Dict[str, Any]:
    fenced = re.search(r"```(?:json)?\s*(\{.*\})\s*```", text, re.DOTALL)
    candidate = fenced.group(1) if fenced else text[text.find("{") : text.rfind("}") + 1]
    if not candidate.strip():
        raise ExtractionParseError("reply contains no JSON object")
    try:
        parsed = json.loads(candidate)
    except json.JSONDecodeError as exc:
        raise ExtractionParseError("reply is not valid JSON: %s" % exc) from exc
    if not isinstance(parsed, dict):
        raise ExtractionParseError("reply must be a JSON object")
    if "task_ir" not in parsed and "nodes" in parsed:
        parsed = {"task_ir": parsed}
    if "task_ir" not in parsed or "nodes" not in parsed["task_ir"]:
        raise ExtractionParseError("reply must contain task_ir.nodes")
    return parsed


_OBSERVABILITY = (
    "environment_state",
    "auditable_tool_output",
    "derived_from_observable",
    "unobservable",
)


def _coerce_observability(verifier: Dict[str, Any]) -> None:
    value = verifier.get("observability")
    if value in _OBSERVABILITY:
        return
    if isinstance(value, str) and value.strip():
        verifier["note"] = value.strip()
    verifier["observability"] = (
        "derived_from_observable" if verifier.get("kind") == "derived" else "environment_state"
    )


def _coerce_equivalents(read: Dict[str, Any], app: str, node_id: str) -> None:
    out = []
    for item in read.get("equivalent_sources") or []:
        if isinstance(item, Mapping):
            item = dict(item)
            item["table"] = qualified_table(app, str(item.get("table", "")))
            item.setdefault("column", "content" if item["table"] == "files.documents" else "rowid")
            item["entity_ref"] = _strip_app(item.get("entity_ref", read.get("entity_ref", "")), app)
            out.append(item)
            continue
        text = str(item)
        if text.startswith("file:"):
            out.append({"table": "files.documents", "column": "content", "entity_ref": text})
        elif text.count(".") >= 2:
            parts = text.split(".")
            out.append(
                {
                    "table": "%s.%s" % (parts[0], parts[1]),
                    "column": parts[2],
                    "entity_ref": _strip_app(read.get("entity_ref", ""), app),
                }
            )
    if out:
        read["equivalent_sources"] = out
    else:
        read.pop("equivalent_sources", None)


def _strip_app(entity_ref: Any, app: str) -> str:
    text = str(entity_ref)
    if text.startswith("derived:") or text == "literal":
        return text
    if "::" in text:
        return (
            "derived:%s:%s" % (_strip_app.current_node, text.split("::", 1)[1])
            if _strip_app.current_node
            else text
        )
    head, sep, tail = text.partition(":")
    if "." in head:
        head = head.rsplit(".", 1)[1]
    return head + sep + tail if sep else text


def _rename(mapping: Dict[str, Any], synonyms: Mapping[str, str]) -> None:
    for old, new in synonyms.items():
        if old in mapping and new not in mapping:
            mapping[new] = mapping.pop(old)


def _resolve_app(app: Any, config: V1ExtractorConfig, aliases: AppAliases) -> str:
    text = str(app or "").strip()
    if not text:
        return "unspecified"
    stems = {v for v in config.app_databases.values() if v}
    if text in stems:
        return text
    canonical = aliases.resolve(text)
    stem = config.app_databases.get(canonical)
    return str(stem) if stem else canonical


def _default_side_effect(
    node_id: str, effect_type: str, ontology: Ontology, index: int, target_ref: str
) -> Dict[str, Any]:
    klass = (
        ontology.default_class_for(effect_type) if effect_type in ontology.effect_types else "R1"
    )
    return {
        "effect_id": "fx-%s-%d" % (node_id, index),
        "effect_type": effect_type,
        "target_ref": target_ref,
        "reversibility_class": klass,
        "commit_scope": "shared_environment" if klass == "R3" else "local_app",
        "compensation_available": klass != "R3",
        "compensating_effect_type": None if klass == "R3" else effect_type,
        "compensation_verifier_id": None,
        "checkpoint_required": klass in ontology.checkpoint_required_classes,
        "checkpoint_id": None,
        "restore_verifier_id": None,
    }


def repair_edges(body: Dict[str, Any]) -> List[Dict[str, Any]]:
    nodes = {str(n.get("node_id")): n for n in body.get("nodes", ())}
    repairs: List[Dict[str, Any]] = []
    edges = list(body.get("edges") or [])
    producers = {
        (str(e["to"]["node_id"]), str(e["to"]["port_id"])): e
        for e in edges
        if isinstance(e.get("from"), Mapping) and isinstance(e.get("to"), Mapping)
    }
    kept: List[Dict[str, Any]] = []
    for edge in edges:
        src, dst = edge.get("from") or {}, edge.get("to") or {}
        src_node, src_port = str(src.get("node_id")), str(src.get("port_id"))
        dst_node, dst_port = str(dst.get("node_id")), str(dst.get("port_id"))
        node = nodes.get(src_node)
        if node is None:
            kept.append(edge)
            continue
        produce_names = {str(p.get("name")) for p in node.get("produces") or []}
        output_ids = {str(p.get("port_id", p.get("name"))) for p in node.get("outputs") or []}
        if src_node == dst_node:
            node["inputs"] = [
                i
                for i in node.get("inputs") or []
                if str(i.get("port_id", i.get("name"))) != dst_port
            ]
            repairs.append(
                {
                    "edge_id": edge.get("edge_id"),
                    "repair": "dropped_self_edge",
                    "node_id": src_node,
                    "port_id": dst_port,
                }
            )
            continue
        if src_port in produce_names or src_port in output_ids:
            kept.append(edge)
            continue
        inputs = {str(i.get("port_id", i.get("name"))): i for i in node.get("inputs") or []}
        source_port = inputs.get(src_port)
        if source_port is None:
            kept.append(edge)
            continue
        target = nodes.get(dst_node)
        if "literal" in source_port:
            if target is not None:
                for port in target.get("inputs") or []:
                    if str(port.get("port_id", port.get("name"))) == dst_port:
                        port["literal"] = source_port["literal"]
                        port["source"] = "literal"
                        port["grounding"] = "literal"
            repairs.append(
                {
                    "edge_id": edge.get("edge_id"),
                    "repair": "literal_copied",
                    "node_id": dst_node,
                    "port_id": dst_port,
                    "value": source_port["literal"],
                }
            )
            continue
        upstream = producers.get((src_node, src_port))
        if upstream is not None and str(upstream["from"]["node_id"]) != src_node:
            rewired = dict(edge)
            rewired["from"] = {
                "node_id": str(upstream["from"]["node_id"]),
                "port_id": str(upstream["from"]["port_id"]),
            }
            kept.append(rewired)
            repairs.append(
                {
                    "edge_id": edge.get("edge_id"),
                    "repair": "rewired_to_producer",
                    "from": rewired["from"],
                }
            )
            continue
        kept.append(edge)
    body["edges"] = kept
    return repairs


_strip_app.current_node = None  # type: ignore[attr-defined]


def normalize_task_ir(
    raw: Mapping[str, Any],
    task: Mapping[str, Any],
    config: V1ExtractorConfig,
    ontology: Ontology,
    aliases: AppAliases,
) -> Dict[str, Any]:
    body = json.loads(json.dumps(raw["task_ir"]))
    repairs = repair_edges(body)
    nodes: List[Dict[str, Any]] = []
    edge_targets = {
        (str(e.get("to", {}).get("node_id")), str(e.get("to", {}).get("port_id")))
        for e in body.get("edges", ())
    }
    for node in body.get("nodes", ()):
        _rename(node, _NODE_SYNONYMS)
        node_id = str(node.get("node_id", "n%d" % (len(nodes) + 1)))
        node["node_id"] = node_id
        _strip_app.current_node = node_id  # type: ignore[attr-defined]
        node["op"] = str(node.get("op", "")).strip().lower()
        node["op"] = _OP_SYNONYMS.get(node["op"], node["op"])
        node["app"] = _resolve_app(node.get("app"), config, aliases)
        node.setdefault("semantic_goal", "")
        node["route_hint"] = [str(r) for r in (node.get("route_hint") or []) if r]
        node.setdefault("critical", False)
        node.pop("predicted_actions", None)
        produces = []
        for produce in node.get("produces") or []:
            derivation = produce.get("derivation") or {}
            if "query" in produce and "derivation" not in produce:
                derivation = {"kind": "sql", "query": produce["query"]}
            if "expr" in produce and "derivation" not in produce:
                derivation = {"kind": "expr", "expr": produce["expr"]}
            derivation.setdefault("kind", "sql" if "query" in derivation else "expr")
            derivation["kind"] = _KIND_SYNONYMS.get(
                str(derivation["kind"]), str(derivation["kind"])
            )
            if derivation["kind"] == "expr" and "expr" not in derivation and "value" in derivation:
                derivation = {"kind": "literal", "value": derivation["value"]}
            if derivation["kind"] == "sql":
                derivation.setdefault("returns", "scalar")
                if derivation["returns"] in ("lastrowid", "rowcount", "none"):
                    derivation["returns"] = "scalar"
            produces.append(
                {
                    "name": str(produce.get("name")),
                    "type": str(produce.get("type") or "Record"),
                    "derivation": derivation,
                }
            )
        node["produces"] = produces
        produce_names = [p["name"] for p in produces]
        for direction in ("inputs", "outputs"):
            ports = []
            for port in node.get(direction) or []:
                _rename(port, _PORT_SYNONYMS)
                port_id = str(port.get("port_id"))
                port.setdefault("type", "Record")
                port.setdefault("cardinality", "one")
                if direction == "outputs":
                    port.setdefault("grounding", "derived:%s:%s" % (node_id, port_id))
                    port.setdefault("source", "upstream")
                else:
                    if "literal" in port:
                        port.setdefault("source", "literal")
                        port.setdefault("grounding", "literal")
                    elif (node_id, port_id) in edge_targets:
                        port["source"] = "upstream"
                        port.setdefault("grounding", "upstream:%s" % port_id)
                    else:
                        port.setdefault("source", "world")
                        port.setdefault("grounding", "unbound:%s" % port_id)
                ports.append(port)
            node[direction] = ports
        if not node["outputs"]:
            node["outputs"] = [
                {
                    "port_id": n,
                    "type": p["type"],
                    "grounding": "derived:%s:%s" % (node_id, n),
                    "source": "upstream",
                    "cardinality": "one",
                }
                for n, p in zip(produce_names, produces)
            ]
        for read in node.get("reads") or []:
            read["table"] = qualified_table(node["app"], str(read.get("table", "")))
            read.setdefault("entity_ref", "%s:*" % read["table"].split(".", 1)[1])
            read["entity_ref"] = _strip_app(read["entity_ref"], node["app"])
            _coerce_equivalents(read, node["app"], node_id)
            read.setdefault("column", "rowid")
        node["reads"] = list(node.get("reads") or [])
        writes = []
        for write in node.get("writes") or []:
            write["table"] = qualified_table(node["app"], str(write.get("table", "")))
            write.setdefault("column", "rowid")
            write.setdefault("entity_ref", "%s:*" % write["table"].split(".", 1)[1])
            write["entity_ref"] = _strip_app(write["entity_ref"], node["app"])
            write.setdefault("effect_type", _DEFAULT_EFFECT.get(node["op"], "modify_record"))
            if "value" not in write and "value_ref" not in write:
                write["value_ref"] = None
            writes.append(write)
        node["writes"] = writes
        effects = list(node.get("side_effects") or [])
        if node["op"] in READ_ONLY_OPS:
            node["writes"] = [] if not writes else writes
            node["side_effects"] = []
            node["reversibility_class"] = "R0"
        else:
            if not effects and writes:
                seen = []
                for write in writes:
                    if write["effect_type"] not in seen:
                        seen.append(write["effect_type"])
                        effects.append(
                            _default_side_effect(
                                node_id,
                                write["effect_type"],
                                ontology,
                                len(seen),
                                write["entity_ref"],
                            )
                        )
            for index, effect in enumerate(effects, 1):
                defaults = _default_side_effect(
                    node_id,
                    str(effect.get("effect_type", "modify_record")),
                    ontology,
                    index,
                    str(effect.get("target_ref", "unknown")),
                )
                for key, value in defaults.items():
                    effect.setdefault(key, value)
                if not effect.get("compensation_available"):
                    effect["compensating_effect_type"] = None
            node["side_effects"] = effects
            classes = [e["reversibility_class"] for e in effects]
            node["reversibility_class"] = (
                ontology.max_class(classes)
                if classes
                else str(node.get("reversibility_class") or "R1")
            )
        verifier = node.get("verifier")
        if verifier:
            _coerce_observability(verifier)
            verifier.setdefault("verifier_id", "v-%s" % node_id)
            verifier.setdefault(
                "kind",
                "sql"
                if str(verifier.get("predicate", "")).lstrip().upper().startswith("SELECT")
                else "derived",
            )
            verifier.setdefault("observability", "environment_state")
            node["verifier"] = verifier
        else:
            node["verifier"] = None
        nodes.append(node)
    edges = []
    for index, edge in enumerate(body.get("edges") or [], 1):
        edge.setdefault("edge_id", "e%d" % index)
        edge.setdefault("kind", "data_dependency")
        edges.append(edge)
    final = body.get("final_verifier")
    if final:
        _coerce_observability(final)
        final.setdefault("verifier_id", "final-%s" % task["id"])
        final.setdefault(
            "kind",
            "sql"
            if str(final.get("predicate", "")).lstrip().upper().startswith("SELECT")
            else "derived",
        )
        final.setdefault("observability", "environment_state")
    return {
        "schema_version": "task-ir/1.0",
        "task_id": str(task["id"]),
        "source_task_id": str(task["id"]),
        "world_id": config.world_id,
        "instruction": str(task["instruction"]),
        "reference_time": config.reference_time,
        "nodes": nodes,
        "edges": edges,
        "final_verifier": final or None,
        "provenance": {
            "unbound_entities": list(raw.get("unbound_entities") or []),
            "notes": str(raw.get("notes") or ""),
            "structural_repairs": repairs,
        },
    }


@dataclass
class ExtractedTaskIR:
    task_id: str
    task_ir: Optional[Dict[str, Any]]
    static_errors: List[str]
    grounding_issues: List[Dict[str, Any]]
    llm_call: Dict[str, Any]

    @property
    def static_valid(self) -> bool:
        return self.task_ir is not None and not self.static_errors


def run_extraction_v1(
    tasks: Sequence[Mapping[str, Any]],
    *,
    config: V1ExtractorConfig,
    schema_graph: SchemaGraph,
    database_dir: Optional[Path],
    ontology: Ontology,
    aliases: AppAliases,
    registry: Optional[ValueTypeRegistry],
    output_dir: Path,
    repository: Path,
    client: Optional[LLMClient] = None,
    saved_replies_dir: Optional[Path] = None,
    workers: int = 1,
    progress: Optional[Callable[[int, int, str, float], None]] = None,
) -> Dict[str, Any]:
    system = config.base.prompt_system.read_text(encoding="utf-8")
    template = config.base.prompt_user.read_text(encoding="utf-8")
    inventory = (
        FileInventory.load(config.file_inventory_path)
        if config.file_inventory_path and config.file_inventory_path.is_file()
        else None
    )
    prompts_dir, replies_dir, ir_dir = (
        output_dir / "prompts",
        output_dir / "replies",
        output_dir / "task_ir",
    )
    prompts_dir.mkdir(parents=True, exist_ok=True)
    prepared = []
    for task in tasks:
        task_id = str(task["id"])
        app_ids = aliases.resolve_all(task.get("apps_involved") or [task.get("app", "")])
        databases = config.databases_for(app_ids)
        samples: Dict[str, Any] = {}
        if database_dir is not None:
            for db in databases:
                path = database_dir / ("%s.sqlite" % db)
                if path.is_file():
                    tables = [
                        t.split(".", 1)[1] for t in schema_graph.tables() if t.startswith(db + ".")
                    ]
                    samples[db] = sample_rows(
                        path, tables, config.sample_rows_per_table, config.max_cell_chars
                    )
        fields = build_prompt_fields_v1(
            task, config, schema_graph, databases, samples, ontology, registry, inventory
        )
        prompt = render_prompt(template, fields)
        (prompts_dir / ("%s.md" % task_id)).write_text(prompt, encoding="utf-8")
        prepared.append((task, task_id, prompt))

    def _fetch(index: int) -> Tuple[int, Optional[LLMResponse], bool, float]:
        started = time.monotonic()
        _, task_id, prompt = prepared[index]
        saved = saved_replies_dir / ("%s.txt" % task_id) if saved_replies_dir else None
        if saved is not None and saved.is_file():
            return (
                index,
                LLMResponse(
                    text=saved.read_text(encoding="utf-8"),
                    model=config.base.model,
                    usage={},
                    request_sha256="",
                ),
                True,
                time.monotonic() - started,
            )
        if client is None:
            return index, None, False, time.monotonic() - started
        return index, client.complete(system, prompt), False, time.monotonic() - started

    fetched: List[Tuple[Optional[LLMResponse], bool]] = [(None, False)] * len(prepared)
    done = 0
    if workers > 1 and client is not None:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            for future in as_completed([pool.submit(_fetch, i) for i in range(len(prepared))]):
                index, response, reused, seconds = future.result()
                fetched[index] = (response, reused)
                done += 1
                if progress:
                    progress(done, len(prepared), prepared[index][1], seconds)
    else:
        for i in range(len(prepared)):
            index, response, reused, seconds = _fetch(i)
            fetched[index] = (response, reused)
            done += 1
            if progress:
                progress(done, len(prepared), prepared[index][1], seconds)

    results: List[ExtractedTaskIR] = []
    for (task, task_id, prompt), (response, reused) in zip(prepared, fetched):
        if response is None:
            continue
        replies_dir.mkdir(parents=True, exist_ok=True)
        (replies_dir / ("%s.txt" % task_id)).write_text(response.text, encoding="utf-8")
        llm_call = {
            "extractor_id": config.base.extractor_id,
            "model": response.model,
            "prompt_sha256": hashlib.sha256((system + prompt).encode("utf-8")).hexdigest(),
            "request_sha256": response.request_sha256,
            "usage": dict(response.usage),
            "called_at": datetime.now(timezone.utc).isoformat(),
            "reused_saved_reply": reused,
        }
        errors: List[str] = []
        task_ir: Optional[Dict[str, Any]] = None
        issues: List[Dict[str, Any]] = []
        usage = dict(response.usage or {})
        details = usage.get("completion_tokens_details") or {}
        if (
            not response.text.strip()
            and usage.get("completion_tokens")
            and (details.get("reasoning_tokens") == usage.get("completion_tokens"))
        ):
            errors.append(
                "REASONING_BUDGET_EXHAUSTED: %s completion tokens spent on reasoning, no answer"
                % usage.get("completion_tokens")
            )
        try:
            parsed = parse_task_ir_reply(response.text)
            task_ir = normalize_task_ir(parsed, task, config, ontology, aliases)
            validate_task_ir(task_ir, repository)
        except ExtractionParseError as exc:
            errors.append("PARSE: %s" % exc)
        except (TaskIRError, SynthesisValidationError) as exc:
            errors.append("STATIC: %s" % exc)
        except Exception as exc:
            errors.append("SCHEMA: %s" % str(exc).splitlines()[0][:300])
        if task_ir is not None:
            issues = validate_grounding(
                task_ir,
                schema_graph,
                database_dir,
                allowed_literals=config.persona_literals,
                inventory=inventory,
            )
            task_ir["provenance"].update(
                {
                    "llm_call": llm_call,
                    "static_errors": errors,
                    "grounding_issues": issues,
                    "review_status": "needs_review",
                }
            )
            ir_dir.mkdir(parents=True, exist_ok=True)
            (ir_dir / ("%s.json" % task_id)).write_text(
                json.dumps(task_ir, indent=1, ensure_ascii=False) + "\n", encoding="utf-8"
            )
        results.append(ExtractedTaskIR(task_id, task_ir, errors, issues, llm_call))

    manifest = {
        "schema_version": "stage-manifest/0.1",
        "stage": "task_ir_v1_extraction",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "mode": (
            "reparse_saved_replies"
            if saved_replies_dir is not None
            else ("dry_run_prompts_only" if client is None else "model_called")
        ),
        "extractor_id": config.base.extractor_id,
        "model": config.base.model if (client is not None or saved_replies_dir) else None,
        "world_id": config.world_id,
        "workers": workers,
        "counts": {
            "tasks": len(tasks),
            "prompts_written": len(prepared),
            "replies": len(results),
            "static_valid": sum(r.static_valid for r in results),
            "grounding_clean": sum(
                1
                for r in results
                if r.static_valid
                and not [i for i in r.grounding_issues if i["severity"] == "error"]
            ),
        },
        "per_task": [
            {
                "task_id": r.task_id,
                "static_errors": r.static_errors,
                "grounding_issue_count": len(r.grounding_issues),
                "grounding_errors": [
                    i["code"] for i in r.grounding_issues if i["severity"] == "error"
                ],
                "usage": r.llm_call.get("usage", {}),
            }
            for r in results
        ],
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=1, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return manifest
