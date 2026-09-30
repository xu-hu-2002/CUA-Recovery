from __future__ import annotations

import re
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Set, Union

from derail.ir.model import parse_entity_ref
from derail.world.facts import date_part, normalize_value, split_table
from derail.world.files import FILE_COLUMNS, FILE_TABLE, FileInventory
from derail.world.schema_graph import SchemaGraph

_PARAM = re.compile(r"(?<![:\w]):([A-Za-z_]\w*)")
_WS = re.compile(r"\s+")
_TIME = re.compile(r"^(\d{1,2}):(\d{2})(?::\d{2})?$")


def _issue(code: str, severity: str, node_id: Optional[str], detail: str) -> Dict[str, Any]:
    return {"code": code, "severity": severity, "node_id": node_id, "detail": detail}


def compile_check(conn: sqlite3.Connection, query: str) -> Optional[str]:
    params = {name: None for name in _PARAM.findall(query)}
    try:
        conn.execute("EXPLAIN " + query, params)
    except sqlite3.Error as exc:
        return str(exc)
    return None


def _open(
    database_dir: Optional[Path], app: str, cache: Dict[str, Optional[sqlite3.Connection]]
) -> Optional[sqlite3.Connection]:
    if app in cache:
        return cache[app]
    conn = None
    if database_dir is not None:
        path = Path(database_dir) / ("%s.sqlite" % app)
        if path.is_file():
            conn = sqlite3.connect("file:%s?mode=ro" % path, uri=True)
    cache[app] = conn
    return conn


def _spellings(value: Any) -> List[str]:
    if value is None or isinstance(value, bool):
        return [str(value).lower()]
    if isinstance(value, (int, float)):
        text = str(value)
        return [text, text.rstrip("0").rstrip(".") if "." in text else text]
    text = _WS.sub(" ", str(value)).strip().lower()
    forms = [text]
    day = date_part(value)
    if day:
        stamp = datetime.strptime(day, "%Y-%m-%d")
        for fmt in ("%B %d", "%b %d", "%B %d, %Y", "%b %d, %Y", "%m/%d", "%m/%d/%Y", "%Y-%m-%d"):
            forms.append(stamp.strftime(fmt).lower())
            forms.append(stamp.strftime(fmt).replace(" 0", " ").lower())
    match = _TIME.match(text)
    if match:
        hour, minute = int(match.group(1)), match.group(2)
        suffix = "am" if hour < 12 else "pm"
        twelve = hour % 12 or 12
        forms += [
            "%d%s" % (twelve, suffix),
            "%d %s" % (twelve, suffix),
            "%d:%s%s" % (twelve, minute, suffix),
            "%d:%s %s" % (twelve, minute, suffix),
            "%d:%s" % (hour, minute),
        ]
        if minute == "00":
            forms += ["%d o'clock" % twelve, "%d" % twelve]
    return [f for f in forms if f]


def literal_allowed(value: Any, instruction: str, allowed_literals: Iterable[Any] = ()) -> bool:
    haystack = _WS.sub(" ", str(instruction)).lower()
    normalized = normalize_value(value)
    for allowed in allowed_literals:
        if normalize_value(allowed) == normalized:
            return True
    if isinstance(value, (list, tuple)):
        return all(literal_allowed(item, instruction, allowed_literals) for item in value)
    return any(form in haystack for form in _spellings(value))


def validate_grounding(
    task_ir: Mapping[str, Any],
    schema_graph: SchemaGraph,
    database_dir: Optional[Union[str, Path]] = None,
    allowed_literals: Sequence[Any] = (),
    inventory: Optional[FileInventory] = None,
) -> List[Dict[str, Any]]:
    issues: List[Dict[str, Any]] = []
    node_ids = {str(n["node_id"]) for n in task_ir["nodes"]}
    created_files = set()
    for node in task_ir["nodes"]:
        for write in node.get("writes", ()):
            if str(write["table"]) == FILE_TABLE:
                created_files.add(str(write["entity_ref"]))
                if str(write["column"]) == "path" and "value" in write:
                    created_files.add("file:%s" % str(write["value"]).lstrip("/"))
    known_tables: Set[str] = set(schema_graph.tables()) | {FILE_TABLE}
    instruction = str(task_ir.get("instruction", ""))
    db_dir = Path(database_dir) if database_dir else None
    conns: Dict[str, Optional[sqlite3.Connection]] = {}
    try:
        for node in task_ir["nodes"]:
            node_id = str(node["node_id"])
            app = str(node["app"])
            if not any(t.startswith(app + ".") for t in known_tables) and app not in (
                "unspecified",
                "files",
            ):
                issues.append(
                    _issue(
                        "APP_WITHOUT_DATABASE",
                        "info",
                        node_id,
                        "%s has no database; only file facts possible" % app,
                    )
                )
            for kind in ("reads", "writes"):
                for ref in node.get(kind, ()):
                    table = str(ref["table"])
                    if table not in known_tables:
                        issues.append(
                            _issue("TABLE_UNKNOWN", "error", node_id, "%s %s" % (kind, table))
                        )
                        continue
                    column = str(ref["column"])
                    if table == FILE_TABLE:
                        if column not in FILE_COLUMNS and column != "*":
                            issues.append(
                                _issue(
                                    "COLUMN_UNKNOWN",
                                    "error",
                                    node_id,
                                    "%s %s.%s" % (kind, table, column),
                                )
                            )
                        entity = str(ref["entity_ref"])
                        if (
                            kind == "reads"
                            and entity.startswith("file:")
                            and inventory is not None
                            and inventory.get(entity) is None
                            and entity not in created_files
                            and not entity.startswith("file::")
                        ):
                            issues.append(
                                _issue(
                                    "ENTITY_MISSING",
                                    "error",
                                    node_id,
                                    "%s not in the file inventory" % entity,
                                )
                            )
                        continue
                    if column not in ("rowid", "*") and not schema_graph.has_column(table, column):
                        issues.append(
                            _issue(
                                "COLUMN_UNKNOWN",
                                "error",
                                node_id,
                                "%s %s.%s" % (kind, table, column),
                            )
                        )
                    parsed = parse_entity_ref(str(ref["entity_ref"]))
                    if parsed[0] == "derived" and parsed[1] not in node_ids:
                        issues.append(
                            _issue("DERIVED_REF_UNKNOWN", "error", node_id, ref["entity_ref"])
                        )
                    elif parsed[0] == "row" and parsed[2] != "*" and not parsed[2].isdigit():
                        issues.append(
                            _issue(
                                "ENTITY_REF_NOT_ROWID",
                                "error",
                                node_id,
                                "%s: row ids are integers" % ref["entity_ref"],
                            )
                        )
                    elif parsed[0] == "row" and parsed[2] != "*":
                        conn = _open(db_dir, split_table(table)[0], conns)
                        if conn is not None and parsed[2].isdigit():
                            exists = conn.execute(
                                'SELECT 1 FROM "%s" WHERE rowid = ?' % parsed[1], (int(parsed[2]),)
                            ).fetchone()
                            if exists is None:
                                issues.append(
                                    _issue(
                                        "ENTITY_MISSING",
                                        "error",
                                        node_id,
                                        "%s has no row %s" % (table, parsed[2]),
                                    )
                                )
                    if kind == "writes" and "value" not in ref and not ref.get("value_ref"):
                        issues.append(
                            _issue(
                                "WRITE_VALUE_MISSING", "error", node_id, "%s.%s" % (table, column)
                            )
                        )
            for produce in node.get("produces", ()):
                derivation = produce["derivation"]
                if derivation.get("kind") == "file" and inventory is not None:
                    path = str(derivation.get("path", ""))
                    if (
                        path
                        and not path.startswith(":")
                        and inventory.get(path) is None
                        and "file:%s" % path.lstrip("/") not in created_files
                    ):
                        issues.append(
                            _issue(
                                "ENTITY_MISSING",
                                "error",
                                node_id,
                                "file %s not in the inventory" % path,
                            )
                        )
                if derivation.get("kind") != "sql":
                    continue
                query_app = str(derivation.get("app") or app)
                conn = _open(db_dir, query_app, conns)
                if conn is None:
                    issues.append(
                        _issue(
                            "SQL_UNCHECKED",
                            "warning",
                            node_id,
                            "%s: no database copy for %s" % (produce["name"], query_app),
                        )
                    )
                    continue
                error = compile_check(conn, str(derivation["query"]))
                if error:
                    issues.append(
                        _issue(
                            "SQL_COMPILE_ERROR",
                            "error",
                            node_id,
                            "%s: %s" % (produce["name"], error),
                        )
                    )
            verifier = node.get("verifier")
            if verifier and verifier.get("kind") == "sql":
                conn = _open(db_dir, str(verifier.get("app") or app), conns)
                if conn is not None:
                    error = compile_check(conn, str(verifier.get("predicate", "")))
                    if error:
                        issues.append(
                            _issue("SQL_COMPILE_ERROR", "error", node_id, "verifier: %s" % error)
                        )
            for port in node.get("inputs", ()):
                grounding = str(port.get("grounding", ""))
                if grounding.startswith("unbound:"):
                    issues.append(
                        _issue(
                            "UNBOUND_INPUT",
                            "error",
                            node_id,
                            "%s <- %s: ground it as a file fact or a query"
                            % (port["port_id"], grounding),
                        )
                    )
                    continue
                if (
                    "literal" in port
                    and port["literal"] not in ("", None)
                    and not literal_allowed(port["literal"], instruction, allowed_literals)
                ):
                    issues.append(
                        _issue(
                            "LITERAL_NOT_IN_INSTRUCTION",
                            "error",
                            node_id,
                            "%s = %r" % (port["port_id"], port["literal"]),
                        )
                    )
        final = task_ir.get("final_verifier")
        parts = final.get("parts", ()) if final and final.get("kind") == "all_of" else [final]
        for part in parts:
            if not part or part.get("kind") != "sql":
                continue
            app = str(part.get("app") or task_ir["nodes"][-1]["app"])
            conn = _open(db_dir, app, conns)
            if conn is not None:
                error = compile_check(conn, str(part.get("predicate", "")))
                if error:
                    issues.append(
                        _issue("SQL_COMPILE_ERROR", "error", None, "final_verifier: %s" % error)
                    )
    finally:
        for conn in conns.values():
            if conn is not None:
                conn.close()
    return issues
