from __future__ import annotations

import hashlib
import json
import re
import shutil
import sqlite3
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Set, Tuple, Union

import yaml

from derail.derived.schema import validate_schema
from derail.ir.expr import ExpressionError, evaluate
from derail.ir.model import dag_index, parse_entity_ref, validate_task_ir
from derail.world.facts import entity_id, split_table
from derail.world.files import FILE_TABLE, FileInventory, csv_rows, extract_text
from derail.world.state import database_and_table_digests, snapshot_row, table_columns, table_names
from derail.world.volatile import VolatileColumns

INTERPRETER_VERSION = "gold-interpreter/1.1"
GOLD_SCHEMA_VERSION = "gold-lineage/1.0"
READ_ONLY_OPS = frozenset(
    {"retrieve", "resolve", "filter", "compare", "aggregate", "verify", "decide", "confirm"}
)
_BINDABLE = (str, int, float, bytes, type(None))
_WRITE_ACTIONS = {
    sqlite3.SQLITE_INSERT,
    sqlite3.SQLITE_UPDATE,
    sqlite3.SQLITE_DELETE,
    sqlite3.SQLITE_CREATE_TABLE,
    sqlite3.SQLITE_DROP_TABLE,
    sqlite3.SQLITE_ALTER_TABLE,
    sqlite3.SQLITE_CREATE_INDEX,
    sqlite3.SQLITE_DROP_INDEX,
    sqlite3.SQLITE_TRANSACTION,
}
_IDENT = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")


class GoldInterpreterError(RuntimeError):
    def __init__(self, code: str, message: str):
        super().__init__("%s: %s" % (code, message))
        self.code = code


@dataclass(frozen=True)
class InterpreterConfig:
    exclude_tables: Tuple[str, ...] = ("sqlite_sequence",)
    exclude_table_patterns: Tuple[str, ...] = ()
    require_declared_writes: bool = True
    audit_reads: bool = True
    max_rows_per_value: int = 500
    evaluated_verifier_kinds: Tuple[str, ...] = ("sql", "derived")
    volatile: Optional[VolatileColumns] = None

    @classmethod
    def from_yaml(
        cls, path: Union[str, Path], repo_root: Optional[Union[str, Path]] = None
    ) -> "InterpreterConfig":
        path = Path(path)
        raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        if raw.get("schema_version") != "gold-interpreter-config/1.0":
            raise ValueError("unsupported gold interpreter config %r" % raw.get("schema_version"))
        digest = raw.get("state_digest", {})
        execution = raw.get("execution", {})
        volatile = None
        if raw.get("volatile_columns_config"):
            root = Path(repo_root) if repo_root else path.resolve().parents[2]
            volatile = VolatileColumns.from_yaml(root / raw["volatile_columns_config"])
        return cls(
            exclude_tables=tuple(digest.get("exclude_tables", ("sqlite_sequence",))),
            exclude_table_patterns=tuple(digest.get("exclude_table_patterns", ())),
            require_declared_writes=bool(execution.get("require_declared_writes", True)),
            audit_reads=bool(execution.get("audit_reads", True)),
            max_rows_per_value=int(execution.get("max_rows_per_value", 500)),
            evaluated_verifier_kinds=tuple(
                execution.get("evaluated_verifier_kinds", ("sql", "derived"))
            ),
            volatile=volatile,
        )


@dataclass
class WorldCopy:
    """Isolated copies of a world's databases, one connection per application."""

    world_id: str
    paths: Dict[str, Path]
    connections: Dict[str, sqlite3.Connection] = field(default_factory=dict)
    files_root: Optional[Path] = None
    inventory: Optional[FileInventory] = None
    source_keys: Dict[str, Tuple[str, int, int]] = field(default_factory=dict)

    @classmethod
    def open(
        cls,
        world_id: str,
        sources: Mapping[str, Union[str, Path]],
        workdir: Union[str, Path],
        files_root: Optional[Union[str, Path]] = None,
    ) -> "WorldCopy":
        target = Path(workdir)
        target.mkdir(parents=True, exist_ok=True)
        paths: Dict[str, Path] = {}
        for app, source in sorted(sources.items()):
            source_path = Path(source)
            if not source_path.is_file():
                raise GoldInterpreterError("WORLD_SOURCE_MISSING", "%s: %s" % (app, source_path))
            destination = target / source_path.name
            shutil.copyfile(source_path, destination)
            for suffix in ("-wal", "-shm"):
                sidecar = source_path.with_name(source_path.name + suffix)
                if sidecar.is_file():
                    shutil.copyfile(sidecar, destination.with_name(destination.name + suffix))
            paths[app] = destination
        world = cls(world_id=world_id, paths=paths)
        for app, source in sorted(sources.items()):
            stat = Path(source).stat()
            world.source_keys[app] = (str(Path(source).resolve()), stat.st_size, stat.st_mtime_ns)
        if files_root is not None and Path(files_root).is_dir():
            destination = target / "files"
            if destination.exists():
                shutil.rmtree(destination)
            shutil.copytree(files_root, destination)
            world.files_root = destination
            world.inventory = FileInventory.build(destination)
        for app, path in paths.items():
            conn = sqlite3.connect(str(path))
            conn.row_factory = sqlite3.Row
            world.connections[app] = conn
        return world

    def connection(self, app: str) -> sqlite3.Connection:
        try:
            return self.connections[app]
        except KeyError:
            raise GoldInterpreterError(
                "UNKNOWN_APP", "world %s has no database for app %r" % (self.world_id, app)
            ) from None

    def close(self) -> None:
        for conn in self.connections.values():
            conn.close()
        self.connections.clear()

    def __enter__(self) -> "WorldCopy":
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()


def _jsonable(value: Any) -> Any:
    if isinstance(value, sqlite3.Row):
        return {key: _jsonable(value[key]) for key in value.keys()}
    if isinstance(value, (bytes, bytearray, memoryview)):
        return {"__bytes_len__": len(value)}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    return value


def _bindable(scope: Mapping[str, Any]) -> Dict[str, Any]:
    out = {key: value for key, value in scope.items() if isinstance(value, _BINDABLE)}
    env = scope.get("V")
    if isinstance(env, Mapping):
        for node_id, produced in env.items():
            for name, value in produced.items():
                if not isinstance(value, _BINDABLE):
                    continue
                if name not in out:
                    out[name] = value
                out["%s__%s" % (node_id, name)] = value
    return out


class _ReadAuditor:
    def __init__(self) -> None:
        self.reads: Set[Tuple[str, str]] = set()

    def __call__(self, action: int, arg1: Any, arg2: Any, dbname: Any, source: Any) -> int:
        if action in _WRITE_ACTIONS:
            return sqlite3.SQLITE_DENY
        if action == sqlite3.SQLITE_READ and arg1 and not str(arg1).startswith("sqlite_"):
            self.reads.add((str(arg1), str(arg2) if arg2 else "rowid"))
        return sqlite3.SQLITE_OK


class _DigestCache:
    def __init__(
        self,
        world: "WorldCopy",
        exclude_tables: Sequence[str],
        exclude_columns: Mapping[str, Mapping[str, Set[str]]],
        shared: Dict[Tuple[str, int, int, str], Tuple[str, Dict[str, str]]],
    ):
        self.world = world
        self.exclude_tables = tuple(exclude_tables)
        self.exclude_columns = exclude_columns
        self.shared = shared
        self.per_app: Dict[str, Tuple[int, str, Dict[str, str]]] = {}

    def _refresh(self) -> None:
        for app, conn in self.world.connections.items():
            changes = conn.total_changes
            cached = self.per_app.get(app)
            if cached is not None and cached[0] == changes:
                continue
            columns = self.exclude_columns.get(app)
            column_key = json.dumps(
                {t: sorted(c) for t, c in (columns or {}).items()}, sort_keys=True
            )
            source = self.world.source_keys.get(app)
            shared_key = (
                (source[0], source[1], source[2], column_key + "|" + "|".join(self.exclude_tables))
                if source and changes == 0
                else None
            )
            if shared_key is not None and shared_key in self.shared:
                db_digest, tables = self.shared[shared_key]
            else:
                db_digest, tables = database_and_table_digests(conn, self.exclude_tables, columns)
                if shared_key is not None:
                    self.shared[shared_key] = (db_digest, tables)
            self.per_app[app] = (changes, db_digest, tables)

    def state(self) -> str:
        self._refresh()
        hasher = hashlib.sha256()
        for app in sorted(self.world.connections):
            hasher.update(("%s:%s\n" % (app, self.per_app[app][1])).encode())
        return hasher.hexdigest()

    def tables(self) -> Dict[Tuple[str, str], str]:
        self._refresh()
        return {
            (app, table): digest
            for app, (_, _, tables) in self.per_app.items()
            for table, digest in tables.items()
        }


class GoldInterpreter:
    def __init__(self, config: Optional[InterpreterConfig] = None):
        self.config = config or InterpreterConfig()
        self._column_cache: Dict[Tuple[str, str], List[str]] = {}
        self._effective_excludes: Tuple[str, ...] = tuple(self.config.exclude_tables)
        self._digest_cache: Dict[Tuple[str, int, int, str], Tuple[str, Dict[str, str]]] = {}

    def run(
        self,
        task_ir: Mapping[str, Any],
        world: WorldCopy,
        repository: Union[str, Path],
        overrides: Optional[Mapping[Tuple[str, str], Any]] = None,
    ) -> Dict[str, Any]:
        validate_task_ir(task_ir, repository)
        overrides = dict(overrides or {})
        dag = dag_index(task_ir)
        binding = self._edge_bindings(task_ir)
        exclude_columns = self._volatile_map(world)
        self._effective_excludes = self._excluded_tables(world)
        env: Dict[str, Dict[str, Any]] = {}
        values: List[Dict[str, Any]] = []
        timeline: List[Dict[str, Any]] = []
        writes_gold: List[Dict[str, Any]] = []
        verifier_results: List[Dict[str, Any]] = []
        resolved_reads: List[Dict[str, Any]] = []
        reads_undeclared: List[Dict[str, Any]] = []
        skipped: List[str] = []
        base_scope = {"reference_time": task_ir.get("reference_time")}
        digests = _DigestCache(world, self._effective_excludes, exclude_columns, self._digest_cache)

        def _digest() -> str:
            digest = digests.state()
            if world.inventory is not None:
                files = "|".join(
                    "%s:%s" % (e.path, e.sha256)
                    for e in sorted(world.inventory.entries.values(), key=lambda x: x.path)
                )
                digest = hashlib.sha256((digest + "\nfiles:" + files).encode("utf-8")).hexdigest()
            return digest

        initial_digest = _digest()
        previous_digest = initial_digest
        for node_id in dag.order:
            node = dag.nodes[node_id]
            env[node_id] = {}
            if any(dep in skipped for dep in dag.incoming[node_id]) or not self._control_allows(
                node_id, task_ir, env, base_scope
            ):
                skipped.append(node_id)
                timeline.append(
                    {"node_id": node_id, "state_sha256": previous_digest, "changed": False}
                )
                continue
            scope = dict(base_scope)
            scope.update(self._bind_inputs(node, binding, env))
            scope["V"] = env
            before_tables = digests.tables() if self.config.require_declared_writes else {}
            before_rows = self._snapshot_declared_rows(node, env, world)
            actual_reads: Set[Tuple[str, str]] = set()
            for produce in node.get("produces", ()):
                value = self._derive(produce, node, scope, world, env, actual_reads)
                if (node_id, str(produce["name"])) in overrides:
                    value = overrides[(node_id, str(produce["name"]))]
                scope[str(produce["name"])] = value
                env[node_id][str(produce["name"])] = value
                values.append(
                    {
                        "node_id": node_id,
                        "name": str(produce["name"]),
                        "type": str(produce["type"]),
                        "value": _jsonable(value),
                    }
                )
            self._apply_writes(node, scope, env, world)
            self._apply_file_writes(node, scope, env, world)
            for conn in world.connections.values():
                conn.commit()
            writes_gold.extend(self._collect_writes(node, env, before_rows, world))
            if self.config.require_declared_writes:
                self._check_undeclared_writes(node, before_tables, digests.tables())
            if self.config.audit_reads:
                reads_undeclared.extend(self._audit_reads(node, actual_reads, world))
            resolved_reads.extend(self._resolve_reads(node, env, world))
            result = self._evaluate_verifier(node.get("verifier"), node_id, scope, node, world)
            if result is not None:
                verifier_results.append(result)
            digest = _digest()
            timeline.append(
                {"node_id": node_id, "state_sha256": digest, "changed": digest != previous_digest}
            )
            previous_digest = digest
        final_scope = dict(base_scope)
        final_scope["V"] = env
        final = self._evaluate_final(task_ir.get("final_verifier"), final_scope, task_ir, world)
        record = {
            "schema_version": GOLD_SCHEMA_VERSION,
            "task_id": str(task_ir["task_id"]),
            "world_id": world.world_id,
            "initial_state_sha256": initial_digest,
            "node_order": list(dag.order),
            "values": values,
            "state_timeline": timeline,
            "writes_gold": writes_gold,
            "verifier_results": verifier_results,
            "final_verifier_passed": final,
            "resolved_reads": resolved_reads,
            "reads_undeclared": reads_undeclared,
            "interpreter_version": INTERPRETER_VERSION,
            "provenance": {
                "skipped_nodes": skipped,
                "overrides": [
                    {"node_id": n, "name": p, "value": _jsonable(v)}
                    for (n, p), v in overrides.items()
                ],
                "world_paths": {app: str(path) for app, path in world.paths.items()},
            },
        }
        validate_schema(record, "gold_lineage.schema.json", Path(repository))
        return record

    def _columns(self, world: WorldCopy, app: str, table: str) -> List[str]:
        key = (app, table)
        if key not in self._column_cache:
            self._column_cache[key] = table_columns(world.connection(app), table)
        return self._column_cache[key]

    def _excluded_tables(self, world: WorldCopy) -> Tuple[str, ...]:
        names = set(self.config.exclude_tables)
        for conn in world.connections.values():
            for table in table_names(conn):
                if any(re.search(pattern, table) for pattern in self.config.exclude_table_patterns):
                    names.add(table)
        return tuple(sorted(names))

    def _volatile_map(self, world: WorldCopy) -> Dict[str, Dict[str, Set[str]]]:
        if self.config.volatile is None:
            return {}
        out: Dict[str, Dict[str, Set[str]]] = {}
        for app, conn in world.connections.items():
            tables = [(t, table_columns(conn, t)) for t in table_names(conn)]
            out[app] = self.config.volatile.excluded_for(app, tables)
        return out

    def _is_volatile(self, table: str, column: str) -> bool:
        return self.config.volatile is not None and self.config.volatile.is_volatile(table, column)

    @staticmethod
    def _edge_bindings(
        task_ir: Mapping[str, Any],
    ) -> Dict[Tuple[str, str], Optional[Tuple[str, str]]]:
        binding: Dict[Tuple[str, str], Optional[Tuple[str, str]]] = {}
        for edge in task_ir["edges"]:
            target = (str(edge["to"]["node_id"]), str(edge["to"]["port_id"]))
            if edge["kind"] == "control_dependency":
                binding[target] = None
                continue
            binding[target] = (str(edge["from"]["node_id"]), str(edge["from"]["port_id"]))
        return binding

    def _bind_inputs(
        self,
        node: Mapping[str, Any],
        binding: Mapping[Tuple[str, str], Optional[Tuple[str, str]]],
        env: Mapping[str, Mapping[str, Any]],
    ) -> Dict[str, Any]:
        node_id = str(node["node_id"])
        scope: Dict[str, Any] = {}
        for port in node.get("inputs", ()):
            port_id = str(port["port_id"])
            if (node_id, port_id) in binding and binding[(node_id, port_id)] is None:
                continue
            source = binding.get((node_id, port_id))
            if source is not None:
                upstream_node, upstream_port = source
                if upstream_port not in env.get(upstream_node, {}):
                    raise GoldInterpreterError(
                        "UNBOUND_PORT",
                        "%s.%s reads %s.%s which produced nothing"
                        % (node_id, port_id, upstream_node, upstream_port),
                    )
                scope[port_id] = env[upstream_node][upstream_port]
            elif "literal" in port:
                scope[port_id] = port["literal"]
            elif port.get("source") == "world":
                scope[port_id] = port["grounding"]
            else:
                raise GoldInterpreterError(
                    "UNBOUND_PORT", "%s.%s has no value" % (node_id, port_id)
                )
        return scope

    def _control_allows(
        self,
        node_id: str,
        task_ir: Mapping[str, Any],
        env: Mapping[str, Mapping[str, Any]],
        base_scope: Mapping[str, Any],
    ) -> bool:
        for edge in task_ir["edges"]:
            if edge["kind"] != "control_dependency" or str(edge["to"]["node_id"]) != node_id:
                continue
            predicate = edge.get("predicate")
            if not predicate:
                continue
            scope = dict(base_scope)
            scope["V"] = env
            try:
                if not evaluate(str(predicate), scope):
                    return False
            except ExpressionError as exc:
                raise GoldInterpreterError(
                    "CONTROL_PREDICATE_ERROR", "%s: %s" % (edge["edge_id"], exc)
                ) from exc
        return True

    def _derive(
        self,
        produce: Mapping[str, Any],
        node: Mapping[str, Any],
        scope: Mapping[str, Any],
        world: WorldCopy,
        env: Mapping[str, Mapping[str, Any]],
        actual_reads: Set[Tuple[str, str]],
    ) -> Any:
        derivation = produce["derivation"]
        kind = derivation["kind"]
        name = "%s.%s" % (node["node_id"], produce["name"])
        if kind == "literal":
            return derivation.get("value")
        if kind == "upstream":
            port_id = str(derivation.get("port_id", produce["name"]))
            if port_id not in scope:
                raise GoldInterpreterError(
                    "UNBOUND_PORT", "%s copies missing port %s" % (name, port_id)
                )
            return scope[port_id]
        if kind == "expr":
            try:
                return evaluate(str(derivation["expr"]), scope)
            except ExpressionError as exc:
                raise GoldInterpreterError("DERIVATION_ERROR", "%s: %s" % (name, exc)) from exc
        if kind == "sql":
            return self._run_sql(derivation, node, scope, world, name, actual_reads)
        if kind == "create":
            if str(derivation.get("table") or "") == FILE_TABLE or str(node["app"]) == "files":
                return self._create_file(produce, node, scope, world, env)
            return self._create_row(produce, node, scope, world, env)
        if kind == "file":
            return self._read_file(derivation, node, scope, world, name, actual_reads)
        raise GoldInterpreterError("DERIVATION_ERROR", "%s: unknown kind %r" % (name, kind))

    def _run_sql(
        self,
        derivation: Mapping[str, Any],
        node: Mapping[str, Any],
        scope: Mapping[str, Any],
        world: WorldCopy,
        name: str,
        actual_reads: Set[Tuple[str, str]],
    ) -> Any:
        app = str(derivation.get("app") or node["app"])
        conn = world.connection(app)
        query = str(derivation["query"])
        returns = str(derivation.get("returns", "scalar"))
        auditor = _ReadAuditor()
        conn.set_authorizer(auditor)
        try:
            cursor = conn.execute(query, _bindable(scope))
            rows = (
                None
                if returns in ("none", "lastrowid", "rowcount")
                else cursor.fetchmany(self.config.max_rows_per_value + 1)
            )
        except sqlite3.DatabaseError as exc:
            if "not authorized" in str(exc):
                raise GoldInterpreterError(
                    "SQL_DERIVATION_WRITES",
                    "%s: derivations are read-only; use writes (%s)" % (name, query[:60]),
                ) from exc
            raise GoldInterpreterError("SQL_ERROR", "%s: %s" % (name, exc)) from exc
        finally:
            conn.set_authorizer(None)
        for table, column in auditor.reads:
            actual_reads.add(("%s.%s" % (app, table), column))
        if returns in ("none", "lastrowid", "rowcount"):
            raise GoldInterpreterError(
                "DERIVATION_ERROR",
                "%s: returns %s belongs to writes, not sql derivations" % (name, returns),
            )
        if len(rows) > self.config.max_rows_per_value:
            raise GoldInterpreterError("VALUE_TOO_LARGE", "%s exceeds max_rows_per_value" % name)
        if returns == "scalar":
            return rows[0][0] if rows else None
        if returns == "row":
            return _jsonable(rows[0]) if rows else None
        if returns == "rows":
            return [_jsonable(row) for row in rows]
        if returns == "column":
            return [row[0] for row in rows]
        raise GoldInterpreterError("DERIVATION_ERROR", "%s: unknown returns %r" % (name, returns))

    def _file_path(self, derivation: Mapping[str, Any], scope: Mapping[str, Any], name: str) -> str:
        path = str(derivation.get("path", ""))
        if path.startswith(":"):
            value = scope.get(path[1:])
            if not isinstance(value, str):
                raise GoldInterpreterError(
                    "DERIVATION_ERROR", "%s: file path port %s is not a string" % (name, path)
                )
            path = value
        if path.startswith("file:"):
            path = path[len("file:") :]
        return path

    def _read_file(
        self,
        derivation: Mapping[str, Any],
        node: Mapping[str, Any],
        scope: Mapping[str, Any],
        world: WorldCopy,
        name: str,
        actual_reads: Set[Tuple[str, str]],
    ) -> Any:
        if world.inventory is None:
            raise GoldInterpreterError(
                "FILES_UNAVAILABLE", "%s reads a file but the world has no file tree" % name
            )
        path = self._file_path(derivation, scope, name)
        entry = world.inventory.get(path)
        op = str(derivation.get("op", "text"))
        if op == "exists":
            actual_reads.add((FILE_TABLE, "path"))
            return entry is not None
        if entry is None:
            raise GoldInterpreterError("FILE_MISSING", "%s: %s not in the world" % (name, path))
        actual_reads.add((FILE_TABLE, "content"))
        text = (
            entry.content
            if entry.content is not None
            else extract_text(Path(world.files_root) / entry.path)
        )
        if text is None:
            raise GoldInterpreterError(
                "FILE_BINARY", "%s: %s has no extractable text" % (name, path)
            )
        if op == "text":
            return text
        if op == "lines":
            return text.splitlines()
        if op in ("regex", "regex_all"):
            pattern = str(derivation.get("pattern", ""))
            try:
                matches = re.findall(pattern, text, re.MULTILINE)
            except re.error as exc:
                raise GoldInterpreterError(
                    "DERIVATION_ERROR", "%s: bad pattern: %s" % (name, exc)
                ) from exc
            if op == "regex":
                return matches[0] if matches else None
            return matches
        if op == "json":
            try:
                return json.loads(text)
            except ValueError as exc:
                raise GoldInterpreterError(
                    "DERIVATION_ERROR", "%s: not JSON: %s" % (name, exc)
                ) from exc
        if op == "csv_rows":
            return csv_rows(text)
        raise GoldInterpreterError("DERIVATION_ERROR", "%s: unknown file op %r" % (name, op))

    def _create_file(
        self,
        produce: Mapping[str, Any],
        node: Mapping[str, Any],
        scope: Mapping[str, Any],
        world: WorldCopy,
        env: Mapping[str, Mapping[str, Any]],
    ) -> str:
        node_id = str(node["node_id"])
        derivation = produce["derivation"]
        entity_ref = "derived:%s:%s" % (node_id, produce["name"])
        file_writes = [w for w in node.get("writes", ()) if str(w["table"]) == FILE_TABLE]
        own = [w for w in file_writes if w["entity_ref"] == entity_ref]
        literal_entities = {
            str(w["entity_ref"]) for w in file_writes if str(w["entity_ref"]).startswith("file:")
        }
        cells = own or (
            [w for w in file_writes if str(w["entity_ref"]) in literal_entities]
            if len(literal_entities) == 1
            else []
        )
        if not cells:
            raise GoldInterpreterError(
                "CREATE_WITHOUT_WRITES",
                "%s declares no file content for %s" % (node_id, entity_ref),
            )
        if world.files_root is None:
            raise GoldInterpreterError(
                "FILES_UNAVAILABLE", "%s creates a file but the world has no file tree" % node_id
            )
        path_cell = next((w for w in cells if str(w["column"]) == "path"), None)
        content_cell = next((w for w in cells if str(w["column"]) == "content"), None)
        if path_cell is not None:
            path = str(self._write_value(path_cell, scope, env, node_id))
        elif derivation.get("path"):
            path = self._file_path(derivation, scope, "%s.%s" % (node_id, produce["name"]))
        elif len(literal_entities) == 1:
            path = next(iter(literal_entities))[len("file:") :]
        else:
            raise GoldInterpreterError(
                "WRITE_VALUE_MISSING",
                "%s: a file create needs a path (write, derivation or file: entity)" % node_id,
            )
        if content_cell is None:
            raise GoldInterpreterError(
                "WRITE_VALUE_MISSING", "%s: a file create needs a content write" % node_id
            )
        path = path.lstrip("/")
        if path.startswith("file:"):
            path = path[len("file:") :]
        target = Path(world.files_root) / path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(
            str(self._write_value(content_cell, scope, env, node_id)), encoding="utf-8"
        )
        world.inventory = FileInventory.build(world.files_root)
        return "file:%s" % path

    def _apply_file_writes(
        self,
        node: Mapping[str, Any],
        scope: Mapping[str, Any],
        env: Mapping[str, Mapping[str, Any]],
        world: WorldCopy,
    ) -> None:
        node_id = str(node["node_id"])
        created = {
            "derived:%s:%s" % (node_id, p["name"])
            for p in node.get("produces", ())
            if p["derivation"].get("kind") == "create"
        }
        if created:
            produced_files = {
                v
                for v in env.get(node_id, {}).values()
                if isinstance(v, str) and v.startswith("file:")
            }
            created |= produced_files
        for write in node.get("writes", ()):
            if str(write["table"]) != FILE_TABLE or write["entity_ref"] in created:
                continue
            if world.files_root is None:
                raise GoldInterpreterError(
                    "FILES_UNAVAILABLE", "%s writes a file but the world has no file tree" % node_id
                )
            entity = str(write["entity_ref"])
            path = entity[len("file:") :] if entity.startswith("file:") else entity
            target = Path(world.files_root) / path
            column = str(write["column"])
            if column == "*":
                if target.exists():
                    target.unlink()
                continue
            if column != "content":
                raise GoldInterpreterError(
                    "WRITE_VALUE_MISSING", "%s: only content can be written on files" % node_id
                )
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(str(self._write_value(write, scope, env, node_id)), encoding="utf-8")
        if world.files_root is not None and any(
            str(w["table"]) == FILE_TABLE for w in node.get("writes", ())
        ):
            world.inventory = FileInventory.build(world.files_root)

    def _write_value(
        self,
        write: Mapping[str, Any],
        scope: Mapping[str, Any],
        env: Mapping[str, Mapping[str, Any]],
        node_id: str,
    ) -> Any:
        if "value" in write:
            return write["value"]
        ref = write.get("value_ref")
        if ref is None:
            raise GoldInterpreterError(
                "WRITE_VALUE_MISSING",
                "%s.%s.%s has neither value nor value_ref"
                % (node_id, write["table"], write["column"]),
            )
        if str(ref).startswith("derived:"):
            _, other, name = str(ref).split(":", 2)
            if name not in env.get(other, {}):
                raise GoldInterpreterError(
                    "WRITE_VALUE_MISSING", "%s: %s not produced" % (node_id, ref)
                )
            return env[other][name]
        if ref in scope:
            return scope[str(ref)]
        raise GoldInterpreterError(
            "WRITE_VALUE_MISSING", "%s: value_ref %r unknown" % (node_id, ref)
        )

    def _create_row(
        self,
        produce: Mapping[str, Any],
        node: Mapping[str, Any],
        scope: Mapping[str, Any],
        world: WorldCopy,
        env: Mapping[str, Mapping[str, Any]],
    ) -> int:
        node_id = str(node["node_id"])
        entity_ref = "derived:%s:%s" % (node_id, produce["name"])
        cells = [w for w in node.get("writes", ()) if w["entity_ref"] == entity_ref]
        if not cells:
            raise GoldInterpreterError(
                "CREATE_WITHOUT_WRITES", "%s declares no cells for %s" % (node_id, entity_ref)
            )
        tables = {str(w["table"]) for w in cells}
        if len(tables) != 1:
            raise GoldInterpreterError(
                "CREATE_MULTI_TABLE", "%s creates %s in several tables" % (node_id, entity_ref)
            )
        app, table = split_table(tables.pop())
        columns = [str(w["column"]) for w in cells]
        params = [self._write_value(w, scope, env, node_id) for w in cells]
        conn = world.connection(app)
        try:
            cursor = conn.execute(
                'INSERT INTO "%s" (%s) VALUES (%s)'
                % (table, ", ".join('"%s"' % c for c in columns), ", ".join("?" for _ in columns)),
                params,
            )
        except sqlite3.Error as exc:
            raise GoldInterpreterError("SQL_ERROR", "%s create: %s" % (node_id, exc)) from exc
        return int(cursor.lastrowid)

    def _apply_writes(
        self,
        node: Mapping[str, Any],
        scope: Mapping[str, Any],
        env: Mapping[str, Mapping[str, Any]],
        world: WorldCopy,
    ) -> None:
        node_id = str(node["node_id"])
        created = {
            "derived:%s:%s" % (node_id, p["name"])
            for p in node.get("produces", ())
            if p["derivation"].get("kind") == "create"
        }
        groups: Dict[Tuple[str, str, int], List[Mapping[str, Any]]] = {}
        for write in node.get("writes", ()):
            if write["entity_ref"] in created or str(write["table"]) == FILE_TABLE:
                continue
            target = self._resolve_write_target(write, env)
            if target is None:
                raise GoldInterpreterError(
                    "WRITE_TARGET_UNRESOLVED", "%s.%s" % (node_id, write["entity_ref"])
                )
            groups.setdefault(target, []).append(write)
        for (app, table, rowid), cells in groups.items():
            conn = world.connection(app)
            deletes = [
                w for w in cells if str(w["effect_type"]).startswith("delete") or w["column"] == "*"
            ]
            if deletes:
                conn.execute('DELETE FROM "%s" WHERE rowid = ?' % table, (rowid,))
                continue
            assignments = [
                (str(w["column"]), self._write_value(w, scope, env, node_id)) for w in cells
            ]
            try:
                conn.execute(
                    'UPDATE "%s" SET %s WHERE rowid = ?'
                    % (table, ", ".join('"%s" = ?' % c for c, _ in assignments)),
                    [v for _, v in assignments] + [rowid],
                )
            except sqlite3.Error as exc:
                raise GoldInterpreterError("SQL_ERROR", "%s update: %s" % (node_id, exc)) from exc

    def _resolve_write_target(
        self, write: Mapping[str, Any], env: Mapping[str, Mapping[str, Any]]
    ) -> Optional[Tuple[str, str, int]]:
        app, table = split_table(str(write["table"]))
        parsed = parse_entity_ref(str(write["entity_ref"]))
        if parsed[0] == "row" and parsed[2] != "*":
            if not str(parsed[2]).isdigit():
                raise GoldInterpreterError(
                    "WRITE_TARGET_NOT_ROWID", "%s is not a row id" % write["entity_ref"]
                )
            return app, table, int(parsed[2])
        if parsed[0] == "derived":
            value = env.get(parsed[1], {}).get(parsed[2])
            if value is None:
                return None
            if isinstance(value, Mapping):
                value = value.get("id", value.get("rowid"))
            if isinstance(value, bool) or not isinstance(value, int):
                raise GoldInterpreterError(
                    "WRITE_TARGET_NOT_ROWID", "%s resolves to %r" % (write["entity_ref"], value)
                )
            return app, table, value
        return None

    def _snapshot_declared_rows(
        self, node: Mapping[str, Any], env: Mapping[str, Mapping[str, Any]], world: WorldCopy
    ) -> Dict[Tuple[str, str, int], Optional[Dict[str, Any]]]:
        snapshots: Dict[Tuple[str, str, int], Optional[Dict[str, Any]]] = {}
        for write in node.get("writes", ()):
            if str(write["table"]) == FILE_TABLE:
                continue
            target = self._resolve_write_target(write, env)
            if target is not None and target not in snapshots:
                snapshots[target] = snapshot_row(world.connection(target[0]), target[1], target[2])
        return snapshots

    def _effect_class(self, node: Mapping[str, Any], effect_type: str) -> str:
        for effect in node.get("side_effects", ()):
            if effect.get("effect_type") == effect_type:
                return str(effect["reversibility_class"])
        return str(node["reversibility_class"])

    def _collect_writes(
        self,
        node: Mapping[str, Any],
        env: Mapping[str, Mapping[str, Any]],
        before: Mapping[Tuple[str, str, int], Optional[Dict[str, Any]]],
        world: WorldCopy,
    ) -> List[Dict[str, Any]]:
        out: List[Dict[str, Any]] = []
        for write in node.get("writes", ()):
            if str(write["table"]) == FILE_TABLE:
                entity = str(write["entity_ref"])
                if entity.startswith("derived:"):
                    _, other, name = entity.split(":", 2)
                    produced = env.get(other, {}).get(name)
                    if isinstance(produced, str):
                        entity = produced if produced.startswith("file:") else "file:%s" % produced
                entry = world.inventory.get(entity) if world.inventory else None
                out.append(
                    {
                        "node_id": str(node["node_id"]),
                        "table": FILE_TABLE,
                        "column": str(write["column"]),
                        "entity_ref": entity,
                        "entity": entity,
                        "value": (
                            entry.content if entry and write["column"] == "content" else None
                        ),
                        "old_value": None,
                        "effect_type": str(write["effect_type"]),
                        "reversibility_class": self._effect_class(node, str(write["effect_type"])),
                        "volatile": False,
                    }
                )
                continue
            target = self._resolve_write_target(write, env)
            if target is None:
                raise GoldInterpreterError(
                    "WRITE_TARGET_UNRESOLVED", "%s.%s" % (node["node_id"], write["entity_ref"])
                )
            app, table, rowid = target
            after = snapshot_row(world.connection(app), table, rowid)
            column = str(write["column"])
            old_row = before.get(target)
            out.append(
                {
                    "node_id": str(node["node_id"]),
                    "table": str(write["table"]),
                    "column": column,
                    "entity_ref": str(write["entity_ref"]),
                    "entity": entity_id(str(write["table"]), rowid),
                    "value": _jsonable(after.get(column)) if after and column != "*" else None,
                    "old_value": _jsonable(old_row.get(column))
                    if old_row and column != "*"
                    else None,
                    "effect_type": str(write["effect_type"]),
                    "reversibility_class": self._effect_class(node, str(write["effect_type"])),
                    "volatile": self._is_volatile(str(write["table"]), column),
                }
            )
        return out

    def _check_undeclared_writes(
        self,
        node: Mapping[str, Any],
        before: Mapping[Tuple[str, str], str],
        after: Mapping[Tuple[str, str], str],
    ) -> None:
        declared = {
            split_table(str(write["table"]))
            for write in node.get("writes", ())
            if str(write["table"]) != FILE_TABLE
        }
        changed = sorted(
            key for key in set(before) | set(after) if before.get(key) != after.get(key)
        )
        undeclared = [key for key in changed if key not in declared]
        if undeclared:
            raise GoldInterpreterError(
                "UNDECLARED_WRITE",
                "%s changed %s without declaring it"
                % (node["node_id"], ", ".join("%s.%s" % key for key in undeclared)),
            )

    def _primary_keys(self, world: WorldCopy, table: str) -> Set[str]:
        app, name = split_table(table)
        key = (app, "pk:" + name)
        if key not in self._column_cache:
            try:
                rows = world.connection(app).execute('PRAGMA table_info("%s")' % name).fetchall()
                self._column_cache[key] = [str(row[1]) for row in rows if row[5]]
            except GoldInterpreterError:
                self._column_cache[key] = []
        return set(self._column_cache[key])

    def _audit_reads(
        self, node: Mapping[str, Any], actual: Set[Tuple[str, str]], world: WorldCopy
    ) -> List[Dict[str, Any]]:
        declared = {(str(r["table"]), str(r["column"])) for r in node.get("reads", ())}
        declared_tables_all = {t for t, c in declared if c == "*"}
        out = []
        for table, column in sorted(actual):
            if (table, column) in declared or table in declared_tables_all or column == "rowid":
                continue
            if column in self._primary_keys(world, table):
                continue
            out.append({"node_id": str(node["node_id"]), "table": table, "column": column})
        return out

    def _resolve_reads(
        self, node: Mapping[str, Any], env: Mapping[str, Mapping[str, Any]], world: WorldCopy
    ) -> List[Dict[str, Any]]:
        node_id = str(node["node_id"])
        produced = env.get(node_id, {})
        out = []
        for read in node.get("reads", ()):
            entity_ref = str(read["entity_ref"])
            entity_set: Optional[List[Any]] = None
            coarse = False
            if entity_ref.endswith(":*"):
                app, table = split_table(str(read["table"]))
                entity_set = self._entity_set_from_values(produced.values())
                coarse = entity_set is None
            elif entity_ref.startswith("derived:"):
                _, other, name = entity_ref.split(":", 2)
                value = env.get(other, {}).get(name)
                entity_set = self._entity_set_from_values([value]) if value is not None else None
                coarse = entity_set is None
            else:
                parsed = parse_entity_ref(entity_ref)
                entity_set = (
                    [int(parsed[2])] if parsed[0] == "row" and parsed[2].isdigit() else [parsed[-1]]
                )
            filter_columns = list(
                read.get("filter_columns") or self._filter_columns(node, read, world)
            )
            out.append(
                {
                    "node_id": node_id,
                    "table": str(read["table"]),
                    "column": str(read["column"]),
                    "entity_ref": entity_ref,
                    "entity_set": entity_set,
                    "filter_columns": filter_columns,
                    "coarse": coarse or bool(read.get("coarse")),
                }
            )
        return out

    @staticmethod
    def _entity_set_from_values(values: Any) -> Optional[List[Any]]:
        ids: List[Any] = []
        for value in values:
            if isinstance(value, bool):
                continue
            if isinstance(value, int):
                ids.append(value)
            elif isinstance(value, Mapping) and isinstance(value.get("id"), int):
                ids.append(value["id"])
            elif isinstance(value, list):
                for item in value:
                    if isinstance(item, Mapping) and isinstance(item.get("id"), int):
                        ids.append(item["id"])
                    elif isinstance(item, int) and not isinstance(item, bool):
                        ids.append(item)
        return sorted(set(ids)) if ids else None

    def _filter_columns(
        self, node: Mapping[str, Any], read: Mapping[str, Any], world: WorldCopy
    ) -> List[str]:
        app, table = split_table(str(read["table"]))
        try:
            columns = set(self._columns(world, app, table))
        except GoldInterpreterError:
            return []
        found: List[str] = []
        for produce in node.get("produces", ()):
            derivation = produce["derivation"]
            if derivation.get("kind") != "sql":
                continue
            query = str(derivation["query"])
            upper = query.upper()
            where = upper.find(" WHERE ")
            if where < 0:
                continue
            end = len(query)
            for stop in (" ORDER BY ", " GROUP BY ", " LIMIT "):
                pos = upper.find(stop, where)
                if 0 <= pos < end:
                    end = pos
            for ident in _IDENT.findall(query[where:end]):
                if ident in columns and ident not in found:
                    found.append(ident)
        return found

    def _evaluate_predicate(
        self,
        verifier: Mapping[str, Any],
        scope: Mapping[str, Any],
        default_app: str,
        world: WorldCopy,
    ) -> Tuple[bool, Any]:
        kind = str(verifier["kind"])
        predicate = str(verifier.get("predicate", ""))
        if kind == "sql":
            conn = world.connection(str(verifier.get("app") or default_app))
            params = _bindable(scope)
            if "expected" in verifier and isinstance(verifier["expected"], _BINDABLE):
                params["expected"] = verifier["expected"]
            try:
                row = conn.execute(predicate, params).fetchone()
            except sqlite3.Error as exc:
                raise GoldInterpreterError(
                    "VERIFIER_ERROR", "%s: %s" % (verifier["verifier_id"], exc)
                ) from exc
            return (row is not None and bool(row[0])), (_jsonable(row) if row is not None else None)
        if kind == "derived":
            env = dict(scope)
            env["expected"] = verifier.get("expected")
            try:
                result = evaluate(predicate, env)
            except ExpressionError as exc:
                raise GoldInterpreterError(
                    "VERIFIER_ERROR", "%s: %s" % (verifier["verifier_id"], exc)
                ) from exc
            return bool(result), _jsonable(result)
        raise GoldInterpreterError("VERIFIER_ERROR", "unsupported verifier kind %r" % kind)

    def _evaluate_verifier(
        self,
        verifier: Optional[Mapping[str, Any]],
        node_id: str,
        scope: Mapping[str, Any],
        node: Mapping[str, Any],
        world: WorldCopy,
    ) -> Optional[Dict[str, Any]]:
        if not verifier or verifier.get("kind") not in self.config.evaluated_verifier_kinds:
            return None
        passed, detail = self._evaluate_predicate(verifier, scope, str(node["app"]), world)
        return {
            "node_id": node_id,
            "verifier_id": str(verifier["verifier_id"]),
            "passed": passed,
            "detail": detail,
        }

    def _evaluate_final(
        self,
        verifier: Optional[Mapping[str, Any]],
        scope: Mapping[str, Any],
        task_ir: Mapping[str, Any],
        world: WorldCopy,
    ) -> Optional[bool]:
        if not verifier:
            return None
        if verifier.get("kind") == "all_of":
            results = [
                self._evaluate_final(part, scope, task_ir, world)
                for part in verifier.get("parts", ())
            ]
            if any(r is False for r in results):
                return False
            return None if all(r is None for r in results) else True
        if verifier.get("kind") not in self.config.evaluated_verifier_kinds:
            return None
        default_app = str(verifier.get("app") or task_ir["nodes"][-1]["app"])
        passed, _ = self._evaluate_predicate(verifier, scope, default_app, world)
        if passed:
            return True
        for alternative in verifier.get("equivalent_final_states", ()):
            merged = dict(verifier)
            merged.update(alternative)
            if self._evaluate_predicate(merged, scope, default_app, world)[0]:
                return True
        return False
