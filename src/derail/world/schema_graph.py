from __future__ import annotations

import hashlib
import re
import sqlite3
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple, Union

import yaml

from derail.world.facts import qualified_table
from derail.world.state import table_names

SCHEMA_GRAPH_VERSION = "schema-graph/1.0"


@dataclass(frozen=True)
class PrincipalTable:
    table: str
    column: str


@dataclass(frozen=True)
class SchemaGraphConfig:
    exclude_tables: Tuple[str, ...]
    id_suffix: str
    email_column_pattern: str
    principal_tables: Tuple[PrincipalTable, ...]
    polymorphic_pairs: Tuple[Tuple[str, str], ...]
    confidence: Mapping[str, float]
    min_confidence_for_generation: float
    config_sha256: str

    @classmethod
    def from_yaml(cls, path: Union[str, Path]) -> "SchemaGraphConfig":
        text = Path(path).read_text(encoding="utf-8")
        raw = yaml.safe_load(text) or {}
        if raw.get("schema_version") != "schema-graph-config/1.0":
            raise ValueError("unsupported schema graph config %r" % raw.get("schema_version"))
        inference = raw.get("inference", {})
        return cls(
            exclude_tables=tuple(raw.get("exclude_tables", ())),
            id_suffix=str(inference.get("id_suffix", "_id")),
            email_column_pattern=str(inference.get("email_column_pattern", "(^|_)email$")),
            principal_tables=tuple(
                PrincipalTable(str(item["table"]), str(item["column"]))
                for item in inference.get("principal_tables", ())
            ),
            polymorphic_pairs=tuple(
                (str(item["type_column"]), str(item["id_column"]))
                for item in inference.get("polymorphic_pairs", ())
            ),
            confidence={str(k): float(v) for k, v in (raw.get("confidence") or {}).items()},
            min_confidence_for_generation=float(raw.get("min_confidence_for_generation", 0.7)),
            config_sha256=hashlib.sha256(text.encode("utf-8")).hexdigest(),
        )


def _file_sha256(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def _table_record(conn: sqlite3.Connection, table: str) -> Dict[str, Any]:
    columns = []
    primary_key = []
    for _, name, ctype, notnull, default, pk in conn.execute('PRAGMA table_info("%s")' % table):
        columns.append(
            {
                "name": str(name),
                "type": str(ctype or ""),
                "notnull": bool(notnull),
                "primary_key": bool(pk),
                "default": default,
            }
        )
        if pk:
            primary_key.append((int(pk), str(name)))
    ddl = conn.execute(
        "SELECT sql FROM sqlite_master WHERE type = 'table' AND name = ?", (table,)
    ).fetchone()
    row_count = int(conn.execute('SELECT count(*) FROM "%s"' % table).fetchone()[0])
    return {
        "name": table,
        "columns": columns,
        "row_count": row_count,
        "primary_key": [name for _, name in sorted(primary_key)],
        "ddl_sha256": hashlib.sha256(str(ddl[0] if ddl else "").encode("utf-8")).hexdigest(),
    }


def _candidate_tables(prefix: str, tables: Sequence[str]) -> Optional[Tuple[str, str]]:
    lowered = {table.lower(): table for table in tables}
    if prefix.lower() in lowered:
        return lowered[prefix.lower()], "id_suffix_exact"
    plural = prefix[:-1] + "ies" if prefix.endswith("y") else None
    for candidate in (prefix + "s", prefix + "es", plural):
        if candidate and candidate.lower() in lowered:
            return lowered[candidate.lower()], "id_suffix_plural"
    return None


def _explicit_relations(conn: sqlite3.Connection, app: str, table: str) -> List[Dict[str, Any]]:
    out = []
    for row in conn.execute('PRAGMA foreign_key_list("%s")' % table):
        _, _, ref_table, from_col, to_col = row[0], row[1], row[2], row[3], row[4]
        out.append(
            {
                "from_table": qualified_table(app, table),
                "from_column": str(from_col),
                "to_table": qualified_table(app, str(ref_table)),
                "to_column": str(to_col or "rowid"),
                "kind": "explicit",
                "rule": "foreign_key",
                "confidence": 1.0,
                "type_column": None,
            }
        )
    return out


def _inferred_relations(
    app: str,
    table: Dict[str, Any],
    tables: Sequence[str],
    config: SchemaGraphConfig,
    principal: Optional[PrincipalTable],
    explicit: Sequence[Tuple[str, str]],
) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    names = [column["name"] for column in table["columns"]]
    poly_id_columns = {
        id_col
        for type_col, id_col in config.polymorphic_pairs
        if type_col in names and id_col in names
    }
    email_re = re.compile(config.email_column_pattern)
    for column in names:
        if (table["name"], column) in explicit:
            continue
        if column in poly_id_columns:
            type_col = [t for t, i in config.polymorphic_pairs if i == column][0]
            out.append(
                {
                    "from_table": qualified_table(app, table["name"]),
                    "from_column": column,
                    "to_table": None,
                    "to_column": None,
                    "kind": "polymorphic",
                    "rule": "polymorphic_pair",
                    "confidence": config.confidence.get("polymorphic_pair", 0.5),
                    "type_column": type_col,
                }
            )
            continue
        if column.endswith(config.id_suffix) and column != config.id_suffix:
            prefix = column[: -len(config.id_suffix)]
            target = _candidate_tables(prefix, [t for t in tables if t != table["name"]])
            if target is not None:
                out.append(
                    {
                        "from_table": qualified_table(app, table["name"]),
                        "from_column": column,
                        "to_table": qualified_table(app, target[0]),
                        "to_column": "id",
                        "kind": "inferred",
                        "rule": target[1],
                        "type_column": None,
                        "confidence": config.confidence.get(target[1], 0.8),
                    }
                )
            continue
        if principal is not None and email_re.search(column) and table["name"] != principal.table:
            out.append(
                {
                    "from_table": qualified_table(app, table["name"]),
                    "from_column": column,
                    "to_table": qualified_table(app, principal.table),
                    "to_column": principal.column,
                    "kind": "inferred",
                    "rule": "email_principal",
                    "confidence": config.confidence.get("email_principal", 0.7),
                    "type_column": None,
                }
            )
    return out


def build_schema_graph(
    sources: Mapping[str, Union[str, Path]],
    config: SchemaGraphConfig,
    graph_id: str,
    progress: Optional[Callable[[int, int, str], None]] = None,
    provenance: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    databases: List[Dict[str, Any]] = []
    relations: List[Dict[str, Any]] = []
    excluded = set(config.exclude_tables)
    apps = sorted(sources)
    for index, app in enumerate(apps, 1):
        path = Path(sources[app])
        conn = sqlite3.connect("file:%s?mode=ro" % path, uri=True)
        try:
            tables = [name for name in table_names(conn) if name not in excluded]
            records = [_table_record(conn, name) for name in tables]
            principal = next(
                (
                    p
                    for p in config.principal_tables
                    if p.table in tables
                    and any(
                        c["name"] == p.column
                        for r in records
                        if r["name"] == p.table
                        for c in r["columns"]
                    )
                ),
                None,
            )
            explicit: List[Dict[str, Any]] = []
            for name in tables:
                explicit.extend(_explicit_relations(conn, app, name))
            explicit_keys = [(r["from_table"].split(".", 1)[1], r["from_column"]) for r in explicit]
            relations.extend(explicit)
            for record in records:
                relations.extend(
                    _inferred_relations(app, record, tables, config, principal, explicit_keys)
                )
            databases.append(
                {
                    "app": app,
                    "file_sha256": _file_sha256(path),
                    "source_path": str(path),
                    "tables": records,
                }
            )
        finally:
            conn.close()
        if progress:
            progress(index, len(apps), app)
    inferred = sum(1 for r in relations if r["kind"] != "explicit")
    return {
        "schema_version": SCHEMA_GRAPH_VERSION,
        "graph_id": graph_id,
        "databases": databases,
        "relations": relations,
        "excluded_tables": sorted(excluded),
        "config_sha256": config.config_sha256,
        "inferred_ratio": round(inferred / len(relations), 4) if relations else 0.0,
        "min_confidence_for_generation": config.min_confidence_for_generation,
        "provenance": dict(provenance or {}),
    }


@dataclass
class SchemaGraph:
    record: Mapping[str, Any]
    _tables: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    _outgoing: Dict[str, List[Dict[str, Any]]] = field(default_factory=dict)
    _incoming: Dict[str, List[Dict[str, Any]]] = field(default_factory=dict)

    @classmethod
    def from_record(cls, record: Mapping[str, Any]) -> "SchemaGraph":
        graph = cls(record)
        for database in record["databases"]:
            for table in database["tables"]:
                graph._tables[qualified_table(database["app"], table["name"])] = table
        for relation in record["relations"]:
            graph._outgoing.setdefault(relation["from_table"], []).append(relation)
            if relation["to_table"]:
                graph._incoming.setdefault(relation["to_table"], []).append(relation)
        return graph

    def tables(self) -> List[str]:
        return sorted(self._tables)

    def record_for(self, table: str) -> Dict[str, Any]:
        return self._tables[table]

    def columns(self, table: str) -> List[str]:
        return [column["name"] for column in self._tables[table]["columns"]]

    def has_column(self, table: str, column: str) -> bool:
        return table in self._tables and column in self.columns(table)

    def references(
        self, table: str, column: Optional[str] = None, min_confidence: float = 0.0
    ) -> List[Dict[str, Any]]:
        out = self._outgoing.get(table, [])
        return [
            r
            for r in out
            if (column is None or r["from_column"] == column)
            and float(r.get("confidence", 1.0)) >= min_confidence
        ]

    def referenced_by(self, table: str, min_confidence: float = 0.0) -> List[Dict[str, Any]]:
        return [
            r
            for r in self._incoming.get(table, [])
            if float(r.get("confidence", 1.0)) >= min_confidence
        ]

    def generation_relations(self) -> List[Dict[str, Any]]:
        threshold = float(self.record.get("min_confidence_for_generation", 0.7))
        return [r for r in self.record["relations"] if float(r.get("confidence", 1.0)) >= threshold]

    def foreign_key_path(
        self, source: str, target: str, max_hops: int = 3
    ) -> Optional[List[Dict[str, Any]]]:
        frontier: List[Tuple[str, List[Dict[str, Any]]]] = [(source, [])]
        seen = {source}
        while frontier:
            next_frontier: List[Tuple[str, List[Dict[str, Any]]]] = []
            for table, path in frontier:
                if len(path) >= max_hops:
                    continue
                steps = [(r["to_table"], r) for r in self._outgoing.get(table, []) if r["to_table"]]
                steps += [(r["from_table"], r) for r in self._incoming.get(table, [])]
                for nxt, relation in steps:
                    if nxt in seen:
                        continue
                    if nxt == target:
                        return path + [relation]
                    seen.add(nxt)
                    next_frontier.append((nxt, path + [relation]))
            frontier = next_frontier
        return None
