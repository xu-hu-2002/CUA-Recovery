"""Deterministic state digests and row snapshots over a set of SQLite databases.

The gold interpreter hashes the world after every node (``gold-lineage/1.0``
``state_timeline``); replay verification compares the same digest after reset + changelog
replay (execution doc v1.2 section 4.3).  Both must agree byte for byte, so the digest is
defined here once: tables sorted by name, rows ordered by rowid, values JSON-encoded.

Bookkeeping tables (``sqlite_sequence``, ``_changelog`` ...) are excluded through the
``exclude_tables`` argument, which callers read from configuration.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
from typing import Tuple, Any, Dict, Iterable, List, Mapping, Optional, Sequence


def table_names(conn: sqlite3.Connection) -> List[str]:
    """Same rule as ``_tables`` in infra/snapshot/changelog_replay.py (SQLite < 3.37 returns no
    rows for ``PRAGMA table_list``; fall back to ``sqlite_master`` minus virtual/shadow tables)."""

    try:
        rows = conn.execute("PRAGMA table_list").fetchall()
    except sqlite3.OperationalError:
        rows = []
    if rows:
        return sorted(
            str(row[1])
            for row in rows
            if str(row[2]) == "table"
            and str(row[1]) not in ("sqlite_schema", "sqlite_temp_schema")
        )
    rows = conn.execute("SELECT name, sql FROM sqlite_master WHERE type = 'table'").fetchall()
    virtual = [
        str(n)
        for n, sql in rows
        if str(sql or "").lstrip().upper().startswith("CREATE VIRTUAL TABLE")
    ]
    return sorted(
        str(n)
        for n, _ in rows
        if str(n) not in virtual and not any(str(n).startswith(v + "_") for v in virtual)
    )


def table_columns(conn: sqlite3.Connection, table: str) -> List[str]:
    return [str(row[1]) for row in conn.execute('PRAGMA table_info("%s")' % table)]


def _has_rowid(conn: sqlite3.Connection, table: str) -> bool:
    sql = conn.execute(
        "SELECT sql FROM sqlite_master WHERE type = 'table' AND name = ?", (table,)
    ).fetchone()
    return bool(sql) and "WITHOUT ROWID" not in str(sql[0]).upper()


def iter_rows(
    conn: sqlite3.Connection, table: str, exclude_columns: Iterable[str] = ()
) -> Iterable[Dict[str, Any]]:
    """Yield rows as ``{column: value}`` with ``_rowid`` first, in rowid order."""

    skip = set(exclude_columns)
    columns = [c for c in table_columns(conn, table) if c not in skip]
    quoted = ", ".join('"%s"' % column for column in columns)
    if _has_rowid(conn, table):
        cursor = conn.execute('SELECT rowid, %s FROM "%s" ORDER BY rowid' % (quoted, table))
        for row in cursor:
            yield dict([("_rowid", row[0])] + list(zip(columns, row[1:])))
    else:
        cursor = conn.execute('SELECT %s FROM "%s" ORDER BY %s' % (quoted, table, quoted))
        for row in cursor:
            yield dict(zip(columns, row))


def _encode(value: Any) -> Any:
    if isinstance(value, (bytes, bytearray, memoryview)):
        return {"__bytes_sha256__": hashlib.sha256(bytes(value)).hexdigest()}
    if isinstance(value, float):
        return float(format(value, ".15g"))
    return value


def database_digest(
    conn: sqlite3.Connection,
    exclude_tables: Sequence[str] = (),
    exclude_columns: Optional[Mapping[str, Iterable[str]]] = None,
) -> str:
    """sha256 over every included table's rows in canonical order.

    ``exclude_columns`` maps table -> columns left out of the hash (volatile columns, D-012).
    """

    excluded = set(exclude_tables)
    skip = exclude_columns or {}
    hasher = hashlib.sha256()
    for table in table_names(conn):
        if table in excluded:
            continue
        hasher.update(("\x00table:%s\n" % table).encode("utf-8"))
        for row in iter_rows(conn, table, skip.get(table, ())):
            encoded = json.dumps(
                {key: _encode(value) for key, value in row.items()},
                sort_keys=True,
                ensure_ascii=False,
                separators=(",", ":"),
            )
            hasher.update(encoded.encode("utf-8"))
            hasher.update(b"\n")
    return hasher.hexdigest()


def state_digest(
    connections: Mapping[str, sqlite3.Connection],
    exclude_tables: Sequence[str] = (),
    exclude_columns: Optional[Mapping[str, Mapping[str, Iterable[str]]]] = None,
) -> str:
    """Digest of a whole world: per-database digests combined in app order.

    ``exclude_columns`` is ``{app: {table: columns}}`` (volatile columns, D-012).
    """

    hasher = hashlib.sha256()
    per_app = exclude_columns or {}
    for app in sorted(connections):
        digest = database_digest(connections[app], exclude_tables, per_app.get(app))
        hasher.update(("%s:%s\n" % (app, digest)).encode())
    return hasher.hexdigest()


def snapshot_row(conn: sqlite3.Connection, table: str, rowid: int) -> Optional[Dict[str, Any]]:
    """Return ``{column: value}`` for ``rowid`` or ``None`` when the row does not exist."""

    columns = table_columns(conn, table)
    quoted = ", ".join('"%s"' % column for column in columns)
    row = conn.execute('SELECT %s FROM "%s" WHERE rowid = ?' % (quoted, table), (rowid,)).fetchone()
    if row is None:
        return None
    return dict(zip(columns, row))


def table_digests(
    conn: sqlite3.Connection,
    exclude_tables: Sequence[str] = (),
    exclude_columns: Optional[Mapping[str, Iterable[str]]] = None,
) -> Dict[str, str]:
    """Per-table digests; lets a caller tell *which* tables changed between two points."""

    excluded = set(exclude_tables)
    skip = exclude_columns or {}
    digests: Dict[str, str] = {}
    for table in table_names(conn):
        if table in excluded:
            continue
        hasher = hashlib.sha256()
        for row in iter_rows(conn, table, skip.get(table, ())):
            hasher.update(
                json.dumps(
                    {key: _encode(value) for key, value in row.items()},
                    sort_keys=True,
                    ensure_ascii=False,
                    separators=(",", ":"),
                ).encode("utf-8")
            )
            hasher.update(b"\n")
        digests[table] = hasher.hexdigest()
    return digests


def database_and_table_digests(
    conn: sqlite3.Connection,
    exclude_tables: Sequence[str] = (),
    exclude_columns: Optional[Mapping[str, Iterable[str]]] = None,
) -> Tuple[str, Dict[str, str]]:
    """One pass over the rows yielding exactly ``database_digest`` and ``table_digests``."""

    excluded = set(exclude_tables)
    skip = exclude_columns or {}
    whole = hashlib.sha256()
    per_table: Dict[str, str] = {}
    for table in table_names(conn):
        if table in excluded:
            continue
        whole.update(("\x00table:%s\n" % table).encode("utf-8"))
        hasher = hashlib.sha256()
        for row in iter_rows(conn, table, skip.get(table, ())):
            encoded = json.dumps(
                {key: _encode(value) for key, value in row.items()},
                sort_keys=True,
                ensure_ascii=False,
                separators=(",", ":"),
            ).encode("utf-8")
            whole.update(encoded)
            whole.update(b"\n")
            hasher.update(encoded)
            hasher.update(b"\n")
        per_table[table] = hasher.hexdigest()
    return whole.hexdigest(), per_table
