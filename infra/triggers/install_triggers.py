#!/usr/bin/env python3
"""Install the change log on SQLite databases."""

from __future__ import annotations

import argparse
import glob
import json
import os
import sqlite3
import sys
from typing import Dict, Iterable, List, Sequence

CHANGELOG_TABLE = "_changelog"
CURSOR_TABLE = "_cursor"
TRIGGER_PREFIX = "trg_derail_"
DEFAULT_SKIP = ("sqlite_sequence", "sqlite_stat1", CHANGELOG_TABLE, CURSOR_TABLE)
CURSOR_UNSET = -1

CHANGELOG_DDL = """
CREATE TABLE IF NOT EXISTS {log} (
  seq INTEGER PRIMARY KEY AUTOINCREMENT,
  ts REAL NOT NULL,
  tbl TEXT NOT NULL,
  rowid INTEGER NOT NULL,
  op TEXT NOT NULL CHECK (op IN ('INSERT', 'UPDATE', 'DELETE')),
  old_json TEXT,
  new_json TEXT,
  action_index INTEGER NOT NULL
)
"""
CURSOR_DDL = """
CREATE TABLE IF NOT EXISTS {cursor} (
  id INTEGER PRIMARY KEY CHECK (id = 1),
  action_index INTEGER NOT NULL,
  updated_at REAL NOT NULL
)
"""


def _quote(name: str) -> str:
    return '"%s"' % name.replace('"', '""')


def user_tables(conn: sqlite3.Connection, skip: Sequence[str] = DEFAULT_SKIP) -> List[str]:
    table_types = {}
    try:
        table_types = {str(row[1]): str(row[2]) for row in conn.execute("PRAGMA table_list")}
    except sqlite3.OperationalError:
        pass
    rows = conn.execute(
        "SELECT name, sql FROM sqlite_master WHERE type = 'table' ORDER BY name"
    ).fetchall()
    virtual = [
        str(n) for n, sql in rows if sql and sql.lstrip().upper().startswith("CREATE VIRTUAL TABLE")
    ]
    out = []
    for name, sql in rows:
        if name in skip or name.startswith("sqlite_"):
            continue
        if table_types.get(str(name), "table") != "table":
            continue
        if any(str(name).startswith(v + "_") for v in virtual):
            continue
        if sql and sql.lstrip().upper().startswith("CREATE VIRTUAL TABLE"):
            continue
        if sql and "WITHOUT ROWID" in sql.upper():
            continue
        out.append(str(name))
    return out


def table_columns(conn: sqlite3.Connection, table: str) -> List[str]:
    return [str(row[1]) for row in conn.execute("PRAGMA table_info(%s)" % _quote(table))]


def _json_object(alias: str, columns: Iterable[str]) -> str:
    parts = ", ".join(
        "'%s', %s.%s" % (column.replace("'", "''"), alias, _quote(column)) for column in columns
    )
    return "json_object(%s)" % parts


def trigger_statements(table: str, columns: Sequence[str]) -> Dict[str, str]:
    cursor_expr = "COALESCE((SELECT action_index FROM %s WHERE id = 1), %d)" % (
        CURSOR_TABLE,
        CURSOR_UNSET,
    )
    base = (
        "CREATE TRIGGER %s AFTER %s ON %s BEGIN INSERT INTO %s "
        "(ts, tbl, rowid, op, old_json, new_json, action_index) VALUES "
        "((julianday('now') - 2440587.5) * 86400.0, '%s', %s, '%s', %s, %s, %s); END"
    )
    log = CHANGELOG_TABLE
    quoted = _quote(table)
    return {
        "insert": base
        % (
            _quote(TRIGGER_PREFIX + table + "_insert"),
            "INSERT",
            quoted,
            log,
            table.replace("'", "''"),
            "NEW.rowid",
            "INSERT",
            "NULL",
            _json_object("NEW", columns),
            cursor_expr,
        ),
        "update": base
        % (
            _quote(TRIGGER_PREFIX + table + "_update"),
            "UPDATE",
            quoted,
            log,
            table.replace("'", "''"),
            "NEW.rowid",
            "UPDATE",
            _json_object("OLD", columns),
            _json_object("NEW", columns),
            cursor_expr,
        ),
        "delete": base
        % (
            _quote(TRIGGER_PREFIX + table + "_delete"),
            "DELETE",
            quoted,
            log,
            table.replace("'", "''"),
            "OLD.rowid",
            "DELETE",
            _json_object("OLD", columns),
            "NULL",
            cursor_expr,
        ),
    }


def installed_triggers(conn: sqlite3.Connection) -> List[str]:
    rows = conn.execute(
        "SELECT name FROM sqlite_master WHERE type = 'trigger' AND name LIKE ? ORDER BY name",
        (TRIGGER_PREFIX + "%",),
    ).fetchall()
    return [str(row[0]) for row in rows]


def install(conn: sqlite3.Connection, skip: Sequence[str] = DEFAULT_SKIP) -> Dict[str, int]:
    conn.execute(CHANGELOG_DDL.format(log=CHANGELOG_TABLE))
    conn.execute(CURSOR_DDL.format(cursor=CURSOR_TABLE))
    conn.execute(
        "INSERT OR IGNORE INTO %s (id, action_index, updated_at) VALUES (1, ?, "
        "(julianday('now') - 2440587.5) * 86400.0)" % CURSOR_TABLE,
        (CURSOR_UNSET,),
    )
    for trigger in installed_triggers(conn):
        conn.execute("DROP TRIGGER IF EXISTS %s" % _quote(trigger))
    tables = user_tables(conn, skip)
    count = 0
    for table in tables:
        for statement in trigger_statements(table, table_columns(conn, table)).values():
            conn.execute(statement)
            count += 1
    conn.commit()
    return {"tables": len(tables), "triggers": count}


def uninstall(conn: sqlite3.Connection, drop_log: bool = False) -> int:
    triggers = installed_triggers(conn)
    for trigger in triggers:
        conn.execute("DROP TRIGGER IF EXISTS %s" % _quote(trigger))
    if drop_log:
        conn.execute("DROP TABLE IF EXISTS %s" % CHANGELOG_TABLE)
        conn.execute("DROP TABLE IF EXISTS %s" % CURSOR_TABLE)
    conn.commit()
    return len(triggers)


def set_cursor(conn: sqlite3.Connection, action_index: int) -> None:
    conn.execute(
        "INSERT INTO %s (id, action_index, updated_at) VALUES (1, ?, "
        "(julianday('now') - 2440587.5) * 86400.0) ON CONFLICT(id) DO UPDATE SET "
        "action_index = excluded.action_index, updated_at = excluded.updated_at" % CURSOR_TABLE,
        (int(action_index),),
    )
    conn.commit()


def read_changelog(
    conn: sqlite3.Connection, since_seq: int = 0, db: str = ""
) -> List[Dict[str, object]]:
    rows = conn.execute(
        "SELECT seq, ts, tbl, rowid, op, old_json, new_json, action_index FROM %s "
        "WHERE seq > ? ORDER BY seq" % CHANGELOG_TABLE,
        (int(since_seq),),
    ).fetchall()
    return [
        {
            "schema_version": "changelog-row/1.0",
            "db": db,
            "seq": int(r[0]),
            "ts": float(r[1]),
            "tbl": str(r[2]),
            "rowid": int(r[3]),
            "op": str(r[4]),
            "old_json": r[5],
            "new_json": r[6],
            "action_index": int(r[7]),
        }
        for r in rows
    ]


def _expand(paths: Sequence[str], globs: Sequence[str]) -> List[str]:
    out: List[str] = list(paths)
    for pattern in globs:
        out.extend(sorted(glob.glob(pattern)))
    seen = []
    for path in out:
        real = os.path.realpath(path)
        if os.path.islink(path) or not os.path.isfile(path) or os.path.getsize(path) == 0:
            continue
        if real not in seen:
            seen.append(real)
    return seen


def main(argv: Sequence[str] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("databases", nargs="*")
    parser.add_argument("--glob", action="append", default=[])
    parser.add_argument("--uninstall", action="store_true")
    parser.add_argument("--drop-log", action="store_true", help="with --uninstall: drop the log")
    args = parser.parse_args(argv)
    targets = _expand(args.databases, args.glob)
    if not targets:
        print("no databases matched", file=sys.stderr)
        return 2
    report = {}
    for path in targets:
        conn = sqlite3.connect(path)
        try:
            if args.uninstall:
                report[path] = {"dropped_triggers": uninstall(conn, args.drop_log)}
            else:
                report[path] = install(conn)
        finally:
            conn.close()
        print(
            "[%d/%d] %s %s" % (len(report), len(targets), path, json.dumps(report[path])),
            file=sys.stderr,
        )
    print(json.dumps(report, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
