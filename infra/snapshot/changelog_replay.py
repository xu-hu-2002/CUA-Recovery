#!/usr/bin/env python3
"""Step-level snapshots by changelog replay."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import sqlite3
from typing import Dict, Iterable, List, Mapping, Optional, Sequence

BOOKKEEPING = ("sqlite_sequence", "sqlite_stat1", "_changelog", "_cursor", "_seed_meta", "_meta")


def _quote(name: str) -> str:
    return '"%s"' % name.replace('"', '""')


def apply_row(conn: sqlite3.Connection, row: Mapping[str, object]) -> None:
    table = _quote(str(row["tbl"]))
    op = str(row["op"])
    if op == "DELETE":
        conn.execute("DELETE FROM %s WHERE rowid = ?" % table, (int(row["rowid"]),))
        return
    values = json.loads(str(row["new_json"]))
    columns = list(values)
    if op == "INSERT":
        conn.execute(
            "INSERT OR REPLACE INTO %s (rowid, %s) VALUES (?, %s)"
            % (table, ", ".join(_quote(c) for c in columns), ", ".join("?" for _ in columns)),
            [int(row["rowid"])] + [values[c] for c in columns],
        )
    elif op == "UPDATE":
        conn.execute(
            "UPDATE %s SET %s WHERE rowid = ?"
            % (table, ", ".join("%s = ?" % _quote(c) for c in columns)),
            [values[c] for c in columns] + [int(row["rowid"])],
        )
    else:
        raise ValueError("unknown changelog op %r" % op)


def replay(
    baseline: str,
    rows: Iterable[Mapping[str, object]],
    out: str,
    until_action: Optional[int] = None,
    until_seq: Optional[int] = None,
) -> Dict[str, int]:
    shutil.copyfile(baseline, out)
    conn = sqlite3.connect(out)
    applied = skipped = 0
    try:
        for (name,) in conn.execute(
            "SELECT name FROM sqlite_master WHERE type = 'trigger' AND name LIKE 'trg_recovery_%'"
        ).fetchall():
            conn.execute("DROP TRIGGER IF EXISTS %s" % _quote(str(name)))
        conn.execute("PRAGMA foreign_keys = OFF")
        for row in sorted(rows, key=lambda r: int(r["seq"])):
            if until_seq is not None and int(row["seq"]) > until_seq:
                skipped += 1
                continue
            if until_action is not None and int(row["action_index"]) > until_action:
                skipped += 1
                continue
            apply_row(conn, row)
            applied += 1
        conn.commit()
    finally:
        conn.close()
    return {"applied": applied, "skipped": skipped}


def load_volatile_rules(path: Optional[str]) -> Dict[str, object]:
    if not path:
        return {"patterns": [], "keep": {}}
    with open(path, "r", encoding="utf-8") as handle:
        raw = json.load(handle)
    return {"patterns": list(raw.get("patterns", [])), "keep": dict(raw.get("keep", {}))}


def volatile_columns(
    app: str, table: str, columns: Sequence[str], rules: Mapping[str, object]
) -> List[str]:
    keep = set(rules.get("keep", {}).get("%s.%s" % (app, table), []))  # type: ignore[union-attr]
    patterns = rules.get("patterns", [])
    return [
        c
        for c in columns
        if c not in keep and any(re.search(p, c) for p in patterns)  # type: ignore[union-attr]
    ]


def _encode(value: object) -> object:
    if isinstance(value, (bytes, bytearray, memoryview)):
        return {"__bytes_sha256__": hashlib.sha256(bytes(value)).hexdigest()}
    if isinstance(value, float):
        return float(format(value, ".15g"))
    return value


def _tables(conn: sqlite3.Connection) -> List[str]:
    try:
        listed = conn.execute("PRAGMA table_list").fetchall()
    except sqlite3.OperationalError:
        listed = []
    if listed:
        return sorted(
            str(row[1])
            for row in listed
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


def digest(
    path: str,
    exclude: Sequence[str] = BOOKKEEPING,
    rules: Optional[Mapping[str, object]] = None,
    app: Optional[str] = None,
) -> str:
    app = app or os.path.splitext(os.path.basename(path))[0]
    conn = sqlite3.connect("file:%s?mode=ro" % path, uri=True)
    hasher = hashlib.sha256()
    try:
        for table in _tables(conn):
            if table in exclude:
                continue
            columns = [str(r[1]) for r in conn.execute("PRAGMA table_info(%s)" % _quote(table))]
            if rules:
                skip = set(volatile_columns(app, table, columns, rules))
                columns = [c for c in columns if c not in skip]
            hasher.update(("\x00table:%s\n" % table).encode("utf-8"))
            quoted = ", ".join(_quote(c) for c in columns)
            for row in conn.execute(
                "SELECT rowid, %s FROM %s ORDER BY rowid" % (quoted, _quote(table))
            ):
                record = {"_rowid": row[0]}
                for column, value in zip(columns, row[1:]):
                    record[column] = _encode(value)
                hasher.update(
                    json.dumps(
                        record, sort_keys=True, ensure_ascii=False, separators=(",", ":")
                    ).encode("utf-8")
                )
                hasher.update(b"\n")
    finally:
        conn.close()
    return hasher.hexdigest()


def load_rows(path: str) -> List[Dict[str, object]]:
    with open(path, "r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def main(argv: Sequence[str] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)
    rep = sub.add_parser("replay")
    rep.add_argument("--baseline", required=True)
    rep.add_argument("--changelog", required=True, help="changelog-row/1.0 JSON lines")
    rep.add_argument("--out", required=True)
    rep.add_argument("--until-action", type=int)
    rep.add_argument("--until-seq", type=int)
    dig = sub.add_parser("digest")
    dig.add_argument("database")
    for sub_parser in (rep, dig):
        sub_parser.add_argument("--volatile-columns", help="volatile_columns.json (D-012)")
    args = parser.parse_args(argv)
    rules = load_volatile_rules(args.volatile_columns)
    if args.command == "replay":
        report = replay(
            args.baseline, load_rows(args.changelog), args.out, args.until_action, args.until_seq
        )
        app = os.path.splitext(os.path.basename(args.baseline))[0]
        report["digest"] = digest(args.out, rules=rules, app=app)
        print(json.dumps(report))
        return 0
    print(digest(args.database, rules=rules))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
