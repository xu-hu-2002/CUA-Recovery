"""Build SQLite databases from ``miniworld-seed/1.0`` files.

A seed file carries, per table, the verbatim ``CREATE TABLE`` statement of the source database
and a list of rows.  Building from a seed instead of committing binary ``.sqlite`` files keeps
the fixture reviewable in diffs and reproducible on every machine.

Entry points
------------
``load_seed(path) -> dict``
``build_database(seed, path) -> Path``          one database file from one seed
``build_world(seeds, out_dir) -> Dict[str, Path]``  ``{app: db_path}`` for several seeds
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any, Dict, Mapping, Union

SEED_SCHEMA_VERSION = "miniworld-seed/1.0"


def load_seed(path: Union[str, Path]) -> Dict[str, Any]:
    seed = json.loads(Path(path).read_text(encoding="utf-8"))
    if seed.get("schema_version") != SEED_SCHEMA_VERSION:
        raise ValueError("unsupported seed version %r in %s" % (seed.get("schema_version"), path))
    for field in ("database", "tables"):
        if field not in seed:
            raise ValueError("seed %s lacks %r" % (path, field))
    return seed


def build_database(seed: Mapping[str, Any], path: Union[str, Path]) -> Path:
    """Create ``path`` from ``seed``; an existing file is replaced."""

    target = Path(path)
    if target.exists():
        target.unlink()
    target.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(target))
    try:
        conn.execute("PRAGMA foreign_keys = OFF")
        for table in seed["tables"]:
            conn.execute(table["ddl"])
            columns = list(table["columns"])
            if not columns:
                continue
            placeholders = ", ".join("?" for _ in columns)
            quoted = ", ".join('"%s"' % column for column in columns)
            conn.executemany(
                'INSERT INTO "%s" (%s) VALUES (%s)' % (table["name"], quoted, placeholders),
                [list(row) for row in table["rows"]],
            )
        conn.commit()
        # Rebuild from a single, deterministic page layout so digests are stable across runs.
        conn.execute("VACUUM")
    finally:
        conn.close()
    return target


def build_world(
    seeds: Mapping[str, Union[str, Path]], out_dir: Union[str, Path]
) -> Dict[str, Path]:
    """Build every seed into ``out_dir/<database>.sqlite`` and return ``{app: path}``."""

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    built: Dict[str, Path] = {}
    for app, seed_path in sorted(seeds.items()):
        seed = load_seed(seed_path)
        built[app] = build_database(seed, out / ("%s.sqlite" % seed["database"]))
    return built
