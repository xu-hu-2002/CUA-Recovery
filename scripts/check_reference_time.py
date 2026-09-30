#!/usr/bin/env python3
"""Verify the pinned reference time and the ${TODAY+-k} variables against the databases (D-008).

    DERAIL_VM_DB_DIR=<dumps> python scripts/check_reference_time.py \
        --variables third_party/MyPCBench/tasks/final/variables.json \
        --probe JAMAICA_CHECK_IN=cheskepdia.bookings.check_in:2 ...

Checks (1) config ``v1.reference_time`` == ``<db_dir>/_seed_meta.json`` bake_reference_time,
(2) every ``--probe NAME=app.table.column:rowid`` expands the variable with the reference date
and compares it with the database cell.  Exit 1 on any mismatch; a drift means the image was
rebuilt or re-anchored and gold values must be recomputed.
"""

from __future__ import annotations

import argparse
import json
import os
import sqlite3
import sys
from pathlib import Path

REPO_ROOT = Path(os.environ.get("DERAIL_REPO_ROOT", Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(REPO_ROOT / "src"))

from derail.ir.extract import V1ExtractorConfig  # noqa: E402
from derail.ir.rubric_check import expand_variables  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "--config", type=Path, default=REPO_ROOT / "configs/synthesis/task_ir_v1_extractor.yaml"
    )
    parser.add_argument("--variables", type=Path, required=True)
    parser.add_argument("--db-dir", type=Path, default=os.environ.get("DERAIL_VM_DB_DIR"))
    parser.add_argument("--probe", action="append", default=[], help="NAME=app.table.column:rowid")
    args = parser.parse_args()
    if not args.db_dir:
        raise SystemExit("need --db-dir or DERAIL_VM_DB_DIR")
    config = V1ExtractorConfig.from_yaml(args.config, REPO_ROOT)
    failures = []
    meta_path = Path(args.db_dir) / "_seed_meta.json"
    if meta_path.is_file():
        bake = json.loads(meta_path.read_text(encoding="utf-8")).get("bake_reference_time")
        status = "ok" if bake == config.reference_time else "MISMATCH"
        print("reference_time config=%s seed_meta=%s %s" % (config.reference_time, bake, status))
        if status != "ok":
            failures.append("reference_time")
    else:
        print("WARNING: %s missing; bake time not verified" % meta_path)
    variables = expand_variables(
        json.loads(args.variables.read_text(encoding="utf-8")), config.reference_time
    )
    for probe in args.probe:
        name, _, target = probe.partition("=")
        location, _, rowid = target.partition(":")
        app, table, column = location.split(".")
        conn = sqlite3.connect(
            "file:%s?mode=ro" % (Path(args.db_dir) / ("%s.sqlite" % app)), uri=True
        )
        cell = conn.execute(
            'SELECT "%s" FROM "%s" WHERE rowid = ?' % (column, table), (int(rowid),)
        ).fetchone()
        conn.close()
        actual = cell[0] if cell else None
        expected = variables.get(name)
        status = (
            "ok" if actual is not None and str(actual)[:10] == str(expected)[:10] else "MISMATCH"
        )
        print("%s expected=%s actual=%s %s" % (name, expected, actual, status))
        if status != "ok":
            failures.append(name)
    print("failures: %s" % (failures or "none"))
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
