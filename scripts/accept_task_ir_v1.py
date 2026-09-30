#!/usr/bin/env python3
"""Freeze the validated Task IRs of one or more runs into an accepted set (Phase 2 output).

    python scripts/accept_task_ir_v1.py --run data/synthesis/task_ir_v1/full_v1_run_01/reparse_02 \
        --run .../full_v1_run_01/fix_02 --out data/synthesis/task_ir_v1/accepted

Later runs override earlier ones per task.  A task is accepted when its rubric check says
``auto_validated`` (literal or lexical).  Writes ``task_ir/<task>.json`` and
``gold_lineage/<task>.gold_lineage.json`` copies, ``accepted.jsonl`` (task, run, verdict code)
and ``manifest.json`` with counts by verdict so the Phase 2 gate (>= 80% auto_validated) is
read off one file.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from collections import Counter
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "--run", action="append", required=True, help="run dir with task_ir/ and rubric_check/"
    )
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--total-tasks", type=int, default=184)
    args = parser.parse_args()
    latest: dict = {}
    for run in [Path(r) for r in args.run]:
        checks = run / "rubric_check" / "rubric_checks.jsonl"
        if not checks.is_file():
            print("skip %s: no rubric_check" % run, file=sys.stderr)
            continue
        for line in checks.read_text(encoding="utf-8").splitlines():
            record = json.loads(line)
            latest[record["task_id"]] = (run, record)
    (args.out / "task_ir").mkdir(parents=True, exist_ok=True)
    (args.out / "gold_lineage").mkdir(parents=True, exist_ok=True)
    rows, counts = [], Counter()
    for task_id, (run, record) in sorted(latest.items()):
        counts[(record["verdict"], record.get("code"))] += 1
        if record["verdict"] != "auto_validated":
            continue
        shutil.copyfile(
            run / "task_ir" / ("%s.json" % task_id), args.out / "task_ir" / ("%s.json" % task_id)
        )
        gold = run / "rubric_check" / ("%s.gold_lineage.json" % task_id)
        if gold.is_file():
            shutil.copyfile(gold, args.out / "gold_lineage" / ("%s.gold_lineage.json" % task_id))
        rows.append({"task_id": task_id, "run": str(run), "code": record.get("code")})
    (args.out / "accepted.jsonl").write_text(
        "".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8"
    )
    accepted = len(rows)
    manifest = {
        "schema_version": "accepted-ir-set/1.0",
        "runs": [str(r) for r in args.run],
        "tasks_seen": len(latest),
        "accepted": accepted,
        "total_tasks": args.total_tasks,
        "accepted_share": round(accepted / args.total_tasks, 4),
        "phase2_gate_0.8": accepted / args.total_tasks >= 0.8,
        "by_verdict": {
            "%s/%s" % k: v
            for k, v in sorted(counts.items(), key=lambda kv: (kv[0][0], str(kv[0][1])))
        },
    }
    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=1) + "\n", encoding="utf-8")
    print(json.dumps(manifest), file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
