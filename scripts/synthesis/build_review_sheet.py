#!/usr/bin/env python3
"""Human-review sheet for tasks whose rubric check could not decide."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

DEFAULT_CODES = "NO_COMPARABLE_EXPECTATIONS,IR_RUBRIC_MISMATCH,FINAL_VERIFIER_FAILED"


def _short(value, limit: int = 160) -> str:
    text = json.dumps(value, ensure_ascii=False) if not isinstance(value, str) else value
    return text if len(text) <= limit else text[: limit - 3] + "..."


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--run", action="append", required=True)
    parser.add_argument("--tasks", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--codes", default=DEFAULT_CODES)
    parser.add_argument("--max-values", type=int, default=12)
    args = parser.parse_args()
    codes = {c.strip() for c in args.codes.split(",") if c.strip()}
    tasks = {t["id"]: t for t in json.loads(args.tasks.read_text(encoding="utf-8"))}
    latest = {}
    for run in [Path(r) for r in args.run]:
        checks = run / "rubric_check" / "rubric_checks.jsonl"
        if not checks.is_file():
            continue
        for line in checks.read_text(encoding="utf-8").splitlines():
            record = json.loads(line)
            latest[record["task_id"]] = (run, record)
    lines = ["# Human review sheet", "", "Codes: %s" % ", ".join(sorted(codes)), ""]
    count = 0
    for task_id, (run, record) in sorted(latest.items()):
        if record.get("verdict") == "auto_validated" or record.get("code") not in codes:
            continue
        count += 1
        task = tasks.get(task_id, {})
        lines += ["## %s (%s, run %s)" % (task_id, record.get("code"), run.name), ""]
        lines += ["**Instruction**: %s" % task.get("instruction", "(missing)"), ""]
        rubrics = (task.get("grading") or {}).get("rubrics") or []
        if rubrics:
            lines.append("**Rubrics**:")
            lines += [
                "- (%.2f) %s" % (float(r.get("weight", 0)), r.get("criterion")) for r in rubrics
            ]
            lines.append("")
        gold_path = run / "rubric_check" / ("%s.gold_lineage.json" % task_id)
        if gold_path.is_file():
            gold = json.loads(gold_path.read_text(encoding="utf-8"))
            lines.append(
                "**Gold**: final verifier %s; %d values"
                % (gold.get("final_verifier_passed"), len(gold.get("values", ())))
            )
            for value in gold.get("values", ())[: args.max_values]:
                lines.append(
                    "- %s.%s = %s" % (value["node_id"], value["name"], _short(value.get("value")))
                )
            lines.append("")
        if record.get("unmatched"):
            lines.append("**Unmatched expectations**: %s" % _short(record["unmatched"], 400))
            lines.append("")
        if record.get("error"):
            lines += ["**Error**: %s" % _short(record["error"], 300), ""]
        lines += ["**Verdict**: `______` (accept / reject) -- reason:", ""]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("review sheet: %d tasks -> %s" % (count, args.out), file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
