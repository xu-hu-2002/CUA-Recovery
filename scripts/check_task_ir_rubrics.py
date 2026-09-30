#!/usr/bin/env python3
"""Gold-execute extracted Task IRs and check them against rubric expectations (doc 5.3).

For every ``task_ir/<task_id>.json`` in ``--ir-dir``: copy the databases named by
``DERAIL_VM_DB_DIR`` (or ``--db-dir``), run the gold interpreter, extract comparable
expectations from the task's rubrics + ``variables.json``, and write
``<out>/<task_id>.gold_lineage.json``, ``<out>/rubric_checks.jsonl`` and ``<out>/report.md``
(auto_validated / needs_review with the unmatched expectations).  Progress on stderr.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
import time
from collections import Counter
from pathlib import Path

REPO_ROOT = Path(os.environ.get("DERAIL_REPO_ROOT", Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(REPO_ROOT / "src"))

from derail.ir.gold_interpreter import (  # noqa: E402
    GoldInterpreter,
    InterpreterConfig,
    WorldCopy,
)
from derail.ir.model import load_task_ir  # noqa: E402
from derail.ir.rubric_check import RubricCheckConfig, check, context_values, extract_expectations  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--ir-dir", type=Path, required=True)
    parser.add_argument("--tasks", type=Path, required=True)
    parser.add_argument("--variables", type=Path, required=True)
    parser.add_argument("--db-dir", type=Path, default=os.environ.get("DERAIL_VM_DB_DIR"))
    parser.add_argument("--world-id", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--interpreter-config",
        type=Path,
        default=REPO_ROOT / "configs/synthesis/gold_interpreter_v1.yaml",
    )
    parser.add_argument(
        "--rubric-config", type=Path, default=REPO_ROOT / "configs/synthesis/rubric_check_v1.yaml"
    )
    parser.add_argument("--workdir", default=os.environ.get("DERAIL_TMP_ROOT"))
    parser.add_argument(
        "--files-root",
        default=os.environ.get("DERAIL_VM_FILES_DIR"),
        help="copy of /home/user (file facts); default DERAIL_VM_FILES_DIR",
    )
    args = parser.parse_args()
    if not args.db_dir:
        raise SystemExit("need --db-dir or DERAIL_VM_DB_DIR")
    tasks = {t["id"]: t for t in json.loads(args.tasks.read_text(encoding="utf-8"))}
    variables = json.loads(args.variables.read_text(encoding="utf-8"))
    interpreter = GoldInterpreter(InterpreterConfig.from_yaml(args.interpreter_config))
    rubric_config = RubricCheckConfig.from_yaml(args.rubric_config)
    files = sorted(Path(args.ir_dir).glob("*.json"))
    args.out.mkdir(parents=True, exist_ok=True)
    print(
        "reading %d IR files, databases %s\nwriting %s" % (len(files), args.db_dir, args.out),
        file=sys.stderr,
    )
    results, started = [], time.time()
    with tempfile.TemporaryDirectory(dir=args.workdir) as tmp:
        for index, path in enumerate(files, 1):
            tick = time.time()
            task_id = path.stem
            record = {"task_id": task_id, "verdict": "needs_review", "code": None}
            try:
                task_ir = load_task_ir(path, REPO_ROOT)
                apps = sorted({n["app"] for n in task_ir["nodes"]})
                sources = {
                    app: Path(args.db_dir) / ("%s.sqlite" % app)
                    for app in apps
                    if (Path(args.db_dir) / ("%s.sqlite" % app)).is_file()
                }
                with WorldCopy.open(
                    args.world_id,
                    sources,
                    Path(tmp) / ("run_%d" % index),
                    files_root=args.files_root,
                ) as world:
                    gold = interpreter.run(task_ir, world, REPO_ROOT)
                    extra = context_values(task_ir, gold, world.connections)
                (args.out / ("%s.gold_lineage.json" % task_id)).write_text(
                    json.dumps(gold, indent=1, ensure_ascii=False) + "\n", encoding="utf-8"
                )
                expectations = extract_expectations(
                    tasks.get(task_id, {}), variables, rubric_config
                )
                record = check(
                    task_id,
                    gold,
                    expectations,
                    rubric_config,
                    extra,
                    task=tasks.get(task_id),
                    task_ir=task_ir,
                )
            except (
                Exception
            ) as exc:  # schema, static, SQL or expression failure: recorded, not fatal
                record.update({"code": "GOLD_EXECUTION_FAILED", "error": str(exc)[:500]})
            results.append(record)
            elapsed = time.time() - started
            print(
                "[%d/%d] %s %s %.1fs | elapsed %dm eta %dm"
                % (
                    index,
                    len(files),
                    task_id,
                    record["verdict"],
                    time.time() - tick,
                    elapsed // 60,
                    (elapsed / index * (len(files) - index)) // 60,
                ),
                file=sys.stderr,
            )
    (args.out / "rubric_checks.jsonl").write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in results), encoding="utf-8"
    )
    counts = Counter((r["verdict"], r.get("code")) for r in results)
    lines = ["# Rubric check report", "", "| verdict | code | tasks |", "|---|---|---|"]
    lines += [
        "| %s | %s | %d |" % (v, c or "-", n)
        for (v, c), n in sorted(counts.items(), key=lambda kv: (kv[0][0], str(kv[0][1] or "")))
    ]
    lines += ["", "## needs_review", ""]
    for r in results:
        if r["verdict"] != "auto_validated":
            detail = (
                r.get("error")
                or (
                    "final verifier returned false"
                    if r.get("code") == "FINAL_VERIFIER_FAILED"
                    else ""
                )
                or "; ".join("%s=%r" % (e["kind"], e["value"]) for e in r.get("unmatched", []))
            )
            lines.append("- **%s** %s: %s" % (r["task_id"], r.get("code"), detail))
    (args.out / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"%s/%s" % k: v for k, v in counts.items()}), file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
