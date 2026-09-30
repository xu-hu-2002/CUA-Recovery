#!/usr/bin/env python3
"""Run the gold interpreter over Task IRs."""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
import time
from pathlib import Path
from typing import Dict, List

REPO_ROOT = Path(os.environ.get("DERAIL_REPO_ROOT", Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(REPO_ROOT / "src"))

from derail.ir.gold_interpreter import (  # noqa: E402
    GoldInterpreter,
    GoldInterpreterError,
    InterpreterConfig,
    WorldCopy,
)
from derail.ir.model import load_task_ir  # noqa: E402
from derail.world.sqlite_fixture import build_world  # noqa: E402


def _pairs(values: List[str]) -> Dict[str, Path]:
    out: Dict[str, Path] = {}
    for item in values:
        app, _, path = item.partition("=")
        if not path:
            raise SystemExit("expected app=path, got %r" % item)
        out[app] = Path(path)
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--task-ir", nargs="+", required=True, help="task-ir/1.0 JSON files")
    parser.add_argument("--seed", nargs="*", default=[], help="app=miniworld-seed.json")
    parser.add_argument("--db", nargs="*", default=[], help="app=database.sqlite")
    parser.add_argument("--world-id", required=True)
    parser.add_argument(
        "--config", default=str(REPO_ROOT / "configs/synthesis/gold_interpreter_v1.yaml")
    )
    parser.add_argument("--out", required=True, help="output directory (must be on the data disk)")
    parser.add_argument(
        "--files-root",
        default=os.environ.get("DERAIL_VM_FILES_DIR"),
        help="copy of /home/user (file facts); default DERAIL_VM_FILES_DIR",
    )
    parser.add_argument(
        "--workdir",
        default=os.environ.get("DERAIL_TMP_ROOT"),
        help="scratch root for database copies (default: DERAIL_TMP_ROOT)",
    )
    args = parser.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    config = InterpreterConfig.from_yaml(args.config)
    print("reading %s\nwriting %s" % (", ".join(args.task_ir), out), file=sys.stderr)
    tasks = [Path(p) for p in args.task_ir]
    errors = []
    started = time.time()
    with tempfile.TemporaryDirectory(dir=args.workdir) as tmp:
        sources = dict(_pairs(args.db))
        if args.seed:
            sources.update(build_world(_pairs(args.seed), Path(tmp) / "seeded"))
        for index, path in enumerate(tasks, 1):
            tick = time.time()
            task_id = path.stem
            try:
                task_ir = load_task_ir(path, REPO_ROOT)
                task_id = task_ir["task_id"]
                with WorldCopy.open(
                    args.world_id,
                    sources,
                    Path(tmp) / ("run_%d" % index),
                    files_root=args.files_root,
                ) as world:
                    record = GoldInterpreter(config).run(task_ir, world, REPO_ROOT)
                (out / ("%s.gold_lineage.json" % task_id)).write_text(
                    json.dumps(record, indent=1, ensure_ascii=False) + "\n", encoding="utf-8"
                )
            except (GoldInterpreterError, ValueError) as exc:
                errors.append({"task": str(path), "error": str(exc)})
            elapsed = time.time() - started
            eta = elapsed / index * (len(tasks) - index)
            print(
                "[%d/%d] %s %.1fs | elapsed %dm eta %dm"
                % (index, len(tasks), task_id, time.time() - tick, elapsed // 60, eta // 60),
                file=sys.stderr,
            )
    if errors:
        (out / "errors.jsonl").write_text(
            "".join(json.dumps(e) + "\n" for e in errors), encoding="utf-8"
        )
        print("%d task(s) failed; see %s" % (len(errors), out / "errors.jsonl"), file=sys.stderr)
    manifest = {
        "config": str(args.config),
        "world_id": args.world_id,
        "tasks": len(tasks),
        "failed": len(errors),
        "seconds": round(time.time() - started, 2),
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1) + "\n", encoding="utf-8")
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
