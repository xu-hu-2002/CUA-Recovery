#!/usr/bin/env python3
"""Split the MyPCBench task set into shard files for ROCK collection."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
DEFAULT_TASKS = REPO / "third_party/MyPCBench/tasks/final/all_tasks_with_grading.json"
OUT_DIR = REPO / "configs/mypcbench_task_shards"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tasks-file", type=Path, default=DEFAULT_TASKS)
    parser.add_argument("--shards", type=int, default=4)
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    args = parser.parse_args()

    tasks = json.loads(args.tasks_file.read_text(encoding="utf-8"))
    if not isinstance(tasks, list) or not tasks:
        raise SystemExit(f"任务文件不是非空 list：{args.tasks_file}")
    if args.shards < 1:
        raise SystemExit("--shards 必须 >= 1")

    args.out_dir.mkdir(parents=True, exist_ok=True)

    shards: list[list[dict]] = [[] for _ in range(args.shards)]
    for index, task in enumerate(tasks):
        shards[index % args.shards].append(task)

    for index, shard in enumerate(shards, start=1):
        path = args.out_dir / f"shard_{index}_of_{args.shards}.json"
        path.write_text(json.dumps(shard, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        apps = sorted({t.get("app", "?") for t in shard})
        print(f"[shards] {path.relative_to(REPO)}: {len(shard)} 任务, apps={apps}")

    smoke = args.out_dir / "smoke_one.json"
    smoke.write_text(
        json.dumps([tasks[0]], indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    first = tasks[0]
    print(f"[shards] {smoke.relative_to(REPO)}: 1 任务 "
          f"(id={first.get('id', '?')}, app={first.get('app', '?')})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
