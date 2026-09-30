"""采集任务来源（configs/collection/sources.yaml）：MyPCBench 原始任务或 ReRail 组合 workflow。

两种来源最终都变成官方 runner / run_rollout_v1.py 能吃的任务列表（每项至少有 ``id`` 与
``instruction``）。ReRail 行的转换与 phase5 的 ``task_config`` 相同（只带 id + instruction，
env.reset 不需要别的字段），额外字段只作 provenance；``grading`` 取合成 bundle 里该
workflow 的组合 rubric（``rubrics_dir``/<workflow_id>.json，derail.gen.graft.compose_rubric
写出的 MyPCBench grading 形状），官方 run_mypcbench.load_tasks 要求每个任务都有
``grading.rubrics``。缺 rubric 的 workflow 按 ``on_missing_rubric``（error | skip）处理。

    python -m derail.rollout.tasks --source rerail_workflows --out <run_root>/_tasks/rerail.json

输出一行 JSON 摘要到 stdout（``tasks_file`` / ``tasks`` / ``graded``），供采集脚本记进 manifest。
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional

import yaml

REPO_ROOT = Path(os.environ.get("DERAIL_REPO_ROOT", Path(__file__).resolve().parents[3]))
SOURCES_CONFIG = REPO_ROOT / "configs" / "collection" / "sources.yaml"


def source_spec(name: str, config_path: Path = SOURCES_CONFIG) -> Dict[str, Any]:
    sources = yaml.safe_load(config_path.read_text(encoding="utf-8"))["sources"]
    for spec in sources:
        if spec["source_benchmark"] == name:
            return dict(spec)
    known = [spec["source_benchmark"] for spec in sources]
    raise ValueError(f"未知 task_source {name!r}；{config_path} 里只有 {known}")


def _rerail_grading(workflow_id: str, rubrics_dir: Path) -> Optional[Dict[str, Any]]:
    path = rubrics_dir / ("%s.json" % workflow_id)
    if not path.is_file():
        return None
    rubric = json.loads(path.read_text(encoding="utf-8"))
    if not rubric.get("rubrics"):
        return None
    return {"type": rubric["type"], "rubrics": rubric["rubrics"]}


def _rerail_task(row: Mapping[str, Any]) -> Dict[str, Any]:
    return {
        "id": row["id"],
        "instruction": row["instruction"],
        "derail_task_source": "rerail_workflows",
        "workflow_id": row["task_id"],
        "variant": row["variant"],
        "origin": row["origin"],
        "bucket": row["bucket"],
        "world_variant": row["world_variant"],
    }


def load_source_tasks(
    name: str, tasks_file: Optional[Path] = None, config_path: Path = SOURCES_CONFIG
) -> List[Dict[str, Any]]:
    spec = source_spec(name, config_path)
    path = tasks_file or REPO_ROOT / spec["tasks_file"]
    rows = json.loads(Path(path).read_text(encoding="utf-8"))
    if spec["task_format"] == "mypcbench_graded":
        return rows
    if spec["task_format"] == "rerail_rollout_tasks":
        origins = set(spec.get("origins") or ())
        worlds = set(spec.get("world_variants") or ())
        rubrics_dir = REPO_ROOT / spec["rubrics_dir"]
        on_missing = spec.get("on_missing_rubric", "error")
        if on_missing not in ("error", "skip"):
            raise ValueError(f"{name}: on_missing_rubric 只能是 error | skip")
        tasks, missing = [], []
        for row in rows:
            if (origins and row["origin"] not in origins) or (
                worlds and row["world_variant"] not in worlds
            ):
                continue
            grading = _rerail_grading(str(row["task_id"]), rubrics_dir)
            if grading is None:
                missing.append(str(row["id"]))
                continue
            tasks.append({**_rerail_task(row), "grading": grading})
        if missing and on_missing == "error":
            raise ValueError(
                f"{name}: {len(missing)} 个 workflow 在 {rubrics_dir} 没有组合 rubric"
                f"（如 {missing[:3]}）；先重跑合成生成 rubrics/，或设 on_missing_rubric: skip"
            )
        if missing:
            print(f"[tasks] {name}: 跳过 {len(missing)} 个缺 rubric 的 workflow", file=sys.stderr)
        return tasks
    raise ValueError(f"{name}: 不支持的 task_format {spec['task_format']!r}")


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--source", required=True)
    parser.add_argument("--tasks-file", type=Path, help="覆盖 sources.yaml 的 tasks_file（如分片）")
    parser.add_argument("--out", type=Path, help="写出 runner 格式任务；缺省只打印默认 tasks_file")
    parser.add_argument("--config", type=Path, default=SOURCES_CONFIG)
    args = parser.parse_args(argv)
    spec = source_spec(args.source, args.config)
    if args.out is None:
        print(args.tasks_file or REPO_ROOT / spec["tasks_file"])
        return 0
    tasks = load_source_tasks(args.source, args.tasks_file, args.config)
    if not tasks:
        raise SystemExit(f"{args.source}: 过滤后没有任务")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(tasks, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    graded = sum(bool((task.get("grading") or {}).get("rubrics")) for task in tasks)
    print(json.dumps({"tasks_file": str(args.out), "tasks": len(tasks), "graded": graded}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
