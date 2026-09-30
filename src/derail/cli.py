from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from derail.canonical.actions import action_from_dict, action_to_dict
from derail.canonical.summaries import summarize_action
from derail.replay.executor import compile_pyautogui

REQUIRED_PATHS = (
    "README.md",
    "main.py",
    "configs/benchmark/derail_v1.yaml",
    "schemas/canonical_action.schema.json",
    "schemas/derived_build_manifest.schema.json",
    "schemas/adjudication.schema.json",
    "schemas/replay_verification.schema.json",
    "schemas/replay_plan.schema.json",
    "schemas/prefix_audit.schema.json",
    "schemas/native_history.schema.json",
    "schemas/taxonomy_summary.schema.json",
    "schemas/task_fragment.schema.json",
    "schemas/grounded_task_module.schema.json",
    "schemas/empirical_skeleton.schema.json",
    "schemas/compatibility_edge.schema.json",
    "schemas/generation_record.schema.json",
    "schemas/synthesis_config.schema.json",
    "src/derail/canonical/actions.py",
    "src/derail/construction/repair.py",
    "src/derail/synthesis/pipeline.py",
    "src/derail/synthesis/skeletons.py",
    "src/derail/replay/executor.py",
    "src/derail/adapters/holo31.py",
    "scripts/collection/serve_open_source.sh",
    "scripts/collection/collect_all.sh",
    "scripts/collection/collect_trajectories.sh",
    "scripts/collection/watch_progress.sh",
    "scripts/benchmark/prepare_annotations.py",
    "scripts/benchmark/build_benchmark.py",
    "scripts/benchmark/replay_instance.py",
    "scripts/takeover/run.sh",
    "scripts/takeover/build_native_history.py",
    "scripts/takeover/run_evaluation.py",
    "scripts/analysis/summarize_open_taxonomy.py",
    "scripts/benchmark/validate_benchmark.py",
    "scripts/synthesis/synthesize_long_horizon_tasks.py",
    "scripts/synthesis/build_task_synthesis_index.py",
)


def repository_root() -> Path:
    return Path(__file__).resolve().parents[2]


def check_repository() -> int:
    root = repository_root()
    missing = [path for path in REQUIRED_PATHS if not (root / path).exists()]
    if missing:
        print("仓库结构不完整，缺少：")
        for path in missing:
            print("- %s" % path)
        return 1
    print("DERAIL repository skeleton 检查通过（%d 个关键路径）。" % len(REQUIRED_PATHS))
    return 0


def inspect_action(raw_json: str) -> int:
    action = action_from_dict(json.loads(raw_json))
    output = {
        "canonical": action_to_dict(action),
        "summary_en": summarize_action(action, "en"),
        "summary_zh": summarize_action(action, "zh"),
        "pyautogui_audit": compile_pyautogui(action),
    }
    print(json.dumps(output, ensure_ascii=False, indent=2))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="DERAIL benchmark toolkit")
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("check-repository", help="检查首版仓库关键文件")
    inspect_parser = subparsers.add_parser("inspect-action", help="审计 canonical action")
    inspect_parser.add_argument("action_json", help="单个 canonical action JSON")
    return parser


def main(argv: Sequence[str] = ()) -> int:
    parser = build_parser()
    args = parser.parse_args(list(argv) if argv else None)
    if args.command == "check-repository":
        return check_repository()
    if args.command == "inspect-action":
        return inspect_action(args.action_json)
    parser.error("未知命令")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
