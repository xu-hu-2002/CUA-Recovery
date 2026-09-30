#!/usr/bin/env python3
"""Archive one judged takeover cell into the independent persistent tree."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any

from takeover_judge_selection import resolve_annotation


def _read(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"expected JSON object: {path}")
    return value


def _validate_human_source_judge(task: Path, launch: dict[str, Any]) -> None:
    source_judge = launch.get("human_source_judge")
    annotation = launch.get("human_annotation")
    if not isinstance(source_judge, dict) or not isinstance(annotation, dict):
        raise RuntimeError(f"missing human source-judge provenance: {task}")
    required = {
        "authority": "human_annotation",
        "failure_eligible": True,
        "trajectory_id": launch.get("trajectory_id"),
        "source_trajectory_sha256": annotation.get("source_trajectory_sha256"),
        "reviewer_id": annotation.get("annotator_id"),
        "root_cause_action_index": annotation.get("root_cause_action_index"),
    }
    if any(source_judge.get(key) != value for key, value in required.items()):
        raise RuntimeError(f"invalid human source-judge provenance: {task}")


def _bundle_provenance(task: Path) -> dict[str, Any]:
    manifest = _read(task / "takeover_manifest.json")
    annotation = manifest.get("human_annotation")
    if not isinstance(annotation, dict):
        raise RuntimeError(f"missing human annotation provenance: {task}")
    required = ("trajectory_id", "source_trajectory_sha256", "root_cause_action_index",
                "annotator_id", "annotation_sha256", "annotation_uri")
    if any(annotation.get(key) in (None, "") for key in required):
        raise RuntimeError(f"incomplete human annotation provenance: {task}")
    resolve_annotation(annotation["annotation_uri"], annotation["annotation_sha256"])
    return manifest


def _episode_records(cell: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    records, retries = [], []
    for task in sorted(
        path for path in cell.iterdir()
        if path.is_dir() and not path.name.startswith("_")
        and (path / "osworld_full_traj_result.json").is_file()
    ):
        launch_path = task / "takeover_launch.json"
        launch = _read(launch_path) if launch_path.is_file() else _bundle_provenance(task)
        if launch_path.is_file():
            _validate_human_source_judge(task, launch)
        detail = _read(task / "osworld_full_traj_result.json")
        records.append({"task_id": task.name, "launch": launch, "judge": detail})
        if detail.get("error"):
            retries.append({"task_id": task.name, "error": detail["error"]})
    return records, retries


def _write(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cell", type=Path, required=True)
    parser.add_argument("--scores", type=Path, required=True)
    parser.add_argument("--csv", type=Path, required=True)
    parser.add_argument("--archive-root", type=Path, required=True)
    parser.add_argument("--judge-model", required=True)
    parser.add_argument("--source-agent", required=True)
    parser.add_argument("--target-agent", required=True)
    parser.add_argument("--condition", required=True)
    parser.add_argument("--depth", type=int, required=True)
    parser.add_argument("--exclusions", type=Path)
    args = parser.parse_args()
    destination = (
        args.archive_root / args.judge_model / args.source_agent / args.target_agent
        / args.condition / f"d{args.depth}"
    )
    records, retries = _episode_records(args.cell)
    destination.mkdir(parents=True, exist_ok=True)
    shutil.copy2(args.scores, destination / "scores.json")
    shutil.copy2(args.csv, destination / "aggregate.csv")
    if args.exclusions and args.exclusions.is_file():
        shutil.copy2(args.exclusions, destination / "judge_exclusions.json")
    _write(destination / "judge_records.json", records)
    _write(destination / "retry_ledger.json", retries)
    _write(destination / "archive_manifest.json", {
        "judge_model": args.judge_model,
        "source_agent": args.source_agent,
        "target_agent": args.target_agent,
        "condition": args.condition,
        "depth": args.depth,
        "record_count": len(records),
        "judge_error_count": len(retries),
        "exclusion_manifest": bool(args.exclusions and args.exclusions.is_file()),
        "source_judge_authority": "human_annotation",
    })
    print(f"judge archive: {destination}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
