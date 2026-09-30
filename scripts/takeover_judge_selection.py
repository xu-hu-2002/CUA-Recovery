#!/usr/bin/env python3
"""Shared eligibility rules for takeover rubric and EAR judging."""

from __future__ import annotations

import json
import hashlib
import sys
from pathlib import Path


EXCLUSION_FILE = "judge_exclusions.json"
REPOSITORY = Path(__file__).resolve().parents[1]


def load_exclusions(run_dir: Path) -> dict[tuple[int, str, str], dict]:
    path = run_dir / EXCLUSION_FILE
    if not path.is_file():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != 1 or not isinstance(payload.get("exclusions"), list):
        raise ValueError(f"invalid judge exclusion manifest: {path}")
    records = {}
    for item in payload["exclusions"]:
        key = (int(item["depth"]), str(item["condition"]), str(item["task_id"]))
        if key in records:
            raise ValueError(f"duplicate judge exclusion: {key}")
        if item.get("classification") != "PROTOCOL_EXCLUSION" or not item.get("reason"):
            raise ValueError(f"invalid judge exclusion record: {item}")
        records[key] = item
    return records


def excluded_task_ids(run_dir: Path, depth: int, condition: str) -> set[str]:
    return {
        task_id for (record_depth, record_condition, task_id) in load_exclusions(run_dir)
        if record_depth == depth and record_condition == condition
    }


def completed_task_dirs(cell: Path, excluded: set[str], required_file: str) -> list[Path]:
    """Episodes the judges may see: result.txt=1.0 (runner completion marker, not success)
    and no protocol_exclusion.json (scripts/10_run_takeover_rollout.py)."""
    found = []
    for task in sorted(cell.iterdir()):
        if not task.is_dir() or task.name.startswith("_") or task.name in excluded:
            continue
        if (task / "protocol_exclusion.json").exists():
            continue
        try:
            completed = float((task / "result.txt").read_text(encoding="utf-8").strip()) == 1.0
        except (OSError, UnicodeError, ValueError):
            continue
        if completed and (task / required_file).is_file():
            found.append(task)
    return found


def resolve_annotation(uri: str, expected_sha256: str = "") -> Path:
    supplied = Path(uri)
    candidates = [supplied, REPOSITORY / supplied, REPOSITORY / "takeovewr_annotation/human_labels" / supplied.name]
    path = next((candidate for candidate in candidates if candidate.is_file()), None)
    if path is None:
        raise FileNotFoundError(f"human annotation is unavailable: {uri}")
    if expected_sha256:
        observed = hashlib.sha256(path.read_bytes()).hexdigest()
        if observed != expected_sha256:
            raise ValueError(f"human annotation SHA mismatch: {path}")
    return path.resolve()


def stage_for_judge(run_dir: Path, depth: int, condition: str, staging: Path) -> tuple[int, int, str]:
    cell = run_dir / f"depth_{depth}" / condition
    excluded = excluded_task_ids(run_dir, depth, condition)
    tasks = completed_task_dirs(cell, excluded, "rubric_bundle.json")
    if not tasks:
        raise RuntimeError(f"no result.txt=1.0 episodes with rubric bundles under {cell}")
    wanted = {task.name: task.resolve() for task in tasks}
    sync_staging(staging, wanted)
    judged, models = existing_judges(wanted.values())
    return len(wanted), len(excluded), f"{judged} {','.join(sorted(models)) or '-'}"


def sync_staging(staging: Path, wanted: dict[str, Path]) -> None:
    staging.mkdir(parents=True, exist_ok=True)
    for old in staging.iterdir():
        if old.name not in wanted and old.is_symlink():
            old.unlink()
    for name, target in wanted.items():
        link = staging / name
        if link.exists() and not link.is_symlink():
            raise RuntimeError(f"staging collision is not a symlink: {link}")
        if link.is_symlink():
            link.unlink()
        link.symlink_to(target, target_is_directory=True)


def existing_judges(tasks) -> tuple[int, set[str]]:
    judged, models = 0, set()
    for task in tasks:
        detail = task / "osworld_full_traj_result.json"
        if not detail.is_file():
            continue
        payload = json.loads(detail.read_text(encoding="utf-8"))
        judged += 1
        if payload.get("model"):
            models.add(str(payload["model"]).strip())
    return judged, models


def main() -> int:
    if len(sys.argv) != 5:
        raise SystemExit("usage: takeover_judge_selection.py RUN_DIR DEPTH CONDITION STAGING")
    run_dir, depth, condition, staging = Path(sys.argv[1]).resolve(), int(sys.argv[2]), sys.argv[3], Path(sys.argv[4]).resolve()
    completed, excluded, judge_summary = stage_for_judge(run_dir, depth, condition, staging)
    print(completed, excluded, judge_summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
