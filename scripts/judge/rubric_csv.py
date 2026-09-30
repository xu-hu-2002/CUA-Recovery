#!/usr/bin/env python3
"""Build the per-rubric CSV for one takeover cell (depth x condition)."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
from takeover_judge_selection import completed_task_dirs, excluded_task_ids

REPOSITORY = Path(__file__).resolve().parents[2]
if str(REPOSITORY / "src") not in sys.path:
    sys.path.insert(0, str(REPOSITORY / "src"))
from recovery.evaluation.metrics import rubric_verdict  # noqa: E402
CONDITIONS = ("unaware", "notified", "diagnosed")
JUDGE_RESULT_NAME = "rubric_judge_result.json"
RUBRIC_DETAIL_NAME = "osworld_full_traj_result.json"
AWARENESS_RESULT_NAME = "error_awareness_judge.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run-dir", type=Path, required=True,
                        help="Takeover run root holding depth_<N>/<condition>/")
    parser.add_argument("--depth", type=int, required=True)
    parser.add_argument("--condition", choices=CONDITIONS, required=True)
    parser.add_argument("--label", default="",
                        help="Filename stem prefix (default: run-dir basename)")
    parser.add_argument("--out-dir", type=Path, default=REPOSITORY / "artifacts" / "takeover",
                        help="Where the CSV lands (default: <repo>/artifacts/takeover)")
    parser.add_argument("--allow-partial", action="store_true",
                        help="Write the CSV from the judged subset instead of exiting")
    return parser.parse_args()


def stage(task_dirs: list[Path], staging: Path) -> None:
    """Point a flat staging dir at the completed episodes (idempotent)."""

    staging.mkdir(parents=True, exist_ok=True)
    wanted = {task_dir.name for task_dir in task_dirs}
    for stale in staging.iterdir():
        if stale.is_symlink() and stale.name not in wanted:
            stale.unlink()
    for task_dir in task_dirs:
        link = staging / task_dir.name
        if link.is_symlink():
            link.unlink()
        link.symlink_to(task_dir)


def rubric_row(task_dir: Path) -> dict[str, object] | None:
    """Read one judged episode, or None when the judge has not produced it."""

    detail = task_dir / RUBRIC_DETAIL_NAME
    if not detail.is_file():
        return None
    payload = json.loads(detail.read_text(encoding="utf-8"))
    results = payload.get("rubric_results")
    if not results:
        return None
    bundle = json.loads((task_dir / "rubric_bundle.json").read_text(encoding="utf-8"))
    rho, passed = rubric_verdict(results)
    return {
        "task_category": bundle["grading_manifest"]["category"],
        "task_id": task_dir.name,
        "rubric_score": round(100 * rho, 2),
        "perfect_pass": int(passed),
        "n_rubrics": len(results),
        **awareness_columns(task_dir),
        **{f"R{i}": r["score"] for i, r in enumerate(results, 1)},
        **{f"R{i}_weight": r["weight"] for i, r in enumerate(results, 1)},
    }


def awareness_columns(task_dir: Path) -> dict[str, object]:
    """Error-Awareness verdict for this episode, blank when it has not been judged."""
    path = task_dir / AWARENESS_RESULT_NAME
    if not path.is_file():
        return {"error_aware": ""}
    return {"error_aware": int(bool(json.loads(path.read_text(encoding="utf-8"))["aware"]))}


def write_csv(rows: list[dict[str, object]], out_path: Path) -> None:
    width = max(int(row["n_rubrics"]) for row in rows)
    fields = ["task_category", "task_id", "rubric_score", "perfect_pass", "n_rubrics",
              "error_aware"]
    fields += [f"R{i}" for i in range(1, width + 1)]
    fields += [f"R{i}_weight" for i in range(1, width + 1)]
    rows.sort(key=lambda row: (row["task_category"], row["task_id"]))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, restval="")
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    args = parse_args()
    if args.depth < 0:
        raise SystemExit("--depth must be non-negative")
    run_dir = args.run_dir.resolve()
    cell = run_dir / f"depth_{args.depth}" / args.condition
    if not cell.is_dir():
        raise SystemExit(f"no such takeover cell: {cell}")

    excluded = excluded_task_ids(run_dir, args.depth, args.condition)
    task_dirs = completed_task_dirs(cell, excluded, "rubric_bundle.json")
    if not task_dirs:
        raise SystemExit(f"no completed episodes (result.txt=1.0) under {cell}")
    staging = run_dir / f"_judge_d{args.depth}_{args.condition}_ok"
    stage(task_dirs, staging)

    rows, pending = [], []
    for task_dir in task_dirs:
        row = rubric_row(task_dir)
        (rows if row is not None else pending).append(row if row is not None else task_dir)

    total = len(task_dirs)
    print(f"cell        : depth_{args.depth}/{args.condition}")
    print(f"completed   : {total} episodes (runner errors excluded)")
    print(f"excluded    : {len(excluded)} protocol exclusion(s)")
    print(f"judged      : {len(rows)}  pending: {len(pending)}")
    print(f"staging dir : {staging}")
    if pending and not args.allow_partial:
        print("\nJudge the staged episodes first, then re-run this script:")
        print(f"  bash {REPOSITORY / 'scripts' / 'judge' / 'run_judge.sh'} {staging}")
        print("(long run -- start it under tmux; --allow-partial writes the "
              "judged subset instead)")
        print("NOTE: every row in one CSV must come from ONE judge config. A "
              "rollout still in flight keeps adding episodes, so judge the "
              "stragglers with the same judge that graded the earlier rows.")
        return 1

    label = args.label or run_dir.name
    out_path = args.out_dir / f"{label}_d{args.depth}_{args.condition}.csv"
    write_csv(rows, out_path)
    perfect = sum(int(row["perfect_pass"]) for row in rows)
    mean = sum(float(row["rubric_score"]) for row in rows) / len(rows)
    print(f"\nRubric % {mean:.1f} | Perfect {perfect}/{len(rows)} "
          f"({perfect / len(rows):.1%}) -> {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
