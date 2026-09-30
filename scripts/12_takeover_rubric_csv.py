#!/usr/bin/env python3
"""Build the per-rubric CSV for one takeover cell (depth x condition).

A "cell" is one <run-dir>/depth_<N>/<condition> directory.  Only episodes the
runner actually completed (result.txt == 1.0) belong in the table: result.txt
0.0 marks a runner error, not a task failure, and averaging those in would
answer a different question.

The script does three things and stops:

1.  Symlinks the completed episodes into a staging dir, so the rubric judge
    bills only those and not the runner errors sitting next to them.
2.  Reports how many of them are already judged, printing the judge command
    for the rest.  Judging is a paid, long-running step -- this script never
    launches it (repo convention: long runs are started under tmux).
3.  Writes <label>_d<N>_<condition>.csv once every episode is judged.

    python3 scripts/12_takeover_rubric_csv.py \
        --run-dir artifacts/model_outputs/takeover/takeover_v1/evocua_32b_to_qwen3_8_27b \
        --depth 0 --condition unaware --label evocua32b_to_qwen3.8_27B

``--label`` carries the human-facing agent spelling ("qwen3.8_27B"), which is
not mechanically derivable from the agent_id directory name; it defaults to the
run directory's basename.
"""

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

REPOSITORY = Path(__file__).resolve().parent.parent
if str(REPOSITORY / "src") not in sys.path:
    sys.path.insert(0, str(REPOSITORY / "src"))
from derail.evaluation.metrics import rubric_verdict  # noqa: E402
CONDITIONS = ("unaware", "notified", "diagnosed")
# Written by judge_results.py / osworld_full_traj_judge.py respectively.
JUDGE_RESULT_NAME = "rubric_judge_result.json"
# Post-takeover episodes only. The source failure prefix is scored from the human rubric
# review bound in selection_manifest.json (rubric_review_uri), never from this LLM-judge file.
RUBRIC_DETAIL_NAME = "osworld_full_traj_result.json"
# Written by scripts/13_takeover_error_awareness.py.
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
    """Point a flat staging dir at the completed episodes (idempotent).

    The judge writes its output through the symlink into the real task dir,
    so the staging dir stays a pure index and nothing is duplicated.
    """

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
        return None  # judge errored; the payload carries an "error" key instead
    bundle = json.loads((task_dir / "rubric_bundle.json").read_text(encoding="utf-8"))
    # ρ and V recomputed per criterion (paper 02:12), not the judge's rounded score/passed.
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
    """Error-Awareness verdict for this episode, blank when it has not been judged.

    Written by scripts/13_takeover_error_awareness.py.  Blank is not 0: an
    episode with no verdict scores 0 in the EAR denominator, but saying so here
    would make an unjudged cell indistinguishable from a judged "not aware".
    """
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
        print(f"  bash {REPOSITORY / 'scripts' / 'run_judge.sh'} {staging}")
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
