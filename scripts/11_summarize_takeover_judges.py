#!/usr/bin/env python3
"""Pass@k and Rubric Score per takeover depth x condition (paper Sec. 2, 02:12).

Each state is run ``aggregation.repeats`` times (configs/judges/default.yaml,
default 3).  A state counts as solved when any run succeeds, V = 1 iff every
rubric criterion is satisfied (recomputed per criterion, not the judge's rounded
``passed``), and its Rubric Score is the score of that run: the successful run,
or the best-scoring one when all fail (derail.evaluation.metrics.pass_at_k).

The denominator is every state eligible at the depth (selection_manifest.json
at or above the first run root, the same eligibility as the EAR script).  A state with
no valid judgement in any run -- n/a, infra failure, never run, or judged by a
model other than the configured judge -- counts as a failure unless
``aggregation.missing_counts_as_failure`` is false (then excluded and reported).

Repeat layout: each repeat is a run root with the same
``depth_<N>/<condition>/<task_id>/`` tree.  An --output-root holding
``repeat_<k>/`` directories (scripts/rock/run_takeover.sh) is expanded to them;
otherwise --output-root is repeat 1 and --repeat-root adds the others.

    python3 scripts/11_summarize_takeover_judges.py --output-root RUN \\
        --depths 0 5 10 --conditions unaware
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
REPOSITORY = SCRIPT_DIR.parent
for path in (SCRIPT_DIR, REPOSITORY / "src"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))
from derail.evaluation.metrics import pass_at_k, rubric_verdict  # noqa: E402
from judge_model_registry import configured_model, load_config  # noqa: E402

_SPEC = importlib.util.spec_from_file_location("takeover_ear", SCRIPT_DIR / "13_takeover_error_awareness.py")
EAR = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(EAR)

CONDITIONS = ("unaware", "notified", "diagnosed")
RUBRIC_DETAIL_NAME = "osworld_full_traj_result.json"


def run_verdict(task_dir: Path, judge_model: str) -> tuple[float, bool] | None:
    """(ρ, V) of one judged run, None when it has no valid judgement by the configured judge."""
    path = task_dir / RUBRIC_DETAIL_NAME
    if not path.is_file():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    results = payload.get("rubric_results")
    if not results or payload.get("model") != judge_model:
        return None
    return rubric_verdict(results)


def summarize_cell(roots: list[Path], depth: int, condition: str, *, repeats: int,
                   missing_counts_as_failure: bool, judge_model: str) -> dict[str, Any]:
    if len(roots) > repeats:
        raise ValueError(f"{len(roots)} run roots but repeats={repeats}")
    if any((parent / "selection_manifest.json").is_file() for parent in (roots[0], *roots[0].parents)):
        units = [EAR.task_id_of(item) for item in EAR.eligible_items(roots[0], depth, condition)]
    else:
        units = sorted({p.name for root in roots
                        for p in (root / f"depth_{depth}" / condition).glob("*/")
                        if not p.name.startswith("_")})
    if not units:
        return {"available": False, "reason": "no_eligible_states"}
    runs = {unit: [run_verdict(root / f"depth_{depth}" / condition / unit, judge_model)
                   for root in roots] for unit in units}
    summary = pass_at_k(runs, units, repeats, missing_counts_as_failure)
    return {"available": True, **asdict(summary),
            "run_roots": [str(root) for root in roots]}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output-root", type=Path, required=True, help="Run root of repeat 1")
    parser.add_argument("--repeat-root", type=Path, action="append", default=[],
                        help="Run root of a further repeat (same depth/condition layout)")
    parser.add_argument("--depths", nargs="*", type=int, default=())
    parser.add_argument("--conditions", nargs="*", default=CONDITIONS)
    parser.add_argument("--repeats", type=int, default=None,
                        help="Runs per state (default: aggregation.repeats)")
    parser.add_argument("--exclude-missing", action="store_true",
                        help="Drop states with no valid judgement from the denominator")
    args = parser.parse_args()
    config = load_config()
    aggregation = config.get("aggregation") or {}
    repeats = args.repeats or int(aggregation["repeats"])
    missing_as_failure = (not args.exclude_missing
                          and bool(aggregation.get("missing_counts_as_failure", True)))
    judge_model = configured_model(config)
    output_root = args.output_root.resolve()
    repeat_dirs = sorted((p for p in output_root.glob("repeat_*") if p.is_dir()),
                         key=lambda p: int(p.name.split("_", 1)[1]))
    roots = [*(repeat_dirs or [output_root]), *(root.resolve() for root in args.repeat_root)]
    conditions = tuple(args.conditions)
    unknown = set(conditions) - set(CONDITIONS)
    if unknown:
        raise RuntimeError(f"unknown takeover conditions: {sorted(unknown)}")
    depths = tuple(args.depths) or tuple(sorted(
        int(p.name.split("_", 1)[1]) for p in roots[0].glob("depth_*") if p.is_dir()))

    comparison = {
        "judge_model": judge_model,
        "repeats": repeats,
        "missing_counts_as_failure": missing_as_failure,
        "by_depth": {
            str(depth): {
                condition: summarize_cell(roots, depth, condition, repeats=repeats,
                                          missing_counts_as_failure=missing_as_failure,
                                          judge_model=judge_model)
                for condition in conditions
            }
            for depth in depths
        },
    }
    target = output_root / "takeover_comparison.json"
    target.write_text(json.dumps(comparison, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
                      encoding="utf-8")
    for depth, cells in comparison["by_depth"].items():
        for condition, cell in cells.items():
            if cell.get("available"):
                print(f"d{depth} {condition}: Pass@{repeats} {cell['pass_at_k']:.3f} "
                      f"({cell['solved_count']}/{cell['unit_count']}) "
                      f"Rubric {cell['rubric_score']:.3f} missing={cell['missing_count']} "
                      f"incomplete={cell['incomplete_count']}")
    print(f"comparison: {target}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
