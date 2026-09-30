#!/usr/bin/env python3
"""Termination calibration on the labelled failure corpus.

Every trajectory in the corpus is a *failure*, so an explicit ``terminate`` carrying a success
status is a miscalibrated stop.  This script quantifies how often that happens, how much step
budget was still available when it happened, and --- the point of the analysis --- whether the
stop decision carries any information about whether the state was still recoverable.  If the
success-claim rate is the same on reversible and irreversible failures, the agent's termination
signal is uninformative about recoverability.

Reads the Phase 0 continuation records (``failure_continuation_records.jsonl``) and the collection
runtime config for the step budget; writes a Markdown and a JSON report.  No new rollouts.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from datetime import date
from pathlib import Path
from statistics import median
from typing import Any, Dict, Iterable, List, Sequence

import yaml


def _load_records(path: Path) -> List[Dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _wilson(successes: int, total: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval; degenerate counts return the point estimate twice."""
    if total == 0:
        return (0.0, 0.0)
    p = successes / total
    denom = 1.0 + z * z / total
    centre = (p + z * z / (2 * total)) / denom
    half = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


def _rate(successes: int, total: int) -> Dict[str, Any]:
    low, high = _wilson(successes, total)
    return {
        "n": successes,
        "of": total,
        "rate": (successes / total) if total else None,
        "ci95": [low, high] if total else None,
    }


def _stop_stats(records: Sequence[Dict[str, Any]], step_budget: int) -> Dict[str, Any]:
    explicit = [r for r in records if r.get("terminated_explicitly")]
    claimed = [r for r in explicit if r.get("terminate_status") == "success"]
    # Budget still unspent at the moment the agent chose to stop.  A stop with plenty of budget
    # left is a decision; a stop at zero is exhaustion.
    remaining = [step_budget - int(r["trajectory_length"]) for r in explicit]
    return {
        "trajectories": len(records),
        "terminated_explicitly": _rate(len(explicit), len(records)),
        "claimed_success_given_explicit": _rate(len(claimed), len(explicit)),
        "claimed_success_overall": _rate(len(claimed), len(records)),
        "remaining_budget_at_stop": {
            "median": median(remaining) if remaining else None,
            "share_with_at_least_half_budget_left": (
                sum(1 for value in remaining if value >= step_budget / 2) / len(remaining)
                if remaining
                else None
            ),
        },
    }


def _difference(left: Dict[str, Any], right: Dict[str, Any]) -> Dict[str, Any]:
    """Two-proportion difference with a normal-approximation interval (stdlib only)."""
    n1, d1 = left["n"], left["of"]
    n2, d2 = right["n"], right["of"]
    if not d1 or not d2:
        return {"difference": None, "ci95": None}
    p1, p2 = n1 / d1, n2 / d2
    se = math.sqrt(p1 * (1 - p1) / d1 + p2 * (1 - p2) / d2)
    return {
        "difference": p1 - p2,
        "ci95": [p1 - p2 - 1.96 * se, p1 - p2 + 1.96 * se],
        "overlaps_zero": abs(p1 - p2) <= 1.96 * se,
    }


def analyse(records: Sequence[Dict[str, Any]], step_budget: int) -> Dict[str, Any]:
    by_reversibility = {
        key: _stop_stats([r for r in records if r.get("reversibility") == key], step_budget)
        for key in ("reversible", "irreversible")
    }
    models = sorted({str(r.get("model")) for r in records})
    return {
        "step_budget": step_budget,
        "overall": _stop_stats(records, step_budget),
        "by_reversibility": by_reversibility,
        "reversibility_contrast": _difference(
            by_reversibility["irreversible"]["claimed_success_given_explicit"],
            by_reversibility["reversible"]["claimed_success_given_explicit"],
        ),
        "by_model": {
            model: _stop_stats([r for r in records if str(r.get("model")) == model], step_budget)
            for model in models
        },
        "primary_type_of_success_claims": dict(
            Counter(
                str(r.get("primary_paper_type"))
                for r in records
                if r.get("terminated_explicitly") and r.get("terminate_status") == "success"
            ).most_common()
        ),
    }


def _pct(value: Any) -> str:
    return "--" if value is None else "%.1f%%" % (100 * value)


def render_markdown(result: Dict[str, Any], *, source: str, generated: str) -> str:
    overall = result["overall"]
    rev = result["by_reversibility"]["reversible"]
    irr = result["by_reversibility"]["irreversible"]
    contrast = result["reversibility_contrast"]
    lines = [
        "# Termination calibration on the labelled failure corpus",
        "",
        "Generated %s from `%s`; step budget %d." % (generated, source, result["step_budget"]),
        "",
        "Every trajectory here is a failure, so an explicit `terminate` with a success status is",
        "a stop the agent should not have made.",
        "",
        "## Headline",
        "",
        "| quantity | value |",
        "|---|---|",
        "| trajectories | %d |" % overall["trajectories"],
        "| ended with an explicit terminate | %s (%d/%d) |"
        % (
            _pct(overall["terminated_explicitly"]["rate"]),
            overall["terminated_explicitly"]["n"],
            overall["terminated_explicitly"]["of"],
        ),
        "| of those, claimed success | %s (%d/%d) |"
        % (
            _pct(overall["claimed_success_given_explicit"]["rate"]),
            overall["claimed_success_given_explicit"]["n"],
            overall["claimed_success_given_explicit"]["of"],
        ),
        "| median step budget still unspent at the stop | %s |"
        % overall["remaining_budget_at_stop"]["median"],
        "| stops made with at least half the budget left | %s |"
        % _pct(overall["remaining_budget_at_stop"]["share_with_at_least_half_budget_left"]),
        "",
        "## Does the stop decision know about recoverability?",
        "",
        "| stratum | trajectories | explicit terminate | claimed success given terminate |",
        "|---|---|---|---|",
    ]
    for name, block in (("reversible", rev), ("irreversible", irr)):
        lines.append(
            "| %s | %d | %s | %s (%d/%d) |"
            % (
                name,
                block["trajectories"],
                _pct(block["terminated_explicitly"]["rate"]),
                _pct(block["claimed_success_given_explicit"]["rate"]),
                block["claimed_success_given_explicit"]["n"],
                block["claimed_success_given_explicit"]["of"],
            )
        )
    if contrast["difference"] is None:
        lines += ["", "One stratum is empty; no contrast reported."]
    else:
        lines += [
            "",
            "Irreversible minus reversible success-claim rate: %s, 95%% CI [%s, %s]. %s"
            % (
                _pct(contrast["difference"]),
                _pct(contrast["ci95"][0]),
                _pct(contrast["ci95"][1]),
                (
                    "The interval covers zero: the termination decision carries no measurable "
                    "information about whether the state was still recoverable."
                    if contrast["overlaps_zero"]
                    else "The interval excludes zero: the two strata differ."
                ),
            ),
        ]
    lines += [
        "",
        "## Per agent",
        "",
        "| agent | trajectories | explicit terminate | claimed success given terminate | "
        "median budget left |",
        "|---|---|---|---|---|",
    ]
    for model, block in result["by_model"].items():
        lines.append(
            "| %s | %d | %s | %s | %s |"
            % (
                model,
                block["trajectories"],
                _pct(block["terminated_explicitly"]["rate"]),
                _pct(block["claimed_success_given_explicit"]["rate"]),
                block["remaining_budget_at_stop"]["median"],
            )
        )
    lines += [
        "",
        "## Primary error type of the miscalibrated stops",
        "",
        "| primary type | count |",
        "|---|---|",
    ]
    for name, count in result["primary_type_of_success_claims"].items():
        lines.append("| %s | %d |" % (name, count))
    lines.append("")
    return "\n".join(lines)


def main(argv: Iterable[str] = ()) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    repo_default = Path(__file__).resolve().parents[1]
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=repo_default,
        help="repository root that repository-relative paths resolve against",
    )
    parser.add_argument(
        "--records",
        default="data/synthesis/phase0_precheck/failure_continuation_records.jsonl",
        help="repository-relative Phase 0 continuation records",
    )
    parser.add_argument(
        "--runtime-config",
        default="configs/collection/mypcbench_runtime.yaml",
        help="repository-relative collection runtime config supplying the step budget",
    )
    parser.add_argument(
        "--output-dir", default="results", help="repository-relative report directory"
    )
    parser.add_argument(
        "--report-date",
        default=date.today().isoformat().replace("-", ""),
        help="date stamp used in the report filenames",
    )
    args = parser.parse_args(list(argv) if argv else None)

    repo = args.repo_root.resolve()
    records_path = repo / args.records
    runtime = yaml.safe_load((repo / args.runtime_config).read_text(encoding="utf-8"))
    step_budget = int(runtime["max_steps"])

    records = _load_records(records_path)
    result = analyse(records, step_budget)
    result["provenance"] = {
        "records": args.records,
        "runtime_config": args.runtime_config,
        "generated": date.today().isoformat(),
    }

    out_dir = repo / args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = "ANALYSIS_termination_calibration_%s" % args.report_date
    (out_dir / (stem + ".md")).write_text(
        render_markdown(result, source=args.records, generated=date.today().isoformat()),
        encoding="utf-8",
    )
    (out_dir / (stem + ".json")).write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({"report": str(out_dir / (stem + '.md')), **result["overall"]},
                     ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
