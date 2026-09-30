#!/usr/bin/env python3
"""False-completion (premature_completion) prevalence and co-occurrence.

Reads the frozen trajectory_labels.csv produced by analyze.py and reports how
often the agent stops and declares success while requirements are unmet, and
which error types travel with it.
"""

from __future__ import annotations

import argparse
import csv
import statistics
from collections import Counter
from pathlib import Path

PC = "premature_completion"
BUDGET = "hit_budget_limit"
CATEGORY = {
    "planning": {"fabricate_data", "misunderstand_task_objective", "lack_of_knowledge", "scope_error"},
    "perception": {"progress_misperception", "detail_misperception", "state_misinterpretation",
                   "ineffective_action"},
    "execution": {"grounding_failure", "incorrect_ui_element",
                  "typing_or_parameter_error", "wrong_target"},
}


def labels(row: dict) -> set[str]:
    return {x for x in row["error_types"].split("|") if x}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", type=Path,
                    default=Path(__file__).with_name("results") / "trajectory_labels.csv")
    args = ap.parse_args()

    rows = list(csv.DictReader(args.labels.open()))
    fail = [r for r in rows if r["state"] == "failure"]
    pc = [r for r in fail if PC in labels(r)]
    other = [r for r in fail if PC not in labels(r)]
    voluntary = [r for r in pc if BUDGET not in labels(r)]

    print(f"trajectories {len(rows)} | failures {len(fail)} | {PC} {len(pc)}"
          f" ({len(pc)/len(fail):.1%} of failures, {len(pc)/len(rows):.1%} of all)")
    print(f"  agent-declared done {len(voluntary)} | also budget-exhausted {len(pc)-len(voluntary)}")

    print(f"\nco-occurring error types (n={len(pc)}), with lift over failures without {PC}:")
    co = Counter(l for r in pc for l in labels(r) - {PC})
    base = Counter(l for r in other for l in labels(r))
    for label, count in co.most_common():
        share, ref = count / len(pc), base[label] / len(other)
        print(f"  {label:30s} {count:4d} {share:6.1%}   non-PC {ref:6.1%}   lift {share/ref if ref else float('inf'):5.2f}")

    print(f"\namong the {len(voluntary)} agent-declared-done cases:")
    for name, members in CATEGORY.items():
        n = sum(1 for r in voluntary if labels(r) & members)
        print(f"  any {name:11s} co-label  {n:4d} {n/len(voluntary):6.1%}")
    unverified = sum(1 for r in voluntary if labels(r) & {"progress_misperception", "fabricate_data"})
    isolated = sum(1 for r in voluntary if labels(r) == {PC})
    print(f"  progress_misperception or fabricate_data {unverified:4d} {unverified/len(voluntary):6.1%}")
    print(f"  no other error type at all               {isolated:4d} {isolated/len(voluntary):6.1%}")

    def score(rs): return [float(r["weighted_rubric_score"]) for r in rs if r["weighted_rubric_score"]]
    print(f"\nweighted rubric score: {PC} mean {statistics.mean(score(pc)):.3f}"
          f" vs other failures {statistics.mean(score(other)):.3f}"
          f" | PC at >=0.8: {sum(1 for s in score(pc) if s >= 0.8)}")

    print("\nper model (share of that model's failures):")
    for model in sorted({r["model"] for r in rows}):
        f = [r for r in fail if r["model"] == model]
        p = [r for r in f if PC in labels(r)]
        print(f"  {model:22s} failures {len(f):4d}  {PC} {len(p):4d}  {len(p)/len(f):6.1%}")


if __name__ == "__main__":
    main()
