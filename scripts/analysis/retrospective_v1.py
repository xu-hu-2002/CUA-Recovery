#!/usr/bin/env python3
"""Compare the parameter detector on legacy failures against human labels."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

REPO_ROOT = Path(os.environ.get("RECOVERY_REPO_ROOT", Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(REPO_ROOT / "src"))

from recovery.failure_analysis.retrospective import (  # noqa: E402
    agreement_summary,
    load_traj,
    retrospective_row,
    trace_from_traj,
)
from recovery.failure_analysis.run import AnalysisConfig  # noqa: E402


def _find_traj(root: Path, model: str, task_id: str) -> Path:
    """Locate traj.jsonl, matching model directories on letters and digits only."""

    key = model.replace("_", "").lower()
    for candidate in sorted(root.glob("*")):
        if candidate.is_dir() and candidate.name.replace("_", "").lower() == key:
            hits = sorted(candidate.glob("*/%s/traj.jsonl" % task_id))
            if hits:
                return hits[0]
    return root / model / "vm0" / task_id / "traj.jsonl"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--records", type=Path, required=True)
    parser.add_argument("--v1-root", type=Path, required=True)
    parser.add_argument("--ir-dir", type=Path, required=True)
    parser.add_argument("--gold-dir", type=Path, required=True)
    parser.add_argument(
        "--tasks",
        type=Path,
        default=REPO_ROOT / "third_party/MyPCBench/tasks/final/all_tasks_with_grading.json",
    )
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    threshold = AnalysisConfig.from_repo(REPO_ROOT).rules.residual_threshold
    instructions = (
        {t["id"]: t["instruction"] for t in json.loads(args.tasks.read_text(encoding="utf-8"))}
        if args.tasks.is_file()
        else {}
    )
    records = [
        json.loads(line)
        for line in args.records.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    args.out.mkdir(parents=True, exist_ok=True)
    rows, skipped, started = [], Counter(), time.time()
    print("reading %d failures\nwriting %s" % (len(records), args.out), file=sys.stderr)
    for index, record in enumerate(records, 1):
        task_id, model = record["task_id"], record["model"]
        gold_path = args.gold_dir / ("%s.gold_lineage.json" % task_id)
        traj_path = _find_traj(args.v1_root, model, task_id)
        if not gold_path.is_file():
            skipped["no_gold"] += 1
            continue
        if not traj_path.is_file():
            skipped["no_traj"] += 1
            continue
        trace = trace_from_traj(
            load_traj(traj_path),
            task_id=task_id,
            agent=model,
            rollout_id=record["trajectory_id"],
            instruction=instructions.get(task_id, ""),
        )
        gold = json.loads(gold_path.read_text(encoding="utf-8"))
        row = retrospective_row(
            trace,
            gold,
            record.get("root_cause_action_index"),
            record.get("primary_paper_type"),
            residual_threshold=threshold,
        )
        row["paper_category"] = record.get("paper_category")
        rows.append(row)
        if index % 25 == 0 or index == len(records):
            print(
                "[%d/%d] elapsed %.0fs" % (index, len(records), time.time() - started),
                file=sys.stderr,
            )
    (args.out / "rows.jsonl").write_text(
        "".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8"
    )
    summary = agreement_summary(rows)
    by_agent = {a: agreement_summary(rs) for a, rs in defaultdict(list, {}).items()}
    groups = defaultdict(list)
    for row in rows:
        groups[("agent", row["agent"])].append(row)
        groups[("category", row.get("paper_category"))].append(row)
    lines = [
        "# Retrospective parameter-detector agreement (E4 pilot)",
        "",
        "skipped: %s" % dict(skipped),
        "",
        "| group | failures | covered | coverage | exact | within 1 |",
        "|---|---|---|---|---|---|",
    ]
    lines.append(
        "| all | %d | %d | %.3f | %.3f | %.3f |"
        % (
            summary["failures"],
            summary["covered"],
            summary["coverage"],
            summary["exact_of_covered"],
            summary["within_one_of_covered"],
        )
    )
    for (kind, key), rs in sorted(groups.items(), key=lambda kv: (kv[0][0], str(kv[0][1]))):
        s = agreement_summary(rs)
        lines.append(
            "| %s=%s | %d | %d | %.3f | %.3f | %.3f |"
            % (
                kind,
                key,
                s["failures"],
                s["covered"],
                s["coverage"],
                s["exact_of_covered"],
                s["within_one_of_covered"],
            )
        )
    (args.out / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines[4:8]), file=sys.stderr)
    del by_agent
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
