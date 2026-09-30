#!/usr/bin/env python3
"""Automatic failure analysis over rollout traces (execution doc v1.2 section 9).

    python scripts/analyze_failures_v1.py --traces <dir with **/rollout_trace.json> \
        --ir-dir <task_ir dir> --gold-dir <gold lineage dir> --out <dir>

For every trace whose outcome is a failure (final_verifier false, budget exhausted, declared
infeasible), writes ``<out>/<rollout_id>.failure_analysis.json`` and ``summary.md``
(detector coverage, residual rate, review routes, type distribution).  Successful traces are
skipped and counted.  Progress on stderr.

With ``--proposals-dir`` (normally ``<human_labels>/auto_analysis``, where the annotation UI
shows it next to the trajectory) each failure also gets ``<trajectory_id>.json``: the automatic
root cause, type and error horizon in the human annotation layout, routed to human verification
or, below the type-confidence threshold of the typing rules (paper 0.7), to human annotation.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import Counter
from pathlib import Path

REPO_ROOT = Path(os.environ.get("DERAIL_REPO_ROOT", Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(REPO_ROOT / "src"))

from derail.derived.schema import validate_schema  # noqa: E402
from derail.derived.layout import sha256_file  # noqa: E402
from derail.failure_analysis.run import (  # noqa: E402
    AnalysisConfig,
    analyze_failure,
    annotation_proposal,
)
from derail.ir.model import load_task_ir  # noqa: E402


def is_failure(outcome: dict) -> bool:
    return (
        outcome.get("final_verifier") is False
        or bool(outcome.get("budget_exhausted"))
        or bool(outcome.get("declared_infeasible"))
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--traces", type=Path, required=True)
    parser.add_argument("--ir-dir", type=Path, required=True)
    parser.add_argument("--gold-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--proposals-dir", type=Path, help="write human-verification proposals here")
    parser.add_argument("--proposal-annotator-id", default="rerail_auto")
    parser.add_argument(
        "--confidence-threshold",
        type=float,
        help="default: residual_threshold of configs/synthesis/typing_rules_v1.yaml",
    )
    args = parser.parse_args()
    config = AnalysisConfig.from_repo(REPO_ROOT)
    threshold = (
        args.confidence_threshold
        if args.confidence_threshold is not None
        else config.rules.residual_threshold
    )
    if args.proposals_dir:
        args.proposals_dir.mkdir(parents=True, exist_ok=True)
    files = sorted(args.traces.rglob("rollout_trace.json")) + sorted(args.traces.glob("*.json"))
    args.out.mkdir(parents=True, exist_ok=True)
    print("reading %d traces\nwriting %s" % (len(files), args.out), file=sys.stderr)
    counts, types, errors, started = Counter(), Counter(), [], time.time()
    for index, path in enumerate(files, 1):
        trace = json.loads(path.read_text(encoding="utf-8"))
        if trace.get("schema_version") != "rollout-trace/1.0":
            continue
        if not is_failure(trace.get("outcome", {})):
            counts["success_skipped"] += 1
            continue
        try:
            task_ir = load_task_ir(args.ir_dir / ("%s.json" % trace["task_id"]), REPO_ROOT)
            gold = json.loads(
                (args.gold_dir / ("%s.gold_lineage.json" % trace["task_id"])).read_text(
                    encoding="utf-8"
                )
            )
            record = analyze_failure(trace, gold, task_ir, config)
            validate_schema(record, "failure_analysis.schema.json", REPO_ROOT)
            (args.out / ("%s.failure_analysis.json" % trace["rollout_id"])).write_text(
                json.dumps(record, indent=1, ensure_ascii=False) + "\n", encoding="utf-8"
            )
            counts["analyzed"] += 1
            counts["residual" if record["residual_code"] else "auto"] += 1
            counts["censored" if record["horizon_censored"] else "horizon_measured"] += 1
            types[record["paper_type"] or "none"] += 1
            if args.proposals_dir:
                provenance = trace.get("provenance") or {}
                trajectory_id = str(provenance.get("trajectory_id") or trace["rollout_id"])
                proposal = annotation_proposal(
                    record,
                    trajectory_id=trajectory_id,
                    source_trajectory_sha256=str(
                        provenance.get("source_trajectory_sha256") or sha256_file(path)
                    ),
                    annotator_id=args.proposal_annotator_id,
                    taxonomy_version=config.rules.taxonomy.taxonomy_version,
                    confidence_threshold=threshold,
                )
                (args.proposals_dir / ("%s.json" % trajectory_id)).write_text(
                    json.dumps(proposal, indent=1, ensure_ascii=False) + "\n", encoding="utf-8"
                )
                counts["route_" + proposal["review_route"]] += 1
        except Exception as exc:  # recorded, not fatal
            errors.append({"trace": str(path), "error": str(exc)[:300]})
        elapsed = time.time() - started
        print(
            "[%d/%d] %s | elapsed %.0fs" % (index, len(files), path.parent.name, elapsed),
            file=sys.stderr,
        )
    lines = ["# Failure analysis summary", "", "| count | n |", "|---|---|"] + [
        "| %s | %d |" % kv for kv in sorted(counts.items())
    ]
    lines += ["", "| paper_type | n |", "|---|---|"] + [
        "| %s | %d |" % kv for kv in types.most_common()
    ]
    if errors:
        (args.out / "errors.jsonl").write_text("".join(json.dumps(e) + "\n" for e in errors))
        lines += ["", "errors: %d (errors.jsonl)" % len(errors)]
    (args.out / "summary.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines[2:]), file=sys.stderr)
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
