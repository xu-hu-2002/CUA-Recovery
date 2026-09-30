#!/usr/bin/env python3
"""Build derail-case/1.0 records from analysed rollouts (doc v1.2 section 10).

    python scripts/build_cases_v1.py --traces <dir> --analyses <dir> --gold-dir <dir> \
        --profiles <profiles.jsonl> --out <bundle dir>

Per failed rollout: prefix repair (state-neutral segments), depth instantiation on the
benchmark depth grid, reversibility stratum from the change log; then dedup across rollouts
and a bundle with the funnel.  Replay verification (10.2) is VM-side and marks cases later.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import yaml

REPO_ROOT = Path(os.environ.get("DERAIL_REPO_ROOT", Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(REPO_ROOT / "src"))

from derail.cases.instantiate import build_cases, dedup_records  # noqa: E402
from derail.cases.package import write_bundle  # noqa: E402
from derail.failure_analysis.retrospective import clean_start_step_budget  # noqa: E402
from derail.cases.repair import repair_prefix  # noqa: E402
from derail.derived.schema import validate_schema  # noqa: E402
from derail.longhorizon.ontology import Ontology  # noqa: E402
from derail.world.volatile import VolatileColumns  # noqa: E402


def _task_ir(ir_dir, task_id):
    if not ir_dir:
        return None
    path = Path(ir_dir) / ("%s.json" % task_id)
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--traces", type=Path, required=True)
    parser.add_argument(
        "--analyses", type=Path, required=True, help="<rollout_id>.failure_analysis.json files"
    )
    parser.add_argument("--gold-dir", type=Path, required=True)
    parser.add_argument(
        "--ir-dir", type=Path, help="task_ir/ directory (gold actions name the node)"
    )
    parser.add_argument(
        "--profiles", type=Path, help="profiles.jsonl from build_latent_profiles.py"
    )
    parser.add_argument(
        "--benchmark-config", type=Path, default=REPO_ROOT / "configs/benchmark/derail_v1.yaml"
    )
    parser.add_argument(
        "--split",
        choices=("test", "train"),
        default="test",
        help="test applies the App. C eligibility rule (d <= h_e); train keeps every depth",
    )
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    benchmark = yaml.safe_load(args.benchmark_config.read_text(encoding="utf-8"))
    depth_grid = list(benchmark["depths"])
    require_error_explicit = bool(
        (benchmark.get("depth_eligibility") or {}).get("require_error_explicit", True)
    )
    ontology = Ontology.from_yaml(REPO_ROOT / "configs/synthesis/ontology_v0.2.yaml")
    volatile = VolatileColumns.from_yaml(REPO_ROOT / "configs/synthesis/volatile_columns_v1.yaml")
    profiles = {}
    if args.profiles and args.profiles.is_file():
        for line in args.profiles.read_text(encoding="utf-8").splitlines():
            record = json.loads(line)
            profiles[record["task_id"]] = record
    traces = sorted(args.traces.rglob("rollout_trace.json")) + sorted(args.traces.glob("*.json"))
    print("reading %d traces\nwriting %s" % (len(traces), args.out), file=sys.stderr)
    all_cases, per_rollout, started = [], [], time.time()
    for index, path in enumerate(traces, 1):
        trace = json.loads(path.read_text(encoding="utf-8"))
        if trace.get("schema_version") != "rollout-trace/1.0":
            continue
        row = {
            "rollout_id": trace["rollout_id"],
            "failed": False,
            "analyzed": False,
            "repaired": False,
            "cases": 0,
        }
        analysis_path = args.analyses / ("%s.failure_analysis.json" % trace["rollout_id"])
        if analysis_path.is_file():
            row["failed"] = True
            analysis = json.loads(analysis_path.read_text(encoding="utf-8"))
            gold = json.loads(
                (args.gold_dir / ("%s.gold_lineage.json" % trace["task_id"])).read_text(
                    encoding="utf-8"
                )
            )
            if analysis.get("root_cause_action_index") is not None and not analysis.get(
                "residual_code"
            ):
                row["analyzed"] = True
                repair = repair_prefix(
                    trace, analysis["root_cause_action_index"], gold["writes_gold"], volatile
                )
                row["repaired"] = repair["status"] in ("repaired", "unchanged")
                cases, _ = build_cases(
                    trace,
                    analysis,
                    gold,
                    repair,
                    profiles.get(trace["task_id"]),
                    depth_grid,
                    ontology,
                    step_budget=int(trace.get("step_budget") or clean_start_step_budget()),
                    task_ir=_task_ir(args.ir_dir, trace["task_id"]),
                    test_eligibility=args.split == "test",
                    require_error_explicit=require_error_explicit,
                )
                for case in cases:
                    validate_schema(case, "derail_case.schema.json", REPO_ROOT)
                row["cases"] = len(cases)
                all_cases.extend(cases)
        per_rollout.append(row)
        print(
            "[%d/%d] %s cases=%d | elapsed %.0fs"
            % (index, len(traces), trace["rollout_id"], row["cases"], time.time() - started),
            file=sys.stderr,
        )
    kept, removed = dedup_records(all_cases)
    manifest = write_bundle(
        args.out,
        kept,
        removed,
        per_rollout,
        depth_grid,
        {"traces": str(args.traces), "analyses": str(args.analyses)},
    )
    print(
        json.dumps({"funnel": manifest["funnel"], "by_depth": manifest["by_depth"]}),
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
