#!/usr/bin/env python3
"""Build the WATCHDOG training inputs from rollout traces, failure analyses and gold lineages."""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from dataclasses import replace
from pathlib import Path

REPO_ROOT = Path(os.environ.get("RECOVERY_REPO_ROOT", Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(REPO_ROOT / "src"))

from recovery.cases.gold_action import gold_action  # noqa: E402
from recovery.derived.schema import validate_schema  # noqa: E402
from recovery.train.build_samples import (  # noqa: E402
    FORMATS,
    BuildConfig,
    balance_check_weights,
    base_samples,
    dataset_manifest,
    detection_samples,
    negative_samples,
    recovery_cases,
    success_samples,
    token_matched,
    verification_samples,
)


def gold_read_facts(gold):
    """``(table, column, entity)`` keys of the rows the gold run read (ledger anchors)."""

    out = set()
    for read in gold.get("resolved_reads", ()):
        table = str(read.get("table") or "")
        columns = read.get("columns") or ([read["column"]] if read.get("column") else [])
        for rowid in read.get("entity_set") or ():
            for column in columns:
                out.add((table, str(column), "%s:%s" % (table.split(".")[-1], rowid)))
    return sorted(out)


def _load(path: Path):
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else None


def _jsonl(path: Path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--traces", type=Path, required=True)
    parser.add_argument("--analyses", type=Path, required=True)
    parser.add_argument("--gold-dir", type=Path, required=True)
    parser.add_argument("--ir-dir", type=Path, required=True)
    parser.add_argument("--profiles", type=Path, help="profiles.jsonl (latent-profile/1.0)")
    parser.add_argument("--config", type=Path, default=REPO_ROOT / "configs/train/sft_v1.yaml")
    parser.add_argument("--format", choices=FORMATS, help="default: the config's format")
    parser.add_argument(
        "--recovery-samples", type=Path, help="success_control: recovery_samples.jsonl to match"
    )
    parser.add_argument("--negatives-per-trace", type=int, default=3, help="steps format")
    parser.add_argument("--seed", type=int, default=0, help="steps format negatives")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    config = BuildConfig.from_yaml(args.config, REPO_ROOT)
    if args.format:
        config = replace(config, format=args.format)
    if config.format == "success_control" and not args.recovery_samples:
        raise SystemExit("success_control needs --recovery-samples")
    rng = random.Random(args.seed)
    profiles = {}
    if args.profiles and args.profiles.is_file():
        for line in args.profiles.read_text(encoding="utf-8").splitlines():
            record = json.loads(line)
            profiles[record["task_id"]] = record
    traces = sorted(args.traces.glob("*.json"))
    args.out.mkdir(parents=True, exist_ok=True)
    samples, rows = [], []
    print(
        "reading %d traces (%s)\nwriting %s" % (len(traces), config.format, args.out),
        file=sys.stderr,
    )
    for index, path in enumerate(traces, 1):
        trace = _load(path)
        if not trace or trace.get("schema_version") != "rollout-trace/1.0":
            continue
        task_id = str(trace["task_id"])
        if config.split_of(task_id) != "train":
            continue
        gold = _load(args.gold_dir / ("%s.gold_lineage.json" % task_id)) or {}
        task_ir = _load(args.ir_dir / ("%s.json" % task_id))
        analysis = _load(args.analyses / ("%s.failure_analysis.json" % trace["rollout_id"]))
        facts = gold_read_facts(gold)
        produced = []
        if config.format == "recovery":
            if analysis is not None:
                produced += recovery_cases(trace, analysis, config, facts)
        elif config.format == "success_control":
            if analysis is None:
                produced += success_samples(trace, config)
        elif analysis is None:
            produced += base_samples(trace, config, facts)
            produced += negative_samples(trace, None, config, args.negatives_per_trace, rng)
        else:
            action = gold_action(analysis, gold, task_ir)
            produced += detection_samples(trace, analysis, config, facts, gold_action=action)
            if task_ir is not None:
                produced += verification_samples(
                    trace, analysis, gold, task_ir, profiles.get(task_id), config
                )
            produced += negative_samples(trace, analysis, config, args.negatives_per_trace, rng)
        samples += produced
        rows.append(
            {
                "rollout_id": trace["rollout_id"],
                "task_id": task_id,
                "failed": analysis is not None,
                "samples": len(produced),
            }
        )
        print(
            "[%d/%d] %s %d samples" % (index, len(traces), trace["rollout_id"], len(produced)),
            file=sys.stderr,
        )
    extra, weights = {}, None
    if config.format == "success_control":
        samples, extra["token_match"] = token_matched(
            samples, _jsonl(args.recovery_samples), config
        )
        if not extra["token_match"]["within_tolerance"]:
            print("warning: control not token-matched: %s" % extra["token_match"], file=sys.stderr)
    if config.format == "steps":
        weights = balance_check_weights(samples, config)
        for sample in samples[:5]:
            validate_schema(sample, "training_sample.schema.json", REPO_ROOT)
    name = "cases.jsonl" if config.format == "recovery" else "samples.jsonl"
    (args.out / name).write_text(
        "".join(json.dumps(s, ensure_ascii=False) + "\n" for s in samples), encoding="utf-8"
    )
    (args.out / "per_trace.jsonl").write_text(
        "".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8"
    )
    manifest = dataset_manifest(samples, config, weights, **extra)
    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=1) + "\n", encoding="utf-8")
    print(json.dumps({k: manifest[k] for k in ("samples", "by_kind", "by_split")}), file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
