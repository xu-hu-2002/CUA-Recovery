#!/usr/bin/env python3
"""Generate verifier-first symbolic long-horizon tasks from grounded modules."""

from __future__ import annotations

import argparse
import json
import subprocess
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List

import yaml

from recovery.derived.layout import atomic_write_json, atomic_write_jsonl, sha256_file
from recovery.derived.schema import validate_schema
from recovery.synthesis.pipeline import SynthesisConfig, SynthesisPipeline


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    records = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError("%s:%d must contain a JSON object" % (path, line_number))
            records.append(value)
    return records


def _git_commit(repository: Path) -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repository,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        text=True,
        check=False,
    )
    return result.stdout.strip() if result.returncode == 0 else "unavailable"


def _input_ref(path: Path) -> Dict[str, str]:
    resolved = path.resolve()
    return {"uri": str(resolved), "sha256": sha256_file(resolved)}


def main(argv: Iterable[str] = ()) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--modules", type=Path, required=True)
    parser.add_argument("--skeletons", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--human-failure-stats", type=Path)
    parser.add_argument("--exclude-model-id")
    args = parser.parse_args(list(argv) if argv else None)
    repository = Path(__file__).resolve().parents[2]

    output_dir = args.output_dir.resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise RuntimeError("output directory exists and is non-empty: %s" % output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    config_raw = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    if not isinstance(config_raw, dict):
        raise ValueError("synthesis config must be a YAML object")
    validate_schema(config_raw, "synthesis_config.schema.json", repository)
    modules = _read_jsonl(args.modules)
    skeletons = _read_jsonl(args.skeletons)
    failure_stats = _read_jsonl(args.human_failure_stats) if args.human_failure_stats else []
    for module in modules:
        validate_schema(module, "grounded_task_module.schema.json", repository)
    for skeleton in skeletons:
        validate_schema(skeleton, "empirical_skeleton.schema.json", repository)

    config = SynthesisConfig.from_dict(config_raw)
    pipeline = SynthesisPipeline(
        modules,
        skeletons,
        config,
        failure_stats,
        excluded_model_id=args.exclude_model_id,
    )
    result = pipeline.run()
    for edge in result.compatibility_edges:
        validate_schema(edge, "compatibility_edge.schema.json", repository)
    for record in (*result.accepted, *result.rejected):
        validate_schema(record, "generation_record.schema.json", repository)

    atomic_write_jsonl(output_dir / "accepted_symbolic_tasks.jsonl", result.accepted)
    atomic_write_jsonl(output_dir / "rejected_tasks.jsonl", result.rejected)
    atomic_write_jsonl(output_dir / "compatibility_edges.jsonl", result.compatibility_edges)
    atomic_write_json(output_dir / "skeleton_weights.json", result.skeleton_weights)
    rejection_counts = Counter(record["rejection_code"] for record in result.rejected)
    inputs = {
        "modules": _input_ref(args.modules),
        "skeletons": _input_ref(args.skeletons),
        "config": _input_ref(args.config),
    }
    if args.human_failure_stats:
        inputs["human_failure_stats"] = _input_ref(args.human_failure_stats)
    manifest = {
        "schema_version": "synthesis-stage-manifest/0.1",
        "status": "symbolic_candidates_ready" if result.accepted else "no_go",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_commit(repository),
        "command": "scripts/synthesis/synthesize_long_horizon_tasks.py",
        "generation_version": config.generation_version,
        "sampling_mode": config.sampling_mode,
        "random_seed": config.seed,
        "inputs": inputs,
        "counts": {
            "input_modules": len(modules),
            "input_skeletons": len(skeletons),
            "compatibility_edges": len(result.compatibility_edges),
            "accepted_symbolic_candidates": len(result.accepted),
            "rejected_attempts": len(result.rejected),
            "rejection_codes": dict(sorted(rejection_counts.items())),
        },
        "gates": {
            "static_structure": "passed" if result.accepted else "failed",
            "natural_language_realization": "pending",
            "round_trip_fidelity": "pending",
            "fixture_reset": "pending",
            "gold_execution": "pending",
            "human_review": "pending",
        },
        "release_eligible": False,
        "next_required_stage": (
            "realize instructions, validate round-trip Task IR, materialize fixtures, "
            "and pass gold execution"
        ),
    }
    atomic_write_json(output_dir / "stage_manifest.json", manifest)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0 if result.accepted else 2


if __name__ == "__main__":
    raise SystemExit(main())
