#!/usr/bin/env python3
"""Build compatibility and empirical-skeleton review indexes from grounded modules."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List

import yaml

from recovery.derived.layout import atomic_write_json, atomic_write_jsonl, sha256_file
from recovery.derived.schema import validate_schema
from recovery.synthesis.compatibility import TypeSystem, build_compatibility_edges
from recovery.synthesis.skeletons import extract_observed_skeletons


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    records = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if line.strip():
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise ValueError("%s:%d is not a JSON object" % (path, line_number))
                records.append(value)
    return records


def main(argv: Iterable[str] = ()) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--modules", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(list(argv) if argv else None)
    repository = Path(__file__).resolve().parents[2]
    output_dir = args.output_dir.resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise RuntimeError("output directory exists and is non-empty: %s" % output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    modules = _read_jsonl(args.modules)
    for module in modules:
        validate_schema(module, "grounded_task_module.schema.json", repository)
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise ValueError("config must be a YAML object")
    validate_schema(config, "synthesis_config.schema.json", repository)
    type_system = TypeSystem.from_config(config.get("type_system", {}))
    edges = build_compatibility_edges(modules, type_system)
    skeletons = extract_observed_skeletons(modules)
    for edge in edges:
        validate_schema(edge, "compatibility_edge.schema.json", repository)
    for skeleton in skeletons:
        validate_schema(skeleton, "empirical_skeleton.schema.json", repository)

    atomic_write_jsonl(output_dir / "compatibility_edges.jsonl", edges)
    atomic_write_jsonl(output_dir / "empirical_skeleton_candidates.jsonl", skeletons)
    manifest = {
        "schema_version": "synthesis-index-manifest/0.1",
        "status": "needs_human_review",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "inputs": {
            "modules": {"uri": str(args.modules.resolve()), "sha256": sha256_file(args.modules)},
            "config": {"uri": str(args.config.resolve()), "sha256": sha256_file(args.config)},
        },
        "counts": {
            "modules": len(modules),
            "compatibility_edges": len(edges),
            "direct_conflict_free_edges": sum(
                edge["environment_status"] == "direct" and not edge["conflicts"]
                for edge in edges
            ),
            "observed_skeleton_candidates": len(skeletons),
        },
        "next_required_stage": (
            "review skeleton clusters and composition rules; set approved rules and "
            "human_approved skeleton status before synthesis"
        ),
    }
    atomic_write_json(output_dir / "stage_manifest.json", manifest)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
