#!/usr/bin/env python3
"""Build the world-graph/0.1 record from the MyPCBench persona seed and reference variables."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

from recovery.derived.layout import atomic_write_json, sha256_file
from recovery.derived.schema import validate_schema
from recovery.longhorizon.world import build_world_graph, load_sources


def main(argv: Iterable[str] = ()) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--persona", type=Path, required=True)
    parser.add_argument("--variables", type=Path, required=True)
    parser.add_argument("--sources-config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[2])
    args = parser.parse_args(list(argv) if argv else None)

    persona = json.loads(args.persona.read_text(encoding="utf-8"))
    variables = json.loads(args.variables.read_text(encoding="utf-8"))
    sources = load_sources(args.sources_config)
    world = build_world_graph(
        persona,
        variables,
        sources,
        provenance={
            "persona_uri": str(args.persona.resolve()),
            "persona_sha256": sha256_file(args.persona),
            "variables_uri": str(args.variables.resolve()),
            "variables_sha256": sha256_file(args.variables),
            "sources_config_sha256": sha256_file(args.sources_config),
        },
    )
    validate_schema(world, "world_graph.schema.json", args.repo_root.resolve())
    args.output_dir.mkdir(parents=True, exist_ok=True)
    atomic_write_json(args.output_dir / "world_graph.json", world)
    by_type = {}
    for entity in world["entities"]:
        by_type[entity["entity_type"]] = by_type.get(entity["entity_type"], 0) + 1
    manifest = {
        "schema_version": "stage-manifest/0.1",
        "stage": "phase0_world_graph",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source": world["source"],
        "counts": {
            "entities": len(world["entities"]),
            "relations": len(world["relations"]),
            "entities_by_type": dict(sorted(by_type.items())),
        },
        "id_confidence": "seed_derived; in-VM store keys not yet verified per app",
        "human_review_status": "pending",
    }
    atomic_write_json(args.output_dir / "stage_manifest.json", manifest)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
