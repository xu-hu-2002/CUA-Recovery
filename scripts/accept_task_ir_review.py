#!/usr/bin/env python3
"""Turn reviewed Task-IR modules into human_verified grounded modules for the synthesis index.

Inputs: one or more ``modules.jsonl`` (later files override earlier ones per task), a verdict
YAML (task-ir-review/0.1), the value-type registry and the ontology.  Output: a schema-valid
``grounded_modules.jsonl`` plus a report of unregistered types for the next registry revision.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

from derail.derived.layout import atomic_write_json, atomic_write_jsonl, sha256_file
from derail.derived.schema import validate_schema
from derail.longhorizon.ontology import Ontology
from derail.longhorizon.review import ReviewVerdicts, apply_review, latest_by_task
from derail.longhorizon.types import ValueTypeRegistry


def main(argv: Iterable[str] = ()) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--modules", type=Path, nargs="+", required=True)
    parser.add_argument("--verdicts", type=Path, required=True)
    parser.add_argument("--value-types", type=Path, required=True)
    parser.add_argument("--ontology", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args(list(argv) if argv else None)
    root = args.repo_root.resolve()

    registry = ValueTypeRegistry.from_yaml(args.value_types)
    ontology = Ontology.from_yaml(args.ontology)
    verdicts = ReviewVerdicts.from_yaml(args.verdicts)
    accepted, report = apply_review(
        latest_by_task(args.modules), verdicts, registry=registry, ontology=ontology
    )
    for module in accepted:
        validate_schema(module, "grounded_task_module.schema.json", root)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    atomic_write_jsonl(args.output_dir / "grounded_modules.jsonl", accepted)
    manifest = {
        "schema_version": "stage-manifest/0.1",
        "stage": "phase1_review_acceptance",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "inputs": {
            "modules": [{"uri": str(p.resolve()), "sha256": sha256_file(p)} for p in args.modules],
            "verdicts": {"uri": str(args.verdicts.resolve()), "sha256": sha256_file(args.verdicts)},
            "value_types": {
                "uri": str(args.value_types.resolve()),
                "sha256": sha256_file(args.value_types),
            },
        },
        "review": {
            "reviewer": verdicts.reviewer,
            "method": verdicts.method,
            "reviewed_at": verdicts.reviewed_at,
        },
        "counts": {
            "accepted": len(accepted),
            "skipped": len(report["skipped"]),
            "unregistered_types": len(report["unregistered_types"]),
        },
        "report": report,
    }
    atomic_write_json(args.output_dir / "stage_manifest.json", manifest)
    print(json.dumps({k: manifest[k] for k in ("counts", "review")}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
