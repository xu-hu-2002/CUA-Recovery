#!/usr/bin/env python3
"""Summarize human-adjudicated open error codes without freezing or merging them."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from derail.annotation.records import Adjudication
from derail.annotation.taxonomy import summarize_open_codes
from derail.derived.layout import DerivedBuild, atomic_write_json, sha256_file, sha256_json
from derail.derived.schema import validate_schema


def _read(path: Path):
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build-dir", type=Path, required=True)
    args = parser.parse_args()

    repository = Path(__file__).resolve().parents[2]
    build_dir = args.build_dir.resolve()
    manifest = _read(build_dir / "build_manifest.json")
    validate_schema(manifest, "derived_build_manifest.schema.json", repository)
    source = manifest["source_collection"]
    build = DerivedBuild(
        root=build_dir.parent,
        build_id=build_dir.name,
        collection_root=Path(source["uri"]),
        collection_id=source["collection_id"],
    )
    if not build.verify_raw_unchanged():
        raise RuntimeError("raw collection tree hash 已改变")

    seed_labels = tuple(manifest["taxonomy"]["seed_labels"])
    adjudications = []
    inputs = []

    for path in sorted((build_dir / "annotations").glob("*/adjudication.json")):
        raw = _read(path)
        validate_schema(raw, "adjudication.schema.json", repository)
        item = Adjudication.from_dict(raw)
        if item.taxonomy_version != manifest["taxonomy"]["version"]:
            raise RuntimeError("adjudication taxonomy_version 与 build 不一致: %s" % path)
        adjudications.append(item)
        inputs.append(
            {
                "case_id": path.parent.name,
                "adjudication_id": item.adjudication_id,
                "uri": str(path.resolve()),
                "sha256": sha256_file(path),
            }
        )

    report = {
        "build_id": manifest["build_id"],
        "taxonomy_version": manifest["taxonomy"]["version"],
        "taxonomy_mode": manifest["taxonomy"]["mode"],
        "seed_labels": sorted(seed_labels),
        **summarize_open_codes(adjudications, seed_labels),
        "input_adjudications": inputs,
        "input_set_sha256": sha256_json(inputs),
        "interpretation_policy": (
            "descriptive human-coded statistics only; no automatic label merge, rename or freeze"
        ),
    }
    validate_schema(report, "taxonomy_summary.schema.json", repository)
    output = build_dir / "reports" / "taxonomy_open_coding_summary.json"
    atomic_write_json(output, report)
    print(json.dumps({"written": str(output), **report}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
