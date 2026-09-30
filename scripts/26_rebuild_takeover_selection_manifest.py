#!/usr/bin/env python3
"""Rebuild one merged takeover selection manifest from bound launch records."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def launch_records(root: Path) -> list[dict]:
    records: dict[str, dict] = {}
    for path in sorted(root.glob("depth_*/*/*/takeover_launch.json")):
        record = json.loads(path.read_text(encoding="utf-8"))
        trajectory_id = str(record["trajectory_id"])
        candidate = {
            "trajectory_id": trajectory_id,
            "canonical_trajectory_uri": record["canonical_trajectory_uri"],
            "canonical_trajectory_sha256": record.get("canonical_trajectory_sha256"),
        }
        # Runs replaying the human-verified repaired prefix (prepare_clean_prefix.py) bind it too.
        for key in ("repaired_trajectory_uri", "repaired_trajectory_sha256"):
            if record.get(key):
                candidate[key] = record[key]
        if trajectory_id in records and records[trajectory_id] != candidate:
            raise RuntimeError(f"conflicting launch provenance: {trajectory_id}")
        records[trajectory_id] = candidate
    if not records:
        raise RuntimeError(f"no takeover launch records under {root}")
    return [records[key] for key in sorted(records)]


def write_manifest(root: Path) -> Path:
    output = root / "selection_manifest.json"
    document = {"schema_version": 1, "included": launch_records(root)}
    payload = json.dumps(document, indent=2, sort_keys=True) + "\n"
    if output.exists() and output.read_text(encoding="utf-8") != payload:
        current = json.loads(output.read_text(encoding="utf-8"))
        proposed = {row["trajectory_id"]: row for row in document["included"]}
        for row in current.get("included", []):
            candidate = proposed.get(row["trajectory_id"])
            current_hash = row.get("canonical_trajectory_sha256")
            proposed_hash = candidate and candidate.get("canonical_trajectory_sha256")
            if current_hash and proposed_hash and current_hash != proposed_hash:
                raise RuntimeError(f"canonical hash conflict: {row['trajectory_id']}")
    output.write_text(payload, encoding="utf-8")
    return output


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("roots", nargs="+", type=Path)
    args = parser.parse_args()
    for root in args.roots:
        output = write_manifest(root.resolve())
        count = len(json.loads(output.read_text(encoding="utf-8"))["included"])
        print(f"{output}: included={count}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
