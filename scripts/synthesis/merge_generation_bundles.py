#!/usr/bin/env python3
"""Merge the selected tasks of several generation bundles into one final set."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from collections import Counter
from pathlib import Path


def _rows(path: Path):
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--bundle", action="append", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--seeds-too", action="store_true", help="also merge selected source tasks")
    args = parser.parse_args()
    bundles = [Path(b) for b in args.bundle]
    chosen = {}
    for bundle in bundles:
        for record in _rows(bundle / "candidates.jsonl"):
            if not record["selected"] or (record["origin"] != "composed" and not args.seeds_too):
                continue
            chosen.setdefault(record["candidate_id"], (bundle, record))
    subdirs = (
        "task_ir",
        "gold_lineage",
        "verifiers",
        "rubrics",
        "mutations",
        "records",
        "realization/replies",
    )
    for sub in subdirs:
        (args.out / sub).mkdir(parents=True, exist_ok=True)
    profiles, hazards, instructions, per_bundle = [], [], [], Counter()
    profile_index = {}
    hazard_index = {}
    instruction_index = {}
    for bundle in bundles:
        profile_index[bundle] = {p["task_id"]: p for p in _rows(bundle / "profiles.jsonl")}
        for h in _rows(bundle / "hazards.jsonl"):
            hazard_index.setdefault((bundle, h["task_id"]), []).append(h)
        for i in _rows(bundle / "realization" / "instructions.jsonl"):
            instruction_index.setdefault((bundle, i["task_id"]), []).append(i)
    for task_id, (bundle, record) in sorted(chosen.items()):
        per_bundle[bundle.name] += 1
        for sub, name in (
            ("task_ir", "%s.json" % task_id),
            ("gold_lineage", "%s.gold_lineage.json" % task_id),
            ("verifiers", "%s.json" % task_id),
            ("rubrics", "%s.json" % task_id),
            ("mutations", "%s.jsonl" % task_id),
            ("records", "%s.json" % task_id),
            ("realization/replies", "%s.json" % task_id),
        ):
            source = bundle / sub / name
            if source.is_file():
                shutil.copyfile(source, args.out / sub / name)
        if task_id in profile_index[bundle]:
            profiles.append(profile_index[bundle][task_id])
        hazards.extend(hazard_index.get((bundle, task_id), []))
        instructions.extend(instruction_index.get((bundle, task_id), []))
    for name, rows in (("profiles.jsonl", profiles), ("hazards.jsonl", hazards)):
        (args.out / name).write_text(
            "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8"
        )
    (args.out / "realization" / "instructions.jsonl").write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in instructions), encoding="utf-8"
    )
    records = [
        json.loads((args.out / "records" / ("%s.json" % t)).read_text(encoding="utf-8"))
        for t in sorted(chosen)
        if (args.out / "records" / ("%s.json" % t)).is_file()
    ]
    (args.out / "candidates.jsonl").write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in records), encoding="utf-8"
    )
    accepted = [i for i in instructions if i["status"] == "accepted"]
    manifest = {
        "schema_version": "generation-bundle-merged/1.0",
        "bundles": [str(b) for b in bundles],
        "tasks": len(chosen),
        "by_bundle": dict(per_bundle),
        "by_bucket": dict(Counter(r["bucket"] for r in records)),
        "by_grafts": dict(Counter(r["grafts"] for r in records)),
        "hazard_records": len(hazards),
        "tasks_with_hazard_variant": sum(1 for r in records if r["variants"]["injectable"] > 0),
        "instructions": len(instructions),
        "instructions_accepted": len(accepted),
        "tasks_with_instruction": len({i["task_id"] for i in accepted}),
    }
    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=1) + "\n", encoding="utf-8")
    lines = ["# Merged generation set", ""] + [
        "- %s: %s" % (k, v) for k, v in manifest.items() if k != "schema_version"
    ]
    (args.out / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(manifest), file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
