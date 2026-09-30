#!/usr/bin/env python3
"""Freeze the train/test split of the composed workflows (paper section 3: 377 / 100).

    python scripts/freeze_source_splits.py [--config configs/synthesis/workflow_splits_v1.yaml]

Workflows are the selected candidates of a generation bundle (``candidates.jsonl``); the split
is ``derail.gen.splits.split_workflows`` over their source tasks.  This replaces the former
source-task split (110/37/37); its frozen output ``data/synthesis/source_splits.json`` is kept
only for the Task-IR extraction development sample.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import yaml

from derail.derived.layout import atomic_write_json, sha256_file
from derail.gen.splits import SPLITS_VERSION, split_workflows


def main(argv: Iterable[str] = ()) -> int:
    root_default = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "--config", type=Path, default=root_default / "configs/synthesis/workflow_splits_v1.yaml"
    )
    parser.add_argument("--repo-root", type=Path, default=root_default)
    args = parser.parse_args(list(argv) if argv else None)
    root = args.repo_root.resolve()
    raw = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    if raw.get("schema_version") != "workflow-splits-config/1.0":
        raise SystemExit("unsupported config %r" % raw.get("schema_version"))

    candidates_path = root / raw["bundle"] / "candidates.jsonl"
    origins = set(raw.get("origins") or ["composed"])
    workflows = {}
    for line in candidates_path.read_text(encoding="utf-8").splitlines():
        record = json.loads(line)
        if record.get("selected") and record.get("origin") in origins:
            workflows[record["candidate_id"]] = record["source_task_ids"]
    result = split_workflows(
        workflows,
        seed=int(raw["seed"]),
        test_size=raw.get("test_size"),
        test_fraction=raw.get("test_fraction"),
        source_disjoint=bool(raw.get("source_disjoint", False)),
    )
    record = {
        "schema_version": SPLITS_VERSION,
        "seed": int(raw["seed"]),
        "test_size": raw.get("test_size"),
        "test_fraction": raw.get("test_fraction"),
        "source_disjoint": bool(raw.get("source_disjoint", False)),
        **result,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "inputs": {
            "candidates": str(candidates_path),
            "candidates_sha256": sha256_file(candidates_path),
            "config": str(args.config.resolve()),
            "config_sha256": sha256_file(args.config),
        },
    }
    output = root / raw["output"]
    atomic_write_json(output, record)
    print(json.dumps({"counts": record["counts"], "output": str(output)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
