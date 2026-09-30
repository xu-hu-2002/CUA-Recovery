#!/usr/bin/env python3
"""Build a hash-locked model/shard workload from the frozen Phase 5 manifest."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPOSITORY = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPOSITORY / "src"))

from recovery.phase5.workload import FrozenPhase5Manifest  # noqa: E402

DEFAULT_MANIFEST = REPOSITORY / "artifacts/phase5/run-manifest-option-a-20260912.json"
DEFAULT_SHA256 = "228c209ed2d892c34adc5539884ab7f591de0dd8a27a38a417f2c49642887126"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--manifest-sha256", default=DEFAULT_SHA256)
    parser.add_argument("--model", required=True)
    parser.add_argument("--shard-count", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--out", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    frozen = FrozenPhase5Manifest(REPOSITORY, args.manifest, args.manifest_sha256)
    workload = frozen.resolve(args.model, args.shard_count, args.shard_index)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(workload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"wrote {workload['combination_count']} combinations to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
