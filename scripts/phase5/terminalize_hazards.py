#!/usr/bin/env python3
"""Merge explicit guest hazard results into an immutable terminal JSONL file."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from derail.phase5.executor import apply_results


def _jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    results = {str(row["injection_id"]): row for row in _jsonl(args.results)}
    merged = apply_results(_jsonl(args.input), results)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(json.dumps(row, sort_keys=True) for row in merged) + "\n")
    print(json.dumps({"records": len(merged), "terminal": True}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
