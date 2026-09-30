#!/usr/bin/env python3
"""Upload one judge archive directory to its frozen OSS tree with parity checks."""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
from pathlib import Path


ALLOWED_ROOT = os.environ.get("JUDGE_OSS_ROOT", "")
LISTING = re.compile(r"^\S+\s+\S+\s+\S+\s+\S+\s+(\d+)\s+\S+\s+\S+\s+(oss://\S+)$")


def local_files(source: Path) -> dict[str, int]:
    return {
        str(path.relative_to(source)): path.stat().st_size
        for path in sorted(source.rglob("*"))
        if path.is_file() and path.name != "ship_ledger.json"
    }


def parse_listing(output: str, destination: str) -> dict[str, int]:
    prefix = destination.rstrip("/") + "/"
    records = {}
    for line in output.splitlines():
        match = LISTING.match(line.strip())
        if match and match.group(2).startswith(prefix):
            relative = match.group(2)[len(prefix):]
            if relative and relative != "ship_ledger.json":
                records[relative] = int(match.group(1))
    return records


def run(command: list[str]) -> str:
    result = subprocess.run(command, capture_output=True, text=True, check=False)
    if result.returncode:
        raise RuntimeError(result.stderr.strip() or result.stdout.strip())
    return result.stdout


def ship(source: Path, destination: str, ossutil: str) -> dict:
    if not ALLOWED_ROOT:
        raise ValueError("set JUDGE_OSS_ROOT to the OSS root that judge archives ship to")
    if not destination.startswith(ALLOWED_ROOT):
        raise ValueError(f"judge destination must be under {ALLOWED_ROOT}")
    expected = local_files(source)
    if not expected:
        raise ValueError(f"judge archive is empty (excluding ship_ledger.json): {source}")
    run([ossutil, "cp", "-r", "-f", f"{source}/", destination.rstrip("/") + "/"])
    observed = parse_listing(run([ossutil, "ls", destination]), destination)
    missing = sorted(set(expected) - set(observed))
    unexpected = sorted(set(observed) - set(expected))
    mismatched = sorted(path for path in expected.keys() & observed.keys() if expected[path] != observed[path])
    return {
        "local_object_count": len(expected),
        "remote_object_count": len(observed),
        "missing_remote": missing,
        "unexpected_remote": unexpected,
        "size_mismatches": mismatched,
        "object_parity": not (missing or unexpected or mismatched),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--destination", required=True)
    parser.add_argument("--ossutil", default=os.environ.get("OSSUTIL") or "ossutil")
    args = parser.parse_args()
    if not args.source.is_dir():
        raise ValueError(f"judge archive does not exist: {args.source}")
    ledger = ship(args.source.resolve(), args.destination, args.ossutil)
    (args.source / "ship_ledger.json").write_text(
        json.dumps(ledger, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    if not ledger["object_parity"]:
        raise RuntimeError(f"judge OSS object parity failed: {ledger}")
    run([args.ossutil, "cp", "-f", str(args.source / "ship_ledger.json"), args.destination])
    print(json.dumps(ledger, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
