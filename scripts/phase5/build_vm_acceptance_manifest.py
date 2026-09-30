#!/usr/bin/env python3
"""Freeze the VM and infra identities used by DERAIL Phase 5 acceptance."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

REPOSITORY = Path(__file__).resolve().parents[2]
INFRA_FILES = (
    "control_api_derail.py",
    "control_api_patch.py",
    "snapshot/changelog_replay.py",
    "sqlite3_wrapper.sh",
    "tracer/systemd-dropin.conf",
    "tracer/trace.js",
    "triggers/install_triggers.py",
    "volatile_columns.json",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def git_revision(path: Path) -> str:
    return subprocess.check_output(
        ["git", "-C", str(path), "rev-parse", "HEAD"], text=True
    ).strip()


def build_manifest(args: argparse.Namespace) -> dict:
    qcow2 = args.qcow2.expanduser().resolve()
    mypcbench = args.mypcbench.expanduser().resolve()
    if not qcow2.is_file():
        raise SystemExit(f"qcow2 not found: {qcow2}")
    if not (mypcbench / ".git").exists():
        raise SystemExit(f"MyPCBench checkout not found: {mypcbench}")
    infra = REPOSITORY / "infra"
    files = {
        relative: sha256_file(infra / relative)
        for relative in INFRA_FILES
    }
    return {
        "schema": "vm-acceptance-manifest/1.0",
        "phase": "derail_phase5",
        "boot_date": args.boot_date,
        "qcow2": {"sha256": sha256_file(qcow2), "size": qcow2.stat().st_size},
        "mypcbench_commit": git_revision(mypcbench),
        "infra_commit": git_revision(REPOSITORY),
        "infra_files": files,
        "acceptance": {
            "trajectory_count": 10,
            "required_digest_matches": 10,
            "max_observation_lag_actions": 1,
            "max_step_overhead_seconds": 1.0,
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--qcow2", type=Path, required=True)
    parser.add_argument("--mypcbench", type=Path, required=True)
    parser.add_argument(
        "--boot-date", default=datetime.now(timezone.utc).date().isoformat()
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    manifest = build_manifest(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"output": str(args.output), "qcow2_sha256": manifest["qcow2"]["sha256"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
