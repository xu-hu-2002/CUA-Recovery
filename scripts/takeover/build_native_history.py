#!/usr/bin/env python3
"""Build one target agent's replay-bound native history after a live conformance probe."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from recovery.adapters import create_native_history_adapter
from recovery.canonical.mypcbench import load_canonical_jsonl
from recovery.derived.layout import atomic_write_json, sha256_file
from recovery.derived.schema import validate_schema
from recovery.replay.verification import ReplayVerification
from recovery.takeover.history import build_native_history_from_replay
from recovery.takeover.protocol import DEFAULT_CONFIG_PATH, load_takeover_config


def _read(path: Path):
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build-dir", type=Path, required=True)
    parser.add_argument("--case-id", required=True)
    parser.add_argument("--depth", type=int, choices=(0, 5, 10, 15, 20, 25), required=True)
    parser.add_argument("--agent-id", required=True)
    parser.add_argument("--renderer-version", required=True)
    parser.add_argument("--conformance-probe", type=Path, required=True)
    parser.add_argument("--conformance-passed", action="store_true")
    parser.add_argument("--takeover-config", type=Path, default=DEFAULT_CONFIG_PATH,
                        help="history.strip_reasoning decides the recorded reasoning_policy")
    args = parser.parse_args()

    if not args.conformance_passed:
        raise RuntimeError("only a live probe with explicit --conformance-passed can produce release history")
    repository = Path(__file__).resolve().parents[2]
    build_dir = args.build_dir.resolve()
    case_path = build_dir / "cases" / args.case_id / "case.json"
    case = _read(case_path)
    validate_schema(case, "benchmark_case.schema.json", repository)
    instance_path = case_path.parent / "depths" / ("d%d" % args.depth) / "instance.json"
    instance = _read(instance_path)
    replay_path = build_dir / "replay" / args.case_id / (instance["instance_id"] + ".json")
    replay_raw = _read(replay_path)
    validate_schema(replay_raw, "replay_verification.schema.json", repository)
    replay = ReplayVerification.from_dict(replay_raw)
    if not replay.accepted_for_release:
        raise RuntimeError("native history needs an accepted real-VM replay")
    probe_path = args.conformance_probe.resolve()
    if not probe_path.is_file():
        raise RuntimeError("conformance probe evidence does not exist")
    adapter = create_native_history_adapter(args.agent_id, case["instruction"])
    steps = load_canonical_jsonl(Path(case["canonical_repaired_uri"]))
    artifact = build_native_history_from_replay(
        adapter,
        steps,
        replay,
        renderer_version=args.renderer_version,
        conformance_probe_sha256=sha256_file(probe_path),
        conformance_passed=True,
        conformance_probe_uri=str(probe_path),
        strip_reasoning=bool(load_takeover_config(args.takeover_config)["history"]["strip_reasoning"]),
    )
    output = (
        case_path.parent
        / "compatibility"
        / args.agent_id
        / (instance["instance_id"] + ".json")
    )
    if output.exists():
        raise RuntimeError("native history artifact already exists, refusing to overwrite")
    validate_schema(artifact.to_dict(), "native_history.schema.json", repository)
    atomic_write_json(output, artifact.to_dict())
    print(json.dumps({"written": str(output), "release_eligible": True}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
