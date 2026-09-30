#!/usr/bin/env python3
"""Replay one corrected depth instance in MyPCBench and persist auditable VM evidence."""

from __future__ import annotations

import argparse
import importlib
import json
from pathlib import Path

from recovery.canonical.mypcbench import load_canonical_jsonl
from recovery.construction.cases import DepthInstance
from recovery.derived.layout import atomic_write_json, sha256_file
from recovery.derived.schema import validate_schema
from recovery.replay.mypcbench import MyPCBenchVMReplayBackend
from recovery.replay.verification import ReplayPlan, execute_replay_plan


def _read(path: Path):
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def _factory(spec: str):
    if ":" not in spec:
        raise RuntimeError("env factory 必须是 module:function")
    module_name, function_name = spec.rsplit(":", 1)
    result = getattr(importlib.import_module(module_name), function_name, None)
    if not callable(result):
        raise RuntimeError("env factory 不可调用: %s" % spec)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build-dir", type=Path, required=True)
    parser.add_argument("--case-id", required=True)
    parser.add_argument("--depth", type=int, choices=(0, 5, 10, 15, 20, 25), required=True)
    parser.add_argument("--attempt-id", required=True)
    parser.add_argument("--qcow2", type=Path, required=True)
    parser.add_argument("--env-factory", required=True)
    parser.add_argument("--expected-state-sha256", default="")
    parser.add_argument("--reviewer-ids", nargs="*", default=())
    parser.add_argument("--sleep-after-s", type=float, default=1.0)
    args = parser.parse_args()

    if not args.attempt_id or any(char in args.attempt_id for char in "/\\"):
        raise RuntimeError("attempt-id 必须是安全路径组件")
    repository = Path(__file__).resolve().parents[2]
    build_dir = args.build_dir.resolve()
    manifest = _read(build_dir / "build_manifest.json")
    validate_schema(manifest, "derived_build_manifest.schema.json", repository)
    case_path = build_dir / "cases" / args.case_id / "case.json"
    case = _read(case_path)
    validate_schema(case, "benchmark_case.schema.json", repository)
    instance_path = case_path.parent / "depths" / ("d%d" % args.depth) / "instance.json"
    instance_raw = _read(instance_path)
    validate_schema(instance_raw, "depth_instance.schema.json", repository)
    instance = DepthInstance.from_dict(instance_raw)

    qcow2 = args.qcow2.resolve()
    source_manifest = _read(Path(manifest["source_collection"]["manifest_uri"]))
    if sha256_file(qcow2) != source_manifest.get("image_sha256"):
        raise RuntimeError("qcow2 与 source collection frozen image SHA-256 不一致")
    for ref_name in ("task_config_ref", "state_probe_config_ref"):
        ref = case[ref_name]
        if not Path(ref["uri"]).is_file() or sha256_file(Path(ref["uri"])) != ref["sha256"]:
            raise RuntimeError("case %s URI/hash 无效" % ref_name)
    task_config = _read(Path(case["task_config_ref"]["uri"]))
    state_probe_config = _read(Path(case["state_probe_config_ref"]["uri"]))
    repaired_path = Path(case["canonical_repaired_uri"])
    steps = load_canonical_jsonl(repaired_path)

    attempt_dir = build_dir / "replay" / args.case_id / "attempts" / args.attempt_id
    if attempt_dir.exists() and any(attempt_dir.iterdir()):
        raise RuntimeError("replay attempt 已存在，拒绝覆盖")
    plan = ReplayPlan(
        attempt_id=args.attempt_id,
        build_id=manifest["build_id"],
        case_id=args.case_id,
        instance=instance,
        snapshot_uri=str(qcow2),
        snapshot_sha256=sha256_file(qcow2),
        canonical_trajectory_uri=str(repaired_path.resolve()),
        canonical_trajectory_sha256=sha256_file(repaired_path),
        expected_state_sha256=args.expected_state_sha256,
        task_id=case["task_id"],
        instruction_sha256=case["instruction_sha256"],
        task_config_uri=case["task_config_ref"]["uri"],
        task_config_sha256=case["task_config_ref"]["sha256"],
        state_probe_config_uri=case["state_probe_config_ref"]["uri"],
        state_probe_config_sha256=case["state_probe_config_ref"]["sha256"],
    )
    validate_schema(plan.to_dict(), "replay_plan.schema.json", repository)
    env_factory = _factory(args.env_factory)
    backend = MyPCBenchVMReplayBackend(
        env_factory=lambda: env_factory(
            qcow2_path=qcow2,
            task_config=task_config,
            attempt_id=args.attempt_id,
        ),
        task_config=task_config,
        state_probe_commands=state_probe_config,
        evidence_dir=attempt_dir / "observations",
        sleep_after_s=args.sleep_after_s,
    )
    try:
        verification = execute_replay_plan(
            plan, steps, backend, reviewer_ids=args.reviewer_ids
        )
    finally:
        backend.close()
    validate_schema(
        verification.to_dict(), "replay_verification.schema.json", repository
    )
    atomic_write_json(attempt_dir / "plan.json", plan.to_dict())
    atomic_write_json(attempt_dir / "verification.json", verification.to_dict())
    if verification.accepted_for_release:
        accepted_path = build_dir / "replay" / args.case_id / (instance.instance_id + ".json")
        if accepted_path.exists():
            raise RuntimeError("instance 已有 accepted replay，拒绝覆盖")
        atomic_write_json(accepted_path, verification.to_dict())
    print(json.dumps(verification.to_dict(), ensure_ascii=False, indent=2))
    return 0 if verification.accepted_for_release else 2


if __name__ == "__main__":
    raise SystemExit(main())
