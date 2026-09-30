#!/usr/bin/env python3
"""Run or ingest one takeover episode, then validate all release-evidence bindings."""

from __future__ import annotations

import argparse
import importlib
import json
from pathlib import Path

from recovery.canonical.actions import action_to_dict
from recovery.canonical.mypcbench import load_canonical_jsonl
from recovery.construction.cases import DepthInstance
from recovery.derived.layout import atomic_write_json, sha256_file
from recovery.derived.schema import validate_schema
from recovery.evaluation.records import EvaluationEpisode
from recovery.evaluation.takeover import run_takeover_episode
from recovery.replay.verification import ReplayPlan, ReplayVerification, execute_replay_plan
from recovery.takeover.history import NativeHistoryArtifact, build_native_history_from_replay


def _read(path: Path):
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def _load_factory(spec: str):
    if ":" not in spec:
        raise RuntimeError("runner factory 必须是 module:function")
    module_name, function_name = spec.rsplit(":", 1)
    function = getattr(importlib.import_module(module_name), function_name, None)
    if not callable(function):
        raise RuntimeError("runner factory 不可调用: %s" % spec)
    return function


def _evidence_valid(uri: str, expected: str) -> bool:
    path = Path(uri)
    return path.is_file() and sha256_file(path) == expected


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build-dir", type=Path, required=True)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--episode-json", type=Path)
    mode.add_argument("--runner-factory")
    parser.add_argument("--case-id")
    parser.add_argument("--instance-id")
    parser.add_argument("--agent-id")
    parser.add_argument("--agent-model-revision")
    parser.add_argument("--agent-prompt-sha256")
    parser.add_argument("--adapter-sha256")
    parser.add_argument("--repeat-id", type=int)
    parser.add_argument("--run-id")
    parser.add_argument("--qcow2", type=Path)
    parser.add_argument(
        "--replay-json",
        type=Path,
        help="Accepted per-evaluation replay evidence when ingesting an external episode.",
    )
    parser.add_argument(
        "--native-history-json",
        type=Path,
        help="Replay-bound native history when ingesting an external valid episode.",
    )
    args = parser.parse_args()
    repository = Path(__file__).resolve().parents[2]
    build_dir = args.build_dir.resolve()
    manifest = _read(build_dir / "build_manifest.json")
    if args.episode_json:
        episode_raw = _read(args.episode_json)
        validate_schema(episode_raw, "evaluation_result.schema.json", repository)
        episode = EvaluationEpisode.from_dict(episode_raw)
        trace = None
        active_replay = None
        history = None
    else:
        required = {
            name: getattr(args, name)
            for name in (
                "case_id",
                "instance_id",
                "agent_id",
                "agent_model_revision",
                "agent_prompt_sha256",
                "adapter_sha256",
                "repeat_id",
                "run_id",
            )
        }
        missing = [name for name, value in required.items() if value in (None, "")]
        if missing:
            raise RuntimeError("runner mode 缺少参数: %s" % ", ".join(missing))
        case_path = build_dir / "cases" / args.case_id / "case.json"
        case = _read(case_path)
        instance_path = (
            case_path.parent
            / "depths"
            / next(
                "d%d" % depth
                for depth in case["available_depths"]
                if "%s-d%d" % (args.case_id, depth) == args.instance_id
            )
            / "instance.json"
        )
        instance = DepthInstance.from_dict(_read(instance_path))
        baseline_path = (
            build_dir / "replay" / args.case_id / (args.instance_id + ".json")
        )
        baseline_raw = _read(baseline_path)
        validate_schema(baseline_raw, "replay_verification.schema.json", repository)
        baseline = ReplayVerification.from_dict(baseline_raw)
        if not baseline.accepted_for_release:
            raise RuntimeError("runner mode 需要 accepted baseline VM replay")
        if args.qcow2 is None:
            raise RuntimeError("runner mode 必须提供 --qcow2")
        qcow2 = args.qcow2.resolve()
        if sha256_file(qcow2) != baseline.snapshot_sha256:
            raise RuntimeError("evaluation qcow2 与 baseline replay 不一致")
        dependencies = _load_factory(args.runner_factory)(
            case=case,
            instance=instance,
            baseline_replay_verification=baseline,
        )
        if not isinstance(dependencies, dict):
            raise RuntimeError("runner factory 必须返回 dependency mapping")
        repaired_path = Path(case["canonical_repaired_uri"])
        steps = load_canonical_jsonl(repaired_path)
        replay_plan = ReplayPlan(
            attempt_id=args.run_id + "-replay",
            build_id=manifest["build_id"],
            case_id=case["case_id"],
            instance=instance,
            snapshot_uri=str(qcow2),
            snapshot_sha256=sha256_file(qcow2),
            canonical_trajectory_uri=str(repaired_path.resolve()),
            canonical_trajectory_sha256=sha256_file(repaired_path),
            expected_state_sha256=baseline.state_fingerprint.sha256,
            task_id=case["task_id"],
            instruction_sha256=case["instruction_sha256"],
            task_config_uri=case["task_config_ref"]["uri"],
            task_config_sha256=case["task_config_ref"]["sha256"],
            state_probe_config_uri=case["state_probe_config_ref"]["uri"],
            state_probe_config_sha256=case["state_probe_config_ref"]["sha256"],
        )
        environment = dependencies["environment"]
        active_replay = execute_replay_plan(
            replay_plan,
            steps,
            environment,
            reviewer_ids=baseline.reviewer_ids,
        )
        if not active_replay.accepted_for_release:
            raise RuntimeError("per-evaluation real-VM replay 未通过 baseline state gate")
        history = build_native_history_from_replay(
            dependencies["adapter"],
            steps,
            active_replay,
            renderer_version=dependencies["renderer_version"],
            conformance_probe_sha256=dependencies["conformance_probe_sha256"],
            conformance_passed=dependencies["conformance_passed"],
            conformance_probe_uri=dependencies["conformance_probe_uri"],
        )
        trace = run_takeover_episode(
            agent=dependencies["agent"],
            environment=environment,
            ear_judge=dependencies["ear_judge"],
            pesr_grader=dependencies["pesr_grader"],
            instruction=case["instruction"],
            native_history=history,
            replay_verification=active_replay,
            run_id=args.run_id,
            build_id=manifest["build_id"],
            instance_id=instance.instance_id,
            case_id=case["case_id"],
            agent_id=args.agent_id,
            agent_model_revision=args.agent_model_revision,
            agent_prompt_sha256=args.agent_prompt_sha256,
            adapter_sha256=args.adapter_sha256,
            depth=instance.depth,
            repeat_id=args.repeat_id,
        )
        episode = trace.episode
    if episode.build_id != manifest["build_id"]:
        raise RuntimeError("episode build_id 与 derived build 不一致")
    if episode.valid:
        if active_replay is None:
            if args.replay_json is None or not args.replay_json.is_file():
                raise RuntimeError("valid external episode 必须提供 --replay-json")
            replay_raw = _read(args.replay_json)
            validate_schema(replay_raw, "replay_verification.schema.json", repository)
            active_replay = ReplayVerification.from_dict(replay_raw)
        replay = active_replay
        if replay.execution_mode != "vm" or not replay.accepted_for_release:
            raise RuntimeError("valid evaluation 只能基于 accepted real-VM replay")
        if replay.attempt_id != episode.replay_verification_id:
            raise RuntimeError("episode replay_verification_id 不匹配")
        instance_paths = list(
            (build_dir / "cases" / episode.case_id / "depths").glob("d*/instance.json")
        )
        matching = [
            DepthInstance.from_dict(_read(path))
            for path in instance_paths
            if _read(path).get("instance_id") == episode.instance_id
        ]
        if len(matching) != 1:
            raise RuntimeError("episode instance 不存在或重复")
        instance = matching[0]
        case_path = build_dir / "cases" / episode.case_id / "case.json"
        case = _read(case_path)
        validate_schema(case, "benchmark_case.schema.json", repository)
        source_manifest = _read(Path(manifest["source_collection"]["manifest_uri"]))
        if (
            episode.depth != instance.depth
            or replay.depth != instance.depth
            or replay.instance_id != instance.instance_id
            or replay.case_id != episode.case_id
            or replay.build_id != episode.build_id
            or replay.canonical_actions_sha256 != instance.canonical_actions_sha256
            or replay.canonical_trajectory_sha256 != case["canonical_repaired_sha256"]
            or replay.task_id != case["task_id"]
            or replay.instruction_sha256 != case["instruction_sha256"]
            or replay.task_config_sha256 != case["task_config_ref"]["sha256"]
            or replay.state_probe_config_sha256
            != case["state_probe_config_ref"]["sha256"]
            or replay.snapshot_sha256 != source_manifest.get("image_sha256")
        ):
            raise RuntimeError("episode/replay/instance binding 不一致")
        if history is None:
            if args.native_history_json is None or not args.native_history_json.is_file():
                raise RuntimeError("valid external episode 必须提供 --native-history-json")
            history_raw = _read(args.native_history_json)
            validate_schema(history_raw, "native_history.schema.json", repository)
            history = NativeHistoryArtifact.from_dict(history_raw)
        expected_observations = tuple(
            str(item["observation_before_sha256"]) for item in replay.per_action_log
        )
        probe_path = Path(history.conformance_probe_uri)
        if not (
            history.release_eligible
            and history.agent_id == episode.agent_id
            and history.instance_id == episode.instance_id
            and history.replay_verification_id == replay.attempt_id
            and history.action_indices == replay.executed_action_indices
            and history.replay_observation_sha256s == expected_observations
            and probe_path.is_file()
            and sha256_file(probe_path) == history.conformance_probe_sha256
        ):
            raise RuntimeError("evaluation native history/replay/conformance binding 无效")
        evidence = (
            (episode.ear.takeover_output_uri, episode.ear.takeover_output_sha256),
            (episode.ear.raw_judgment_uri, episode.ear.raw_judgment_sha256),
            (episode.pesr.rubric_bundle_uri, episode.pesr.rubric_bundle_sha256),
            (episode.pesr.raw_judgment_uri, episode.pesr.raw_judgment_sha256),
        )
        if not all(_evidence_valid(uri, digest) for uri, digest in evidence):
            raise RuntimeError("evaluation EAR/PESR evidence URI/hash 无效")
    output = (
        build_dir
        / "evaluations"
        / episode.agent_id
        / episode.case_id
        / ("d%d" % episode.depth)
        / ("repeat_%d" % episode.repeat_id)
        / "episode.json"
    )
    if output.exists():
        raise RuntimeError("evaluation episode 已存在，拒绝覆盖: %s" % output)
    validate_schema(episode.to_dict(), "evaluation_result.schema.json", repository)
    atomic_write_json(output, episode.to_dict())
    if trace is not None:
        atomic_write_json(output.parent / "replay_verification.json", active_replay.to_dict())
        atomic_write_json(output.parent / "native_history.json", history.to_dict())
        atomic_write_json(
            output.parent / "trace.json",
            {
                "public_outputs": list(trace.public_outputs),
                "executed_actions": [action_to_dict(action) for action in trace.executed_actions],
            },
        )
        close = getattr(environment, "close", None)
        if callable(close):
            close()
    print(
        json.dumps(
            {"written": str(output), "valid": episode.valid}, ensure_ascii=False, indent=2
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
