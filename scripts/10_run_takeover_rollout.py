#!/usr/bin/env python3
"""Run one prefix-takeover rollout and emit MyPCBench judge artifacts."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any


REPOSITORY = Path(__file__).resolve().parents[1]
SRC = REPOSITORY / "src"
HARNESS = REPOSITORY / "third_party" / "MyPCBench" / "agent-harness"
for entry in (str(SRC), str(HARNESS)):
    if entry not in sys.path:
        sys.path.insert(0, entry)

from env import MyPCBenchEnv  # noqa: E402
from run_mypcbench import get_agent, run_single_example  # noqa: E402

from derail.canonical.mypcbench import load_canonical_jsonl  # noqa: E402
from derail.derived.layout import atomic_write_json, sha256_file  # noqa: E402
from derail.mypcbench.agent_config import load_config  # noqa: E402
from derail.mypcbench.takeover_agent import PrefixTakeoverAgent  # noqa: E402
from derail.mypcbench.qwen35_takeover import wrap_qwen35_takeover_target  # noqa: E402
from derail.mypcbench.claude_takeover import wrap_claude_takeover_target  # noqa: E402
from derail.mypcbench.openai_takeover import wrap_openai_takeover_target  # noqa: E402
from derail.rollout.state_probe import EnvironmentHooks  # noqa: E402
from derail.takeover.diagnosis import load_human_diagnosis_evidence  # noqa: E402
from derail.takeover.protocol import (  # noqa: E402
    DEFAULT_CONFIG_PATH,
    ProtocolExclusion,
    load_takeover_config,
    load_takeover_steps,
)

# The shell controller does not retry this exit code: the protocol rejected the episode
# (missing repaired prefix under on_missing_repaired=skip, or replayed-state mismatch).
PROTOCOL_EXCLUSION_EXIT = 3


def _read_object(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"cannot read {label}: {path}") from exc
    if not isinstance(value, dict):
        raise RuntimeError(f"{label} must be a JSON object: {path}")
    return value


def _safe_task_id(task: dict[str, Any]) -> str:
    task_id = task.get("id")
    if (
        not isinstance(task_id, str)
        or not task_id
        or "/" in task_id
        or "\\" in task_id
        or ".." in task_id
        or task_id.startswith(("~", "."))
    ):
        raise RuntimeError(f"unsafe or missing task id: {task_id!r}")
    return task_id


def _result_is_completed(path: Path) -> bool:
    """Only a successful binary completion marker is resumable."""

    try:
        return float(path.read_text(encoding="utf-8").strip()) == 1.0
    except (OSError, UnicodeError, ValueError):
        return False


def _result_is_failed(path: Path) -> bool:
    """Return whether a prior attempt left the binary 0.0 failure marker."""

    try:
        return float(path.read_text(encoding="utf-8").strip()) == 0.0
    except (OSError, UnicodeError, ValueError):
        return False


def _result_exit_code(result: Any) -> int:
    """Map the binary runner marker to the process contract used by retries."""

    try:
        completed = float(result) == 1.0
    except (TypeError, ValueError):
        completed = False
    return 0 if completed else 1


def _takeover_artifact_errors(result_dir: Path) -> list[str]:
    """Return missing or invalid evidence for one completed takeover episode."""

    required = (
        "native_history.json",
        "prefix_replay_log.json",
        "takeover_manifest.json",
        "rubric_bundle.json",
        "traj.jsonl",
    )
    errors = [name for name in required if not (result_dir / name).is_file()]
    trajectory = result_dir / "traj.jsonl"
    if trajectory.is_file() and "PREDICT_CRASH" in trajectory.read_text(
        encoding="utf-8", errors="replace"
    ):
        errors.append("traj.jsonl contains PREDICT_CRASH")
    if not any(result_dir.glob("*.png")):
        errors.append("no screenshots")
    return errors


def _enforce_takeover_artifacts(result_dir: Path, result: Any) -> int:
    """Convert false-positive runner success into a retryable failure."""

    if _result_exit_code(result):
        return 1
    errors = _takeover_artifact_errors(result_dir)
    if not errors:
        return 0
    (result_dir / "result.txt").write_text("0.0\n", encoding="utf-8")
    (result_dir / "incomplete.txt").write_text("\n".join(errors) + "\n", encoding="utf-8")
    print(f"Takeover artifact contract failed: {', '.join(errors)}", file=sys.stderr)
    return 1


def _record_protocol_exclusion(result_dir: Path, reason: str, detail: str) -> int:
    exclusion = result_dir / "protocol_exclusion.json"
    if not exclusion.is_file():
        atomic_write_json(exclusion, {"reason": reason, "detail": detail})
    (result_dir / "result.txt").write_text("0.0\n", encoding="utf-8")
    (result_dir / "incomplete.txt").write_text(f"protocol_exclusion: {reason}\n", encoding="utf-8")
    print(f"PROTOCOL_EXCLUSION: reason={reason} detail={detail}", file=sys.stderr)
    return PROTOCOL_EXCLUSION_EXIT


def _live_target_agent_id(target_agent_id: str) -> str:
    """Resolve the live scaffold without changing the canonical experiment ID."""

    if target_agent_id == "kimi_k3":
        return "kimi_k3_cuabash"
    return target_agent_id


def _runtime_model_name(target_agent_id: str, configured_model: str) -> str:
    """Resolve the endpoint alias used by the live target service."""

    env_names = {
        "evocua_32b": "EVOCUA_MODEL",
        "opencua_72b": "OPENCUA_MODEL",
        "qwen3_8_27b": "QWEN38_MODEL",
        "qwen3_5_35b_a3b": "QWEN35_MODEL",
        "kimi_k3": "KIMI_K3_MODEL",
        "gpt_5_5": "GPT55_MODEL",
        "claude_opus_4_8": "CLAUDE_OPUS_4_8_MODEL",
    }
    env_name = env_names.get(target_agent_id)
    if not env_name:
        return configured_model
    return os.environ.get(env_name, configured_model)


def _assert_bash_environment_bound(target: Any, environment: Any) -> None:
    """Hard-fail when a live cuabash target is detached from its VM."""

    protocol = getattr(target, "protocol", None)
    if not getattr(protocol, "enable_bash", False):
        return
    if getattr(target, "_env", None) is not environment:
        raise RuntimeError("live cuabash target is not bound to the active VM environment")


def _load_human_source_judge(
    annotation_path: Path,
    *,
    trajectory_id: str,
    source_sha256: str,
    annotator_id: str,
    root_cause_action_index: int,
) -> dict[str, Any]:
    """Load the hash-bound human rubric decision paired with an annotation."""

    review_path = annotation_path.parent / "rubric_scores" / annotation_path.name
    review = _read_object(review_path, "human rubric review")
    expected = (
        review.get("reviewer_role") == "human"
        and review.get("reviewer_id") == annotator_id
        and review.get("trajectory_id") == trajectory_id
        and review.get("source_trajectory_sha256") == source_sha256
        and review.get("task_success") is False
    )
    if not expected:
        raise RuntimeError("human rubric review is not bound to the selected failure")
    return {
        "authority": "human_annotation",
        "failure_eligible": True,
        "trajectory_id": trajectory_id,
        "source_trajectory_sha256": source_sha256,
        "review_uri": str(review_path.resolve()),
        "review_sha256": sha256_file(review_path),
        "reviewer_id": annotator_id,
        "root_cause_action_index": root_cause_action_index,
        "scores": dict(review.get("scores", {})),
        "task_success": False,
        "perfect_score_policy": "all human rubric scores equal 1",
        "rubric_bundle_sha256": str(review.get("rubric_bundle_sha256", "")),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-agent", required=True)
    parser.add_argument("--target-agent", required=True)
    parser.add_argument(
        "--condition", choices=("unaware", "notified", "diagnosed"), required=True
    )
    parser.add_argument("--depth", type=int, default=0)
    parser.add_argument("--canonical-trajectory", type=Path, required=True)
    parser.add_argument("--normalization-report", type=Path, required=True)
    parser.add_argument("--task-json", type=Path, required=True)
    parser.add_argument("--annotation-json", type=Path, required=True)
    parser.add_argument("--qcow2", type=Path, required=True)
    parser.add_argument("--qcow2-sha256", default="")
    parser.add_argument("--qcow2-hash-preverified", action="store_true")
    parser.add_argument("--result-dir", type=Path, required=True)
    parser.add_argument("--takeover-config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--repeat", type=int, default=1, help="1-based run index of this state")
    parser.add_argument("--max-steps", type=int, help="default: takeover config max_steps")
    parser.add_argument("--timeout", type=int, help="default: takeover config timeout_seconds")
    parser.add_argument(
        "--prefix-source", choices=("repaired", "original"), help="default: config prefix.source"
    )
    parser.add_argument(
        "--repaired-prefix-dir",
        type=Path,
        help="default: <build_dir>/<prefix.repaired_dir> from the takeover config",
    )
    parser.add_argument(
        "--on-missing-repaired",
        choices=("error", "skip", "fallback_original"),
        help="default: config prefix.on_missing_repaired",
    )
    parser.add_argument("--sleep-after", type=float, default=1.0)
    parser.add_argument("--replay-sleep-after", type=float, default=1.0)
    parser.add_argument("--backend", choices=("qemu", "docker"), default="qemu")
    parser.add_argument("--docker-image", default="ljang/mypcbench-qemu:latest")
    parser.add_argument("--container-name", default="mypcbench-agent")
    parser.add_argument("--worker-index", type=int, default=0)
    parser.add_argument(
        "--skip-completed",
        action="store_true",
        help="Return successfully without booting a VM only when result.txt is 1.0",
    )
    parser.add_argument(
        "--retry-failed-only",
        action="store_true",
        help="Run only tasks that already have result.txt=0.0",
    )
    parser.add_argument(
        "--port-base",
        type=int,
        default=24000,
        help="First host port; each worker receives a disjoint 100-port window",
    )
    parser.add_argument("--persona", default="michael_scott")
    parser.add_argument("--world", default="scranton-office")
    parser.add_argument("--client-password", default="password")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = load_takeover_config(args.takeover_config)
    if args.max_steps is None:
        args.max_steps = int(config["max_steps"])
    if args.timeout is None:
        args.timeout = int(config["timeout_seconds"])
    prefix_config = dict(config["prefix"])
    if args.prefix_source:
        prefix_config["source"] = args.prefix_source
    if args.on_missing_repaired:
        prefix_config["on_missing_repaired"] = args.on_missing_repaired
    if args.max_steps <= 0 or args.timeout <= 0 or args.depth < 0 or args.repeat <= 0:
        raise RuntimeError("max-steps/timeout must be positive and depth must be non-negative")
    if args.worker_index < 0 or not 6000 <= args.port_base <= 65000:
        raise RuntimeError("worker-index must be non-negative and port-base must be 6000..65000")
    worker_port_base = args.port_base + args.worker_index * 100
    if worker_port_base + 27 > 65535:
        raise RuntimeError("worker port window exceeds TCP port 65535")
    canonical_path = args.canonical_trajectory.resolve()
    report_path = args.normalization_report.resolve()
    task_path = args.task_json.resolve()
    qcow2 = args.qcow2.expanduser().resolve()
    if not qcow2.is_file():
        raise RuntimeError(f"qcow2 does not exist: {qcow2}")

    original_steps = load_canonical_jsonl(canonical_path)
    report = _read_object(report_path, "normalization report")
    task = _read_object(task_path, "task config")
    task_id = _safe_task_id(task)
    trajectory_id = str(report.get("trajectory_id", ""))
    if not trajectory_id or canonical_path.parent.name != trajectory_id:
        raise RuntimeError("canonical path and normalization report trajectory IDs differ")
    if report.get("canonical_trajectory_sha256") != sha256_file(canonical_path):
        raise RuntimeError("canonical trajectory SHA-256 differs from normalization report")
    if not (task.get("grading") or {}).get("rubrics"):
        raise RuntimeError("task config has no grading.rubrics for LLM-as-judge")

    diagnosis = load_human_diagnosis_evidence(
        args.annotation_json,
        expected_trajectory_id=trajectory_id,
        expected_source_trajectory_sha256=str(report.get("source_trajectory_sha256", "")),
        maximum_action_index=len(original_steps) - 1,
    )
    human_source_judge = _load_human_source_judge(
        args.annotation_json.resolve(),
        trajectory_id=trajectory_id,
        source_sha256=str(report.get("source_trajectory_sha256", "")),
        annotator_id=diagnosis.annotator_id,
        root_cause_action_index=diagnosis.root_cause_action_index,
    )
    live_target_agent_id = _live_target_agent_id(args.target_agent)
    # Upstream MyPCBench targets (currently Claude) still need their validated
    # document for agent_type/model routing.  load_agent_config intentionally
    # rejects those targets because only DERAIL factory agents have live
    # protocol fields; load_config validates both kinds without weakening the
    # native upstream construction path.
    target_config = load_config(live_target_agent_id)
    agent_type = str(target_config.document["agent_type"])
    configured_model = str(
        target_config.document.get("checkpoint") or target_config.document.get("model") or ""
    )
    if not configured_model:
        raise RuntimeError(f"target config has no model/checkpoint: {target_config.path}")
    target_model = _runtime_model_name(args.target_agent, configured_model)
    os.environ["DERAIL_REPO_ROOT"] = str(REPOSITORY)
    os.environ["DERAIL_AGENT_ID"] = live_target_agent_id
    os.environ["DERAIL_AGENT_MAX_STEPS"] = str(args.max_steps)
    os.environ["MYPCBENCH_HOST_API_PORT"] = str(worker_port_base)
    os.environ["MYPCBENCH_HOST_VNC_PORT"] = str(worker_port_base + 1)
    os.environ["MYPCBENCH_HOST_SSH_PORT"] = str(worker_port_base + 3)
    for offset, guest_port in enumerate(range(3001, 3019), start=10):
        os.environ[f"MYPCBENCH_HOST_APP_PORT_{guest_port}"] = str(
            worker_port_base + offset
        )

    condition_root = args.result_dir.resolve()
    task_result_dir = condition_root / task_id
    task_result_dir.mkdir(parents=True, exist_ok=True)
    result_path = task_result_dir / "result.txt"
    if args.skip_completed and _result_is_completed(result_path):
        print(f"Skipping completed takeover rollout: {task_result_dir}", flush=True)
        return 0
    if args.retry_failed_only and not _result_is_failed(result_path):
        print(f"Skipping task without prior result.txt=0.0: {task_result_dir}", flush=True)
        return 0
    # Keep result.txt=0.0 until the new attempt finishes so an interrupted
    # smoke/recovery run remains resumable. A successful retry must not retain
    # stale failure detail, though, so clear it here.
    for stale in ("incomplete.txt", "protocol_exclusion.json", "prefix_state_verification.json"):
        try:
            (task_result_dir / stale).unlink()
        except FileNotFoundError:
            pass
    try:
        steps, prefix_info = load_takeover_steps(
            canonical_path,
            trajectory_id,
            diagnosis.root_cause_action_index,
            prefix_config,
            args.repaired_prefix_dir,
        )
    except ProtocolExclusion as exc:
        return _record_protocol_exclusion(task_result_dir, exc.reason, exc.detail)
    runtime_dir = condition_root / "_runtime" / f"worker_{args.worker_index}"
    runtime_dir.mkdir(parents=True, exist_ok=True)
    os.environ["MYPCBENCH_RUNTIME_DIR"] = str(runtime_dir)
    qcow2_sha256 = args.qcow2_sha256.strip() or sha256_file(qcow2)
    if len(qcow2_sha256) != 64:
        raise RuntimeError("qcow2 SHA-256 must contain 64 hexadecimal characters")

    env = MyPCBenchEnv(
        docker_image=args.docker_image,
        container_name=args.container_name,
        persona=args.persona,
        world=args.world,
        screen_size=(1280, 800),
        client_password=args.client_password,
        backend=args.backend,
        qcow2_path=str(qcow2),
    )
    try:
        target = get_agent(
            agent_type=agent_type,
            model=target_model,
            screen_size=(1280, 800),
            client_password=args.client_password,
            env=env,
        )
        target = wrap_qwen35_takeover_target(target, args.target_agent)
        target = wrap_claude_takeover_target(target, args.target_agent)
        target = wrap_openai_takeover_target(
            target,
            args.target_agent,
            context_screenshots=int(config["history"]["openai_context_screenshots"]),
        )
        _assert_bash_environment_bound(target, env)
        agent = PrefixTakeoverAgent(
            target_agent=target,
            environment=env,
            task_config=task,
            canonical_steps=steps,
            diagnosis=diagnosis,
            condition=args.condition,
            source_agent=args.source_agent,
            target_agent_id=args.target_agent,
            qcow2_path=qcow2,
            qcow2_sha256=qcow2_sha256,
            qcow2_hash_preverified=args.qcow2_hash_preverified,
            artifact_dir=task_result_dir,
            depth=args.depth,
            replay_sleep_after_s=args.replay_sleep_after,
            strip_reasoning=bool(config["history"]["strip_reasoning"]),
            evocua_cross_agent=bool(config["history"]["evocua_cross_agent"]),
            replay_verification=config["replay_verification"],
            prefix_info=prefix_info,
            environment_hooks=EnvironmentHooks.from_config(Path(config["environment_config"])),
        )
        atomic_write_json(
            task_result_dir / "takeover_launch.json",
            {
                "source_agent": args.source_agent,
                "target_agent": args.target_agent,
                "live_target_agent": live_target_agent_id,
                "target_model": target_model,
                "target_agent_type": agent_type,
                "condition": args.condition,
                "depth": args.depth,
                "repeat": args.repeat,
                "takeover_config_uri": config["config_uri"],
                "takeover_config_sha256": config["config_sha256"],
                **prefix_info,
                "max_steps": args.max_steps,
                "timeout_seconds": args.timeout,
                "worker_index": args.worker_index,
                "worker_port_base": worker_port_base,
                "rock_sandbox_id": os.environ.get("ROCK_SANDBOX_ID") or None,
                "runtime_dir": str(runtime_dir),
                "trajectory_id": trajectory_id,
                "task_id": task_id,
                "canonical_trajectory_uri": str(canonical_path),
                "canonical_trajectory_sha256": sha256_file(canonical_path),
                "normalization_report_uri": str(report_path),
                "task_config_uri": str(task_path),
                "qcow2_uri": str(qcow2),
                "qcow2_sha256": qcow2_sha256,
                "qcow2_hash_preverified": args.qcow2_hash_preverified,
                "human_annotation": diagnosis.to_manifest_dict(),
                "human_source_judge": human_source_judge,
            },
        )
        result = run_single_example(
            agent=agent,
            env=env,
            task=task,
            max_steps=args.max_steps,
            result_dir=str(task_result_dir),
            sleep_after=args.sleep_after,
            timeout=args.timeout,
        )
    finally:
        env.close()
    exclusion = task_result_dir / "protocol_exclusion.json"
    if exclusion.is_file():
        record = _read_object(exclusion, "protocol exclusion")
        return _record_protocol_exclusion(
            task_result_dir, str(record.get("reason")), str(record.get("detail"))
        )
    # result.txt is a binary runner-completion marker. Propagate its failure
    # state to the shard controller so JOB_MAX_ATTEMPTS actually retries it.
    return _enforce_takeover_artifacts(task_result_dir, result)


if __name__ == "__main__":
    raise SystemExit(main())
