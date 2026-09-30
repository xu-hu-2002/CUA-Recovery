"""Snapshot replay plans and state-fingerprint verification gates."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Mapping, Protocol, Sequence, Tuple

from derail.canonical.actions import Action, action_to_dict
from derail.canonical.trajectory import CanonicalStep
from derail.construction.cases import DepthInstance
from derail.derived.layout import DEPTH_GRID, sha256_file
from derail.derived.validation import require_sha256, strict_bool

EXECUTION_MODES = frozenset({"dry_run", "synthetic", "vm"})
REQUIRED_RELEASE_CHECKS = frozenset(
    {
        "snapshot_restored",
        "action_count_match",
        "action_indices_match",
        "action_hashes_present",
        "expected_state_present",
        "state_fingerprint_match",
        "non_screenshot_state_present",
        "replay_observations_present",
        "task_provenance_verified",
    }
)


class ReplayVerificationError(ValueError):
    """Replay was not deterministic, restored, or state-equivalent."""


def _digest(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


@dataclass(frozen=True)
class StateFingerprint:
    components: Mapping[str, str]
    screenshot_sha256: str = ""

    def __post_init__(self) -> None:
        if not self.components:
            raise ReplayVerificationError("state fingerprint 不能只依赖截图或为空")
        if any(
            not str(key).strip() or not str(value).strip()
            for key, value in self.components.items()
        ):
            raise ReplayVerificationError("state fingerprint components 不能为空")
        try:
            for name, value in self.components.items():
                require_sha256(value, "state_fingerprint.components.%s" % name)
            require_sha256(
                self.screenshot_sha256,
                "state_fingerprint.screenshot_sha256",
                allow_empty=True,
            )
        except ValueError as exc:
            raise ReplayVerificationError(str(exc)) from exc

    @property
    def sha256(self) -> str:
        return _digest({"components": dict(self.components)})

    def to_dict(self) -> Dict[str, Any]:
        return {
            "components": dict(self.components),
            "screenshot_sha256": self.screenshot_sha256,
            "sha256": self.sha256,
        }

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "StateFingerprint":
        components = raw.get("components")
        if not isinstance(components, Mapping):
            raise ReplayVerificationError("state_fingerprint.components 必须是 object")
        result = cls(
            components={str(key): str(value) for key, value in components.items()},
            screenshot_sha256=str(raw.get("screenshot_sha256", "")),
        )
        supplied = raw.get("sha256")
        if supplied is not None and str(supplied) != result.sha256:
            raise ReplayVerificationError("state_fingerprint.sha256 与 components 不一致")
        return result


def compare_state_fingerprints(
    expected: StateFingerprint,
    observed: StateFingerprint,
    ignore_components: Sequence[str] = (),
) -> Dict[str, Any]:
    """Component-wise ``state_fingerprint_match``; ignored components are reported only."""

    ignored = frozenset(ignore_components)
    names = sorted(set(expected.components) | set(observed.components))
    differing = [
        name
        for name in names
        if expected.components.get(name) != observed.components.get(name)
    ]
    return {
        "state_fingerprint_match": not [name for name in differing if name not in ignored],
        "mismatched_components": [name for name in differing if name not in ignored],
        "tolerated_components": [name for name in differing if name in ignored],
        "expected": expected.to_dict(),
        "observed": observed.to_dict(),
    }


class SnapshotReplayBackend(Protocol):
    execution_mode: str
    replay_session_id: str

    def restore_snapshot(self, snapshot_uri: str, snapshot_sha256: str) -> None: ...

    def execute(self, action: Action) -> Mapping[str, Any]: ...

    def fingerprint(self) -> StateFingerprint: ...


@dataclass(frozen=True)
class ReplayPlan:
    attempt_id: str
    build_id: str
    case_id: str
    instance: DepthInstance
    snapshot_uri: str
    snapshot_sha256: str
    canonical_trajectory_sha256: str
    canonical_trajectory_uri: str = ""
    expected_state_sha256: str = ""
    task_id: str = ""
    instruction_sha256: str = ""
    task_config_uri: str = ""
    task_config_sha256: str = ""
    state_probe_config_uri: str = ""
    state_probe_config_sha256: str = ""

    def __post_init__(self) -> None:
        required = (
            self.attempt_id,
            self.build_id,
            self.case_id,
            self.snapshot_uri,
            self.snapshot_sha256,
            self.canonical_trajectory_sha256,
            self.canonical_trajectory_uri,
            self.task_id,
            self.instruction_sha256,
            self.task_config_uri,
            self.task_config_sha256,
            self.state_probe_config_uri,
            self.state_probe_config_sha256,
        )
        if any(not value.strip() for value in required):
            raise ReplayVerificationError("replay plan provenance 字段不能为空")
        if self.instance.case_id != self.case_id:
            raise ReplayVerificationError("replay instance/case 不一致")
        try:
            require_sha256(self.snapshot_sha256, "snapshot_sha256")
            require_sha256(self.canonical_trajectory_sha256, "canonical_trajectory_sha256")
            require_sha256(self.expected_state_sha256, "expected_state_sha256", allow_empty=True)
        except ValueError as exc:
            raise ReplayVerificationError(str(exc)) from exc

    def to_dict(self) -> Dict[str, Any]:
        return {
            "attempt_id": self.attempt_id,
            "build_id": self.build_id,
            "case_id": self.case_id,
            "instance": self.instance.to_dict(),
            "snapshot_uri": self.snapshot_uri,
            "snapshot_sha256": self.snapshot_sha256,
            "canonical_trajectory_uri": self.canonical_trajectory_uri,
            "canonical_trajectory_sha256": self.canonical_trajectory_sha256,
            "expected_state_sha256": self.expected_state_sha256,
            "task_id": self.task_id,
            "instruction_sha256": self.instruction_sha256,
            "task_config_uri": self.task_config_uri,
            "task_config_sha256": self.task_config_sha256,
            "state_probe_config_uri": self.state_probe_config_uri,
            "state_probe_config_sha256": self.state_probe_config_sha256,
        }

    def verify_task_provenance(self) -> bool:
        refs = (
            (self.task_config_uri, self.task_config_sha256),
            (self.state_probe_config_uri, self.state_probe_config_sha256),
        )
        try:
            require_sha256(self.instruction_sha256, "instruction_sha256")
            if not self.task_id.strip():
                return False
            refs_valid = all(
                uri
                and Path(uri).is_file()
                and sha256_file(Path(uri)) == expected
                for uri, expected in refs
            )
            if not refs_valid:
                return False
            task_config = json.loads(Path(self.task_config_uri).read_text(encoding="utf-8"))
            probes = json.loads(Path(self.state_probe_config_uri).read_text(encoding="utf-8"))
            instruction = task_config.get("instruction") if isinstance(task_config, dict) else None
            return bool(
                isinstance(task_config, dict)
                and task_config.get("id") == self.task_id
                and isinstance(instruction, str)
                and hashlib.sha256(instruction.encode("utf-8")).hexdigest()
                == self.instruction_sha256
                and isinstance(probes, dict)
                and probes
                and all(
                    isinstance(name, str)
                    and name.strip()
                    and isinstance(command, str)
                    and command.strip()
                    for name, command in probes.items()
                )
            )
        except (ValueError, OSError, json.JSONDecodeError):
            return False


@dataclass(frozen=True)
class ReplayVerification:
    attempt_id: str
    build_id: str
    case_id: str
    instance_id: str
    task_id: str
    root_cause_action_index: int
    depth: int
    replay_end_action_index: int
    canonical_actions_sha256: str
    canonical_trajectory_sha256: str
    snapshot_sha256: str
    instruction_sha256: str
    task_config_sha256: str
    state_probe_config_sha256: str
    vm_session_id: str
    execution_mode: str
    snapshot_restored: bool
    executed_action_indices: Tuple[int, ...]
    per_action_log: Tuple[Mapping[str, Any], ...]
    state_fingerprint: StateFingerprint
    expected_state_sha256: str
    automated_checks: Mapping[str, bool]
    accepted_for_release: bool
    reviewer_ids: Tuple[str, ...]
    rejection_reasons: Tuple[str, ...]

    def __post_init__(self) -> None:
        if self.execution_mode not in EXECUTION_MODES:
            raise ReplayVerificationError("未知 replay execution_mode")
        if any(
            not value.strip()
            for value in (self.attempt_id, self.build_id, self.case_id, self.instance_id)
        ):
            raise ReplayVerificationError("replay verification identity 不能为空")
        if self.depth not in DEPTH_GRID:
            raise ReplayVerificationError("replay depth 不在冻结 grid")
        if self.replay_end_action_index != self.root_cause_action_index + self.depth:
            raise ReplayVerificationError("replay_end 必须等于 root + depth")
        if not self.executed_action_indices:
            raise ReplayVerificationError("cleaned replay actions 不能为空")
        if self.executed_action_indices != tuple(sorted(set(self.executed_action_indices))):
            raise ReplayVerificationError("cleaned replay action indices 必须严格递增且唯一")
        if any(
            index < 0 or index > self.replay_end_action_index
            for index in self.executed_action_indices
        ):
            raise ReplayVerificationError("cleaned replay action 超出 source depth 边界")
        if self.root_cause_action_index not in self.executed_action_indices:
            raise ReplayVerificationError("cleaned replay 不能缺少 root-cause action")
        if len(self.per_action_log) != len(self.executed_action_indices):
            raise ReplayVerificationError("per_action_log 长度与 executed actions 不一致")
        log_indices = tuple(
            int(item.get("action_index_global", -1)) for item in self.per_action_log
        )
        if log_indices != self.executed_action_indices:
            raise ReplayVerificationError("per_action_log action indices 不一致")
        try:
            for field, value in (
                ("canonical_actions_sha256", self.canonical_actions_sha256),
                ("canonical_trajectory_sha256", self.canonical_trajectory_sha256),
                ("snapshot_sha256", self.snapshot_sha256),
            ):
                require_sha256(value, field)
            for item in self.per_action_log:
                require_sha256(item.get("action_sha256"), "per_action_log.action_sha256")
            for name, value in self.automated_checks.items():
                strict_bool(value, "automated_checks.%s" % name)
            strict_bool(self.snapshot_restored, "snapshot_restored")
            strict_bool(self.accepted_for_release, "accepted_for_release")
        except ValueError as exc:
            raise ReplayVerificationError(str(exc)) from exc
        if self.accepted_for_release:
            if self.execution_mode != "vm":
                raise ReplayVerificationError("只有 real VM replay 可 accepted_for_release")
            if not REQUIRED_RELEASE_CHECKS.issubset(self.automated_checks):
                raise ReplayVerificationError("accepted replay 缺少固定 automated checks")
            if not self.snapshot_restored or not all(self.automated_checks.values()):
                raise ReplayVerificationError("accepted replay 的 automated checks 必须全部通过")
            if not self.reviewer_ids or self.rejection_reasons:
                raise ReplayVerificationError("accepted replay 必须有人审且不能有 rejection reason")
            if not self.expected_state_sha256 or (
                self.expected_state_sha256 != self.state_fingerprint.sha256
            ):
                raise ReplayVerificationError("accepted replay 的 expected/fingerprint hash 不一致")
            if not self.vm_session_id.strip():
                raise ReplayVerificationError("accepted replay 必须绑定 VM session")
            for field, value in (
                ("instruction_sha256", self.instruction_sha256),
                ("task_config_sha256", self.task_config_sha256),
                ("state_probe_config_sha256", self.state_probe_config_sha256),
            ):
                try:
                    require_sha256(value, field)
                except ValueError as exc:
                    raise ReplayVerificationError(str(exc)) from exc

    def to_dict(self) -> Dict[str, Any]:
        return {
            "attempt_id": self.attempt_id,
            "build_id": self.build_id,
            "case_id": self.case_id,
            "instance_id": self.instance_id,
            "task_id": self.task_id,
            "root_cause_action_index": self.root_cause_action_index,
            "depth": self.depth,
            "replay_end_action_index": self.replay_end_action_index,
            "canonical_actions_sha256": self.canonical_actions_sha256,
            "canonical_trajectory_sha256": self.canonical_trajectory_sha256,
            "snapshot_sha256": self.snapshot_sha256,
            "instruction_sha256": self.instruction_sha256,
            "task_config_sha256": self.task_config_sha256,
            "state_probe_config_sha256": self.state_probe_config_sha256,
            "vm_session_id": self.vm_session_id,
            "execution_mode": self.execution_mode,
            "snapshot_restored": self.snapshot_restored,
            "executed_action_indices": list(self.executed_action_indices),
            "per_action_log": [dict(item) for item in self.per_action_log],
            "state_fingerprint": self.state_fingerprint.to_dict(),
            "expected_state_sha256": self.expected_state_sha256,
            "automated_checks": dict(self.automated_checks),
            "accepted_for_release": self.accepted_for_release,
            "reviewer_ids": list(self.reviewer_ids),
            "rejection_reasons": list(self.rejection_reasons),
        }

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "ReplayVerification":
        checks = raw.get("automated_checks")
        if not isinstance(checks, Mapping):
            raise ReplayVerificationError("automated_checks 必须是 object")
        return cls(
            attempt_id=str(raw["attempt_id"]),
            build_id=str(raw["build_id"]),
            case_id=str(raw["case_id"]),
            instance_id=str(raw["instance_id"]),
            task_id=str(raw.get("task_id", "")),
            root_cause_action_index=int(raw["root_cause_action_index"]),
            depth=int(raw["depth"]),
            replay_end_action_index=int(raw["replay_end_action_index"]),
            canonical_actions_sha256=str(raw["canonical_actions_sha256"]),
            canonical_trajectory_sha256=str(raw["canonical_trajectory_sha256"]),
            snapshot_sha256=str(raw["snapshot_sha256"]),
            instruction_sha256=str(raw.get("instruction_sha256", "")),
            task_config_sha256=str(raw.get("task_config_sha256", "")),
            state_probe_config_sha256=str(raw.get("state_probe_config_sha256", "")),
            vm_session_id=str(raw.get("vm_session_id", "")),
            execution_mode=str(raw["execution_mode"]),
            snapshot_restored=strict_bool(raw["snapshot_restored"], "snapshot_restored"),
            executed_action_indices=tuple(int(item) for item in raw["executed_action_indices"]),
            per_action_log=tuple(dict(item) for item in raw["per_action_log"]),
            state_fingerprint=StateFingerprint.from_dict(raw["state_fingerprint"]),
            expected_state_sha256=str(raw.get("expected_state_sha256", "")),
            automated_checks={
                str(name): strict_bool(value, "automated_checks.%s" % name)
                for name, value in checks.items()
            },
            accepted_for_release=strict_bool(
                raw["accepted_for_release"], "accepted_for_release"
            ),
            reviewer_ids=tuple(str(item) for item in raw["reviewer_ids"]),
            rejection_reasons=tuple(str(item) for item in raw["rejection_reasons"]),
        )


def execute_replay_plan(
    plan: ReplayPlan,
    steps: Sequence[CanonicalStep],
    backend: SnapshotReplayBackend,
    *,
    reviewer_ids: Sequence[str] = (),
) -> ReplayVerification:
    if backend.execution_mode not in EXECUTION_MODES:
        raise ReplayVerificationError("未知 execution_mode: %s" % backend.execution_mode)
    trajectory_path = Path(plan.canonical_trajectory_uri).resolve()
    if not trajectory_path.is_file():
        raise ReplayVerificationError("replay 缺少 canonical trajectory artifact")
    if sha256_file(trajectory_path) != plan.canonical_trajectory_sha256:
        raise ReplayVerificationError("canonical trajectory artifact SHA-256 不匹配")
    task_provenance_verified = plan.verify_task_provenance()
    assert_plan_provenance = getattr(backend, "assert_plan_provenance", None)
    if backend.execution_mode == "vm":
        if not callable(assert_plan_provenance):
            raise ReplayVerificationError("real VM backend 缺少 plan provenance binding")
        assert_plan_provenance(plan)
    required_indices = plan.instance.executed_action_indices
    steps_by_id = {step.action_index_global: step for step in steps}
    missing = tuple(index for index in required_indices if index not in steps_by_id)
    if missing:
        raise ReplayVerificationError("replay plan 引用了 cleaned trajectory 中不存在的 action: %s" % (missing,))
    selected = [steps_by_id[index] for index in required_indices]
    if _digest([action_to_dict(step.action) for step in selected]) != (
        plan.instance.canonical_actions_sha256
    ):
        raise ReplayVerificationError("replay actions 与 depth instance hash 不一致")

    backend.restore_snapshot(plan.snapshot_uri, plan.snapshot_sha256)
    logs = []
    for step in selected:
        result = dict(backend.execute(step.action))
        log = {
                "action_index_global": step.action_index_global,
                "action_sha256": _digest(action_to_dict(step.action)),
                **result,
            }
        logs.append(log)
    fingerprint = backend.fingerprint()
    expected_present = bool(plan.expected_state_sha256)
    state_match = expected_present and fingerprint.sha256 == plan.expected_state_sha256
    checks = {
        "snapshot_restored": True,
        "action_count_match": len(logs) == len(required_indices),
        "action_indices_match": tuple(item["action_index_global"] for item in logs)
        == required_indices,
        "expected_state_present": expected_present,
        "state_fingerprint_match": state_match,
        "non_screenshot_state_present": bool(fingerprint.components),
        "action_hashes_present": all(item.get("action_sha256") for item in logs),
        "replay_observations_present": all(
            item.get("observation_before_uri")
            and item.get("observation_before_sha256")
            and item.get("observation_after_uri")
            and item.get("observation_after_sha256")
            for item in logs
        ),
        "task_provenance_verified": task_provenance_verified,
    }
    reasons = []
    if backend.execution_mode != "vm":
        reasons.append("dry_run_or_synthetic_replay_is_not_release_evidence")
    if not all(checks.values()):
        reasons.extend(key for key, passed in checks.items() if not passed)
    if not reviewer_ids:
        reasons.append("human_reviewer_missing")
    accepted = backend.execution_mode == "vm" and all(checks.values()) and bool(reviewer_ids)
    verification = ReplayVerification(
        attempt_id=plan.attempt_id,
        build_id=plan.build_id,
        case_id=plan.case_id,
        instance_id=plan.instance.instance_id,
        task_id=plan.task_id,
        root_cause_action_index=plan.instance.root_cause_action_index,
        depth=plan.instance.depth,
        replay_end_action_index=plan.instance.replay_end_action_index,
        canonical_actions_sha256=plan.instance.canonical_actions_sha256,
        canonical_trajectory_sha256=plan.canonical_trajectory_sha256,
        snapshot_sha256=plan.snapshot_sha256,
        instruction_sha256=plan.instruction_sha256,
        task_config_sha256=plan.task_config_sha256,
        state_probe_config_sha256=plan.state_probe_config_sha256,
        vm_session_id=str(getattr(backend, "replay_session_id", "")),
        execution_mode=backend.execution_mode,
        snapshot_restored=True,
        executed_action_indices=required_indices,
        per_action_log=tuple(logs),
        state_fingerprint=fingerprint,
        expected_state_sha256=plan.expected_state_sha256,
        automated_checks=checks,
        accepted_for_release=accepted,
        reviewer_ids=tuple(reviewer_ids),
        rejection_reasons=tuple(reasons),
    )
    bind_verification = getattr(backend, "bind_verification", None)
    if callable(bind_verification):
        bind_verification(verification.attempt_id, verification.state_fingerprint.sha256)
    return verification


class DeterministicSyntheticBackend:
    """State-machine backend for plumbing tests; never release evidence."""

    execution_mode = "synthetic"

    def __init__(self) -> None:
        self.restored = False
        self.actions: list[Mapping[str, Any]] = []
        self.replay_session_id = "synthetic-session"

    def restore_snapshot(self, snapshot_uri: str, snapshot_sha256: str) -> None:
        if not snapshot_uri or not snapshot_sha256:
            raise ReplayVerificationError("synthetic snapshot provenance 不能为空")
        self.restored = True
        self.actions = []

    def execute(self, action: Action) -> Mapping[str, Any]:
        if not self.restored:
            raise ReplayVerificationError("必须先 restore snapshot")
        self.actions.append(action_to_dict(action))
        return {"result": "ok"}

    def fingerprint(self) -> StateFingerprint:
        if not self.restored:
            raise ReplayVerificationError("必须先 restore snapshot")
        return StateFingerprint(components={"synthetic_action_state": _digest(self.actions)})
