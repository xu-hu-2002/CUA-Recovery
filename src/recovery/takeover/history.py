"""Build auditable native history from replay observations and canonical actions."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, replace
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

from recovery.adapters.base import AgentAdapter, HistoryStep
from recovery.canonical.trajectory import CanonicalStep
from recovery.derived.validation import require_sha256, strict_bool
from recovery.replay.verification import ReplayVerification
from recovery.takeover.source_logs import load_trajectory_log


def build_native_history(
    adapter: AgentAdapter, steps: Iterable[HistoryStep]
) -> List[Dict[str, Any]]:
    """Check capability for each step, then render native history."""

    messages: List[Dict[str, Any]] = []
    previous_step_id = -1
    frozen_steps = tuple(steps)
    last_rendered_step_id = -1
    boundary_step_id = -1
    for step in frozen_steps:
        if step.step_id <= previous_step_id:
            raise ValueError("history steps 必须按严格递增的 step_id 排列")
        adapter.capabilities.require(step.action)
        rendered = adapter.render_step(step)
        messages.extend(rendered)
        if rendered:
            last_rendered_step_id = step.step_id
        if step.action.kind not in adapter.capabilities.silent_action_kinds:
            boundary_step_id = step.step_id
        previous_step_id = step.step_id
    if boundary_step_id >= 0 and last_rendered_step_id < boundary_step_id:
        raise ValueError(
            "native history cannot be losslessly truncated at the requested action boundary"
        )
    if messages and messages[0].get("role") != "system":
        system_prompt = getattr(adapter, "system_prompt", None)
        if callable(system_prompt):
            messages.insert(0, {"role": "system", "content": system_prompt()})
        elif getattr(adapter, "history_starts_with_system_message", True):
            raise ValueError("native history adapter omitted the initial system message")
    return messages


REASONING_POLICIES = {True: "source_action_only_reasoning_stripped", False: "preserved_as_recorded"}


@dataclass(frozen=True)
class NativeHistoryArtifact:
    agent_id: str
    renderer_version: str
    action_indices: Tuple[int, ...]
    messages: Tuple[Mapping[str, Any], ...]
    messages_sha256: str
    conformance_probe_sha256: str
    conformance_passed: bool
    replay_verification_id: str = ""
    instance_id: str = ""
    replay_observation_sha256s: Tuple[str, ...] = ()
    conformance_probe_uri: str = ""
    strip_reasoning: bool = True

    def __post_init__(self) -> None:
        payload = json.dumps(
            self.messages, ensure_ascii=False, sort_keys=True, separators=(",", ":")
        )
        actual = hashlib.sha256(payload.encode("utf-8")).hexdigest()
        if self.messages_sha256 != actual:
            raise ValueError("native history messages_sha256 不匹配")
        if self.conformance_passed:
            require_sha256(self.conformance_probe_sha256, "conformance_probe_sha256")
        for value in self.replay_observation_sha256s:
            require_sha256(value, "replay_observation_sha256s")

    @property
    def release_eligible(self) -> bool:
        return (
            self.conformance_passed
            and bool(self.conformance_probe_sha256)
            and bool(self.replay_verification_id)
            and bool(self.instance_id)
            and len(self.replay_observation_sha256s) == len(self.action_indices)
            and bool(self.conformance_probe_uri)
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "agent_id": self.agent_id,
            "renderer_version": self.renderer_version,
            "action_indices": list(self.action_indices),
            "messages": [dict(message) for message in self.messages],
            "messages_sha256": self.messages_sha256,
            "conformance_probe_sha256": self.conformance_probe_sha256,
            "conformance_passed": self.conformance_passed,
            "replay_verification_id": self.replay_verification_id,
            "instance_id": self.instance_id,
            "replay_observation_sha256s": list(self.replay_observation_sha256s),
            "conformance_probe_uri": self.conformance_probe_uri,
            "release_eligible": self.release_eligible,
            "reasoning_policy": REASONING_POLICIES[self.strip_reasoning],
        }

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "NativeHistoryArtifact":
        return cls(
            agent_id=str(raw["agent_id"]),
            renderer_version=str(raw["renderer_version"]),
            action_indices=tuple(int(item) for item in raw["action_indices"]),
            messages=tuple(dict(item) for item in raw["messages"]),
            messages_sha256=str(raw["messages_sha256"]),
            conformance_probe_sha256=str(raw.get("conformance_probe_sha256", "")),
            conformance_passed=strict_bool(
                raw["conformance_passed"], "conformance_passed"
            ),
            replay_verification_id=str(raw.get("replay_verification_id", "")),
            instance_id=str(raw.get("instance_id", "")),
            replay_observation_sha256s=tuple(
                str(item) for item in raw.get("replay_observation_sha256s", ())
            ),
            conformance_probe_uri=str(raw.get("conformance_probe_uri", "")),
            strip_reasoning=raw.get("reasoning_policy") != REASONING_POLICIES[False],
        )


def build_native_history_artifact(
    adapter: AgentAdapter,
    steps: Iterable[HistoryStep],
    *,
    renderer_version: str,
    conformance_probe_sha256: str = "",
    conformance_passed: bool = False,
    strip_reasoning: bool = True,
) -> NativeHistoryArtifact:
    """``strip_reasoning`` records how the steps' ``trajectory_log`` was produced."""

    frozen_steps = tuple(steps)
    expected_indices = tuple(step.action_index_global for step in frozen_steps)
    if expected_indices != tuple(sorted(set(expected_indices))):
        raise ValueError("native history action indices 必须严格递增且唯一")
    messages = tuple(build_native_history(adapter, frozen_steps))
    payload = json.dumps(messages, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    capabilities = getattr(adapter, "capabilities")
    return NativeHistoryArtifact(
        agent_id=capabilities.agent_id,
        renderer_version=renderer_version,
        action_indices=expected_indices,
        messages=messages,
        messages_sha256=hashlib.sha256(payload.encode("utf-8")).hexdigest(),
        conformance_probe_sha256=conformance_probe_sha256,
        conformance_passed=conformance_passed,
        strip_reasoning=strip_reasoning,
    )


def build_native_history_from_replay(
    adapter: AgentAdapter,
    steps: Sequence[CanonicalStep],
    replay_verification: ReplayVerification,
    *,
    renderer_version: str,
    conformance_probe_sha256: str,
    conformance_passed: bool,
    conformance_probe_uri: str,
    strip_reasoning: bool = True,
) -> NativeHistoryArtifact:
    """Render corrected history solely from accepted per-action VM replay evidence."""

    if not replay_verification.accepted_for_release:
        raise ValueError("native history 只能从 accepted real-VM replay 构建")
    steps_by_id = {step.action_index_global: step for step in steps}
    selected = []
    for index, log in zip(
        replay_verification.executed_action_indices,
        replay_verification.per_action_log,
    ):
        if index not in steps_by_id:
            raise ValueError("replay/canonical action index 不一致")
        step = steps_by_id[index]
        if not log.get("observation_before_uri") or not log.get(
            "observation_before_sha256"
        ):
            raise ValueError("replay log 缺少 corrected pre-action observation")
        selected.append(
            HistoryStep(
                step_id=index,
                turn_index=step.turn_index or 0,
                action_index_within_turn=step.action_index_within_turn,
                observation_image_url=str(log["observation_before_uri"]),
                observation_after_image_url=str(log.get("observation_after_uri", "")),
                observation_sha256=str(log["observation_before_sha256"]),
                action=step.action,
                tool_result=str(log.get("result", "ok")),
                trajectory_log=load_trajectory_log(step, strip_reasoning=strip_reasoning),
            )
        )
    artifact = build_native_history_artifact(
        adapter,
        selected,
        renderer_version=renderer_version,
        conformance_probe_sha256=conformance_probe_sha256,
        conformance_passed=conformance_passed,
        strip_reasoning=strip_reasoning,
    )
    return replace(
        artifact,
        replay_verification_id=replay_verification.attempt_id,
        instance_id=replay_verification.instance_id,
        replay_observation_sha256s=tuple(
            str(item["observation_before_sha256"])
            for item in replay_verification.per_action_log
        ),
        conformance_probe_uri=conformance_probe_uri,
    )
