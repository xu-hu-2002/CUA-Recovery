"""Generic takeover loop with immediate EAR and PESR within the takeover step budget."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Protocol, Sequence, Tuple

from derail.canonical.actions import Action, TerminateAction
from derail.derived.layout import sha256_file
from derail.evaluation.records import (
    POST_TAKEOVER_ACTION_BUDGET,
    ErrorAwarenessJudgment,
    EvaluationEpisode,
    PostErrorOutcome,
)
from derail.replay.verification import ReplayVerification
from derail.takeover.history import NativeHistoryArtifact


class TakeoverRunError(RuntimeError):
    """The episode could not be evaluated without violating the protocol."""


@dataclass(frozen=True)
class TakeoverTurn:
    public_output: str
    actions: Tuple[Action, ...]
    stop: bool = False


class TakeoverAgent(Protocol):
    def begin(self, instruction: str, native_history: Sequence[Mapping[str, Any]]) -> None: ...

    def predict(self, observation: Any) -> TakeoverTurn: ...


class TakeoverEnvironment(Protocol):
    def observe(self) -> Any: ...

    def execute(self, action: Action) -> None: ...

    def assert_replay_binding(self, verification: ReplayVerification) -> None: ...


class ImmediateErrorAwarenessJudge(Protocol):
    def judge(self, public_output: str, takeover_observation: Any) -> ErrorAwarenessJudgment: ...


class PostErrorGrader(Protocol):
    def grade(self, action_count: int, termination_reason: str) -> PostErrorOutcome: ...


@dataclass(frozen=True)
class TakeoverRunTrace:
    episode: EvaluationEpisode
    public_outputs: Tuple[str, ...]
    executed_actions: Tuple[Action, ...]


def run_takeover_episode(
    *,
    agent: TakeoverAgent,
    environment: TakeoverEnvironment,
    ear_judge: ImmediateErrorAwarenessJudge,
    pesr_grader: PostErrorGrader,
    instruction: str,
    native_history: NativeHistoryArtifact,
    replay_verification: ReplayVerification,
    run_id: str,
    build_id: str,
    instance_id: str,
    case_id: str,
    agent_id: str,
    agent_model_revision: str,
    agent_prompt_sha256: str,
    adapter_sha256: str,
    depth: int,
    repeat_id: int,
) -> TakeoverRunTrace:
    if not replay_verification.accepted_for_release or replay_verification.execution_mode != "vm":
        raise TakeoverRunError("takeover evaluation 需要 accepted real-VM replay")
    if (
        replay_verification.instance_id != instance_id
        or replay_verification.case_id != case_id
        or replay_verification.build_id != build_id
        or replay_verification.depth != depth
    ):
        raise TakeoverRunError("takeover identity/depth 与 replay verification 不一致")
    if not instruction.strip() or not native_history.messages:
        raise TakeoverRunError("takeover instruction/native history 不能为空")
    instruction_sha256 = hashlib.sha256(instruction.encode("utf-8")).hexdigest()
    if instruction_sha256 != replay_verification.instruction_sha256:
        raise TakeoverRunError("takeover instruction 与 replay task provenance 不一致")
    if not native_history.release_eligible:
        raise TakeoverRunError("native history 缺少 renderer conformance 或 replay binding")
    if native_history.agent_id != agent_id:
        raise TakeoverRunError("native history agent_id 与 takeover agent 不一致")
    if native_history.replay_verification_id != replay_verification.attempt_id:
        raise TakeoverRunError("native history/replay attempt 不一致")
    if native_history.instance_id != instance_id:
        raise TakeoverRunError("native history/instance 不一致")
    if native_history.action_indices != replay_verification.executed_action_indices:
        raise TakeoverRunError("native history actions 与 replay actions 不一致")
    try:
        environment.assert_replay_binding(replay_verification)
    except Exception as exc:
        raise TakeoverRunError("takeover VM session/state 未绑定 verified replay") from exc

    agent.begin(instruction, native_history.messages)
    takeover_observation = environment.observe()
    outputs = []
    executed = []
    termination_reason = "action_budget"
    first_turn = True
    ear = None

    while len(executed) < POST_TAKEOVER_ACTION_BUDGET:
        observation = takeover_observation if first_turn else environment.observe()
        turn = agent.predict(observation)
        outputs.append(turn.public_output)
        if first_turn:
            ear = ear_judge.judge(turn.public_output, takeover_observation)
            output_sha256 = hashlib.sha256(turn.public_output.encode("utf-8")).hexdigest()
            if ear.takeover_output_sha256 != output_sha256:
                raise TakeoverRunError("EAR takeover_output hash 与实际 first output 不一致")
            first_turn = False
        if not turn.actions:
            if turn.stop:
                termination_reason = "agent_text_stop"
                break
            raise TakeoverRunError("agent returned neither actions nor a stop signal")
        for action in turn.actions:
            if len(executed) >= POST_TAKEOVER_ACTION_BUDGET:
                break
            executed.append(action)
            if isinstance(action, TerminateAction):
                termination_reason = "agent_terminate_%s" % action.status
                break
            environment.execute(action)
        if isinstance(executed[-1], TerminateAction):
            break
        if turn.stop:
            termination_reason = "agent_stop_after_actions"
            break

    if ear is None:
        raise TakeoverRunError("EAR judge was not invoked")
    pesr = pesr_grader.grade(len(executed), termination_reason)
    if pesr.post_takeover_action_count != len(executed):
        raise TakeoverRunError("PESR action count 与 takeover trace 不一致")
    for uri, expected, label in (
        (ear.takeover_output_uri, ear.takeover_output_sha256, "EAR takeover output"),
        (ear.raw_judgment_uri, ear.raw_judgment_sha256, "EAR raw judgment"),
        (pesr.rubric_bundle_uri, pesr.rubric_bundle_sha256, "PESR rubric bundle"),
        (pesr.raw_judgment_uri, pesr.raw_judgment_sha256, "PESR raw judgment"),
    ):
        path = Path(uri)
        if not path.is_file() or sha256_file(path) != expected:
            raise TakeoverRunError("%s evidence URI/hash 无法验证" % label)
    episode = EvaluationEpisode(
        run_id=run_id,
        build_id=build_id,
        instance_id=instance_id,
        case_id=case_id,
        agent_id=agent_id,
        agent_model_revision=agent_model_revision,
        agent_prompt_sha256=agent_prompt_sha256,
        adapter_sha256=adapter_sha256,
        depth=depth,
        repeat_id=repeat_id,
        valid=True,
        replay_verification_id=replay_verification.attempt_id,
        ear=ear,
        pesr=pesr,
    )
    return TakeoverRunTrace(
        episode=episode,
        public_outputs=tuple(outputs),
        executed_actions=tuple(executed),
    )
