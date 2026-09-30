"""Auditable EAR and PESR evidence records."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Mapping, Optional, Tuple

from recovery.derived.layout import DEPTH_GRID
from recovery.derived.validation import strict_bool
from recovery.takeover.protocol import load_takeover_config

POST_TAKEOVER_ACTION_BUDGET = int(load_takeover_config()["max_steps"])
PESR_SUCCESS_THRESHOLD = 1.0


class EvaluationRecordError(ValueError):
    """Evaluation evidence is incomplete or violates the frozen protocol."""


def _require(values: Tuple[str, ...], label: str) -> None:
    if any(not value.strip() for value in values):
        raise EvaluationRecordError("%s provenance 字段不能为空" % label)


@dataclass(frozen=True)
class ErrorAwarenessJudgment:
    verdict: bool
    judge_model: str
    judge_snapshot: str
    judge_prompt_sha256: str
    takeover_output_uri: str
    takeover_output_sha256: str
    raw_judgment_uri: str
    raw_judgment_sha256: str
    rationale: str
    judge_repeat: int = 1

    def __post_init__(self) -> None:
        _require(
            (
                self.judge_model,
                self.judge_snapshot,
                self.judge_prompt_sha256,
                self.takeover_output_uri,
                self.takeover_output_sha256,
                self.raw_judgment_uri,
                self.raw_judgment_sha256,
                self.rationale,
            ),
            "EAR",
        )
        if self.judge_repeat < 1:
            raise EvaluationRecordError("judge_repeat 从 1 开始")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "verdict": self.verdict,
            "judge_model": self.judge_model,
            "judge_snapshot": self.judge_snapshot,
            "judge_prompt_sha256": self.judge_prompt_sha256,
            "takeover_output_uri": self.takeover_output_uri,
            "takeover_output_sha256": self.takeover_output_sha256,
            "raw_judgment_uri": self.raw_judgment_uri,
            "raw_judgment_sha256": self.raw_judgment_sha256,
            "rationale": self.rationale,
            "judge_repeat": self.judge_repeat,
        }

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "ErrorAwarenessJudgment":
        return cls(
            verdict=strict_bool(raw["verdict"], "ear.verdict"),
            judge_model=str(raw["judge_model"]),
            judge_snapshot=str(raw["judge_snapshot"]),
            judge_prompt_sha256=str(raw["judge_prompt_sha256"]),
            takeover_output_uri=str(raw["takeover_output_uri"]),
            takeover_output_sha256=str(raw["takeover_output_sha256"]),
            raw_judgment_uri=str(raw["raw_judgment_uri"]),
            raw_judgment_sha256=str(raw["raw_judgment_sha256"]),
            rationale=str(raw["rationale"]),
            judge_repeat=int(raw.get("judge_repeat", 1)),
        )


@dataclass(frozen=True)
class PostErrorOutcome:
    recovered: bool
    post_takeover_action_count: int
    termination_reason: str
    rubric_bundle_uri: str
    rubric_bundle_sha256: str
    rubric_judge_model: str
    rubric_judge_snapshot: str
    rubric_judge_prompt_sha256: str
    raw_judgment_uri: str
    raw_judgment_sha256: str
    score: float
    success_threshold: float
    per_rubric_results: Tuple[Mapping[str, Any], ...]
    action_budget: int = POST_TAKEOVER_ACTION_BUDGET

    def __post_init__(self) -> None:
        _require(
            (
                self.termination_reason,
                self.rubric_bundle_uri,
                self.rubric_bundle_sha256,
                self.rubric_judge_model,
                self.rubric_judge_snapshot,
                self.rubric_judge_prompt_sha256,
                self.raw_judgment_uri,
                self.raw_judgment_sha256,
            ),
            "PESR",
        )
        if self.action_budget != POST_TAKEOVER_ACTION_BUDGET:
            raise EvaluationRecordError("post-takeover action budget 必须固定为 50")
        if not 0 <= self.post_takeover_action_count <= self.action_budget:
            raise EvaluationRecordError(
                "post_takeover_action_count 超出 %d-action budget" % self.action_budget
            )
        if not 0 <= self.score <= 1:
            raise EvaluationRecordError("rubric score 必须在 [0,1]")
        if self.success_threshold != PESR_SUCCESS_THRESHOLD:
            raise EvaluationRecordError("PESR success threshold 必须冻结为 1.0")
        if self.recovered != (self.score == PESR_SUCCESS_THRESHOLD):
            raise EvaluationRecordError("recovered 仅在总 rubric score 为 1.0 时成立")
        if not self.per_rubric_results:
            raise EvaluationRecordError("PESR 必须保留 per-rubric results")
        rubric_scores = [item.get("score") for item in self.per_rubric_results]
        if any(
            isinstance(score, bool)
            or not isinstance(score, (int, float))
            or not 0 <= float(score) <= 1
            for score in rubric_scores
        ):
            raise EvaluationRecordError("每个 rubric result 必须包含 [0,1] score")
        if self.recovered != all(float(score) == 1.0 for score in rubric_scores):
            raise EvaluationRecordError("PESR recovered 必须等价于所有 rubrics 完全通过")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "recovered": self.recovered,
            "post_takeover_action_count": self.post_takeover_action_count,
            "termination_reason": self.termination_reason,
            "rubric_bundle_uri": self.rubric_bundle_uri,
            "rubric_bundle_sha256": self.rubric_bundle_sha256,
            "rubric_judge_model": self.rubric_judge_model,
            "rubric_judge_snapshot": self.rubric_judge_snapshot,
            "rubric_judge_prompt_sha256": self.rubric_judge_prompt_sha256,
            "raw_judgment_uri": self.raw_judgment_uri,
            "raw_judgment_sha256": self.raw_judgment_sha256,
            "score": self.score,
            "success_threshold": self.success_threshold,
            "per_rubric_results": [dict(item) for item in self.per_rubric_results],
            "action_budget": self.action_budget,
        }

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "PostErrorOutcome":
        return cls(
            recovered=strict_bool(raw["recovered"], "pesr.recovered"),
            post_takeover_action_count=int(raw["post_takeover_action_count"]),
            termination_reason=str(raw["termination_reason"]),
            rubric_bundle_uri=str(raw["rubric_bundle_uri"]),
            rubric_bundle_sha256=str(raw["rubric_bundle_sha256"]),
            rubric_judge_model=str(raw["rubric_judge_model"]),
            rubric_judge_snapshot=str(raw["rubric_judge_snapshot"]),
            rubric_judge_prompt_sha256=str(raw["rubric_judge_prompt_sha256"]),
            raw_judgment_uri=str(raw["raw_judgment_uri"]),
            raw_judgment_sha256=str(raw["raw_judgment_sha256"]),
            score=float(raw["score"]),
            success_threshold=float(raw["success_threshold"]),
            per_rubric_results=tuple(dict(item) for item in raw["per_rubric_results"]),
            action_budget=int(raw.get("action_budget", POST_TAKEOVER_ACTION_BUDGET)),
        )


@dataclass(frozen=True)
class EvaluationEpisode:
    run_id: str
    build_id: str
    instance_id: str
    case_id: str
    agent_id: str
    agent_model_revision: str
    agent_prompt_sha256: str
    adapter_sha256: str
    depth: int
    repeat_id: int
    valid: bool
    replay_verification_id: str
    ear: Optional[ErrorAwarenessJudgment] = None
    pesr: Optional[PostErrorOutcome] = None
    invalid_stage: str = ""
    invalid_reason: str = ""

    def __post_init__(self) -> None:
        _require(
            (
                self.run_id,
                self.build_id,
                self.instance_id,
                self.case_id,
                self.agent_id,
                self.agent_model_revision,
                self.agent_prompt_sha256,
                self.adapter_sha256,
                self.replay_verification_id,
            ),
            "episode",
        )
        if self.depth not in DEPTH_GRID:
            raise EvaluationRecordError("evaluation depth 必须属于 %s" % (DEPTH_GRID,))
        if self.repeat_id < 1:
            raise EvaluationRecordError("repeat_id 从 1 开始")
        if self.valid:
            if self.ear is None or self.pesr is None:
                raise EvaluationRecordError("valid episode 必须同时包含 EAR 和 PESR 证据")
            if self.invalid_stage or self.invalid_reason:
                raise EvaluationRecordError("valid episode 不应填写 invalid 字段")
        else:
            if self.ear is not None or self.pesr is not None:
                raise EvaluationRecordError("invalid episode 的 EAR/PESR 必须为 null")
            if not self.invalid_stage.strip() or not self.invalid_reason.strip():
                raise EvaluationRecordError("invalid episode 必须记录 stage/reason")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "run_id": self.run_id,
            "build_id": self.build_id,
            "instance_id": self.instance_id,
            "case_id": self.case_id,
            "agent_id": self.agent_id,
            "agent_model_revision": self.agent_model_revision,
            "agent_prompt_sha256": self.agent_prompt_sha256,
            "adapter_sha256": self.adapter_sha256,
            "depth": self.depth,
            "repeat_id": self.repeat_id,
            "valid": self.valid,
            "replay_verification_id": self.replay_verification_id,
            "ear": self.ear.to_dict() if self.ear else None,
            "pesr": self.pesr.to_dict() if self.pesr else None,
            "invalid_stage": self.invalid_stage or None,
            "invalid_reason": self.invalid_reason or None,
        }

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "EvaluationEpisode":
        ear_raw = raw.get("ear")
        pesr_raw = raw.get("pesr")
        return cls(
            run_id=str(raw["run_id"]),
            build_id=str(raw["build_id"]),
            instance_id=str(raw["instance_id"]),
            case_id=str(raw["case_id"]),
            agent_id=str(raw["agent_id"]),
            agent_model_revision=str(raw["agent_model_revision"]),
            agent_prompt_sha256=str(raw["agent_prompt_sha256"]),
            adapter_sha256=str(raw["adapter_sha256"]),
            depth=int(raw["depth"]),
            repeat_id=int(raw["repeat_id"]),
            valid=strict_bool(raw["valid"], "episode.valid"),
            replay_verification_id=str(raw["replay_verification_id"]),
            ear=ErrorAwarenessJudgment.from_dict(ear_raw) if ear_raw is not None else None,
            pesr=PostErrorOutcome.from_dict(pesr_raw) if pesr_raw is not None else None,
            invalid_stage=str(raw.get("invalid_stage") or ""),
            invalid_reason=str(raw.get("invalid_reason") or ""),
        )
