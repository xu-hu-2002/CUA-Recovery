from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class CandidateFailure:
    rollout_id: str
    rubric_judge_model: str
    rubric_prompt_sha256: str
    judge_score: float
    proposed_failure_summary: str
    human_verified: Optional[bool] = None

    def __post_init__(self) -> None:
        if not 0.0 <= self.judge_score <= 1.0:
            raise ValueError("judge_score 必须归一化到 [0, 1]")
