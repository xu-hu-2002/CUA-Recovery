"""人工裁决后的失败标签。"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class Reversibility(str, Enum):
    REVERSIBLE = "reversible"
    IRREVERSIBLE = "irreversible"


@dataclass(frozen=True)
class FailureAnnotation:
    rollout_id: str
    root_cause_step: int
    error_horizon: int
    error_type: str
    reversibility: Reversibility
    annotator_id: str
    adjudicated: bool = False

    def __post_init__(self) -> None:
        if self.root_cause_step < 0:
            raise ValueError("root_cause_step 不能为负数")
        if self.error_horizon < 0:
            raise ValueError("error_horizon 不能为负数")
        if not self.error_type or not self.annotator_id:
            raise ValueError("error_type 和 annotator_id 不能为空")
