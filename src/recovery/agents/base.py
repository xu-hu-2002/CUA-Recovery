from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class AgentRunSpec:
    agent_id: str
    model: str
    revision: str
    endpoint: str
    temperature: float
    max_steps: int
    prompt_sha256: str
    serving_backend: Optional[str] = None

    def __post_init__(self) -> None:
        if not self.revision:
            raise ValueError("模型 revision/API snapshot 不能为空")
        if self.max_steps <= 0:
            raise ValueError("max_steps 必须为正整数")
        if self.temperature < 0:
            raise ValueError("temperature 不能为负数")
