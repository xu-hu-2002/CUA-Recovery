"""Agent adapter 的公共接口与 capability contract。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, FrozenSet, List, Protocol

from derail.canonical.actions import Action


class ActionNotSupportedError(ValueError):
    """目标 agent 无法无损表达某个 canonical action。"""


@dataclass(frozen=True)
class AgentCapabilities:
    agent_id: str
    action_kinds: FrozenSet[str]
    coordinate_protocol: str
    history_format: str
    # 该历史格式中不代表任何消息的 action kind（如控制流事件）；这些 step 可以
    # 合法地渲染为空，不能作为 native history 的截断边界。
    silent_action_kinds: FrozenSet[str] = frozenset()

    def require(self, action: Action) -> None:
        kind = getattr(action, "kind")
        if kind not in self.action_kinds:
            raise ActionNotSupportedError(
                "%s 不支持 canonical action %s；主 benchmark 禁止静默降级"
                % (self.agent_id, kind)
            )


@dataclass(frozen=True)
class HistoryStep:
    """构建 takeover history 所需的公开信息。

    observation_image_url 可以是 data URL，也可以由调用者在最终发送前解析的
    content-addressed URI。这里不保存或生成 hidden chain-of-thought。
    """

    step_id: int
    observation_image_url: str
    action: Action
    observation_after_image_url: str = ""
    tool_result: str = "ok"
    turn_index: int = 0
    action_index_within_turn: int = 0
    observation_sha256: str = ""
    trajectory_log: str = ""

    @property
    def action_index_global(self) -> int:
        return self.step_id


class AgentAdapter(Protocol):
    capabilities: AgentCapabilities

    def render_step(self, step: HistoryStep) -> List[Dict[str, Any]]: ...

    def tool_definitions(self) -> List[Dict[str, Any]]: ...
