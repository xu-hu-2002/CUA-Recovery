"""Agent adapter interface and capability contract."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, FrozenSet, List, Protocol

from recovery.canonical.actions import Action


class ActionNotSupportedError(ValueError):
    """Target agent cannot losslessly express a canonical action."""


@dataclass(frozen=True)
class AgentCapabilities:
    agent_id: str
    action_kinds: FrozenSet[str]
    coordinate_protocol: str
    history_format: str
    silent_action_kinds: FrozenSet[str] = frozenset()

    def require(self, action: Action) -> None:
        kind = getattr(action, "kind")
        if kind not in self.action_kinds:
            raise ActionNotSupportedError(
                "%s does not support canonical action %s; silent downgrade is not allowed in the main benchmark"
                % (self.agent_id, kind)
            )


@dataclass(frozen=True)
class HistoryStep:
    """Public information needed to build takeover history."""

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
