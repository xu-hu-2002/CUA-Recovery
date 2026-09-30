"""State renderer for the vendored Qwen 3.5 OSWorld agent."""

from __future__ import annotations

import json
import re
from typing import Any, Dict, List

from .base import AgentCapabilities, HistoryStep


_ACTION_LINE = re.compile(r"(?m)^Action:\s*(?P<action>\S.*)$")
_TOOL_CALL = re.compile(r"<tool_call>.*?</tool_call>", re.DOTALL)


class Qwen35StateAdapter:
    """Lower recorded self-rollout evidence into Qwen35VLAgent state records."""

    capabilities = AgentCapabilities(
        agent_id="qwen3_5_35b_a3b",
        action_kinds=frozenset(
            {
                "click", "double_click", "type", "hotkey", "scroll", "move",
                "drag", "wait", "terminate", "sequence", "no_op", "shell",
            }
        ),
        coordinate_protocol="normalized_0_1000",
        history_format="qwen35vl_snapshot_state_public_response",
    )
    history_starts_with_system_message = False

    def __init__(self, instruction: str = "") -> None:
        self.instruction = instruction

    @staticmethod
    def _source_response(trajectory_log: str) -> tuple[str, str, str]:
        try:
            record = json.loads(trajectory_log)
            response = record.get("response", record.get("visible_response"))
        except (KeyError, TypeError, json.JSONDecodeError) as exc:
            raise ValueError("Qwen 3.5 history requires a bound source response") from exc
        if not isinstance(response, str) or not response.strip():
            raise ValueError("Qwen 3.5 source response is empty")
        actions = list(_ACTION_LINE.finditer(response))
        calls = list(_TOOL_CALL.finditer(response))
        if len(actions) == 1 and len(calls) >= 1:
            action = actions[0].group("action").strip()
        elif len(actions) == 0 and len(calls) >= 1:
            action = ""
        elif len(actions) <= 1 and len(calls) == 0:
            action = ""
        else:
            raise ValueError(
                "Qwen 3.5 response has unexpected Action/tool_call structure: "
                f"{len(actions)} Action lines, {len(calls)} tool_calls"
            )
        reasoning = json.dumps(record.get("reasoning", []), ensure_ascii=False)
        return response, action, reasoning

    def render_step(self, step: HistoryStep) -> List[Dict[str, Any]]:
        self.capabilities.require(step.action)
        if step.action_index_within_turn > 0:
            return []
        response, action, reasoning = self._source_response(step.trajectory_log)
        return [
            {
                "role": "qwen35_state",
                "observation_image_url": step.observation_image_url,
                "observation_sha256": step.observation_sha256,
                "action": action,
                "response": response,
                "reasoning": reasoning,
            }
        ]

    def tool_definitions(self) -> List[Dict[str, Any]]:
        return []
