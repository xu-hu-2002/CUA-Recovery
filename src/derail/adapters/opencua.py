"""Exact state renderer for OpenCUA's upstream ``action_history`` protocol."""

from __future__ import annotations

import json
import re
from typing import Any, Dict, List

from .base import AgentCapabilities, HistoryStep


_ACTION_BLOCK = re.compile(
    r"(?:^|\n)## Action:\s*\n(?P<action>.*?)(?=\n\n## |\Z)", re.DOTALL
)
_THOUGHT_BLOCK = re.compile(
    r"(?:^|\n)## Thought:\s*\n(?P<thought>.*?)(?=\n\n## |\Z)", re.DOTALL
)
_CODE_BLOCK = re.compile(
    r"(?:^|\n)## Code:\s*\n(?:```python\s*)?(?P<code>.*?)(?:```)?(?=\n\s*`?\n?## |\Z)",
    re.DOTALL,
)


def _code_action_kind(code: str) -> str:
    patterns = (
        ("move", r"pyautogui\.moveTo\s*\("),
        ("drag", r"pyautogui\.(?:dragTo|dragRel)\s*\("),
        ("double_click", r"pyautogui\.doubleClick\s*\("),
        ("click", r"pyautogui\.click\s*\("),
        ("type", r"pyautogui\.(?:write|typewrite)\s*\("),
        ("hotkey", r"pyautogui\.hotkey\s*\("),
        ("scroll", r"pyautogui\.scroll\s*\("),
        ("horizontal_scroll", r"pyautogui\.hscroll\s*\("),
    )
    matches = [kind for kind, pattern in patterns if re.search(pattern, code)]
    return matches[0] if len(matches) == 1 else ""


class OpenCUAActionHistoryAdapter:
    """Restore recorded source reasoning and actions in upstream action-history mode."""

    capabilities = AgentCapabilities(
        agent_id="opencua_72b",
        action_kinds=frozenset(
            {
                "click", "double_click", "type", "hotkey", "scroll", "move",
                "drag", "wait", "terminate", "sequence",
            }
        ),
        coordinate_protocol="smart_resize_absolute_pixels_qwen25",
        history_format="opencua_observations_actions_cots_action_history",
    )

    def __init__(self, instruction: str = "", *, system_prompt: str = "") -> None:
        self.instruction = instruction
        self._system_prompt = system_prompt

    def system_prompt(self) -> str:
        if not self._system_prompt.strip():
            raise ValueError("OpenCUA renderer requires the exact live system prompt")
        return self._system_prompt

    @staticmethod
    def _source_cot(trajectory_log: str, action_kind: str) -> Dict[str, str]:
        try:
            record = json.loads(trajectory_log)
            response = record.get("response", record.get("visible_response"))
        except (KeyError, TypeError, json.JSONDecodeError) as exc:
            raise ValueError("OpenCUA history requires a bound source response") from exc
        response_text = str(response)
        matches = list(_ACTION_BLOCK.finditer(response_text))
        if not matches and action_kind == "terminate" and response_text.strip():
            return {"action": response_text.strip()}
        selected = matches
        if len(matches) > 1:
            complete_matches = []
            for index, match in enumerate(matches):
                end = matches[index + 1].start() if index + 1 < len(matches) else len(response_text)
                code = _CODE_BLOCK.search(response_text, match.end(), end)
                if code and _code_action_kind(code.group("code")) == action_kind:
                    complete_matches.append(match)
            # OpenCUA may return its accumulated Step history in one response.
            # The upstream executor runs the last complete matching Code block.
            selected = complete_matches[-1:]
        if len(selected) != 1 or not selected[0].group("action").strip():
            raise ValueError("OpenCUA source response has no unique public Action block")
        action_match = selected[0]
        thoughts = [item for item in _THOUGHT_BLOCK.finditer(response_text) if item.start() < action_match.start()]
        cot = {"action": action_match.group("action").strip()}
        if thoughts and thoughts[-1].group("thought").strip():
            cot["thought"] = thoughts[-1].group("thought").strip()
        if record.get("reasoning"):
            cot["recorded_reasoning"] = json.dumps(
                record["reasoning"], ensure_ascii=False, separators=(",", ":")
            )
        return cot

    def render_step(self, step: HistoryStep) -> List[Dict[str, Any]]:
        cot = self._source_cot(step.trajectory_log, step.action.kind)
        state = {
            "role": "opencua_state",
            "observation_image_url": step.observation_image_url,
            "action": cot["action"],
            "cot": cot,
        }
        if step.step_id == 0:
            return [{"role": "system", "content": self.system_prompt()}, state]
        return [state]

    def tool_definitions(self) -> List[Dict[str, Any]]:
        return []
