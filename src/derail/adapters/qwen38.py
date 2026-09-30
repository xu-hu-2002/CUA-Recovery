"""History renderer for the Qwen3.8 DERAIL hybrid GUI/shell scaffold."""

from __future__ import annotations

import json
from typing import Any, Dict, List

from derail.canonical.actions import (
    Action,
    HorizontalScrollAction,
    HotkeyAction,
    KeyTransitionAction,
    MouseButtonTransitionAction,
    NoOpAction,
    ScrollAction,
    SequenceAction,
    ShellAction,
)
from derail.mypcbench.tool_agent import build_computer_tools, build_shell_tool, compose_system_prompt
from derail.replay.executor import compile_pyautogui

from .base import AgentCapabilities, HistoryStep
from .holo31 import Holo31Adapter
from .qwen36 import Qwen36ScaffoldAdapter


class Qwen38ScaffoldAdapter(Qwen36ScaffoldAdapter):
    """Render replay evidence as Qwen3.8 OpenAI-compatible tool history."""

    _SYSTEM_PROMPT_FILE = "qwen3_8_27b_mypcbench_system.txt"

    capabilities = AgentCapabilities(
        agent_id="qwen3_8_27b",
        action_kinds=Holo31Adapter.capabilities.action_kinds
        | frozenset(
            {
                "shell",
                "sequence",
                "key_down",
                "key_up",
                "mouse_down",
                "mouse_up",
                "no_op",
                "horizontal_scroll",
            }
        ),
        coordinate_protocol="absolute_pixels_1280x800",
        history_format="derail_openai_compatible_function_calls",
    )

    @classmethod
    def system_prompt(cls) -> str:
        return compose_system_prompt(cls._SYSTEM_PROMPT_FILE)

    @staticmethod
    def _xy(x_px: int, y_px: int, width: int, height: int) -> tuple[int, int]:
        if width <= 1 or height <= 1:
            raise ValueError("Qwen3.8 history action has an invalid source frame")
        return round(x_px * 1279 / (width - 1)), round(y_px * 799 / (height - 1))

    @staticmethod
    def _write_call(content: str, call_id: str) -> Dict[str, Any]:
        return {
            "id": call_id,
            "type": "function",
            "function": {
                "name": "write",
                "arguments": json.dumps(
                    {
                        "content": content,
                        "press_enter": False,
                        "clear_existing": False,
                    },
                    ensure_ascii=False,
                ),
            },
        }

    @staticmethod
    def _literal_text_key(action: Action) -> str | None:
        if not isinstance(action, HotkeyAction) or len(action.keys) != 1:
            return None
        key = action.keys[0]
        if key == "space":
            return " "
        return key if len(key) == 1 else None

    @staticmethod
    def _bash_call(command: str, call_id: str) -> Dict[str, Any]:
        return {
            "id": call_id,
            "type": "function",
            "function": {
                "name": "bash",
                "arguments": json.dumps(
                    {
                        "commands": [command],
                        "timeout_ms": 120000,
                        "max_output_length": 8192,
                    },
                    ensure_ascii=False,
                ),
            },
        }

    def action_to_calls(self, action: Action, call_id: str) -> List[Dict[str, Any]]:
        self.capabilities.require(action)
        if isinstance(action, SequenceAction):
            calls: List[Dict[str, Any]] = []
            text_parts: List[str] = []
            text_start = 0

            def flush_text() -> None:
                if not text_parts:
                    return
                calls.append(
                    self._write_call("".join(text_parts), f"{call_id}_{text_start:02d}")
                )
                text_parts.clear()

            for index, primitive in enumerate(action.actions):
                literal = self._literal_text_key(primitive)
                if literal is not None:
                    if not text_parts:
                        text_start = index
                    text_parts.append(literal)
                    continue
                flush_text()
                calls.extend(self.action_to_calls(primitive, f"{call_id}_{index:02d}"))
            flush_text()
            return calls
        if isinstance(action, ShellAction):
            return [
                {
                    "id": call_id,
                    "type": "function",
                    "function": {
                        "name": "bash",
                        "arguments": json.dumps(
                            {
                                "commands": list(action.commands),
                                "timeout_ms": action.timeout_ms,
                                "max_output_length": action.max_output_length,
                            },
                            ensure_ascii=False,
                        ),
                    },
                }
            ]
        if isinstance(action, ScrollAction) and abs(action.delta_y) > 30:
            return [self._bash_call(compile_pyautogui(action), call_id)]
        literal = self._literal_text_key(action)
        if literal is not None:
            return [self._write_call(literal, call_id)]
        if isinstance(action, NoOpAction):
            return [
                {
                    "id": call_id,
                    "type": "function",
                    "function": {
                        "name": "wait",
                        "arguments": json.dumps({"seconds": 0}),
                    },
                }
            ]
        if isinstance(
            action,
            (KeyTransitionAction, MouseButtonTransitionAction, HorizontalScrollAction),
        ) or (
            isinstance(action, HotkeyAction)
            and (
                len(action.keys) > 5
                or any(not Holo31Adapter._SAFE_KEY.fullmatch(key) for key in action.keys)
            )
        ):
            return [self._bash_call(compile_pyautogui(action), call_id)]
        return [super().action_to_call(action, call_id)]

    @staticmethod
    def _contains_gui_action(action: Action) -> bool:
        if isinstance(action, SequenceAction):
            return any(not isinstance(item, ShellAction) for item in action.actions)
        return not isinstance(action, ShellAction)

    @staticmethod
    def _result_per_call(
        calls: List[Dict[str, Any]], result_text: str
    ) -> List[str]:
        gui_reply = Holo31Adapter._TOOL_RESULT
        results = [gui_reply] * len(calls)
        shell_positions = [
            index
            for index, call in enumerate(calls)
            if call["function"].get("name") == "bash"
        ]
        if not shell_positions:
            return results
        fallback = result_text.strip() or "(no textual shell result)"
        try:
            payload = json.loads(fallback)
        except json.JSONDecodeError:
            for index in shell_positions:
                results[index] = fallback
            return results
        shell_results = payload.get("shell_results") if isinstance(payload, dict) else None
        if not isinstance(shell_results, list) or len(shell_results) != len(shell_positions):
            for index in shell_positions:
                results[index] = fallback
            return results
        for position, item in zip(shell_positions, shell_results):
            results[position] = json.dumps(
                item.get("result") if isinstance(item, dict) else item,
                ensure_ascii=False,
                sort_keys=True,
            )
        return results

    def render_step(self, step: HistoryStep) -> List[Dict[str, Any]]:
        if not self.instruction.strip():
            raise ValueError("Qwen3.8 native history requires the original task instruction")
        base_id = "derail_step_%04d" % step.step_id
        calls = self.action_to_calls(step.action, base_id)
        content: List[Dict[str, Any]] = []
        if self._contains_gui_action(step.action) and step.observation_image_url:
            content.append(
                {"type": "image_url", "image_url": {"url": step.observation_image_url}}
            )
        content.append(
            {
                "type": "text",
                "text": (
                    f"Task: {self.instruction}\n"
                    + (
                        f"Source trajectory log: {step.trajectory_log}\n"
                        if step.trajectory_log
                        else ""
                    )
                    + "Inspect the current trajectory logs and screenshot and emit the "
                    "next action as a tool call."
                ),
            }
        )
        messages: List[Dict[str, Any]] = []
        if step.step_id == 0:
            messages.append({"role": "system", "content": self.system_prompt()})
        messages.extend(
            [
                {"role": "user", "content": content},
                {"role": "assistant", "content": "(tool call)", "tool_calls": calls},
            ]
        )
        results = self._result_per_call(calls, step.tool_result)
        for call, result in zip(calls, results):
            messages.append(
                {"role": "tool", "tool_call_id": call["id"], "content": result}
            )
        return messages

    def tool_definitions(self) -> List[Dict[str, Any]]:
        return [*build_computer_tools(1279, 799), build_shell_tool()]
