"""History renderer for the frozen RECOVERY kimi-k3 general-VLM scaffold."""

from __future__ import annotations

import copy
import json
import re
from typing import Any, Dict, List

from recovery.canonical.actions import (
    Action,
    ClickAction,
    DragAction,
    HorizontalScrollAction,
    HotkeyAction,
    KeyTransitionAction,
    MouseButtonTransitionAction,
    MoveAction,
    NoOpAction,
    ScrollAction,
    SequenceAction,
    ShellAction,
    TerminateAction,
    TypeAction,
    WaitAction,
)
from recovery.mypcbench.tool_agent import build_computer_tools, compose_system_prompt
from recovery.replay.executor import compile_pyautogui

from .base import ActionNotSupportedError, AgentCapabilities, HistoryStep


class KimiK3ScaffoldAdapter:
    _SYSTEM_PROMPT_FILE = "kimi_k3_cuabash_mypcbench_system.txt"
    _TOOL_RESULT = "Accepted for execution; inspect the next screenshot for the result."
    _SAFE_KEY = re.compile(r"^[A-Za-z0-9_+\-]{1,32}$")
    capabilities = AgentCapabilities(
        agent_id="kimi_k3",
        action_kinds=frozenset(
            {
                "click",
                "double_click",
                "type",
                "hotkey",
                "scroll",
                "move",
                "drag",
                "wait",
                "terminate",
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
        history_format="recovery_openai_compatible_function_calls",
    )

    def __init__(self, instruction: str = "") -> None:
        self.instruction = instruction

    @classmethod
    def system_prompt(cls) -> str:
        return compose_system_prompt(cls._SYSTEM_PROMPT_FILE, has_bash=True)

    @staticmethod
    def _xy(x_px: int, y_px: int, width: int, height: int) -> tuple[int, int]:
        if width <= 1 or height <= 1:
            raise ValueError("Kimi history action has an invalid source frame")
        return round(x_px * 1279 / (width - 1)), round(y_px * 799 / (height - 1))

    def action_to_call(self, action: Action, call_id: str) -> Dict[str, Any]:
        self.capabilities.require(action)

        name: str
        args: Dict[str, Any]
        if isinstance(action, ClickAction):
            x, y = self._xy(action.x_px, action.y_px, action.frame_width, action.frame_height)
            name = action.kind
            args = {"x": x, "y": y, "button": action.button}
        elif isinstance(action, TypeAction):
            if len(action.text) > 10000:
                raise ActionNotSupportedError("write exceeds the live compiler limit of 10000 characters")
            name = "write"
            args = {
                "content": action.text,
                "press_enter": action.press_enter,
                "clear_existing": action.clear_existing,
            }
        elif isinstance(action, HotkeyAction):
            if any(not self._SAFE_KEY.fullmatch(key) for key in action.keys):
                raise ActionNotSupportedError("hotkey contains key names the live compiler rejects")
            name = "hotkey"
            args = {"keys": list(action.keys)}
        elif isinstance(action, ScrollAction):
            if abs(action.delta_y) > 10000:
                raise ActionNotSupportedError("scroll exceeds the live compiler absolute limit of 10000")
            x, y = self._xy(action.x_px, action.y_px, action.frame_width, action.frame_height)
            name = "scroll"
            args = {"x": x, "y": y, "delta_y": action.delta_y}
        elif isinstance(action, MoveAction):
            x, y = self._xy(action.x_px, action.y_px, action.frame_width, action.frame_height)
            name = "move"
            args = {"x": x, "y": y}
        elif isinstance(action, DragAction):
            if action.duration_s > 10:
                raise ActionNotSupportedError("drag exceeds the live compiler limit of 10 seconds")
            start_x, start_y = self._xy(
                action.start_x_px,
                action.start_y_px,
                action.frame_width,
                action.frame_height,
            )
            end_x, end_y = self._xy(
                action.end_x_px,
                action.end_y_px,
                action.frame_width,
                action.frame_height,
            )
            name = "drag"
            args = {
                "start_x": start_x,
                "start_y": start_y,
                "end_x": end_x,
                "end_y": end_y,
                "button": action.button,
                "duration_s": action.duration_s,
            }
        elif isinstance(action, WaitAction):
            if action.seconds > 30:
                raise ActionNotSupportedError("wait exceeds the live compiler limit of 30 seconds")
            name = "wait"
            args = {"seconds": action.seconds}
        elif isinstance(action, TerminateAction):
            name = "answer"
            args = {"status": action.status, "content": action.answer}
        else:
            raise TypeError("unhandled action: %s" % type(action).__name__)

        return {
            "id": call_id,
            "type": "function",
            "function": {"name": name, "arguments": json.dumps(args, ensure_ascii=False)},
        }

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
        if isinstance(action, ScrollAction):
            return [self.action_to_call(action, call_id)]
        if isinstance(action, ShellAction):
            if len(action.commands) != 1:
                raise ValueError("Kimi CUABash history requires exactly one command per call")
            return [
                {
                    "id": call_id,
                    "type": "function",
                    "function": {
                        "name": "bash",
                        "arguments": json.dumps(
                            {"command": action.commands[0]}, ensure_ascii=False
                        ),
                    },
                }
            ]
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
                or any(not self._SAFE_KEY.fullmatch(key) for key in action.keys)
            )
        ):
            return [self._bash_call(compile_pyautogui(action), call_id)]
        return [self.action_to_call(action, call_id)]

    @staticmethod
    def _contains_gui_action(action: Action) -> bool:
        if isinstance(action, SequenceAction):
            return any(not isinstance(item, ShellAction) for item in action.actions)
        return not isinstance(action, ShellAction)

    @classmethod
    def _result_per_call(
        cls, calls: List[Dict[str, Any]], result_text: str
    ) -> List[str]:
        results = [cls._TOOL_RESULT] * len(calls)
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
            raise ValueError("Kimi native history requires the original task instruction")
        base_id = "recovery_step_%04d" % step.step_id
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

    def history_tool_definitions(self) -> List[Dict[str, Any]]:
        """Return the tool schema used by the frozen August collection."""

        tools = copy.deepcopy(self.tool_definitions())
        scroll = next(tool for tool in tools if tool["function"]["name"] == "scroll")
        delta = scroll["function"]["parameters"]["properties"]["delta_y"]
        delta.pop("minimum", None)
        delta.pop("maximum", None)
        return tools

    def tool_definitions(self) -> List[Dict[str, Any]]:
        return build_computer_tools(1279, 799, include_bash=True)
