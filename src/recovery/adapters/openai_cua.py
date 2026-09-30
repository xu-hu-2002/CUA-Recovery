"""History renderer for GPT-5.5 on MyPCBench's ``openai_cuabash`` (Responses API)."""

from __future__ import annotations

import json
from typing import Any, Dict, List

from recovery.canonical.actions import (
    Action,
    ClickAction,
    DragAction,
    HorizontalScrollAction,
    HotkeyAction,
    MoveAction,
    NoOpAction,
    ScrollAction,
    SequenceAction,
    ShellAction,
    TypeAction,
    WaitAction,
    reframe_action,
)

from .base import ActionNotSupportedError, AgentCapabilities, HistoryStep


class OpenAIResponsesHistoryAdapter:
    """Render replay evidence as ``openai_cuabash`` Responses input items."""

    capabilities = AgentCapabilities(
        agent_id="gpt_5_5",
        action_kinds=frozenset(
            {
                "click", "double_click", "type", "hotkey", "scroll", "horizontal_scroll",
                "move", "drag", "wait", "no_op", "shell", "sequence", "terminate",
            }
        ),
        coordinate_protocol="absolute_pixels_1280x800",
        history_format="openai_responses_input_items",
        silent_action_kinds=frozenset({"terminate"}),
    )
    history_starts_with_system_message = False

    def __init__(
        self, instruction: str, *, system_prompt: str, screen_size: tuple = (1280, 800)
    ) -> None:
        if not instruction.strip() or not system_prompt.strip():
            raise ValueError("GPT renderer requires instruction and the live operator primer")
        self.instruction = instruction
        self._primer = system_prompt
        self._width, self._height = screen_size
        self._opened = False

    def _computer_actions(self, action: Action) -> List[Dict[str, Any]]:
        action = reframe_action(action, self._width, self._height)
        if isinstance(action, ClickAction):
            if action.kind == "double_click":
                return [{"type": "double_click", "x": action.x_px, "y": action.y_px}]
            return [{"type": "click", "button": action.button, "x": action.x_px, "y": action.y_px}]
        if isinstance(action, TypeAction):
            return (
                ([{"type": "keypress", "keys": ["ctrl", "a"]}] if action.clear_existing else [])
                + [{"type": "type", "text": action.text}]
                + ([{"type": "keypress", "keys": ["enter"]}] if action.press_enter else [])
            )
        if isinstance(action, HotkeyAction):
            return [{"type": "keypress", "keys": list(action.keys)}]
        if isinstance(action, ScrollAction):
            # CUA scroll is positive-down; pyautogui (canonical) is positive-up.
            return [{"type": "scroll", "x": action.x_px, "y": action.y_px,
                     "scroll_x": 0, "scroll_y": -action.delta_y}]
        if isinstance(action, HorizontalScrollAction):
            return [{"type": "scroll", "x": action.x_px, "y": action.y_px,
                     "scroll_x": -action.delta_x, "scroll_y": 0}]
        if isinstance(action, MoveAction):
            return [{"type": "move", "x": action.x_px, "y": action.y_px}]
        if isinstance(action, DragAction):
            return [{"type": "drag", "path": [
                {"x": action.start_x_px, "y": action.start_y_px},
                {"x": action.end_x_px, "y": action.end_y_px},
            ]}]
        if isinstance(action, WaitAction):
            return [{"type": "wait", "ms": int(round(action.seconds * 1000))}]
        if isinstance(action, NoOpAction):
            return [{"type": "wait", "ms": 0}]
        raise ActionNotSupportedError("GPT computer tool has no action for %s" % action.kind)

    @staticmethod
    def _shell_outputs(tool_result: str, count: int) -> List[List[Dict[str, Any]]]:
        try:
            payload = json.loads(tool_result)
        except json.JSONDecodeError:
            payload = None
        results = payload.get("shell_results") if isinstance(payload, dict) else None
        if isinstance(results, list) and len(results) == count:
            return [list(item["result"]) for item in results]
        text = [{"stdout": tool_result, "stderr": "", "outcome": {"type": "exit", "exit_code": 0}}]
        return [list(text) for _ in range(count)]

    def render_step(self, step: HistoryStep) -> List[Dict[str, Any]]:
        self.capabilities.require(step.action)
        items: List[Dict[str, Any]] = []
        if not self._opened:
            self._opened = True
            items.append({
                "role": "user",
                "content": [
                    {"type": "input_text", "text": (
                        f"{self._primer}\n\nTask: {self.instruction}\n\nPrevious actions:\nNone"
                    )},
                    {"type": "input_image", "image_url": step.observation_image_url},
                ],
            })
        primitives = (
            step.action.actions if isinstance(step.action, SequenceAction) else (step.action,)
        )
        shell_outputs = self._shell_outputs(
            step.tool_result, sum(isinstance(item, ShellAction) for item in primitives)
        )
        batch: List[Dict[str, Any]] = []

        def flush() -> None:
            if not batch:
                return
            call_id = "call_recovery_%04d_%02d" % (step.step_id, len(items))
            items.append({"type": "computer_call", "id": "cu_recovery_%04d_%02d" % (
                step.step_id, len(items)), "call_id": call_id, "actions": list(batch),
                "pending_safety_checks": [], "status": "completed"})
            items.append({"type": "computer_call_output", "call_id": call_id, "output": {
                "type": "computer_screenshot",
                "image_url": step.observation_after_image_url,
                "detail": "original",
            }})
            batch.clear()

        for primitive in primitives:
            if primitive.kind == "terminate":
                continue
            if not isinstance(primitive, ShellAction):
                batch.extend(self._computer_actions(primitive))
                continue
            flush()
            call_id = "call_recovery_%04d_%02d" % (step.step_id, len(items))
            items.append({"type": "shell_call", "id": "sh_recovery_%04d_%02d" % (
                step.step_id, len(items)), "call_id": call_id, "action": {
                "commands": list(primitive.commands),
                "timeout_ms": primitive.timeout_ms,
                "max_output_length": primitive.max_output_length,
            }, "status": "completed"})
            items.append({"type": "shell_call_output", "call_id": call_id,
                          "output": shell_outputs.pop(0)})
        flush()
        if any(item.get("type") == "computer_call_output" for item in items) and not (
            step.observation_after_image_url
        ):
            raise ValueError("GPT computer_call_output requires a replay post-action screenshot")
        return items

    def tool_definitions(self) -> List[Dict[str, Any]]:
        return []
