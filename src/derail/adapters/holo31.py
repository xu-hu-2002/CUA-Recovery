"""Holo-3.1 native OpenAI-style function-calling adapter。

这里注册的是 DERAIL 冻结后的实验 schema，而不是声称模型卡逐项公开了这些参数。
正式实验前必须对 scroll/hotkey/drag 等 primitive 做 action-conformance probe，并把
checkpoint revision、runtime 和 probe 结果写入 run manifest。
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, List, Tuple

from derail.canonical.actions import (
    Action,
    ClickAction,
    DragAction,
    HotkeyAction,
    MoveAction,
    ScrollAction,
    TerminateAction,
    TypeAction,
    WaitAction,
)
from derail.mypcbench.tool_agent import build_computer_tools

from .base import ActionNotSupportedError, AgentCapabilities, HistoryStep


class Holo31Adapter:
    """Render the history consumed by ``NativeToolComputerAgent``.

    The renderer deliberately mirrors the live agent's ``messages`` property:
    one frozen system message followed by user/assistant/tool triples.  It does
    not recreate model reasoning.  ``instruction`` is required when rendering
    because the live user message repeats the task text on every turn.
    """

    _REPO_ROOT = Path(__file__).resolve().parents[3]
    _SYSTEM_PROMPT_PATH = (
        _REPO_ROOT / "prompts" / "agents" / "holo_3_1_35b_a3b_mypcbench_system.txt"
    )
    _TOOL_RESULT = "Accepted for execution; inspect the next screenshot for the result."
    _SAFE_KEY = re.compile(r"^[A-Za-z0-9_+\-]{1,32}$")
    capabilities = AgentCapabilities(
        agent_id="holo_3_1_35b_a3b",
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
            }
        ),
        coordinate_protocol="normalized_0_1000",
        history_format="openai_native_tool_calls",
    )

    def __init__(self, instruction: str = "") -> None:
        self.instruction = instruction

    @classmethod
    def system_prompt(cls) -> str:
        if not cls._SYSTEM_PROMPT_PATH.is_file():
            raise RuntimeError(
                "找不到 %s 冻结 system prompt: %s"
                % (cls.capabilities.agent_id, cls._SYSTEM_PROMPT_PATH)
            )
        return cls._SYSTEM_PROMPT_PATH.read_text(encoding="utf-8").strip()

    @staticmethod
    def _require_frozen_frame(width: int, height: int) -> None:
        if (width, height) != (1280, 800):
            raise ValueError(
                "Holo renderer 只通过了冻结 1280x800 协议，不能静默转换 %dx%d" % (width, height)
            )

    @staticmethod
    def _xy(x_px: int, y_px: int, width: int, height: int) -> Tuple[int, int]:
        Holo31Adapter._require_frozen_frame(width, height)
        # This is the inverse of SafePyAutoGUICompiler, which maps 1000 to
        # width-1/height-1.  Do not use the generic coordinate helper here:
        # its upstream-agent convention divides by the full extent.
        x = round(x_px * 1000 / (width - 1))
        y = round(y_px * 1000 / (height - 1))
        roundtrip_x = round(x * (width - 1) / 1000)
        roundtrip_y = round(y * (height - 1) / 1000)
        if (roundtrip_x, roundtrip_y) != (x_px, y_px):
            raise ActionNotSupportedError(
                "normalized_0_1000 不能无损表示像素 (%d, %d)，round-trip 得到 (%d, %d)"
                % (x_px, y_px, roundtrip_x, roundtrip_y)
            )
        return x, y

    def action_to_call(self, action: Action, call_id: str) -> Dict[str, Any]:
        self.capabilities.require(action)

        name: str
        args: Dict[str, Any]
        if isinstance(action, ClickAction):
            x, y = self._xy(action.x_px, action.y_px, action.frame_width, action.frame_height)
            name = action.kind
            # Exact live schema from build_computer_tools(): target summaries are
            # canonical metadata, not a Holo tool argument.
            args = {"x": x, "y": y, "button": action.button}
        elif isinstance(action, TypeAction):
            if len(action.text) > 10000:
                raise ActionNotSupportedError("write 超过 live compiler 的 10000 字符上限")
            name = "write"
            args = {
                "content": action.text,
                "press_enter": action.press_enter,
                "clear_existing": action.clear_existing,
            }
        elif isinstance(action, HotkeyAction):
            if any(not self._SAFE_KEY.fullmatch(key) for key in action.keys):
                raise ActionNotSupportedError("hotkey 含 live compiler 不接受的按键名")
            name = "hotkey"
            args = {"keys": list(action.keys)}
        elif isinstance(action, ScrollAction):
            if abs(action.delta_y) > 10000:
                raise ActionNotSupportedError("scroll 超过 live compiler 的绝对值 10000 上限")
            x, y = self._xy(action.x_px, action.y_px, action.frame_width, action.frame_height)
            name = "scroll"
            args = {"x": x, "y": y, "delta_y": action.delta_y}
        elif isinstance(action, MoveAction):
            x, y = self._xy(action.x_px, action.y_px, action.frame_width, action.frame_height)
            name = "move"
            args = {"x": x, "y": y}
        elif isinstance(action, DragAction):
            if action.duration_s > 10:
                raise ActionNotSupportedError("drag 超过 live compiler 的 10 秒上限")
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
                raise ActionNotSupportedError("wait 超过 live compiler 的 30 秒上限")
            name = "wait"
            args = {"seconds": action.seconds}
        elif isinstance(action, TerminateAction):
            name = "answer"
            args = {"status": action.status, "content": action.answer}
        else:
            raise TypeError("未处理的 Holo action: %s" % type(action).__name__)

        return {
            "id": call_id,
            "type": "function",
            "function": {"name": name, "arguments": json.dumps(args, ensure_ascii=False)},
        }

    def render_step(self, step: HistoryStep) -> List[Dict[str, Any]]:
        if not self.instruction.strip():
            raise ValueError(
                "Holo native history 需要原始 task instruction；禁止用占位文本伪造 history"
            )
        call_id = "derail_step_%04d" % step.step_id
        call = self.action_to_call(step.action, call_id)
        messages: List[Dict[str, Any]] = []
        if step.step_id == 0:
            messages.append({"role": "system", "content": self.system_prompt()})
        messages.extend([
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": step.observation_image_url}},
                    {
                        "type": "text",
                        "text": (
                            f"Task: {self.instruction}\n"
                            + (
                                f"Source trajectory log: {step.trajectory_log}\n"
                                if step.trajectory_log
                                else ""
                            )
                            + "Inspect the current trajectory logs and screenshot and "
                            "emit the next action as a tool call."
                        ),
                    },
                ],
            },
            {
                "role": "assistant",
                # Recorded source reasoning is preserved in the preceding trajectory log.
                "content": None,
                "tool_calls": [call],
            },
            {
                "role": "tool",
                "tool_call_id": call_id,
                # This is the exact visible acknowledgement stored by
                # NativeToolComputerAgent before the environment executes.
                "content": self._TOOL_RESULT,
            },
        ])
        return messages

    def tool_definitions(self) -> List[Dict[str, Any]]:
        """Return the exact schema used by the live MyPCBench agent."""

        return build_computer_tools(1000, 1000)
