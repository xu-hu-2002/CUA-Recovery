"""History renderer for the upstream EvoCUA S2 protocol."""

from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Mapping, Sequence

from recovery.canonical.actions import (
    Action,
    ClickAction,
    DragAction,
    HotkeyAction,
    KeyTransitionAction,
    MoveAction,
    NoOpAction,
    ScrollAction,
    SequenceAction,
    TerminateAction,
    TypeAction,
    WaitAction,
)
from recovery.canonical.summaries import summarize_action

from .base import ActionNotSupportedError, AgentCapabilities, HistoryStep

_TOOL_CALL_BLOCK = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.DOTALL)


class EvoCUAS2Adapter:
    """Render replay evidence as EvoCUA S2 image/raw-response turns."""

    capabilities = AgentCapabilities(
        agent_id="evocua_32b",
        action_kinds=frozenset(
            {
                "click",
                "double_click",
                "triple_click",
                "type",
                "hotkey",
                "sequence",
                "scroll",
                "move",
                "drag",
                "wait",
                "terminate",
                "key_down",
                "key_up",
                "no_op",
            }
        ),
        coordinate_protocol="normalized_0_999",
        history_format="evocua_s2_raw_response_turns",
    )

    def __init__(self, instruction: str, *, source_agent: str = "evocua_32b") -> None:
        if not instruction.strip():
            raise ValueError("EvoCUA native history 需要非空 task instruction")
        self.instruction = instruction
        self.source_agent = source_agent

    @staticmethod
    def _upstream() -> Any:
        # Lazy import avoids a circular import with the factory.
        from recovery.mypcbench.factory import load_evocua_upstream

        return load_evocua_upstream()

    @classmethod
    def _tools_def(cls) -> Dict[str, Any]:
        module = cls._upstream()
        description = module.S2_DESCRIPTION_PROMPT_TEMPLATE.format(
            resolution_info="* The screen's resolution is 1000x1000."
        )
        return module.build_s2_tools_def(description)

    @classmethod
    def system_prompt(cls) -> str:
        """Return the exact S2 system prompt the live agent builds each turn."""

        module = cls._upstream()
        return module.S2_SYSTEM_PROMPT.format(tools_xml=json.dumps(cls._tools_def()))

    def tool_definitions(self) -> List[Dict[str, Any]]:
        """Return the S2 tool schema."""

        return [self._tools_def()]

    @classmethod
    def extract_tool_calls(
        cls, messages: Sequence[Mapping[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Lift the inline `<tool_call>` JSON into OpenAI-shaped call dicts."""

        calls: List[Dict[str, Any]] = []
        for message in messages:
            if message.get("role") != "assistant":
                continue
            for part in message.get("content", []):
                if not isinstance(part, Mapping) or part.get("type") != "text":
                    continue
                blocks = _TOOL_CALL_BLOCK.findall(str(part.get("text", "")))
                if not blocks:
                    raise ValueError("S2 assistant turn has no <tool_call> block")
                for block in blocks:
                    payload = json.loads(block)
                    calls.append(
                        {
                            "id": "evocua_%04d" % len(calls),
                            "type": "function",
                            "function": {
                                "name": str(payload.get("name", "")),
                                "arguments": json.dumps(
                                    payload.get("arguments", {}), ensure_ascii=False
                                ),
                            },
                        }
                    )
        return calls

    @staticmethod
    def source_response(step: HistoryStep) -> str:
        """Return the source turn's completion exactly as it was recorded."""

        if not step.trajectory_log:
            raise ValueError(
                "EvoCUA history step %d has no source trajectory log; its raw "
                "completion cannot be reconstructed" % step.step_id
            )
        record = json.loads(step.trajectory_log)
        response = record.get("visible_response", record.get("response"))
        if not isinstance(response, str) or not response.strip():
            raise ValueError(
                "EvoCUA history step %d has an empty source completion" % step.step_id
            )
        return response

    @staticmethod
    def _grid(action: Any, x_px: int, y_px: int) -> List[int]:
        return [round(x_px * 999 / action.frame_width), round(y_px * 999 / action.frame_height)]

    @classmethod
    def s2_arguments(cls, action: Action) -> List[Dict[str, Any]]:
        """Lower one canonical action into S2 ``computer_use`` argument objects."""

        cls.capabilities.require(action)
        if isinstance(action, SequenceAction):
            return [item for part in action.actions for item in cls.s2_arguments(part)]
        if isinstance(action, ClickAction):
            name = "double_click" if action.kind == "double_click" else "%s_click" % action.button
            return [{"action": name, "coordinate": cls._grid(action, action.x_px, action.y_px)}]
        if isinstance(action, MoveAction):
            return [{"action": "mouse_move", "coordinate": cls._grid(action, action.x_px, action.y_px)}]
        if isinstance(action, DragAction):
            return [
                {"action": "mouse_move",
                 "coordinate": cls._grid(action, action.start_x_px, action.start_y_px)},
                {"action": "left_click_drag",
                 "coordinate": cls._grid(action, action.end_x_px, action.end_y_px)},
            ]
        if isinstance(action, TypeAction):
            clear = [{"action": "key", "keys": ["ctrl", "a"]}] if action.clear_existing else []
            return clear + [
                {"action": "type", "text": action.text + ("\n" if action.press_enter else "")}
            ]
        if isinstance(action, HotkeyAction):
            return [{"action": "key", "keys": list(action.keys)}]
        if isinstance(action, KeyTransitionAction):
            return [{"action": action.kind, "keys": [action.key]}]
        if isinstance(action, ScrollAction):
            return [{"action": "scroll", "pixels": action.delta_y,
                     "coordinate": cls._grid(action, action.x_px, action.y_px)}]
        if isinstance(action, WaitAction):
            return [{"action": "wait", "time": action.seconds}]
        if isinstance(action, NoOpAction):
            return [{"action": "wait", "time": 0}]
        if isinstance(action, TerminateAction):
            return [{"action": "terminate", "status": action.status}]
        raise ActionNotSupportedError("EvoCUA S2 has no call for %s" % action.kind)

    @classmethod
    def synthesized_response(cls, step: HistoryStep) -> str:
        """S2 completion for a foreign source step: public summary plus its tool calls."""

        calls = "\n".join(
            "<tool_call>\n%s\n</tool_call>"
            % json.dumps({"name": "computer_use", "arguments": arguments}, ensure_ascii=False)
            for arguments in cls.s2_arguments(step.action)
        )
        return "Action: %s\n%s" % (summarize_action(step.action), calls)

    def render_step(self, step: HistoryStep) -> List[Dict[str, Any]]:
        if not step.observation_image_url:
            raise ValueError(
                "EvoCUA history step %d has no pre-action screenshot; S2 pairs "
                "every assistant turn with the image it acted on" % step.step_id
            )
        response = (
            self.source_response(step)
            if self.source_agent == self.capabilities.agent_id
            else self.synthesized_response(step)
        )
        messages: List[Dict[str, Any]] = []
        if step.step_id == 0:
            messages.append({"role": "system", "content": self.system_prompt()})
        messages.append(
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": step.observation_image_url},
                    }
                ],
            }
        )
        messages.append(
            {"role": "assistant", "content": [{"type": "text", "text": response}]}
        )
        return messages
