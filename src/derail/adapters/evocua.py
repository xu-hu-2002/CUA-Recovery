"""History renderer for the upstream EvoCUA S2 protocol.

Its live loop stores the raw model completion in ``self.responses`` and replays
the last ``max_history_turns`` of them verbatim as assistant turns
(``evocua_agent.py:177`` / ``_build_s2_messages``).  For self-takeover the source
transcript itself is therefore the lossless rendering.  For any other source the
assistant turns are rendered from the canonical actions in the S2 grammar
(``Action:`` line plus ``<tool_call>`` JSON), exactly as the other scaffold
renderers lower canonical actions into their own tool calls.
"""

from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Mapping, Sequence

from derail.canonical.actions import (
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
from derail.canonical.summaries import summarize_action

from .base import ActionNotSupportedError, AgentCapabilities, HistoryStep

# S2 asks for the call as inline JSON rather than an OpenAI `tool_calls` field
# (prompts.py:S2_SYSTEM_PROMPT), so conformance checks parse it out of the text.
_TOOL_CALL_BLOCK = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.DOTALL)


class EvoCUAS2Adapter:
    """Render replay evidence as EvoCUA S2 image/raw-response turns."""

    # Mirrors the upstream `computer_use` action enum (prompts.py:build_s2_tools_def)
    # projected onto canonical kinds.  `shell` is absent because EvoCUA is a
    # pure-GUI agent with no bash tool.
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
        # configs/agents/evocua_32b.yaml pins coordinate_type=relative, so the
        # prompt always advertises the 1000x1000 grid regardless of frame size.
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
        # Imported lazily: the live factory imports this module to build the
        # takeover wrapper, so a module-level import would be circular.
        from derail.mypcbench.factory import load_evocua_upstream

        return load_evocua_upstream()

    @classmethod
    def _tools_def(cls) -> Dict[str, Any]:
        module = cls._upstream()
        # coordinate_type=relative -> upstream hardcodes the normalized grid.
        description = module.S2_DESCRIPTION_PROMPT_TEMPLATE.format(
            resolution_info="* The screen's resolution is 1000x1000."
        )
        return module.build_s2_tools_def(description)

    @classmethod
    def system_prompt(cls) -> str:
        """Return the exact S2 system prompt the live agent builds each turn.

        ``load_evocua_upstream`` applies the DERAIL shared environment block to
        ``S2_SYSTEM_PROMPT`` under the same idempotence flag the live factory
        uses, so an offline preflight measures the same prompt the VM run sends.
        """

        module = cls._upstream()
        return module.S2_SYSTEM_PROMPT.format(tools_xml=json.dumps(cls._tools_def()))

    def tool_definitions(self) -> List[Dict[str, Any]]:
        """Return the S2 tool schema.

        EvoCUA ships it inside the system prompt instead of an API `tools`
        field, but it is still the schema every call must satisfy, so
        conformance checks validate against it like any other target.
        """

        return [self._tools_def()]

    @classmethod
    def extract_tool_calls(
        cls, messages: Sequence[Mapping[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Lift the inline `<tool_call>` JSON into OpenAI-shaped call dicts.

        Its presence marks this renderer as a text protocol: callers must not
        expect an OpenAI `tool_calls` field, per-call IDs, or `tool` result
        messages, none of which exist in S2.
        """

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
        # Inverse of upstream adjust_coordinates for coordinate_type=relative (0..999 grid).
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
            # clear_existing executes as ctrl+a before typing (replay.executor).
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
        # Self-takeover: the recorded visible completion (reasoning already stripped
        # by load_trajectory_log unless history.strip_reasoning is false).
        messages.append(
            {"role": "assistant", "content": [{"type": "text", "text": response}]}
        )
        return messages
