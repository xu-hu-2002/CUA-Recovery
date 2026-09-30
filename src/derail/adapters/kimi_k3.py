"""History renderer for the frozen DERAIL kimi-k3 general-VLM scaffold.

kimi-k3 has no native computer-use protocol reachable through the routify
gateway (Responses API not mounted; see docs/ROUTIFY_API_MATRIX.md).  The live
agent uses the DERAIL absolute-pixel OpenAI-compatible scaffold. The takeover
target exposes the CUABash tool surface needed by the annotated August source
collections. It must not be described as a native Kimi CUA protocol.
"""

from __future__ import annotations

import copy
import json
from typing import Any, Dict, List

from derail.canonical.actions import Action, ScrollAction, ShellAction
from derail.mypcbench.tool_agent import build_computer_tools, compose_system_prompt

from .base import AgentCapabilities
from .holo31 import Holo31Adapter
from .qwen38 import Qwen38ScaffoldAdapter


class KimiK3ScaffoldAdapter(Qwen38ScaffoldAdapter):
    _SYSTEM_PROMPT_FILE = "kimi_k3_cuabash_mypcbench_system.txt"
    capabilities = AgentCapabilities(
        agent_id="kimi_k3",
        action_kinds=Qwen38ScaffoldAdapter.capabilities.action_kinds,
        coordinate_protocol="absolute_pixels_1280x800",
        history_format="derail_openai_compatible_function_calls",
    )

    @classmethod
    def system_prompt(cls) -> str:
        return compose_system_prompt(cls._SYSTEM_PROMPT_FILE, has_bash=True)

    def history_tool_definitions(self) -> List[Dict[str, Any]]:
        """Return the tool schema used by the frozen August collection."""

        tools = copy.deepcopy(self.tool_definitions())
        scroll = next(tool for tool in tools if tool["function"]["name"] == "scroll")
        delta = scroll["function"]["parameters"]["properties"]["delta_y"]
        delta.pop("minimum", None)
        delta.pop("maximum", None)
        return tools

    def action_to_calls(self, action: Action, call_id: str) -> List[Dict[str, Any]]:
        if isinstance(action, ScrollAction):
            return [Holo31Adapter.action_to_call(self, action, call_id)]
        if not isinstance(action, ShellAction):
            return super().action_to_calls(action, call_id)
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

    def tool_definitions(self) -> List[Dict[str, Any]]:
        return build_computer_tools(1279, 799, include_bash=True)
