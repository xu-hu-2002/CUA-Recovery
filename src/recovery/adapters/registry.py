"""Truthful registry for target-agent native-history support."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from functools import partial
from typing import Any, Callable, Dict, Optional, Tuple

from .base import AgentAdapter
from .claude import ClaudeNativeHistoryAdapter
from .evocua import EvoCUAS2Adapter
from .kimi_k3 import KimiK3ScaffoldAdapter
from .openai_cua import OpenAIResponsesHistoryAdapter
from .opencua import OpenCUAActionHistoryAdapter
from .qwen35 import QWEN35_AGENT_IDS, Qwen35StateAdapter

TARGET_AGENT_IDS: Tuple[str, ...] = (
    "qwen3_5_35b_a3b",
    "rerail_35b_a3b",
    "evocua_32b",
    "opencua_72b",
    "gpt_5_5",
    "claude_opus_4_8",
    "kimi_k3",
)


class NativeHistoryUnavailableError(RuntimeError):
    """No lossless renderer is implemented for the requested live protocol."""


@dataclass(frozen=True)
class NativeHistoryRegistration:
    agent_id: str
    family: str
    live_source: str
    history_structure: str
    protocol_origin: str
    renderer_path: str
    renderer_implemented: bool
    conformance_required: bool
    blocker: str = ""
    factory: Optional[Callable[..., AgentAdapter]] = None

    def __post_init__(self) -> None:
        if self.renderer_implemented != (self.factory is not None):
            raise ValueError("renderer_implemented and factory must agree")
        if self.renderer_implemented and self.blocker:
            raise ValueError("an implemented renderer cannot declare a blocker")
        if not self.renderer_implemented and not self.blocker:
            raise ValueError("an unimplemented renderer must record a specific blocker")

    def create(self, instruction: str, **kwargs: Any) -> AgentAdapter:
        if not self.renderer_implemented or self.factory is None:
            raise NativeHistoryUnavailableError(
                "%s native-history unavailable: %s" % (self.agent_id, self.blocker)
            )
        if not instruction.strip():
            raise ValueError("native-history renderer needs a non-empty task instruction")
        return self.factory(instruction, **kwargs)

    def status_dict(self) -> Dict[str, object]:
        data = asdict(self)
        data.pop("factory")
        data["release_ready_without_probe"] = False
        return data


_REGISTRY: Dict[str, NativeHistoryRegistration] = {
    **{
        agent_id: NativeHistoryRegistration(
            agent_id=agent_id,
            family="qwen35_osworld_vendored",
            live_source="third_party/MyPCBench/agent-harness/agents/qwen_cua.py:QwenOSWorldAgent",
            history_structure="processed-image chat history with folding/context fitting",
            protocol_origin="vendored_paper_results",
            renderer_path="recovery.adapters.qwen35.Qwen35StateAdapter",
            renderer_implemented=True,
            conformance_required=True,
            factory=partial(Qwen35StateAdapter, agent_id=agent_id),
        )
        for agent_id in sorted(QWEN35_AGENT_IDS)
    },
    "evocua_32b": NativeHistoryRegistration(
        agent_id="evocua_32b",
        family="evocua_s2",
        live_source="third_party/EvoCUA/mm_agents/evocua/evocua_agent.py:EvoCUAAgent",
        history_structure="S2 resized-image user turns plus raw XML/JSON assistant responses",
        protocol_origin="upstream_evocua_s2",
        renderer_path="recovery.adapters.evocua.EvoCUAS2Adapter",
        renderer_implemented=True,
        conformance_required=True,
        factory=EvoCUAS2Adapter,
    ),
    "kimi_k3": NativeHistoryRegistration(
        agent_id="kimi_k3",
        family="kimi_general_vlm_scaffold",
        live_source="src/recovery/mypcbench/tool_agent.py:NativeToolComputerAgent",
        history_structure="OpenAI-compatible system/user/assistant-tool/tool messages",
        protocol_origin="recovery_scaffold_frozen_schema_via_gateway",
        renderer_path="recovery.adapters.kimi_k3.KimiK3ScaffoldAdapter",
        renderer_implemented=True,
        conformance_required=True,
        factory=KimiK3ScaffoldAdapter,
    ),
    "opencua_72b": NativeHistoryRegistration(
        agent_id="opencua_72b",
        family="opencua",
        live_source=(
            "third_party/OpenCUA-OSWorld/mm_agents/opencua/opencua_agent.py:OpenCUAAgent"
        ),
        history_structure="self.observations/actions/cots in action_history mode",
        protocol_origin="upstream_opencua_action_history",
        renderer_path="recovery.adapters.opencua.OpenCUAActionHistoryAdapter",
        renderer_implemented=True,
        conformance_required=True,
        factory=OpenCUAActionHistoryAdapter,
    ),
    "gpt_5_5": NativeHistoryRegistration(
        agent_id="gpt_5_5",
        family="openai_computer_use",
        live_source=(
            "third_party/MyPCBench/agent-harness/agents/openai_cuabash.py:OpenAICUAToolsAgent"
        ),
        history_structure=(
            "stateless Responses input items: user primer, computer_call/computer_call_output, "
            "shell_call/shell_call_output (injected via recovery.mypcbench.openai_takeover)"
        ),
        protocol_origin="openai_responses_builtin_computer_and_shell",
        renderer_path="recovery.adapters.openai_cua.OpenAIResponsesHistoryAdapter",
        renderer_implemented=True,
        conformance_required=True,
        factory=OpenAIResponsesHistoryAdapter,
    ),
    "claude_opus_4_8": NativeHistoryRegistration(
        agent_id="claude_opus_4_8",
        family="anthropic_computer_use",
        live_source=(
            "third_party/MyPCBench/agent-harness/agents/claude_cuabash.py:ClaudeCUAAgent"
        ),
        history_structure=(
            "Anthropic Messages assistant tool_use plus user tool_result blocks, with "
            "provider computer/bash/text-editor tools"
        ),
        protocol_origin="anthropic_messages_computer_use_20251124",
        renderer_path="recovery.adapters.claude.ClaudeNativeHistoryAdapter",
        renderer_implemented=True,
        conformance_required=True,
        factory=ClaudeNativeHistoryAdapter,
    ),
}


def get_registration(agent_id: str) -> NativeHistoryRegistration:
    try:
        return _REGISTRY[agent_id]
    except KeyError as exc:
        raise NativeHistoryUnavailableError("unregistered target agent: %s" % agent_id) from exc


def create_native_history_adapter(
    agent_id: str, instruction: str, **kwargs: Any
) -> AgentAdapter:
    return get_registration(agent_id).create(instruction, **kwargs)


def registry_status() -> Tuple[Dict[str, object], ...]:
    return tuple(_REGISTRY[agent_id].status_dict() for agent_id in TARGET_AGENT_IDS)
