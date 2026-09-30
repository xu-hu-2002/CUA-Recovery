"""Truthful registry for target-agent native-history support."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Callable, Dict, Optional, Tuple

from .base import AgentAdapter
from .claude import ClaudeNativeHistoryAdapter
from .evocua import EvoCUAS2Adapter
from .holo31 import Holo31Adapter
from .kimi_k3 import KimiK3ScaffoldAdapter
from .openai_cua import OpenAIResponsesHistoryAdapter
from .opencua import OpenCUAActionHistoryAdapter
from .qwen35 import Qwen35StateAdapter
from .qwen36 import Qwen36ScaffoldAdapter
from .qwen38 import Qwen38ScaffoldAdapter

TARGET_AGENT_IDS: Tuple[str, ...] = (
    "qwen3_5_35b_a3b",
    "evocua_32b",
    "qwen3_6_27b",
    "qwen3_8_27b",
    "holo_3_1_35b_a3b",
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
            raise ValueError("renderer_implemented 与 factory 必须一致")
        if self.renderer_implemented and self.blocker:
            raise ValueError("已实现 renderer 不能同时声明 blocker")
        if not self.renderer_implemented and not self.blocker:
            raise ValueError("未实现 renderer 必须记录具体 blocker")

    def create(self, instruction: str, **kwargs: Any) -> AgentAdapter:
        if not self.renderer_implemented or self.factory is None:
            raise NativeHistoryUnavailableError(
                "%s native-history unavailable: %s" % (self.agent_id, self.blocker)
            )
        if not instruction.strip():
            raise ValueError("native-history renderer 需要非空 task instruction")
        return self.factory(instruction, **kwargs)

    def status_dict(self) -> Dict[str, object]:
        data = asdict(self)
        data.pop("factory")
        data["release_ready_without_probe"] = False
        return data


_REGISTRY: Dict[str, NativeHistoryRegistration] = {
    "qwen3_5_35b_a3b": NativeHistoryRegistration(
        agent_id="qwen3_5_35b_a3b",
        family="qwen35_osworld_vendored",
        live_source="third_party/MyPCBench/agent-harness/agents/qwen_cua.py:QwenOSWorldAgent",
        history_structure="processed-image chat history with folding/context fitting",
        protocol_origin="vendored_paper_results",
        renderer_path="derail.adapters.qwen35.Qwen35StateAdapter",
        renderer_implemented=True,
        conformance_required=True,
        factory=Qwen35StateAdapter,
    ),
    "evocua_32b": NativeHistoryRegistration(
        agent_id="evocua_32b",
        family="evocua_s2",
        live_source="third_party/EvoCUA/mm_agents/evocua/evocua_agent.py:EvoCUAAgent",
        history_structure="S2 resized-image user turns plus raw XML/JSON assistant responses",
        protocol_origin="upstream_evocua_s2",
        renderer_path="derail.adapters.evocua.EvoCUAS2Adapter",
        renderer_implemented=True,
        conformance_required=True,
        factory=EvoCUAS2Adapter,
    ),
    "qwen3_6_27b": NativeHistoryRegistration(
        agent_id="qwen3_6_27b",
        family="qwen36_general_vlm_scaffold",
        live_source="src/derail/mypcbench/tool_agent.py:NativeToolComputerAgent",
        history_structure="OpenAI-compatible system/user/assistant-tool/tool messages",
        protocol_origin="derail_scaffold_not_native_qwen_cua",
        renderer_path="derail.adapters.qwen36.Qwen36ScaffoldAdapter",
        renderer_implemented=True,
        conformance_required=True,
        factory=Qwen36ScaffoldAdapter,
    ),
    "qwen3_8_27b": NativeHistoryRegistration(
        agent_id="qwen3_8_27b",
        family="qwen38_general_vlm_scaffold",
        live_source="src/derail/mypcbench/tool_agent.py:NativeToolComputerAgent",
        history_structure=(
            "OpenAI-compatible system/user/assistant-tool/tool messages with GUI and VM-shell "
            "calls/results"
        ),
        protocol_origin="derail_scaffold_not_native_qwen_cua",
        renderer_path="derail.adapters.qwen38.Qwen38ScaffoldAdapter",
        renderer_implemented=True,
        conformance_required=True,
        factory=Qwen38ScaffoldAdapter,
    ),
    "kimi_k3": NativeHistoryRegistration(
        agent_id="kimi_k3",
        family="kimi_general_vlm_scaffold",
        live_source="src/derail/mypcbench/tool_agent.py:NativeToolComputerAgent",
        history_structure="OpenAI-compatible system/user/assistant-tool/tool messages",
        protocol_origin="derail_scaffold_frozen_schema_via_routify",
        renderer_path="derail.adapters.kimi_k3.KimiK3ScaffoldAdapter",
        renderer_implemented=True,
        conformance_required=True,
        factory=KimiK3ScaffoldAdapter,
    ),
    "holo_3_1_35b_a3b": NativeHistoryRegistration(
        agent_id="holo_3_1_35b_a3b",
        family="holo31",
        live_source="src/derail/mypcbench/tool_agent.py:NativeToolComputerAgent",
        history_structure="OpenAI-compatible system/user/assistant-tool/tool messages",
        protocol_origin="frozen_derail_experimental_schema",
        renderer_path="derail.adapters.holo31.Holo31Adapter",
        renderer_implemented=True,
        conformance_required=True,
        factory=Holo31Adapter,
    ),
    "opencua_72b": NativeHistoryRegistration(
        agent_id="opencua_72b",
        family="opencua",
        live_source=(
            "third_party/OpenCUA-OSWorld/mm_agents/opencua/opencua_agent.py:OpenCUAAgent"
        ),
        history_structure="self.observations/actions/cots in action_history mode",
        protocol_origin="upstream_opencua_action_history",
        renderer_path="derail.adapters.opencua.OpenCUAActionHistoryAdapter",
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
            "shell_call/shell_call_output (injected via derail.mypcbench.openai_takeover)"
        ),
        protocol_origin="openai_responses_builtin_computer_and_shell",
        renderer_path="derail.adapters.openai_cua.OpenAIResponsesHistoryAdapter",
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
        renderer_path="derail.adapters.claude.ClaudeNativeHistoryAdapter",
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
