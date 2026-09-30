"""Canonical action 与各 agent 原生历史格式之间的无损转换。"""

from .base import ActionNotSupportedError, AgentCapabilities, HistoryStep
from .claude import ClaudeNativeHistoryAdapter
from .holo31 import Holo31Adapter
from .openai_cua import OpenAIResponsesHistoryAdapter
from .opencua import OpenCUAActionHistoryAdapter
from .qwen35 import Qwen35StateAdapter
from .qwen36 import Qwen36ScaffoldAdapter
from .qwen38 import Qwen38ScaffoldAdapter
from .registry import (
    TARGET_AGENT_IDS,
    NativeHistoryUnavailableError,
    create_native_history_adapter,
    get_registration,
    registry_status,
)

__all__ = [
    "ActionNotSupportedError",
    "AgentCapabilities",
    "ClaudeNativeHistoryAdapter",
    "HistoryStep",
    "Holo31Adapter",
    "NativeHistoryUnavailableError",
    "OpenAIResponsesHistoryAdapter",
    "OpenCUAActionHistoryAdapter",
    "Qwen35StateAdapter",
    "Qwen36ScaffoldAdapter",
    "Qwen38ScaffoldAdapter",
    "TARGET_AGENT_IDS",
    "create_native_history_adapter",
    "get_registration",
    "registry_status",
]
