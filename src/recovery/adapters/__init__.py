"""Lossless conversion between canonical actions and native agent history formats."""

from .base import ActionNotSupportedError, AgentCapabilities, HistoryStep
from .claude import ClaudeNativeHistoryAdapter
from .openai_cua import OpenAIResponsesHistoryAdapter
from .opencua import OpenCUAActionHistoryAdapter
from .qwen35 import Qwen35StateAdapter
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
    "NativeHistoryUnavailableError",
    "OpenAIResponsesHistoryAdapter",
    "OpenCUAActionHistoryAdapter",
    "Qwen35StateAdapter",
    "TARGET_AGENT_IDS",
    "create_native_history_adapter",
    "get_registration",
    "registry_status",
]
