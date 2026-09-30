"""History renderer for the frozen DERAIL Qwen3.6 general-VLM scaffold."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Tuple

from derail.mypcbench.tool_agent import build_computer_tools

from .base import AgentCapabilities
from .holo31 import Holo31Adapter


class Qwen36ScaffoldAdapter(Holo31Adapter):
    _REPO_ROOT = Path(__file__).resolve().parents[3]
    _SYSTEM_PROMPT_PATH = (
        _REPO_ROOT / "prompts" / "agents" / "qwen3_6_27b_mypcbench_system.txt"
    )
    capabilities = AgentCapabilities(
        agent_id="qwen3_6_27b",
        action_kinds=Holo31Adapter.capabilities.action_kinds,
        coordinate_protocol="absolute_pixels_1280x800",
        history_format="derail_openai_compatible_function_calls",
    )

    @staticmethod
    def _xy(x_px: int, y_px: int, width: int, height: int) -> Tuple[int, int]:
        if (width, height) != (1280, 800):
            raise ValueError(
                "Qwen3.6 scaffold 使用冻结绝对像素 1280x800，不能静默转换 %dx%d"
                % (width, height)
            )
        return x_px, y_px

    def tool_definitions(self) -> List[Dict[str, Any]]:
        """Return the exact absolute-pixel schema used by the live scaffold."""

        return build_computer_tools(1279, 799)

