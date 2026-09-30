"""Lossless renderer for recorded Anthropic Messages takeover history."""

from __future__ import annotations

import copy
import base64
import io
import json
from pathlib import Path
from typing import Any, Dict, List, Mapping

from .base import AgentCapabilities, HistoryStep
from .kimi_k3 import KimiK3ScaffoldAdapter


class ClaudeNativeHistoryAdapter:
    """Restore recorded Claude turns; canonical actions are used only for replay."""

    capabilities = AgentCapabilities(
        agent_id="claude_opus_4_8",
        action_kinds=KimiK3ScaffoldAdapter.capabilities.action_kinds,
        coordinate_protocol="anthropic_computer_1280x720",
        history_format="anthropic_recorded_messages",
        silent_action_kinds=frozenset({"terminate"}),
    )

    def __init__(self, instruction: str, *, system_prompt: str) -> None:
        if not instruction.strip() or not system_prompt.strip():
            raise ValueError("Claude renderer requires instruction and live system prompt")
        self.instruction = instruction
        self._system_prompt = system_prompt
        self._initial_image = ""
        self._opened = False

    def system_prompt(self) -> str:
        return self._system_prompt

    @staticmethod
    def _trajectory_record(step: HistoryStep) -> Mapping[str, Any]:
        try:
            record = json.loads(step.trajectory_log)
        except json.JSONDecodeError as exc:
            raise ValueError("Claude trajectory log is not valid JSON") from exc
        if not isinstance(record, Mapping):
            raise ValueError("Claude source row has no recorded Anthropic native turn")
        return record

    @staticmethod
    def _tool_ids(message: Mapping[str, Any], block_type: str, id_key: str) -> List[str]:
        content = message.get("content")
        if not isinstance(content, list):
            raise ValueError("Claude native message content must be an array")
        return [
            str(block[id_key])
            for block in content
            if isinstance(block, Mapping) and block.get("type") == block_type
        ]

    @staticmethod
    def _replay_image(path_text: str) -> Dict[str, Any]:
        from PIL import Image

        if path_text.startswith("data:image/") and "," in path_text:
            image_source: Any = io.BytesIO(base64.b64decode(path_text.split(",", 1)[1]))
        else:
            path = Path(path_text)
            if not path.is_file():
                raise ValueError(f"Claude replay screenshot does not exist: {path}")
            image_source = path
        with Image.open(image_source) as image:
            resized = image.convert("RGB").resize((1280, 720), Image.Resampling.LANCZOS)
            buffer = io.BytesIO()
            resized.save(buffer, format="PNG")
        return {
            "type": "image",
            "source": {
                "type": "base64",
                "media_type": "image/png",
                "data": base64.b64encode(buffer.getvalue()).decode("ascii"),
            },
        }

    @classmethod
    def _is_stripped_image(cls, block: Any) -> bool:
        return (
            isinstance(block, Mapping)
            and block.get("type") == "image"
            and not isinstance(block.get("source"), Mapping)
        )

    @classmethod
    def _is_stripped_placeholder(cls, block: Any) -> bool:
        return cls._is_stripped_image(block) or (
            isinstance(block, Mapping)
            and block.get("type") == "text"
            and str(block.get("text", "")).strip()
            in {"[image removed]", "<stripped:base64 image>"}
        )

    @classmethod
    def _restore_removed_image(cls, message: Dict[str, Any], path_text: str) -> None:
        content = message.get("content")
        if not isinstance(content, list):
            raise ValueError("Claude native message content must be an array")
        replacement = cls._replay_image(path_text)
        restored = False
        for index, block in enumerate(content):
            if cls._is_stripped_placeholder(block):
                content[index] = copy.deepcopy(replacement)
                restored = True
            elif isinstance(block, dict) and block.get("type") == "tool_result":
                nested = block.get("content")
                if not isinstance(nested, list):
                    continue
                for nested_index, item in enumerate(nested):
                    if cls._is_stripped_placeholder(item):
                        nested[nested_index] = copy.deepcopy(replacement)
                        restored = True
        if not restored:
            content.append(replacement)

    @classmethod
    def _pending_action_result(
        cls, tool_ids: List[str], after_image: str
    ) -> Dict[str, Any]:
        blocks: List[Dict[str, Any]] = []
        for index, tool_id in enumerate(tool_ids):
            content: List[Dict[str, Any]] = [{"type": "text", "text": "Action executed."}]
            if index == len(tool_ids) - 1:
                content.append(cls._replay_image(after_image))
            blocks.append(
                {
                    "type": "tool_result",
                    "tool_use_id": tool_id,
                    "content": content,
                    "is_error": False,
                }
            )
        return {"role": "user", "content": blocks}

    def _opening_user(self, turn: Mapping[str, Any]) -> Any:
        if self._opened:
            return None
        self._opened = True
        return turn.get("initial_user") or turn.get("task_user")

    def render_step(self, step: HistoryStep) -> List[Dict[str, Any]]:
        if not self._initial_image:
            self._initial_image = step.observation_image_url
        record = self._trajectory_record(step)
        if record.get("control_flow_row") is True:
            return []
        turn = record.get("anthropic_native_turn")
        if not isinstance(turn, Mapping):
            raise ValueError("Claude source row has no recorded Anthropic native turn")
        if not turn.get("is_group_end"):
            return []
        assistant = copy.deepcopy(turn.get("assistant"))
        result_user = copy.deepcopy(turn.get("result_user"))
        if not isinstance(assistant, dict) or assistant.get("role") != "assistant":
            raise ValueError("Claude native turn has no assistant message")
        tool_ids = self._tool_ids(assistant, "tool_use", "id")
        if not tool_ids:
            if result_user is not None:
                raise ValueError(
                    "Claude tool-free assistant turn unexpectedly has a result message"
                )
            messages: List[Dict[str, Any]] = []
            opening = self._opening_user(turn)
            if opening is not None:
                initial_user = copy.deepcopy(opening)
                if not isinstance(initial_user, dict) or initial_user.get("role") != "user":
                    raise ValueError("Claude first native turn has no initial user message")
                self._restore_removed_image(initial_user, self._initial_image)
                messages.extend(
                    [{"role": "system", "content": self._system_prompt}, initial_user]
                )
            messages.append(assistant)
            return messages

        if result_user is not None:
            if not isinstance(result_user, dict) or result_user.get("role") != "user":
                raise ValueError("Claude native tool turn has no following result message")
            result_ids = self._tool_ids(result_user, "tool_result", "tool_use_id")
            if result_ids != tool_ids or len(set(tool_ids)) != len(tool_ids):
                raise ValueError("Claude native tool_use/tool_result IDs do not match exactly")

        messages = []
        opening = self._opening_user(turn)
        if opening is not None:
            initial_user = copy.deepcopy(opening)
            if not isinstance(initial_user, dict) or initial_user.get("role") != "user":
                raise ValueError("Claude first native turn has no initial user message")
            self._restore_removed_image(initial_user, self._initial_image)
            messages.extend(
                [{"role": "system", "content": self._system_prompt}, initial_user]
            )
        after_image = step.observation_after_image_url
        if not after_image:
            raise ValueError("Claude native result requires a replay post-action screenshot")
        if result_user is None:
            result_user = self._pending_action_result(tool_ids, after_image)
        else:
            self._restore_removed_image(result_user, after_image)
        messages.extend([assistant, result_user])
        return messages

    def tool_definitions(self) -> List[Dict[str, Any]]:
        return []
