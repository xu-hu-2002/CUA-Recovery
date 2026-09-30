"""Native Anthropic Messages state injection for the vendored Claude agent."""

from __future__ import annotations

import copy
import base64
import io
from typing import Any, Mapping, Sequence

from derail.mypcbench.tool_agent import takeover_condition_prompt


class ClaudeTakeoverTarget:
    def __init__(self, target: Any) -> None:
        if not isinstance(getattr(target, "messages", None), list):
            raise TypeError("Claude target does not expose mutable native messages")
        self._target = target
        if not isinstance(getattr(target, "last_trajectory_tool_messages", None), list):
            target.last_trajectory_tool_messages = []
        if not isinstance(getattr(target, "agent_metadata", None), dict):
            target.agent_metadata = {}
        self._instruction = ""

    def __getattr__(self, name: str) -> Any:
        return getattr(self._target, name)

    @property
    def native_history_system_prompt(self) -> str:
        return str(self._target.system_prompt)

    def reset(self, *args: Any, **kwargs: Any) -> None:
        self._target.reset(*args, **kwargs)
        self._instruction = ""

    @staticmethod
    def _strip_null_tool_use_caller(messages: Sequence[Mapping[str, Any]]) -> None:
        """Drop recorded `tool_use.caller: null`, which the gateway rejects."""

        for message in messages:
            content = message.get("content")
            if not isinstance(content, list):
                continue
            for block in content:
                if (
                    isinstance(block, dict)
                    and block.get("type") == "tool_use"
                    and "caller" in block
                    and block["caller"] is None
                ):
                    del block["caller"]

    @staticmethod
    def _validate_turns(messages: Sequence[Mapping[str, Any]]) -> None:
        if not messages or messages[0].get("role") != "user":
            raise ValueError("Claude history must start with a user message")
        roles = [item.get("role") for item in messages]
        if roles != [
            "user" if index % 2 == 0 else "assistant" for index in range(len(messages))
        ]:
            raise ValueError("Claude history must alternate complete user/assistant turns")

    def seed_native_history(
        self,
        instruction: str,
        native_history: Sequence[Mapping[str, Any]],
        **kwargs: Any,
    ) -> None:
        frozen = copy.deepcopy([dict(item) for item in native_history])
        self._strip_null_tool_use_caller(frozen)
        system = frozen.pop(0) if frozen else {}
        if system != {"role": "system", "content": self.native_history_system_prompt}:
            raise ValueError("Claude history system prompt does not match the live agent")
        self._validate_turns(frozen)
        prompt = takeover_condition_prompt(
            str(kwargs.get("condition", "unaware")),
            str(kwargs.get("diagnosis", "")),
            kwargs.get("root_cause_action_index"),
            str(kwargs.get("hint", "")),
        )
        if frozen[-1].get("role") == "assistant":
            frozen.append({"role": "user", "content": []})
        if prompt:
            content = frozen[-1].get("content")
            if not isinstance(content, list):
                raise ValueError("Claude result message content must be an array")
            content.append({"type": "text", "text": prompt})
        self._target.messages = frozen
        self._target.actions_log = []
        self._instruction = instruction

    def predict(self, instruction: str, observation: Mapping[str, Any]):
        if instruction != self._instruction:
            raise ValueError("predict instruction differs from seeded Claude task")
        return self._target.predict(instruction, observation)

    @staticmethod
    def _screenshot_base64(screenshot: bytes) -> str:
        from PIL import Image

        with Image.open(io.BytesIO(screenshot)) as image:
            resized = image.convert("RGB").resize((1280, 720), Image.Resampling.LANCZOS)
            buffer = io.BytesIO()
            resized.save(buffer, format="PNG")
        return base64.b64encode(buffer.getvalue()).decode("ascii")

    @staticmethod
    def _has_image(content: Sequence[Mapping[str, Any]]) -> bool:
        for block in content:
            if block.get("type") == "image":
                return True
            nested = block.get("content") if block.get("type") == "tool_result" else None
            if isinstance(nested, list) and any(
                isinstance(item, Mapping) and item.get("type") == "image" for item in nested
            ):
                return True
        return False

    def preflight_next_request(
        self, instruction: str, observation: Mapping[str, Any]
    ) -> dict[str, Any]:
        """Build the exact live Anthropic payload without calling the model."""

        if instruction != self._instruction:
            raise ValueError("preflight instruction differs from seeded Claude task")
        screenshot = observation.get("screenshot")
        if not isinstance(screenshot, bytes):
            raise ValueError("Claude preflight requires screenshot bytes")
        encoded = self._screenshot_base64(screenshot)
        self._target._resolve_pending_tool_uses(encoded)
        content = self._target.messages[-1].get("content", [])
        if isinstance(content, list) and not self._has_image(content):
            content.append({
                "type": "image",
                "source": {"type": "base64", "media_type": "image/png", "data": encoded},
            })
        self._target._trim_images()
        payload = {
            "model": self._target.model,
            "max_tokens": self._target.max_tokens,
            "tools": copy.deepcopy(self._target.tools),
            "messages": copy.deepcopy(self._target.messages),
            "system": [{"type": "text", "text": self._target.system_prompt}],
        }
        if self._target.betas:
            payload["betas"] = list(self._target.betas)
        return payload


def wrap_claude_takeover_target(target: Any, agent_id: str) -> Any:
    if agent_id != "claude_opus_4_8":
        return target
    return ClaudeTakeoverTarget(target)
