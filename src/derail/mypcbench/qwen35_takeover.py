"""Exact state injection wrapper for MyPCBench's vendored Qwen 3.5 agent."""

from __future__ import annotations

import base64
import importlib
import urllib.parse
from pathlib import Path
from typing import Any, Mapping, Sequence

from derail.mypcbench.tool_agent import takeover_condition_prompt


def _image_bytes(value: str) -> bytes:
    if value.startswith("data:image/"):
        try:
            return base64.b64decode(value.split(",", 1)[1], validate=True)
        except (IndexError, ValueError) as exc:
            raise ValueError("invalid Qwen 3.5 history image data URL") from exc
    parsed = urllib.parse.urlparse(value) if value.startswith("file://") else None
    path = Path(urllib.parse.unquote(parsed.path)) if parsed else Path(value)
    if not path.is_file():
        raise ValueError(f"Qwen 3.5 history screenshot does not exist: {value}")
    return path.read_bytes()


class Qwen35TakeoverTarget:
    """Delegate normal behavior while restoring Qwen35VLAgent's public state."""

    def __init__(self, target: Any) -> None:
        inner = getattr(target, "_inner", None)
        if inner is None or not callable(getattr(inner, "restore_state", None)):
            raise TypeError("Qwen 3.5 target does not expose restore_state")
        self._target = target
        self._inner = inner
        self._instruction = ""
        self._condition_prompt = ""

    def __getattr__(self, name: str) -> Any:
        return getattr(self._target, name)

    def reset(self, *args: Any, **kwargs: Any) -> None:
        self._target.reset(*args, **kwargs)
        self._instruction = ""
        self._condition_prompt = ""

    def _process_screenshots(self, records: Sequence[Mapping[str, Any]]) -> list[str]:
        process_image = None
        for cls in self._inner.__class__.__mro__:
            module = importlib.import_module(cls.__module__)
            candidate = getattr(module, "process_image", None)
            if callable(candidate):
                process_image = candidate
                break
        if not callable(process_image):
            raise RuntimeError("Qwen 3.5 live class hierarchy has no process_image function")
        return [
            process_image(
                _image_bytes(str(record["observation_image_url"])),
                min_pixels=self._inner.min_pixels,
                max_pixels=self._inner.max_pixels,
            )
            for record in records
        ]

    @staticmethod
    def _restored_state(
        records: Sequence[Mapping[str, Any]], screenshots: Sequence[str]
    ) -> dict[str, Any]:
        return {
            "thoughts": [],
            "actions": [str(record["action"]) for record in records],
            "observations": [],
            "responses": [str(record["response"]) for record in records],
            "reasonings": [str(record.get("reasoning", "")) for record in records],
            "screenshots": list(screenshots),
            "folded_prefix_k": 0,
            "_last_reasoning": "",
        }

    def seed_native_history(
        self,
        instruction: str,
        native_history: Sequence[Mapping[str, Any]],
        **kwargs: Any,
    ) -> None:
        records = [dict(record) for record in native_history]
        if not instruction.strip() or not records:
            raise ValueError("Qwen 3.5 takeover instruction and history must be non-empty")
        if any(record.get("role") != "qwen35_state" for record in records):
            raise ValueError("Qwen 3.5 history contains a non-native state record")
        screenshots = self._process_screenshots(records)
        self._inner.restore_state(self._restored_state(records, screenshots))
        self._inner._update_folding_state(len(screenshots))
        self._instruction = instruction
        self._condition_prompt = takeover_condition_prompt(
            str(kwargs.get("condition", "unaware")),
            str(kwargs.get("diagnosis", "")),
            kwargs.get("root_cause_action_index"),
            str(kwargs.get("hint", "")),
        )

    def predict(self, instruction: str, observation: Mapping[str, Any]):
        if instruction != self._instruction:
            raise ValueError("predict instruction differs from seeded Qwen 3.5 task")
        effective = instruction
        if self._condition_prompt:
            effective = f"{instruction}\n\n{self._condition_prompt}"
        return self._target.predict(effective, observation)

    def preflight_next_request(
        self, instruction: str, observation: Mapping[str, Any]
    ) -> Mapping[str, Any]:
        """Run native request shaping and /tokenize without model completion."""

        base = self._inner.__class__.__mro__[1]
        original = base.call_llm

        def stop_before_completion(_inner: Any, _payload: Any, _model: str) -> str:
            return (
                "Action: Wait.\n<tool_call><function=computer_use>"
                "<parameter=action>wait</parameter><parameter=time>1</parameter>"
                "</function></tool_call>"
            )

        base.call_llm = stop_before_completion
        try:
            self.predict(instruction, observation)
        finally:
            base.call_llm = original
        metadata = getattr(self._inner, "last_step_metadata", None)
        if not isinstance(metadata, Mapping) or not metadata.get("max_model_len"):
            raise RuntimeError("Qwen 3.5 native token preflight emitted no context metadata")
        return dict(metadata)


def wrap_qwen35_takeover_target(target: Any, agent_id: str) -> Any:
    if agent_id != "qwen3_5_35b_a3b":
        return target
    return Qwen35TakeoverTarget(target)
