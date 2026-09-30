"""Responses-API history injection for MyPCBench's ``openai_cuabash`` (GPT-5.5) agent."""

from __future__ import annotations

import base64
import copy
from datetime import datetime
from typing import Any, Mapping, Sequence

from recovery.mypcbench.tool_agent import NativeToolComputerAgent, takeover_condition_prompt


class OpenAITakeoverTarget:
    def __init__(self, target: Any, *, context_screenshots: int) -> None:
        if not isinstance(getattr(target, "_history", None), list):
            raise TypeError("GPT target does not expose a client-side Responses history")
        if context_screenshots <= 0:
            raise ValueError("context_screenshots must be positive")
        self._target = target
        target._zdr_stateless = True
        target._zdr_keep_images = context_screenshots
        self._instruction = ""
        self._condition_prompt = ""

    def __getattr__(self, name: str) -> Any:
        return getattr(self._target, name)

    @property
    def native_history_system_prompt(self) -> str:
        """The primer the live agent sends on its first call (openai_cuabash.predict)."""

        return self._target._operator_prompt.format(
            CLIENT_PASSWORD=self._target.client_password,
            CURRENT_DATE=datetime.today().strftime("%A, %B %d, %Y"),
        )

    def reset(self, *args: Any, **kwargs: Any) -> None:
        self._target.reset(*args, **kwargs)
        self._instruction = ""
        self._condition_prompt = ""

    @staticmethod
    def _resolve_images(items: list[dict[str, Any]]) -> None:
        resolve = NativeToolComputerAgent._resolve_history_image_url
        for item in items:
            if item.get("type") == "computer_call_output":
                item["output"]["image_url"] = resolve(str(item["output"]["image_url"]))
            for part in item.get("content") or []:
                if isinstance(part, dict) and part.get("type") == "input_image":
                    part["image_url"] = resolve(str(part["image_url"]))

    def seed_native_history(
        self,
        instruction: str,
        native_history: Sequence[Mapping[str, Any]],
        **kwargs: Any,
    ) -> None:
        items = copy.deepcopy([dict(item) for item in native_history])
        opening = items[0] if items else {}
        text = (opening.get("content") or [{}])[0].get("text", "")
        if opening.get("role") != "user" or not str(text).startswith(
            self.native_history_system_prompt
        ):
            raise ValueError("GPT history does not open with the live operator primer")
        calls = [item["call_id"] for item in items if item.get("type") in {"computer_call", "shell_call"}]
        outputs = [
            item["call_id"]
            for item in items
            if item.get("type") in {"computer_call_output", "shell_call_output"}
        ]
        if calls != outputs or len(set(calls)) != len(calls):
            raise ValueError("GPT history calls and outputs do not pair exactly")
        self._resolve_images(items)
        self._target._history = items
        self._target.previous_response_id = None
        self._target.pending_items.clear()
        self._target._last_computer_call_id = None
        self._target._pending_safety_checks = []
        self._instruction = instruction
        self._condition_prompt = takeover_condition_prompt(
            str(kwargs.get("condition", "unaware")),
            str(kwargs.get("diagnosis", "")),
            kwargs.get("root_cause_action_index"),
            str(kwargs.get("hint", "")),
        )

    def predict(self, instruction: str, observation: Mapping[str, Any]):
        if instruction != self._instruction:
            raise ValueError("predict instruction differs from seeded GPT task")
        screenshot = observation.get("screenshot")
        if self._condition_prompt and isinstance(screenshot, bytes):
            self._target.pending_items.append({
                "role": "user",
                "content": [
                    {"type": "input_text", "text": self._condition_prompt},
                    {"type": "input_image", "image_url": "data:image/png;base64,"
                     + base64.b64encode(screenshot).decode("ascii")},
                ],
            })
            self._condition_prompt = ""
        return self._target.predict(instruction, observation)


def wrap_openai_takeover_target(target: Any, agent_id: str, *, context_screenshots: int) -> Any:
    if agent_id != "gpt_5_5":
        return target
    return OpenAITakeoverTarget(target, context_screenshots=context_screenshots)
