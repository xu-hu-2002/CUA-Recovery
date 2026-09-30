"""Lossless GPT Responses state persistence contract."""

from __future__ import annotations

import tempfile
import unittest
from unittest import mock
from pathlib import Path

from agents.openai_cuabash import (
    OpenAIContinuationError,
    ScreenshotUnavailableError,
    _OpenAICUABase,
)


class NativeResponsesStateTests(unittest.TestCase):
    def test_missing_screenshot_is_recovered_without_dropping_call_id(self) -> None:
        class Env:
            def __init__(self) -> None:
                self.calls = 0

            def _get_obs(self):
                self.calls += 1
                return {"screenshot": b"png" if self.calls == 2 else None}

        agent = object.__new__(_OpenAICUABase)
        agent.env = Env()
        agent._last_computer_call_id = "call-1"
        agent._screenshot_retry_limit = 3
        with mock.patch("agents.openai_cuabash.time.sleep"):
            self.assertEqual(agent._recover_screenshot(None), b"png")
        self.assertEqual(agent._last_computer_call_id, "call-1")

    def test_missing_screenshot_exhaustion_is_explicit_infra_error(self) -> None:
        class Env:
            @staticmethod
            def _get_obs():
                return {"screenshot": None}

        agent = object.__new__(_OpenAICUABase)
        agent.env = Env()
        agent._last_computer_call_id = "call-1"
        agent._screenshot_retry_limit = 2
        with mock.patch("agents.openai_cuabash.time.sleep"):
            with self.assertRaises(ScreenshotUnavailableError):
                agent._recover_screenshot(None)

    def test_empty_output_continuation_is_bounded(self) -> None:
        agent = object.__new__(_OpenAICUABase)
        agent.pending_items = []
        agent._empty_output_retries = 0
        agent._empty_output_retry_limit = 1
        agent._queue_native_continue()
        self.assertEqual(agent.pending_items[0]["role"], "user")
        with self.assertRaises(OpenAIContinuationError):
            agent._queue_native_continue()

    def test_state_preserves_encrypted_items_and_has_stable_hashes(self) -> None:
        agent = object.__new__(_OpenAICUABase)
        agent._zdr_stateless = True
        agent._history = [{"type": "reasoning", "encrypted_content": "cipher"}]
        agent._response_audit = [{"request_sha256": "a" * 64}]

        state = agent.native_responses_state
        self.assertEqual(state["history"][0]["encrypted_content"], "cipher")
        self.assertEqual(
            agent._payload_sha256({"b": 1, "a": 2}),
            agent._payload_sha256({"a": 2, "b": 1}),
        )

    def test_runner_checkpoints_full_state_without_stripping_images(self) -> None:
        from run_mypcbench import _persist_native_responses_state

        agent = object.__new__(_OpenAICUABase)
        agent._zdr_stateless = True
        agent._history = [
            {
                "type": "computer_call_output",
                "call_id": "call-1",
                "output": {"image_url": "data:image/png;base64,exact"},
            }
        ]
        agent._response_audit = []
        with tempfile.TemporaryDirectory() as directory:
            _persist_native_responses_state(agent, directory)
            payload = Path(directory, "native_responses_state.json").read_text()
        self.assertIn("data:image/png;base64,exact", payload)
        self.assertIn("call-1", payload)


if __name__ == "__main__":
    unittest.main()
