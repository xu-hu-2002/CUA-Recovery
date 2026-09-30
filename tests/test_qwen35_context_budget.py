import os
import base64
import sys
import types
import unittest
from pathlib import Path
from unittest import mock
from io import BytesIO

from PIL import Image


HARNESS_ROOT = Path(__file__).resolve().parents[1] / "third_party/MyPCBench/agent-harness"
if str(HARNESS_ROOT) not in sys.path:
    sys.path.insert(0, str(HARNESS_ROOT))

from agents.qwen_cua import _Qwen35VLPatched  # noqa: E402


def _messages(turns: int):
    instruction = (
        "\nPlease generate the next move according to the UI screenshot, instruction "
        "and previous actions.\n\nInstruction: do the task\n\nPrevious actions:\nNone"
    )
    result = [{"role": "system", "content": [{"type": "text", "text": "system"}]}]
    for index in range(turns):
        if index == 0:
            content = [
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{index}"}},
                {"type": "text", "text": instruction},
            ]
        else:
            content = [
                {"type": "text", "text": "<tool_response>\n"},
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{index}"}},
                {"type": "text", "text": "\n</tool_response>"},
            ]
        result.append({"role": "user", "content": content})
        if index < turns - 1:
            result.append(
                {"role": "assistant", "content": [{"type": "text", "text": f"action {index + 1}"}]}
            )
    return result


def _agent(turns: int):
    agent = object.__new__(_Qwen35VLPatched)
    agent.screenshots = [f"shot-{index}" for index in range(turns)]
    agent.actions = [f"action-{index + 1}" for index in range(max(0, turns - 1))]
    agent.collapse_text = "This screenshot has been collapsed."
    agent.max_tokens = 200
    agent.enable_thinking = True
    agent.last_step_metadata = {}
    return agent


def _counter(base: int, image_tokens: int, assistant_tokens: int, max_model_len: int):
    def count(_self, messages, _model):
        images = sum(
            1
            for message in messages
            for part in (message.get("content") or [])
            if isinstance(part, dict) and part.get("type") == "image_url"
        )
        assistants = sum(message.get("role") == "assistant" for message in messages)
        return base + images * image_tokens + assistants * assistant_tokens, max_model_len

    return count


class Qwen35ContextBudgetTests(unittest.TestCase):
    def test_tokenizer_surrogate_preserves_dimensions_and_not_real_hash(self):
        image = Image.new("RGB", (64, 32), (1, 2, 3))
        buffer = BytesIO()
        image.save(buffer, format="PNG")
        original_url = "data:image/png;base64," + base64.b64encode(
            buffer.getvalue()
        ).decode("ascii")
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": original_url}}
                ],
            }
        ]
        surrogate = _Qwen35VLPatched._tokenizer_surrogate_messages(messages)
        surrogate_url = surrogate[0]["content"][0]["image_url"]["url"]
        self.assertNotEqual(surrogate_url, original_url)
        self.assertEqual(messages[0]["content"][0]["image_url"]["url"], original_url)
        with Image.open(BytesIO(base64.b64decode(surrogate_url.split(",", 1)[1]))) as result:
            self.assertEqual(result.size, (64, 32))

    def test_oldest_images_are_collapsed_and_current_image_is_preserved(self):
        agent = _agent(4)
        agent._tokenize_messages = types.MethodType(_counter(100, 100, 40, 500), agent)
        fitted, metadata = agent._fit_messages_to_context(_messages(4), "model", 150)
        self.assertEqual(metadata["collapsed_step_ids"], [1, 2, 3])
        self.assertEqual(metadata["dropped_turn_ids"], [])
        self.assertEqual(metadata["final_prompt_tokens"], 320)
        images = [
            part
            for message in fitted
            for part in (message.get("content") or [])
            if isinstance(part, dict) and part.get("type") == "image_url"
        ]
        self.assertEqual(len(images), 1)
        self.assertTrue(images[0]["image_url"]["url"].endswith(",3"))

    def test_complete_oldest_pairs_are_dropped_after_image_folding(self):
        agent = _agent(4)
        agent._tokenize_messages = types.MethodType(_counter(300, 50, 100, 650), agent)
        fitted, metadata = agent._fit_messages_to_context(_messages(4), "model", 200)
        self.assertEqual(metadata["dropped_turn_ids"], [1, 2])
        self.assertEqual(metadata["retained_start_step"], 3)
        first_user = next(message for message in fitted if message["role"] == "user")
        first_text = "".join(
            part.get("text", "") for part in first_user["content"] if part.get("type") == "text"
        )
        self.assertIn("Instruction: do the task", first_text)
        self.assertIn("Step 1: action-1", first_text)
        self.assertIn("Step 2: action-2", first_text)
        self.assertEqual(
            [message["role"] for message in fitted],
            ["system", "user", "assistant", "user"],
        )

    def test_under_budget_request_is_byte_equivalent(self):
        agent = _agent(4)
        agent._tokenize_messages = types.MethodType(_counter(100, 100, 40, 1000), agent)
        original = _messages(4)
        fitted, metadata = agent._fit_messages_to_context(original, "model", 10)
        self.assertEqual(fitted, original)
        self.assertEqual(metadata["status"], "not_needed")
        self.assertEqual(
            metadata["original_messages_sha256"], metadata["final_messages_sha256"]
        )

    def test_minimal_context_overflow_fails_without_touching_current_image(self):
        agent = _agent(1)
        agent._tokenize_messages = types.MethodType(_counter(500, 100, 0, 500), agent)
        with self.assertRaisesRegex(RuntimeError, "current screenshot exceed"):
            agent._fit_messages_to_context(_messages(1), "model", 200)
        self.assertEqual(agent.last_step_metadata["status"], "error")

    def test_tokenizer_failure_is_hard_failure_with_metadata(self):
        agent = _agent(1)

        def fail(_self, _messages_arg, _model):
            raise RuntimeError("tokenizer unavailable")

        agent._tokenize_messages = types.MethodType(fail, agent)
        with mock.patch.dict(
            os.environ,
            {"MYPCBENCH_QWEN_CONTEXT_POLICY": "tokenize_oldest_first_v1"},
            clear=False,
        ):
            with self.assertRaisesRegex(RuntimeError, "tokenizer unavailable"):
                agent.call_llm(
                    {"messages": _messages(1), "max_tokens": 200}, "model"
                )
        self.assertEqual(agent.last_step_metadata["status"], "error")
        self.assertNotIn("api_key", agent.last_step_metadata)


if __name__ == "__main__":
    unittest.main()
