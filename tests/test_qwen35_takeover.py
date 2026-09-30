import base64
import json
import unittest
from types import SimpleNamespace

from derail.adapters.base import HistoryStep
from derail.adapters.qwen35 import Qwen35StateAdapter
from derail.canonical.actions import WaitAction
from derail.mypcbench.qwen35_takeover import Qwen35TakeoverTarget
from derail.takeover.history import build_native_history


def process_image(data, *, min_pixels, max_pixels):
    assert data == b"png"
    assert (min_pixels, max_pixels) == (1, 2)
    return base64.b64encode(data).decode("ascii")


class _Inner:
    min_pixels = 1
    max_pixels = 2

    def __init__(self):
        self.state = None

    def restore_state(self, state):
        self.state = state

    def _update_folding_state(self, count):
        self.state["folded_prefix_k"] = max(0, count - 1)


class _Target:
    def __init__(self):
        self._inner = _Inner()
        self.predicted_instruction = ""

    def reset(self, *_args, **_kwargs):
        pass

    def predict(self, instruction, observation):
        self.predicted_instruction = instruction
        return "response", ["WAIT"]


class _CompletionBase(_Inner):
    def call_llm(self, payload, model):
        raise AssertionError("completion request must be intercepted")


class _RequestShaper(_CompletionBase):
    def __init__(self):
        super().__init__()
        self.last_step_metadata = {}

    def call_llm(self, payload, model):
        self.last_step_metadata = {
            "status": "not_needed",
            "final_prompt_tokens": 123,
            "max_model_len": 32768,
        }
        return super().call_llm(payload, model)


class _PreflightTarget(_Target):
    def __init__(self):
        super().__init__()
        self._inner = _RequestShaper()

    def predict(self, instruction, observation):
        response = self._inner.call_llm({"messages": []}, "model")
        return response, ["WAIT"]


class Qwen35TakeoverTargetTests(unittest.TestCase):
    def test_seed_restores_public_state_and_folding(self):
        target = _Target()
        wrapped = Qwen35TakeoverTarget(target)
        image = "data:image/png;base64," + base64.b64encode(b"png").decode("ascii")
        wrapped.seed_native_history(
            "Do the task",
            [
                {
                    "role": "qwen35_state",
                    "observation_image_url": image,
                    "action": "Click Save.",
                    "response": "Action: Click Save.\n<tool_call>x</tool_call>",
                },
                {
                    "role": "qwen35_state",
                    "observation_image_url": image,
                    "action": "Wait.",
                    "response": "Action: Wait.\n<tool_call>y</tool_call>",
                },
            ],
            condition="unaware",
        )
        state = target._inner.state
        self.assertEqual(state["actions"], ["Click Save.", "Wait."])
        self.assertEqual(len(state["screenshots"]), 2)
        self.assertEqual(state["folded_prefix_k"], 1)
        self.assertEqual(wrapped.predict("Do the task", {}), ("response", ["WAIT"]))
        self.assertEqual(target.predicted_instruction, "Do the task")

    def test_seed_rejects_non_native_record(self):
        wrapped = Qwen35TakeoverTarget(_Target())
        with self.assertRaisesRegex(ValueError, "non-native"):
            wrapped.seed_native_history(
                "Do the task", [{"role": "assistant"}], condition="unaware"
            )

    def test_preflight_runs_request_shaping_but_intercepts_completion(self):
        target = _PreflightTarget()
        wrapped = Qwen35TakeoverTarget(target)
        wrapped._instruction = "Do the task"
        metadata = wrapped.preflight_next_request("Do the task", {"screenshot": b"png"})
        self.assertEqual(metadata["final_prompt_tokens"], 123)
        self.assertEqual(metadata["max_model_len"], 32768)
        with self.assertRaisesRegex(AssertionError, "must be intercepted"):
            target._inner.call_llm({}, "model")


class Qwen35NativeHistoryTests(unittest.TestCase):
    def test_history_starts_at_state_records_without_system(self):
        adapter = Qwen35StateAdapter(instruction="Do the task")
        step = HistoryStep(
            step_id=0,
            observation_image_url="artifact://screenshots/step-0.png",
            action=WaitAction(kind="wait", seconds=1.0),
            trajectory_log=json.dumps(
                {
                    "response": (
                        "Action: Wait.\n<tool_call><function=computer_use>"
                        "<parameter=action>wait</parameter>"
                        "<parameter=time>1</parameter></function></tool_call>"
                    ),
                    "reasoning": [],
                }
            ),
        )
        messages = build_native_history(adapter, (step,))
        self.assertEqual([message["role"] for message in messages], ["qwen35_state"])

    def test_headerless_undeclared_adapter_still_raises(self):
        class HeaderlessAdapter:
            capabilities = SimpleNamespace(
                require=lambda action: None, silent_action_kinds=frozenset()
            )

            def render_step(self, step):
                return [{"role": "custom_state"}]

        with self.assertRaisesRegex(ValueError, "omitted the initial system message"):
            build_native_history(
                HeaderlessAdapter(),
                (HistoryStep(step_id=0, observation_image_url="x", action=WaitAction(kind="wait", seconds=0.0)),),
            )


if __name__ == "__main__":
    unittest.main()
