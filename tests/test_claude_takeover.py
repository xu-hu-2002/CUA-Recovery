import json
import tempfile
import unittest
from pathlib import Path

from PIL import Image

from derail.adapters.claude import ClaudeNativeHistoryAdapter
from derail.canonical.actions import ShellAction, TerminateAction
from derail.canonical.trajectory import CanonicalStep
from derail.mypcbench.claude_takeover import ClaudeTakeoverTarget
from derail.adapters.base import HistoryStep
from derail.takeover.history import build_native_history
from derail.takeover.source_logs import _claude_control_row, _claude_native_turn


def _png(path: Path) -> None:
    Image.new("RGB", (1280, 800), "white").save(path)


class ClaudeNativeHistoryTests(unittest.TestCase):
    def test_row_alignment_counts_bash_as_one_and_computer_calls_individually(self) -> None:
        messages = [
            {"role": "user", "content": []},
            {"role": "assistant", "content": [
                {"type": "tool_use", "id": "b1", "name": "bash"},
                {"type": "tool_use", "id": "b2", "name": "bash"},
            ]},
            {"role": "user", "content": []},
            {"role": "assistant", "content": [
                {"type": "tool_use", "id": "c1", "name": "computer"},
                {"type": "tool_use", "id": "c2", "name": "computer"},
            ]},
            {"role": "user", "content": []},
        ]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "messages.json"
            path.write_text(json.dumps(messages), encoding="utf-8")
            self.assertTrue(_claude_native_turn(path, 1)["is_group_end"])
            self.assertFalse(_claude_native_turn(path, 2)["is_group_end"])
            self.assertTrue(_claude_native_turn(path, 3)["is_group_end"])

    def test_preserves_recorded_ids_and_restores_replay_images(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            before, after = Path(directory) / "before.png", Path(directory) / "after.png"
            _png(before)
            _png(after)
            turn = {
                "is_group_end": True,
                "initial_user": {"role": "user", "content": [
                    {"type": "text", "text": "Task: do it"},
                    {"type": "text", "text": "[image removed]"},
                ]},
                "assistant": {"role": "assistant", "content": [
                    {"type": "thinking", "thinking": "recorded", "signature": "sig"},
                    {"type": "tool_use", "id": "tool-1", "name": "bash", "input": {"command": "pwd"}},
                ]},
                "result_user": {"role": "user", "content": [
                    {"type": "tool_result", "tool_use_id": "tool-1", "content": [{"type": "text", "text": "/tmp"}], "is_error": False},
                    {"type": "text", "text": "[image removed]"},
                ]},
            }
            messages = ClaudeNativeHistoryAdapter("do it", system_prompt="live").render_step(
                HistoryStep(
                    step_id=0,
                    observation_image_url=str(before),
                    observation_after_image_url=str(after),
                    action=ShellAction(kind="shell", commands=("pwd",)),
                    trajectory_log=json.dumps({"anthropic_native_turn": turn}),
                )
            )
        self.assertEqual(messages[2]["content"][1]["id"], "tool-1")
        self.assertEqual(messages[2]["content"][0]["signature"], "sig")
        self.assertEqual(messages[1]["content"][1]["type"], "image")
        self.assertEqual(messages[3]["content"][1]["type"], "image")

    def test_restores_image_blocks_whose_source_was_stripped_to_a_string(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            before, after = Path(directory) / "before.png", Path(directory) / "after.png"
            _png(before)
            _png(after)
            turn = {
                "is_group_end": True,
                "initial_user": {"role": "user", "content": [
                    {"type": "text", "text": "Task: do it"},
                    {"type": "image", "source": "<stripped:base64 image>"},
                ]},
                "assistant": {"role": "assistant", "content": [
                    {"type": "tool_use", "id": "tool-1", "name": "computer", "input": {"action": "screenshot"}},
                ]},
                "result_user": {"role": "user", "content": [
                    {"type": "tool_result", "tool_use_id": "tool-1", "content": [
                        {"type": "text", "text": "ok"},
                        {"type": "image", "source": "<stripped:base64 image>"},
                    ], "is_error": False},
                ]},
            }
            messages = ClaudeNativeHistoryAdapter("do it", system_prompt="live").render_step(
                HistoryStep(
                    step_id=0,
                    observation_image_url=str(before),
                    observation_after_image_url=str(after),
                    action=ShellAction(kind="shell", commands=("pwd",)),
                    trajectory_log=json.dumps({"anthropic_native_turn": turn}),
                )
            )
        for image in (messages[1]["content"][1], messages[3]["content"][0]["content"][1]):
            self.assertEqual(image["type"], "image")
            self.assertIsInstance(image["source"], dict)
            self.assertEqual(image["source"]["type"], "base64")
            self.assertTrue(image["source"]["data"])

    def test_seeding_strips_null_tool_use_caller_without_touching_input(self) -> None:
        class Target:
            system_prompt = "live"
            messages = []
            actions_log = []

            def reset(self, *_args): self.messages = []

        inner = Target()
        target = ClaudeTakeoverTarget(inner)
        history = [
            {"role": "system", "content": "live"},
            {"role": "user", "content": [{"type": "text", "text": "Task: do it"}]},
            {"role": "assistant", "content": [
                {"type": "tool_use", "id": "a", "name": "bash", "input": {"command": "pwd"}, "caller": None},
                {"type": "tool_use", "id": "b", "name": "bash", "input": {"command": "ls"}, "caller": {"type": "direct"}},
            ]},
            {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "a"}]},
        ]
        target.seed_native_history("do it", history)
        first, second = inner.messages[1]["content"]
        self.assertNotIn("caller", first)
        self.assertEqual(second["caller"], {"type": "direct"})
        self.assertEqual([block["id"] for block in inner.messages[1]["content"]], ["a", "b"])
        self.assertIsNone(history[2]["content"][0]["caller"])

    def test_rejects_orphan_tool_result(self) -> None:
        adapter = ClaudeNativeHistoryAdapter("do it", system_prompt="live")
        turn = {
            "is_group_end": True,
            "initial_user": {"role": "user", "content": []},
            "assistant": {"role": "assistant", "content": [{"type": "tool_use", "id": "a"}]},
            "result_user": {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "b"}]},
        }
        with self.assertRaisesRegex(ValueError, "do not match"):
            adapter.render_step(HistoryStep(
                step_id=0, observation_image_url="missing", observation_after_image_url="missing",
                action=ShellAction(kind="shell", commands=("pwd",)),
                trajectory_log=json.dumps({"anthropic_native_turn": turn}),
            ))

    def test_preserves_tool_free_terminal_turn_for_live_continuation(self) -> None:
        turn = {
            "is_group_end": True,
            "initial_user": {"role": "user", "content": []},
            "assistant": {
                "role": "assistant",
                "content": [{"type": "text", "text": "The task is complete."}],
            },
            "result_user": None,
        }
        messages = ClaudeNativeHistoryAdapter(
            "do it", system_prompt="live"
        ).render_step(HistoryStep(
            step_id=0,
            observation_image_url="data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk+A8AAQUBAScY42YAAAAASUVORK5CYII=",
            observation_after_image_url="",
            action=ShellAction(kind="shell", commands=("pwd",)),
            trajectory_log=json.dumps({"anthropic_native_turn": turn}),
        ))
        self.assertEqual(messages[-1], turn["assistant"])


class ClaudeTakeoverTargetTests(unittest.TestCase):
    def test_initializes_fields_required_by_upstream_reset(self) -> None:
        class Target:
            system_prompt = "live"
            messages = []
            actions_log = []

            def reset(self, *_args):
                self.messages.clear()
                self.actions_log.clear()
                self.last_trajectory_tool_messages.clear()
                self.agent_metadata.clear()

        inner = Target()
        target = ClaudeTakeoverTarget(inner)
        target.reset()
        self.assertEqual(inner.last_trajectory_tool_messages, [])
        self.assertEqual(inner.agent_metadata, {})

    def test_seeds_messages_and_injects_notified_prompt(self) -> None:
        class Target:
            system_prompt = "live"
            messages = []
            actions_log = ["old"]
            def reset(self, *_args): self.messages = []
            def predict(self, instruction, observation): return instruction, observation

        inner = Target()
        target = ClaudeTakeoverTarget(inner)
        target.seed_native_history("do it", [
            {"role": "system", "content": "live"},
            {"role": "user", "content": [{"type": "text", "text": "Task: do it"}]},
            {"role": "assistant", "content": [{"type": "tool_use", "id": "a"}]},
            {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "a"}]},
        ], condition="notified")
        self.assertIn("Takeover notice:", inner.messages[-1]["content"][-1]["text"])
        self.assertEqual(inner.actions_log, [])

    def test_preflight_builds_native_anthropic_payload_without_api_call(self) -> None:
        class Target:
            system_prompt = "live"
            model = "claude-opus-4-8"
            max_tokens = 4096
            tools = [{"type": "computer_20251124", "name": "computer"}]
            betas = ["computer-use-2025-11-24"]
            messages = []
            actions_log = []

            def reset(self, *_args): self.messages = []
            def _resolve_pending_tool_uses(self, _image): pass
            def _trim_images(self): pass

        inner = Target()
        target = ClaudeTakeoverTarget(inner)
        target.seed_native_history("do it", [
            {"role": "system", "content": "live"},
            {"role": "user", "content": [{"type": "text", "text": "Task: do it"}]},
            {"role": "assistant", "content": [{"type": "tool_use", "id": "a"}]},
            {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "a"}]},
        ])
        with tempfile.TemporaryDirectory() as directory:
            screenshot = Path(directory) / "screen.png"
            _png(screenshot)
            payload = target.preflight_next_request(
                "do it", {"screenshot": screenshot.read_bytes()}
            )
        self.assertEqual(payload["model"], "claude-opus-4-8")
        self.assertEqual(payload["betas"], ["computer-use-2025-11-24"])
        self.assertEqual(payload["messages"][-1]["content"][-1]["type"], "image")

    def test_tool_free_assistant_gets_new_live_user_observation(self) -> None:
        class Target:
            system_prompt = "live"
            model = "claude-opus-4-8"
            max_tokens = 4096
            tools = []
            betas = []
            messages = []
            actions_log = []

            def reset(self, *_args): self.messages = []
            def _resolve_pending_tool_uses(self, _image): pass
            def _trim_images(self): pass

        target = ClaudeTakeoverTarget(Target())
        target.seed_native_history("do it", [
            {"role": "system", "content": "live"},
            {"role": "user", "content": [{"type": "text", "text": "Task"}]},
            {"role": "assistant", "content": [{"type": "text", "text": "Done"}]},
        ])
        with tempfile.TemporaryDirectory() as directory:
            screenshot = Path(directory) / "screen.png"
            _png(screenshot)
            payload = target.preflight_next_request(
                "do it", {"screenshot": screenshot.read_bytes()}
            )
        self.assertEqual(payload["messages"][-1]["role"], "user")
        self.assertEqual(payload["messages"][-1]["content"][-1]["type"], "image")


class ClaudeTruncatedPrefixTests(unittest.TestCase):
    def _adapter(self) -> ClaudeNativeHistoryAdapter:
        return ClaudeNativeHistoryAdapter("do it", system_prompt="live")

    def _step(self, step_id: int, log: dict, *, action=None) -> HistoryStep:
        return HistoryStep(
            step_id=step_id,
            observation_image_url=(
                "data:image/png;base64,"
                "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk+A8AAQUBAScY42YAAAAASUVORK5CYII="
            ),
            observation_after_image_url=(
                "data:image/png;base64,"
                "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk+A8AAQUBAScY42YAAAAASUVORK5CYII="
            ),
            action=action or ShellAction(kind="shell", commands=("pwd",)),
            trajectory_log=json.dumps(log),
        )

    def test_mid_group_step_renders_no_messages(self) -> None:
        step = self._step(
            1,
            {"anthropic_native_turn": {"is_group_end": False, "assistant": None, "result_user": None}},
        )
        self.assertEqual(self._adapter().render_step(step), [])

    def test_control_flow_row_renders_no_messages(self) -> None:
        step = self._step(1, {"control_flow_row": True})
        self.assertEqual(self._adapter().render_step(step), [])

    def test_truncated_prefix_synthesizes_unrecorded_result(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            after = Path(directory) / "after.png"
            _png(after)
            step = self._step(
                0,
                {
                    "anthropic_native_turn": {
                        "is_group_end": True,
                        "initial_user": {"role": "user", "content": [{"type": "text", "text": "Task"}]},
                        "assistant": {"role": "assistant", "content": [
                            {"type": "tool_use", "id": "tool-a", "name": "computer", "input": {"action": "screenshot"}},
                            {"type": "tool_use", "id": "tool-b", "name": "computer", "input": {"action": "click"}},
                        ]},
                        "result_user": None,
                    }
                },
            )
            step = HistoryStep(
                step_id=step.step_id,
                observation_image_url=step.observation_image_url,
                action=step.action,
                observation_after_image_url=str(after),
                trajectory_log=step.trajectory_log,
            )
            messages = self._adapter().render_step(step)
        system, initial, assistant, result = messages
        self.assertEqual(system, {"role": "system", "content": "live"})
        self.assertEqual(initial["role"], "user")
        self.assertEqual(assistant["role"], "assistant")
        self.assertEqual(result["role"], "user")
        self.assertEqual(
            [block["tool_use_id"] for block in result["content"]], ["tool-a", "tool-b"]
        )
        for block in result["content"]:
            self.assertEqual(block["content"][0], {"type": "text", "text": "Action executed."})
            self.assertFalse(block["is_error"])
        self.assertEqual(result["content"][-1]["content"][-1]["type"], "image")

    def test_silent_terminate_row_cannot_be_the_truncation_boundary(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            after = Path(directory) / "after.png"
            _png(after)
            turn = {
                "is_group_end": True,
                "initial_user": {"role": "user", "content": [{"type": "text", "text": "Task"}]},
                "assistant": {"role": "assistant", "content": [
                    {"type": "tool_use", "id": "tool-a", "name": "computer", "input": {"action": "screenshot"}},
                ]},
                "result_user": None,
            }
            steps = [
                self._step(0, {"anthropic_native_turn": turn}),
                self._step(
                    1,
                    {"control_flow_row": True},
                    action=TerminateAction(kind="terminate", status="failure"),
                ),
            ]
            messages = build_native_history(self._adapter(), steps)
        self.assertEqual(messages[-1]["role"], "user")
        self.assertEqual(
            [block["tool_use_id"] for block in messages[-1]["content"]], ["tool-a"]
        )

    def test_truncation_boundary_without_rendered_message_still_fails(self) -> None:
        steps = [
            self._step(
                0,
                {"anthropic_native_turn": {"is_group_end": False, "assistant": None, "result_user": None}},
            ),
        ]
        with self.assertRaisesRegex(ValueError, "losslessly truncated"):
            build_native_history(self._adapter(), steps)


class ClaudeControlRowDetectionTests(unittest.TestCase):
    def _terminate_step(self, status: str) -> CanonicalStep:
        return CanonicalStep(
            step_id=0,
            action=TerminateAction(kind="terminate", status=status),
            observation_before_sha256="0" * 64,
            source_agent="claude_opus_4_8",
        )

    def test_gate_requires_done_and_terminate(self) -> None:
        self.assertTrue(_claude_control_row({"done": True}, self._terminate_step("failure")))
        self.assertTrue(_claude_control_row({"done": True}, self._terminate_step("success")))
        self.assertFalse(
            _claude_control_row(
                {"done": True},
                CanonicalStep(
                    step_id=0,
                    action=ShellAction(kind="shell", commands=("pwd",)),
                    observation_before_sha256="0" * 64,
                    source_agent="claude_opus_4_8",
                ),
            )
        )
        self.assertFalse(_claude_control_row({}, self._terminate_step("failure")))

    def test_unbound_row_returns_none_only_when_allowed(self) -> None:
        messages = [
            {"role": "user", "content": []},
            {"role": "assistant", "content": [
                {"type": "tool_use", "id": "b1", "name": "bash"},
            ]},
            {"role": "user", "content": []},
        ]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "messages.json"
            path.write_text(json.dumps(messages), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "no matching native assistant action"):
                _claude_native_turn(path, 2)
            self.assertIsNone(_claude_native_turn(path, 2, allow_unbound=True))


if __name__ == "__main__":
    unittest.main()
