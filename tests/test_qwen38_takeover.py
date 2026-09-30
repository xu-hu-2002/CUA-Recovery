"""Qwen3.8 hybrid prefix conversion and takeover injection tests."""

import json
import pathlib
import tempfile
import types
import unittest

from derail.adapters.base import HistoryStep
from derail.adapters.qwen38 import Qwen38ScaffoldAdapter
from derail.canonical.actions import ClickAction, ShellAction
from derail.canonical.mypcbench import _normalize_shell_tool_row
from derail.mypcbench.tool_agent import (
    NativeToolComputerAgent,
    qwen38_protocol,
    takeover_condition_prompt,
)
from derail.takeover.history import build_native_history


class _FakeCompletions:
    def __init__(self, messages):
        self.responses = list(messages)
        self.requests = []

    def create(self, **kwargs):
        self.requests.append(kwargs)
        message = self.responses.pop(0)
        return types.SimpleNamespace(
            choices=[types.SimpleNamespace(message=message)]
        )


class _FakeClient:
    def __init__(self, messages):
        self.completions = _FakeCompletions(messages)
        self.chat = types.SimpleNamespace(completions=self.completions)


class Qwen38TakeoverTests(unittest.TestCase):
    def _history(self):
        adapter = Qwen38ScaffoldAdapter("Finish the task")
        return build_native_history(
            adapter,
            (
                HistoryStep(
                    step_id=0,
                    observation_image_url="file:///shell-state.png",
                    action=ShellAction(kind="shell", commands=("pwd",)),
                    tool_result=(
                        '{"shell_results":[{"call_id":"source-call",'
                        '"result":{"stdout":"/home/user\\n","stderr":"",'
                        '"outcome":{"type":"exit","exit_code":0}}}]}'
                    ),
                ),
                HistoryStep(
                    step_id=1,
                    observation_image_url="data:image/png;base64,cHJlZml4",
                    action=ClickAction(kind="click", x_px=20, y_px=30),
                ),
            ),
        )

    def test_history_uses_actions_and_shell_results_as_primary_evidence(self):
        history = self._history()
        self.assertEqual(
            [message["role"] for message in history],
            ["system", "user", "assistant", "tool", "user", "assistant", "tool"],
        )
        self.assertFalse(
            any(part.get("type") == "image_url" for part in history[1]["content"])
        )
        self.assertIn("/home/user", history[3]["content"])
        self.assertTrue(
            any(part.get("type") == "image_url" for part in history[4]["content"])
        )
        self.assertEqual(history[5]["tool_calls"][0]["function"]["name"], "click")
        self.assertIn("Accepted for execution", history[6]["content"])

    def test_seeded_history_is_sent_before_current_state_and_keeps_instruction(self):
        call = {
            "id": "next",
            "type": "function",
            "function": {
                "name": "click",
                "arguments": json.dumps({"x": 100, "y": 200}),
            },
        }
        client = _FakeClient(
            [types.SimpleNamespace(content=None, tool_calls=[call])]
        )
        agent = NativeToolComputerAgent(
            "Qwen/Qwen3.8-27B",
            (1280, 800),
            qwen38_protocol(),
            client=client,
        )
        history = self._history()
        agent.seed_native_history(
            "Finish the task", history, condition="notified"
        )

        _response, actions = agent.predict(
            "Finish the task", {"screenshot": b"current-png"}
        )

        self.assertEqual(actions, ["pyautogui.click(100, 200, button='left')"])
        sent = client.completions.requests[0]["messages"]
        self.assertEqual(sent[:-1], history)
        current_text = next(
            part["text"] for part in sent[-1]["content"] if part["type"] == "text"
        )
        self.assertIn("Task: Finish the task", current_text)
        self.assertIn("current trajectory logs and screenshot", current_text)
        self.assertIn("serious errors", current_text)
        self.assertNotIn("root-cause", current_text)
        with self.assertRaisesRegex(ValueError, "differs"):
            agent.predict("A different task", {"screenshot": b"current-png"})

    def test_three_prompt_layers_have_the_intended_information_order(self):
        self.assertEqual(takeover_condition_prompt("unaware"), "")
        notified = takeover_condition_prompt("notified")
        self.assertIn("serious errors", notified)
        self.assertIn("caused the overall task to fail", notified)
        self.assertNotIn("root-cause action", notified)
        diagnosed = takeover_condition_prompt(
            "diagnosed", "The last click selected the wrong account.", 17
        )
        self.assertIn("action_index_global=17", diagnosed)
        self.assertIn("selected the wrong account", diagnosed)
        self.assertIn("human annotation rationale", diagnosed)
        self.assertIn("Fix the errors and complete the original task", diagnosed)
        with self.assertRaises(ValueError):
            takeover_condition_prompt("diagnosed")
        with self.assertRaises(ValueError):
            takeover_condition_prompt("unaware", "leak")
        hint = "The meeting on step 4 was booked in the wrong room."
        self.assertEqual(takeover_condition_prompt("hinted", hint=hint), hint)
        with self.assertRaises(ValueError):
            takeover_condition_prompt("hinted")
        with self.assertRaises(ValueError):
            takeover_condition_prompt("notified", hint=hint)

    def test_public_claude_bash_log_normalizes_without_a_screenshot(self):
        action, result = _normalize_shell_tool_row(
            {
                "agent_metadata": {
                    "tool_messages": [
                        {
                            "type": "tool_use",
                            "id": "bash-1",
                            "name": "bash",
                            "input": {"command": "ls /home/user/Documents"},
                        },
                        {
                            "type": "tool_result",
                            "tool_use_id": "bash-1",
                            "content": "report.odt",
                            "is_error": False,
                        },
                    ]
                }
            }
        )
        self.assertIsInstance(action, ShellAction)
        self.assertEqual(action.commands, ("ls /home/user/Documents",))
        self.assertIn("report.odt", result)

    def test_sparse_prefix_still_gets_one_system_message(self):
        adapter = Qwen38ScaffoldAdapter("Finish the task")
        history = build_native_history(
            adapter,
            (
                HistoryStep(
                    step_id=4,
                    observation_image_url="data:image/png;base64,cHJlZml4",
                    action=ClickAction(kind="click", x_px=20, y_px=30),
                ),
            ),
        )
        self.assertEqual(history[0]["role"], "system")
        self.assertEqual(sum(message["role"] == "system" for message in history), 1)

    def test_local_replay_image_is_resolved_to_data_url_at_injection(self):
        with tempfile.TemporaryDirectory() as directory:
            screenshot = pathlib.Path(directory) / "before.png"
            screenshot.write_bytes(b"\x89PNG\r\n\x1a\nreplay-evidence")
            resolved = NativeToolComputerAgent._resolve_history_image_url(str(screenshot))
        self.assertTrue(resolved.startswith("data:image/png;base64,"))

    def test_live_qwen_shell_turn_records_command_output_for_the_next_turn(self):
        call = {
            "id": "bash-next",
            "type": "function",
            "function": {
                "name": "bash",
                "arguments": json.dumps({"commands": ["pwd"]}),
            },
        }

        class Environment:
            def _execute_command(self, command, shell=False):
                self.seen = (command, shell)
                return {"output": "/home/user\n", "error": "", "returncode": 0}

        environment = Environment()
        client = _FakeClient(
            [types.SimpleNamespace(content=None, tool_calls=[call])]
        )
        agent = NativeToolComputerAgent(
            "Qwen/Qwen3.8-27B",
            (1280, 800),
            qwen38_protocol(),
            client=client,
            environment=environment,
        )

        response, actions = agent.predict(
            "Finish the task", {"screenshot": b"current-png"}
        )

        self.assertEqual(actions, [])
        self.assertEqual(environment.seen, ("pwd", True))
        stdout = json.loads(response)["tool_results"][0]["outputs"][0]["stdout"]
        self.assertEqual(stdout, "/home/user\n")
        self.assertEqual(agent.last_trajectory_tool_messages[0]["name"], "bash")
        self.assertIn("/home/user", agent._turns[-1][-1]["content"])


if __name__ == "__main__":
    unittest.main()
