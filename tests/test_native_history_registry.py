"""Capability gates for the planned takeover agents."""

import json
import unittest

from derail.adapters import (
    TARGET_AGENT_IDS,
    NativeHistoryUnavailableError,
    create_native_history_adapter,
    get_registration,
    registry_status,
)
from derail.adapters.kimi_k3 import KimiK3ScaffoldAdapter
from derail.adapters.opencua import OpenCUAActionHistoryAdapter
from derail.adapters.qwen35 import Qwen35StateAdapter
from derail.canonical.actions import (
    ClickAction,
    HotkeyAction,
    MoveAction,
    SequenceAction,
    TerminateAction,
)
from derail.mypcbench.tool_agent import build_computer_tools
from derail.adapters.base import HistoryStep


class NativeHistoryRegistryTests(unittest.TestCase):
    def test_registry_covers_exact_targets(self) -> None:
        statuses = registry_status()
        self.assertEqual(tuple(item["agent_id"] for item in statuses), TARGET_AGENT_IDS)
        implemented = {
            item["agent_id"] for item in statuses if item["renderer_implemented"]
        }
        self.assertEqual(
            implemented,
            {
                "kimi_k3", "opencua_72b", "qwen3_5_35b_a3b", "claude_opus_4_8", "evocua_32b",
                "gpt_5_5", "rerail_35b_a3b",
            },
        )
        self.assertTrue(all(item["release_ready_without_probe"] is False for item in statuses))

    def test_unregistered_target_hard_fails(self) -> None:
        with self.assertRaisesRegex(NativeHistoryUnavailableError, "unregistered"):
            create_native_history_adapter("gpt_4o", "Do the task")

    def test_opencua_restores_action_history_with_thought(self) -> None:
        adapter = create_native_history_adapter(
            "opencua_72b", "Do the task", system_prompt="live prompt"
        )
        self.assertIsInstance(adapter, OpenCUAActionHistoryAdapter)
        trajectory_log = json.dumps(
            {
                "visible_response": (
                    "# Step 1:\n## Thought:\nprivate rationale\n\n"
                    "## Action:\nClick the Save button.\n\n## Code:\npyautogui.click(1,2)"
                )
            }
        )
        messages = adapter.render_step(
            HistoryStep(
                step_id=0,
                observation_image_url="/tmp/replay.png",
                action=ClickAction(kind="click", x_px=1, y_px=2),
                trajectory_log=trajectory_log,
            )
        )
        self.assertEqual(messages[0], {"role": "system", "content": "live prompt"})
        self.assertEqual(messages[1]["action"], "Click the Save button.")
        self.assertEqual(messages[1]["cot"]["thought"], "private rationale")

    def test_opencua_selects_action_matching_recorded_action_kind(self) -> None:
        adapter = create_native_history_adapter(
            "opencua_72b", "Do the task", system_prompt="live prompt"
        )
        trajectory_log = json.dumps(
            {
                "response": (
                    "# Step 1:\n## Thought:\nfirst\n\n## Action:\nClick channel.\n\n"
                    "## Code:\n```python\npyautogui.click(192,530)\n```\n`\n"
                    "## Thought:\nretry\n\n## Action:\nClick again.\n\n"
                    "## Code:\n```python\npyautogui.click(193,531)\n```\n`\n"
                    "## Thought:\nsecond\n\n## Action:\nMove into chat.\n\n"
                    "## Code:\n```python\npyautogui.moveTo(791,443)\n```\n"
                )
            }
        )
        messages = adapter.render_step(
            HistoryStep(
                step_id=0,
                observation_image_url="/tmp/replay.png",
                action=MoveAction(
                    kind="move", x_px=786, y_px=436, frame_width=1280, frame_height=800
                ),
                trajectory_log=trajectory_log,
            )
        )
        self.assertEqual(messages[1]["action"], "Move into chat.")
        self.assertEqual(messages[1]["cot"]["thought"], "second")

    def test_opencua_preserves_terminal_response_without_action_block(self) -> None:
        adapter = create_native_history_adapter(
            "opencua_72b", "Do the task", system_prompt="live prompt"
        )
        messages = adapter.render_step(
            HistoryStep(
                step_id=37,
                observation_image_url="/tmp/replay.png",
                action=TerminateAction(kind="terminate", status="failure", answer="PREDICT_CRASH"),
                trajectory_log=json.dumps({"response": "调用不在 PyAutoGUI 白名单中"}),
            )
        )
        self.assertEqual(messages[0]["action"], "调用不在 PyAutoGUI 白名单中")
        self.assertEqual(messages[0]["cot"], {"action": "调用不在 PyAutoGUI 白名单中"})

    def test_qwen35_renderer_preserves_public_native_response(self) -> None:
        adapter = create_native_history_adapter("qwen3_5_35b_a3b", "Do the task")
        self.assertIsInstance(adapter, Qwen35StateAdapter)
        response = (
            "Action: Click Save.\n"
            "<tool_call><function=computer_use><parameter=action>left_click"
            "</parameter><parameter=coordinate>[10,20]</parameter>"
            "</function></tool_call>"
        )
        records = adapter.render_step(
            HistoryStep(
                step_id=0,
                observation_image_url="/tmp/replay.png",
                observation_sha256="abc",
                action=ClickAction(kind="click", x_px=10, y_px=20),
                trajectory_log=json.dumps(
                    {"response": response, "reasoning": ["inspect save button"]}
                ),
            )
        )
        self.assertEqual(records[0]["role"], "qwen35_state")
        self.assertEqual(records[0]["action"], "Click Save.")
        self.assertEqual(records[0]["response"], response)
        self.assertIn("inspect save button", records[0]["reasoning"])

    def test_qwen35_renderer_rejects_ambiguous_source_response(self) -> None:
        adapter = create_native_history_adapter("qwen3_5_35b_a3b", "Do the task")
        with self.assertRaisesRegex(ValueError, "one Action line"):
            adapter.render_step(
                HistoryStep(
                    step_id=0,
                    observation_image_url="/tmp/replay.png",
                    action=ClickAction(kind="click", x_px=10, y_px=20),
                    trajectory_log=json.dumps({"visible_response": "Action: Click Save."}),
                )
            )

    def test_qwen35_renderer_accepts_batched_multi_call_turn(self) -> None:
        adapter = create_native_history_adapter("qwen3_5_35b_a3b", "Do the task")
        response = (
            "Action: Scroll down on the page.\n\n"
            "<tool_call>\n<function=computer_use>\n<parameter=action>\nmouse_move\n"
            "</parameter>\n<parameter=coordinate>\n[471, 624]\n</parameter>\n"
            "</function>\n</tool_call>\n"
            "<tool_call>\n<function=computer_use>\n<parameter=action>\nscroll\n"
            "</parameter>\n<parameter=pixels>\n-5\n</parameter>\n"
            "</function>\n</tool_call>"
        )
        records = adapter.render_step(
            HistoryStep(
                step_id=3,
                turn_index=3,
                observation_image_url="/tmp/replay.png",
                observation_sha256="0" * 64,
                action=MoveAction(kind="move", x_px=603, y_px=499),
                trajectory_log=json.dumps({"response": response}),
            )
        )
        self.assertEqual(records[0]["role"], "qwen35_state")
        self.assertEqual(records[0]["action"], "Scroll down on the page.")
        self.assertEqual(records[0]["response"], response)

    def test_qwen35_renderer_does_not_duplicate_multi_action_turn(self) -> None:
        adapter = create_native_history_adapter("qwen3_5_35b_a3b", "Do the task")
        self.assertEqual(
            adapter.render_step(
                HistoryStep(
                    step_id=1,
                    turn_index=0,
                    action_index_within_turn=1,
                    observation_image_url="/tmp/replay.png",
                    action=ClickAction(kind="click", x_px=20, y_px=30),
                    trajectory_log="",
                )
            ),
            [],
        )

    def test_kimi_k3_is_explicitly_scaffold_and_reuses_absolute_pixels(self) -> None:
        registration = get_registration("kimi_k3")
        self.assertIn("frozen_schema_via_gateway", registration.protocol_origin)
        adapter = registration.create("Do the task")
        self.assertIsInstance(adapter, KimiK3ScaffoldAdapter)
        call = adapter.action_to_call(
            ClickAction(kind="click", x_px=1279, y_px=799), "call-1"
        )
        arguments = json.loads(call["function"]["arguments"])
        self.assertEqual((arguments["x"], arguments["y"]), (1279, 799))
        self.assertEqual(
            adapter.tool_definitions(), build_computer_tools(1279, 799, include_bash=True)
        )
        current_scroll = next(
            item for item in adapter.tool_definitions() if item["function"]["name"] == "scroll"
        )
        history_scroll = next(
            item
            for item in adapter.history_tool_definitions()
            if item["function"]["name"] == "scroll"
        )
        current_delta = current_scroll["function"]["parameters"]["properties"]["delta_y"]
        history_delta = history_scroll["function"]["parameters"]["properties"]["delta_y"]
        self.assertEqual((current_delta["minimum"], current_delta["maximum"]), (-30, 30))
        self.assertNotIn("minimum", history_delta)
        self.assertNotIn("maximum", history_delta)
        self.assertIn("bash", {item["function"]["name"] for item in adapter.tool_definitions()})
        self.assertIn("Kimi", adapter.system_prompt())

    def test_kimi_k3_collapses_evocua_character_macros_to_write(self) -> None:
        adapter = create_native_history_adapter("kimi_k3", "Do the task")
        action = SequenceAction(
            kind="sequence",
            actions=tuple(
                HotkeyAction(kind="hotkey", keys=(key,))
                for key in ("M", "i", "c", "h", "a", "e", "l", "space", ":")
            ),
        )
        calls = adapter.action_to_calls(action, "call-1")
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0]["function"]["name"], "write")
        self.assertEqual(
            json.loads(calls[0]["function"]["arguments"])["content"],
            "Michael :",
        )

    def test_kimi_k3_reprojects_source_coordinates_to_live_frame(self) -> None:
        adapter = create_native_history_adapter("kimi_k3", "Do the task")
        call = adapter.action_to_calls(
            ClickAction(
                kind="click",
                x_px=1023,
                y_px=767,
                frame_width=1024,
                frame_height=768,
            ),
            "call-1",
        )[0]
        self.assertEqual(
            json.loads(call["function"]["arguments"]),
            {"x": 1279, "y": 799, "button": "left"},
        )


if __name__ == "__main__":
    unittest.main()
