"""Holo-3.1 native function-call history 的 golden behavior。"""

import json
import unittest

from derail.adapters.base import HistoryStep
from derail.adapters.holo31 import Holo31Adapter
from derail.canonical.actions import DragAction, ScrollAction
from derail.takeover.history import build_native_history


class Holo31AdapterTests(unittest.TestCase):
    def test_scroll_coordinates_and_tool_call_ids(self) -> None:
        adapter = Holo31Adapter(instruction="Save the document")
        step = HistoryStep(
            step_id=0,
            observation_image_url="artifact://screenshots/step-7.png",
            action=ScrollAction(kind="scroll", x_px=640, y_px=400, delta_y=-4),
            tool_result="ok",
        )
        messages = build_native_history(adapter, (step,))
        self.assertEqual(
            [message["role"] for message in messages],
            ["system", "user", "assistant", "tool"],
        )
        self.assertEqual(messages[1]["content"][0]["type"], "image_url")
        self.assertIn("Task: Save the document", messages[1]["content"][1]["text"])
        tool_call = messages[2]["tool_calls"][0]
        self.assertEqual(tool_call["id"], "derail_step_0000")
        self.assertEqual(tool_call["function"]["name"], "scroll")
        args = json.loads(tool_call["function"]["arguments"])
        self.assertEqual((args["x"], args["y"]), (500, 501))
        self.assertEqual(messages[3]["tool_call_id"], tool_call["id"])

    def test_drag_is_single_native_tool_call(self) -> None:
        adapter = Holo31Adapter()
        action = DragAction(
            kind="drag",
            start_x_px=128,
            start_y_px=80,
            # 1151/719 are the exact 1280x800 pixels represented by (900, 900)
            # after the live compiler's normalized-coordinate round trip.
            end_x_px=1151,
            end_y_px=719,
        )
        call = adapter.action_to_call(action, "call-1")
        args = json.loads(call["function"]["arguments"])
        self.assertEqual(call["function"]["name"], "drag")
        self.assertEqual((args["start_x"], args["start_y"]), (100, 100))
        self.assertEqual((args["end_x"], args["end_y"]), (900, 900))

    def test_declared_tools_cover_capabilities(self) -> None:
        adapter = Holo31Adapter()
        declared = {tool["function"]["name"] for tool in adapter.tool_definitions()}
        expected = {
            "click",
            "double_click",
            "write",
            "hotkey",
            "scroll",
            "move",
            "drag",
            "wait",
            "answer",
        }
        self.assertEqual(declared, expected)

    def test_render_requires_real_instruction(self) -> None:
        adapter = Holo31Adapter()
        step = HistoryStep(
            step_id=0,
            observation_image_url="artifact://screenshots/step-0.png",
            action=ScrollAction(kind="scroll", x_px=640, y_px=400, delta_y=-4),
        )
        with self.assertRaisesRegex(ValueError, "instruction"):
            build_native_history(adapter, (step,))


if __name__ == "__main__":
    unittest.main()
