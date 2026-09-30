"""Canonical action、坐标和 replay 的基础测试。"""

import unittest

from derail.canonical.actions import (
    CanonicalActionError,
    DragAction,
    HotkeyAction,
    MouseButtonTransitionAction,
    ShellAction,
    action_from_dict,
    action_to_dict,
)
from derail.canonical.coordinates import normalized_to_pixel, pixel_to_normalized
from derail.canonical.summaries import summarize_action
from derail.replay.executor import CanonicalExecutor, RecordingBackend, compile_pyautogui


class CanonicalActionTests(unittest.TestCase):
    def test_click_round_trip_and_summary(self) -> None:
        action = action_from_dict(
            {"kind": "click", "x_px": 912, "y_px": 356, "target": "Save button"}
        )
        self.assertEqual(action_to_dict(action)["x_px"], 912)
        self.assertEqual(summarize_action(action), "Click Save button at (912, 356).")
        self.assertIn("pyautogui.click", compile_pyautogui(action))

    def test_rejects_out_of_bounds_coordinate(self) -> None:
        with self.assertRaises(CanonicalActionError):
            action_from_dict({"kind": "click", "x_px": 1280, "y_px": 10})

    def test_rejects_unknown_field(self) -> None:
        with self.assertRaises(CanonicalActionError):
            action_from_dict({"kind": "wait", "seconds": 1, "typo": True})

    def test_hotkey_list_is_frozen_tuple(self) -> None:
        action = action_from_dict({"kind": "hotkey", "keys": ["ctrl", "s"]})
        self.assertIsInstance(action, HotkeyAction)
        self.assertEqual(action.keys, ("ctrl", "s"))
        self.assertEqual(action_to_dict(action)["keys"], ["ctrl", "s"])

    def test_coordinate_protocols_use_explicit_maximum(self) -> None:
        self.assertEqual(pixel_to_normalized(640, 1280, 1000), 500)
        self.assertEqual(pixel_to_normalized(640, 1280, 999), 500)
        self.assertEqual(normalized_to_pixel(500, 1280, 1000), 640)

    def test_drag_executor_has_press_move_release_sequence(self) -> None:
        action = DragAction(
            kind="drag",
            start_x_px=10,
            start_y_px=20,
            end_x_px=100,
            end_y_px=120,
        )
        backend = RecordingBackend()
        CanonicalExecutor(backend).execute(action)
        self.assertEqual(
            backend.calls,
            [
                ("move_to", 10, 20, 0.0),
                ("mouse_down", "left"),
                ("move_to", 100, 120, 0.5),
                ("mouse_up", "left"),
            ],
        )

    def test_mouse_button_transition_round_trip_and_execution(self) -> None:
        action = action_from_dict({"kind": "mouse_down", "button": "right"})
        self.assertIsInstance(action, MouseButtonTransitionAction)
        self.assertEqual(action_to_dict(action), {"kind": "mouse_down", "button": "right"})
        self.assertEqual(summarize_action(action), "Hold mouse button right.")
        self.assertEqual(compile_pyautogui(action), "pyautogui.mouseDown(button='right')")

        backend = RecordingBackend()
        CanonicalExecutor(backend).execute(action)
        CanonicalExecutor(backend).execute(
            MouseButtonTransitionAction(kind="mouse_up", button="right")
        )
        self.assertEqual(
            backend.calls,
            [("mouse_down", "right"), ("mouse_up", "right")],
        )

    def test_mouse_button_transition_rejects_unknown_button(self) -> None:
        with self.assertRaises(CanonicalActionError):
            MouseButtonTransitionAction(kind="mouse_down", button="primary")

    def test_shell_action_round_trip_without_python_compilation(self) -> None:
        raw = {
            "kind": "shell",
            "commands": ["pwd", "sqlite3 app.db '.tables'"],
            "timeout_ms": 90000,
            "max_output_length": 12000,
        }
        action = action_from_dict(raw)
        self.assertIsInstance(action, ShellAction)
        self.assertEqual(action.commands, tuple(raw["commands"]))
        self.assertEqual(action_to_dict(action), raw)
        self.assertIn("Run", summarize_action(action))
        with self.assertRaises(TypeError):
            CanonicalExecutor(RecordingBackend()).execute(action)


if __name__ == "__main__":
    unittest.main()
