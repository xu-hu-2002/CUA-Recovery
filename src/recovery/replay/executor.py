"""Safe execution of canonical actions."""

from __future__ import annotations

import time
from typing import Any, List, Protocol, Sequence, Tuple

from recovery.canonical.actions import (
    Action,
    ClickAction,
    DragAction,
    HotkeyAction,
    HorizontalScrollAction,
    KeyTransitionAction,
    MouseButtonTransitionAction,
    MoveAction,
    NoOpAction,
    ScrollAction,
    SequenceAction,
    ShellAction,
    TerminateAction,
    TypeAction,
    WaitAction,
)


class DesktopBackend(Protocol):
    def click(self, x: int, y: int, button: str, clicks: int = 1) -> None: ...

    def write(self, text: str, interval_s: float = 0.0) -> None: ...

    def press(self, key: str) -> None: ...

    def hotkey(self, keys: Sequence[str]) -> None: ...

    def key_down(self, key: str) -> None: ...

    def key_up(self, key: str) -> None: ...

    def move_to(self, x: int, y: int, duration_s: float = 0.0) -> None: ...

    def mouse_down(self, button: str) -> None: ...

    def mouse_up(self, button: str) -> None: ...

    def scroll(self, delta_y: int) -> None: ...

    def horizontal_scroll(self, delta_x: int) -> None: ...

    def sleep(self, seconds: float) -> None: ...


class PyAutoGUIBackend:
    """Real desktop backend; imports pyautogui on instantiation."""

    def __init__(self) -> None:
        try:
            import pyautogui  # type: ignore
        except ImportError as exc:
            raise RuntimeError("real replay needs optional dependencies: pip install -e '.[desktop]'") from exc
        self._pyautogui = pyautogui

    def click(self, x: int, y: int, button: str, clicks: int = 1) -> None:
        self._pyautogui.click(x, y, button=button, clicks=clicks)

    def write(self, text: str, interval_s: float = 0.0) -> None:
        self._pyautogui.write(text, interval=interval_s)

    def press(self, key: str) -> None:
        self._pyautogui.press(key)

    def hotkey(self, keys: Sequence[str]) -> None:
        self._pyautogui.hotkey(*keys)

    def key_down(self, key: str) -> None:
        self._pyautogui.keyDown(key)

    def key_up(self, key: str) -> None:
        self._pyautogui.keyUp(key)

    def move_to(self, x: int, y: int, duration_s: float = 0.0) -> None:
        self._pyautogui.moveTo(x, y, duration=duration_s)

    def mouse_down(self, button: str) -> None:
        self._pyautogui.mouseDown(button=button)

    def mouse_up(self, button: str) -> None:
        self._pyautogui.mouseUp(button=button)

    def scroll(self, delta_y: int) -> None:
        self._pyautogui.scroll(delta_y)

    def horizontal_scroll(self, delta_x: int) -> None:
        self._pyautogui.hscroll(delta_x)

    def sleep(self, seconds: float) -> None:
        time.sleep(seconds)


class RecordingBackend:
    """Backend that only records calls, for tests and dry runs."""

    def __init__(self) -> None:
        self.calls: List[Tuple[Any, ...]] = []

    def click(self, x: int, y: int, button: str, clicks: int = 1) -> None:
        self.calls.append(("click", x, y, button, clicks))

    def write(self, text: str, interval_s: float = 0.0) -> None:
        self.calls.append(
            ("write", text) if interval_s == 0 else ("write", text, interval_s)
        )

    def press(self, key: str) -> None:
        self.calls.append(("press", key))

    def hotkey(self, keys: Sequence[str]) -> None:
        self.calls.append(("hotkey", tuple(keys)))

    def key_down(self, key: str) -> None:
        self.calls.append(("key_down", key))

    def key_up(self, key: str) -> None:
        self.calls.append(("key_up", key))

    def move_to(self, x: int, y: int, duration_s: float = 0.0) -> None:
        self.calls.append(("move_to", x, y, duration_s))

    def mouse_down(self, button: str) -> None:
        self.calls.append(("mouse_down", button))

    def mouse_up(self, button: str) -> None:
        self.calls.append(("mouse_up", button))

    def scroll(self, delta_y: int) -> None:
        self.calls.append(("scroll", delta_y))

    def horizontal_scroll(self, delta_x: int) -> None:
        self.calls.append(("horizontal_scroll", delta_x))

    def sleep(self, seconds: float) -> None:
        self.calls.append(("sleep", seconds))


class CanonicalExecutor:
    """Map validated actions to backend primitives."""

    def __init__(self, backend: DesktopBackend, select_all_keys: Sequence[str] = ("ctrl", "a")):
        self.backend = backend
        self.select_all_keys = tuple(select_all_keys)

    def execute(self, action: Action) -> None:
        if isinstance(action, SequenceAction):
            for primitive in action.actions:
                self.execute(primitive)
            return
        if isinstance(action, ClickAction):
            clicks = 2 if action.kind == "double_click" else 1
            self.backend.click(action.x_px, action.y_px, action.button, clicks)
            return
        if isinstance(action, TypeAction):
            if action.clear_existing:
                self.backend.hotkey(self.select_all_keys)
            self.backend.write(action.text, action.interval_s)
            if action.press_enter:
                self.backend.press("enter")
            return
        if isinstance(action, HotkeyAction):
            self.backend.hotkey(action.keys)
            return
        if isinstance(action, KeyTransitionAction):
            if action.kind == "key_down":
                self.backend.key_down(action.key)
            else:
                self.backend.key_up(action.key)
            return
        if isinstance(action, MouseButtonTransitionAction):
            if action.kind == "mouse_down":
                self.backend.mouse_down(action.button)
            else:
                self.backend.mouse_up(action.button)
            return
        if isinstance(action, NoOpAction):
            return
        if isinstance(action, ScrollAction):
            self.backend.move_to(action.x_px, action.y_px)
            self.backend.scroll(action.delta_y)
            return
        if isinstance(action, HorizontalScrollAction):
            self.backend.move_to(action.x_px, action.y_px)
            self.backend.horizontal_scroll(action.delta_x)
            return
        if isinstance(action, MoveAction):
            self.backend.move_to(action.x_px, action.y_px)
            return
        if isinstance(action, DragAction):
            self.backend.move_to(action.start_x_px, action.start_y_px)
            self.backend.mouse_down(action.button)
            self.backend.move_to(action.end_x_px, action.end_y_px, action.duration_s)
            self.backend.mouse_up(action.button)
            return
        if isinstance(action, WaitAction):
            self.backend.sleep(action.seconds)
            return
        if isinstance(action, ShellAction):
            raise TypeError("shell actions require an isolated VM backend")
        if isinstance(action, TerminateAction):
            return
        raise TypeError("unhandled action: %s" % type(action).__name__)


def compile_pyautogui(action: Action) -> str:
    """Render readable PyAutoGUI code for auditing; never executed."""

    if isinstance(action, SequenceAction):
        return "\n".join(compile_pyautogui(item) for item in action.actions)
    if isinstance(action, ClickAction):
        method = "doubleClick" if action.kind == "double_click" else "click"
        return "pyautogui.%s(%d, %d, button=%r)" % (
            method,
            action.x_px,
            action.y_px,
            action.button,
        )
    if isinstance(action, TypeAction):
        commands = []
        if action.clear_existing:
            commands.append("pyautogui.hotkey('ctrl', 'a')")
        commands.append(
            "pyautogui.write(%r%s)"
            % (action.text, ", interval=%r" % action.interval_s if action.interval_s else "")
        )
        if action.press_enter:
            commands.append("pyautogui.press('enter')")
        return "; ".join(commands)
    if isinstance(action, HotkeyAction):
        return "pyautogui.hotkey(%s)" % ", ".join(repr(key) for key in action.keys)
    if isinstance(action, KeyTransitionAction):
        method = "keyDown" if action.kind == "key_down" else "keyUp"
        return "pyautogui.%s(%r)" % (method, action.key)
    if isinstance(action, MouseButtonTransitionAction):
        method = "mouseDown" if action.kind == "mouse_down" else "mouseUp"
        return "pyautogui.%s(button=%r)" % (method, action.button)
    if isinstance(action, NoOpAction):
        return "# no desktop action: %s" % action.reason
    if isinstance(action, ScrollAction):
        return "pyautogui.moveTo(%d, %d); pyautogui.scroll(%d)" % (
            action.x_px,
            action.y_px,
            action.delta_y,
        )
    if isinstance(action, HorizontalScrollAction):
        return "pyautogui.moveTo(%d, %d); pyautogui.hscroll(%d)" % (
            action.x_px,
            action.y_px,
            action.delta_x,
        )
    if isinstance(action, MoveAction):
        return "pyautogui.moveTo(%d, %d)" % (action.x_px, action.y_px)
    if isinstance(action, DragAction):
        return (
            "pyautogui.moveTo(%d, %d); pyautogui.dragTo(%d, %d, duration=%r, button=%r)"
            % (
                action.start_x_px,
                action.start_y_px,
                action.end_x_px,
                action.end_y_px,
                action.duration_s,
                action.button,
            )
        )
    if isinstance(action, WaitAction):
        return "time.sleep(%r)" % action.seconds
    if isinstance(action, ShellAction):
        return "\n".join("# VM shell: %s" % command for command in action.commands)
    if isinstance(action, TerminateAction):
        return "# terminate(status=%r, answer=%r)" % (action.status, action.answer)
    raise TypeError("unhandled action: %s" % type(action).__name__)
