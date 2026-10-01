"""Canonical desktop actions."""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields, replace
from typing import Any, Dict, Mapping, Tuple, Type, Union

DEFAULT_FRAME_WIDTH = 1280
DEFAULT_FRAME_HEIGHT = 800
VALID_MOUSE_BUTTONS = frozenset({"left", "right", "middle"})


class CanonicalActionError(ValueError):
    """Input cannot be losslessly interpreted as a canonical action."""


def _validate_frame(width: int, height: int) -> None:
    if width <= 0 or height <= 0:
        raise CanonicalActionError("frame_width and frame_height must be positive integers")


def _validate_point(x_px: int, y_px: int, width: int, height: int, name: str) -> None:
    _validate_frame(width, height)
    if not (0 <= x_px < width and 0 <= y_px < height):
        raise CanonicalActionError(
            "%s=(%d, %d) is outside the %dx%d screenshot" % (name, x_px, y_px, width, height)
        )


def _validate_button(button: str) -> None:
    if button not in VALID_MOUSE_BUTTONS:
        raise CanonicalActionError("unsupported mouse button: %s" % button)


@dataclass(frozen=True)
class ClickAction:
    kind: str
    x_px: int
    y_px: int
    button: str = "left"
    target: str = ""
    frame_width: int = DEFAULT_FRAME_WIDTH
    frame_height: int = DEFAULT_FRAME_HEIGHT

    def __post_init__(self) -> None:
        if self.kind not in {"click", "double_click"}:
            raise CanonicalActionError("ClickAction.kind must be click or double_click")
        _validate_point(self.x_px, self.y_px, self.frame_width, self.frame_height, "click")
        _validate_button(self.button)


@dataclass(frozen=True)
class TypeAction:
    kind: str
    text: str
    press_enter: bool = False
    clear_existing: bool = False
    interval_s: float = 0.0

    def __post_init__(self) -> None:
        if self.kind != "type":
            raise CanonicalActionError("TypeAction.kind must be type")
        if not isinstance(self.text, str):
            raise CanonicalActionError("TypeAction.text must be a string")
        if isinstance(self.interval_s, bool) or not isinstance(self.interval_s, (int, float)):
            raise CanonicalActionError("TypeAction.interval_s must be a number")
        if self.interval_s < 0:
            raise CanonicalActionError("TypeAction.interval_s must not be negative")


@dataclass(frozen=True)
class HotkeyAction:
    kind: str
    keys: Tuple[str, ...]

    def __post_init__(self) -> None:
        if self.kind != "hotkey":
            raise CanonicalActionError("HotkeyAction.kind must be hotkey")
        if not 1 <= len(self.keys) <= 32:
            raise CanonicalActionError("hotkey must contain 1 to 32 keys")
        if any(not key.strip() for key in self.keys):
            raise CanonicalActionError("hotkey must not contain empty keys")


@dataclass(frozen=True)
class KeyTransitionAction:
    kind: str
    key: str

    def __post_init__(self) -> None:
        if self.kind not in {"key_down", "key_up"}:
            raise CanonicalActionError("KeyTransitionAction.kind must be key_down or key_up")
        if not isinstance(self.key, str) or not self.key.strip():
            raise CanonicalActionError("key transition must contain a non-empty key")


@dataclass(frozen=True)
class MouseButtonTransitionAction:
    kind: str
    button: str = "left"

    def __post_init__(self) -> None:
        if self.kind not in {"mouse_down", "mouse_up"}:
            raise CanonicalActionError(
                "MouseButtonTransitionAction.kind must be mouse_down or mouse_up"
            )
        _validate_button(self.button)


@dataclass(frozen=True)
class NoOpAction:
    kind: str
    reason: str

    def __post_init__(self) -> None:
        if self.kind != "no_op":
            raise CanonicalActionError("NoOpAction.kind must be no_op")
        if not isinstance(self.reason, str) or not self.reason.strip():
            raise CanonicalActionError("no_op.reason must not be empty")


@dataclass(frozen=True)
class ScrollAction:
    kind: str
    x_px: int
    y_px: int
    delta_y: int
    frame_width: int = DEFAULT_FRAME_WIDTH
    frame_height: int = DEFAULT_FRAME_HEIGHT

    def __post_init__(self) -> None:
        if self.kind != "scroll":
            raise CanonicalActionError("ScrollAction.kind must be scroll")
        _validate_point(self.x_px, self.y_px, self.frame_width, self.frame_height, "scroll")
        if self.delta_y == 0:
            raise CanonicalActionError("scroll.delta_y must not be 0")


@dataclass(frozen=True)
class HorizontalScrollAction:
    kind: str
    x_px: int
    y_px: int
    delta_x: int
    frame_width: int = DEFAULT_FRAME_WIDTH
    frame_height: int = DEFAULT_FRAME_HEIGHT

    def __post_init__(self) -> None:
        if self.kind != "horizontal_scroll":
            raise CanonicalActionError("HorizontalScrollAction.kind must be horizontal_scroll")
        _validate_point(self.x_px, self.y_px, self.frame_width, self.frame_height, "horizontal_scroll")
        if self.delta_x == 0:
            raise CanonicalActionError("horizontal_scroll.delta_x must not be 0")


@dataclass(frozen=True)
class MoveAction:
    kind: str
    x_px: int
    y_px: int
    frame_width: int = DEFAULT_FRAME_WIDTH
    frame_height: int = DEFAULT_FRAME_HEIGHT

    def __post_init__(self) -> None:
        if self.kind != "move":
            raise CanonicalActionError("MoveAction.kind must be move")
        _validate_point(self.x_px, self.y_px, self.frame_width, self.frame_height, "move")


@dataclass(frozen=True)
class DragAction:
    kind: str
    start_x_px: int
    start_y_px: int
    end_x_px: int
    end_y_px: int
    button: str = "left"
    duration_s: float = 0.5
    frame_width: int = DEFAULT_FRAME_WIDTH
    frame_height: int = DEFAULT_FRAME_HEIGHT

    def __post_init__(self) -> None:
        if self.kind != "drag":
            raise CanonicalActionError("DragAction.kind must be drag")
        _validate_point(
            self.start_x_px,
            self.start_y_px,
            self.frame_width,
            self.frame_height,
            "drag.start",
        )
        _validate_point(
            self.end_x_px,
            self.end_y_px,
            self.frame_width,
            self.frame_height,
            "drag.end",
        )
        _validate_button(self.button)
        if self.duration_s < 0:
            raise CanonicalActionError("drag.duration_s must not be negative")


@dataclass(frozen=True)
class WaitAction:
    kind: str
    seconds: float

    def __post_init__(self) -> None:
        if self.kind != "wait":
            raise CanonicalActionError("WaitAction.kind must be wait")
        if self.seconds < 0:
            raise CanonicalActionError("wait.seconds must not be negative")


@dataclass(frozen=True)
class ShellAction:
    """One model-issued shell tool call, possibly containing several commands."""

    kind: str
    commands: Tuple[str, ...]
    timeout_ms: int = 120000
    max_output_length: int = 8192

    def __post_init__(self) -> None:
        if self.kind != "shell":
            raise CanonicalActionError("ShellAction.kind must be shell")
        if not self.commands or any(
            not isinstance(command, str) or not command.strip()
            for command in self.commands
        ):
            raise CanonicalActionError("shell.commands must contain non-empty commands")
        if (
            isinstance(self.timeout_ms, bool)
            or not isinstance(self.timeout_ms, int)
            or not 1 <= self.timeout_ms <= 120000
        ):
            raise CanonicalActionError("shell.timeout_ms must be in [1, 120000]")
        if (
            isinstance(self.max_output_length, bool)
            or not isinstance(self.max_output_length, int)
            or not 1 <= self.max_output_length <= 20000
        ):
            raise CanonicalActionError("shell.max_output_length must be in [1, 20000]")


@dataclass(frozen=True)
class TerminateAction:
    kind: str
    status: str
    answer: str = ""

    def __post_init__(self) -> None:
        if self.kind != "terminate":
            raise CanonicalActionError("TerminateAction.kind must be terminate")
        if self.status not in {"success", "failure"}:
            raise CanonicalActionError("terminate.status must be success or failure")


@dataclass(frozen=True)
class SequenceAction:
    """One source runner action composed of multiple executable primitives."""

    kind: str
    actions: Tuple[Any, ...]

    def __post_init__(self) -> None:
        if self.kind != "sequence":
            raise CanonicalActionError("SequenceAction.kind must be sequence")
        if not self.actions:
            raise CanonicalActionError("sequence must contain at least one primitive")
        primitive_types = (
            ClickAction,
            TypeAction,
            HotkeyAction,
            KeyTransitionAction,
            MouseButtonTransitionAction,
            NoOpAction,
            ScrollAction,
            HorizontalScrollAction,
            MoveAction,
            DragAction,
            WaitAction,
            ShellAction,
            TerminateAction,
        )
        if any(not isinstance(action, primitive_types) for action in self.actions):
            raise CanonicalActionError("sequence.actions may only contain typed canonical primitives")
        if any(isinstance(action, SequenceAction) for action in self.actions):
            raise CanonicalActionError("sequence must not be nested")


Action = Union[
    ClickAction,
    TypeAction,
    HotkeyAction,
    KeyTransitionAction,
    MouseButtonTransitionAction,
    NoOpAction,
    ScrollAction,
    HorizontalScrollAction,
    MoveAction,
    DragAction,
    WaitAction,
    ShellAction,
    TerminateAction,
    SequenceAction,
]


def reframe_action(action: Action, frame_width: int, frame_height: int) -> Action:
    """Reproject pointer coordinates into a different screenshot frame."""

    _validate_frame(frame_width, frame_height)

    def coordinate(value: int, old_extent: int, new_extent: int) -> int:
        if old_extent == new_extent:
            return value
        if old_extent <= 1 or new_extent <= 1:
            raise CanonicalActionError("cannot reproject a frame extent smaller than 2")
        return round(value * (new_extent - 1) / (old_extent - 1))

    if isinstance(action, SequenceAction):
        return replace(
            action,
            actions=tuple(
                reframe_action(item, frame_width, frame_height)
                for item in action.actions
            ),
        )
    if isinstance(action, (ClickAction, MoveAction, ScrollAction, HorizontalScrollAction)):
        return replace(
            action,
            x_px=coordinate(action.x_px, action.frame_width, frame_width),
            y_px=coordinate(action.y_px, action.frame_height, frame_height),
            frame_width=frame_width,
            frame_height=frame_height,
        )
    if isinstance(action, DragAction):
        return replace(
            action,
            start_x_px=coordinate(action.start_x_px, action.frame_width, frame_width),
            start_y_px=coordinate(action.start_y_px, action.frame_height, frame_height),
            end_x_px=coordinate(action.end_x_px, action.frame_width, frame_width),
            end_y_px=coordinate(action.end_y_px, action.frame_height, frame_height),
            frame_width=frame_width,
            frame_height=frame_height,
        )
    return action


_ACTION_TYPES: Dict[str, Type[Any]] = {
    "click": ClickAction,
    "double_click": ClickAction,
    "type": TypeAction,
    "hotkey": HotkeyAction,
    "key_down": KeyTransitionAction,
    "key_up": KeyTransitionAction,
    "mouse_down": MouseButtonTransitionAction,
    "mouse_up": MouseButtonTransitionAction,
    "no_op": NoOpAction,
    "scroll": ScrollAction,
    "horizontal_scroll": HorizontalScrollAction,
    "move": MoveAction,
    "drag": DragAction,
    "wait": WaitAction,
    "shell": ShellAction,
    "terminate": TerminateAction,
    "sequence": SequenceAction,
}


def action_to_dict(action: Action) -> Dict[str, Any]:
    """Convert an action to a JSON-friendly dict."""

    if isinstance(action, SequenceAction):
        return {"kind": "sequence", "actions": [action_to_dict(item) for item in action.actions]}
    data = asdict(action)
    if isinstance(action, HotkeyAction):
        data["keys"] = list(action.keys)
    if isinstance(action, ShellAction):
        data["commands"] = list(action.commands)
    return data


def action_from_dict(raw: Mapping[str, Any]) -> Action:
    """Strictly parse a canonical action, rejecting unknown fields."""

    data = dict(raw)
    kind = data.get("kind")
    if kind not in _ACTION_TYPES:
        raise CanonicalActionError("unknown canonical action kind: %r" % kind)

    action_type = _ACTION_TYPES[str(kind)]
    allowed = {field.name for field in fields(action_type)}
    unknown = sorted(set(data) - allowed)
    if unknown:
        raise CanonicalActionError("action has unknown fields: %s" % ", ".join(unknown))

    if kind == "hotkey" and "keys" in data:
        if not isinstance(data["keys"], (list, tuple)):
            raise CanonicalActionError("hotkey.keys must be an array")
        data["keys"] = tuple(str(key) for key in data["keys"])
    if kind == "shell" and "commands" in data:
        if not isinstance(data["commands"], (list, tuple)):
            raise CanonicalActionError("shell.commands must be an array")
        data["commands"] = tuple(str(command) for command in data["commands"])
    if kind == "sequence" and "actions" in data:
        if not isinstance(data["actions"], (list, tuple)):
            raise CanonicalActionError("sequence.actions must be an array")
        data["actions"] = tuple(action_from_dict(item) for item in data["actions"])

    try:
        return action_type(**data)
    except TypeError as exc:
        raise CanonicalActionError("action has missing or mistyped fields: %s" % exc) from exc
