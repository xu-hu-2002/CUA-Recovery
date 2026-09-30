"""Canonical desktop actions.

设计原则：
1. 这里的 typed record 是唯一真值；summary、PyAutoGUI 和 native history 都是派生物。
2. canonical 坐标是截图/环境的真实像素，不使用某个模型私有的 0--1000 网格。
3. 所有坐标在构造对象时立即验证，错误不能拖到真实桌面执行阶段才暴露。
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields, replace
from typing import Any, Dict, Mapping, Tuple, Type, Union

DEFAULT_FRAME_WIDTH = 1280
DEFAULT_FRAME_HEIGHT = 800
VALID_MOUSE_BUTTONS = frozenset({"left", "right", "middle"})


class CanonicalActionError(ValueError):
    """输入不能无损解释为 DERAIL canonical action。"""


def _validate_frame(width: int, height: int) -> None:
    if width <= 0 or height <= 0:
        raise CanonicalActionError("frame_width 和 frame_height 必须为正整数")


def _validate_point(x_px: int, y_px: int, width: int, height: int, name: str) -> None:
    _validate_frame(width, height)
    if not (0 <= x_px < width and 0 <= y_px < height):
        raise CanonicalActionError(
            "%s=(%d, %d) 超出 %dx%d 截图边界" % (name, x_px, y_px, width, height)
        )


def _validate_button(button: str) -> None:
    if button not in VALID_MOUSE_BUTTONS:
        raise CanonicalActionError("不支持的鼠标按键: %s" % button)


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
            raise CanonicalActionError("ClickAction.kind 必须是 click 或 double_click")
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
            raise CanonicalActionError("TypeAction.kind 必须是 type")
        if not isinstance(self.text, str):
            raise CanonicalActionError("TypeAction.text 必须是字符串")
        if isinstance(self.interval_s, bool) or not isinstance(self.interval_s, (int, float)):
            raise CanonicalActionError("TypeAction.interval_s 必须是数字")
        if self.interval_s < 0:
            raise CanonicalActionError("TypeAction.interval_s 不能为负数")


@dataclass(frozen=True)
class HotkeyAction:
    kind: str
    keys: Tuple[str, ...]

    def __post_init__(self) -> None:
        if self.kind != "hotkey":
            raise CanonicalActionError("HotkeyAction.kind 必须是 hotkey")
        if not 1 <= len(self.keys) <= 32:
            raise CanonicalActionError("hotkey 必须包含 1 到 32 个按键")
        if any(not key.strip() for key in self.keys):
            raise CanonicalActionError("hotkey 中不能包含空按键")


@dataclass(frozen=True)
class KeyTransitionAction:
    kind: str
    key: str

    def __post_init__(self) -> None:
        if self.kind not in {"key_down", "key_up"}:
            raise CanonicalActionError("KeyTransitionAction.kind 必须是 key_down 或 key_up")
        if not isinstance(self.key, str) or not self.key.strip():
            raise CanonicalActionError("key transition 必须包含非空按键")


@dataclass(frozen=True)
class MouseButtonTransitionAction:
    kind: str
    button: str = "left"

    def __post_init__(self) -> None:
        if self.kind not in {"mouse_down", "mouse_up"}:
            raise CanonicalActionError(
                "MouseButtonTransitionAction.kind 必须是 mouse_down 或 mouse_up"
            )
        _validate_button(self.button)


@dataclass(frozen=True)
class NoOpAction:
    kind: str
    reason: str

    def __post_init__(self) -> None:
        if self.kind != "no_op":
            raise CanonicalActionError("NoOpAction.kind 必须是 no_op")
        if not isinstance(self.reason, str) or not self.reason.strip():
            raise CanonicalActionError("no_op.reason 不能为空")


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
            raise CanonicalActionError("ScrollAction.kind 必须是 scroll")
        _validate_point(self.x_px, self.y_px, self.frame_width, self.frame_height, "scroll")
        if self.delta_y == 0:
            raise CanonicalActionError("scroll.delta_y 不能为 0")


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
            raise CanonicalActionError("HorizontalScrollAction.kind 必须是 horizontal_scroll")
        _validate_point(self.x_px, self.y_px, self.frame_width, self.frame_height, "horizontal_scroll")
        if self.delta_x == 0:
            raise CanonicalActionError("horizontal_scroll.delta_x 不能为 0")


@dataclass(frozen=True)
class MoveAction:
    kind: str
    x_px: int
    y_px: int
    frame_width: int = DEFAULT_FRAME_WIDTH
    frame_height: int = DEFAULT_FRAME_HEIGHT

    def __post_init__(self) -> None:
        if self.kind != "move":
            raise CanonicalActionError("MoveAction.kind 必须是 move")
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
            raise CanonicalActionError("DragAction.kind 必须是 drag")
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
            raise CanonicalActionError("drag.duration_s 不能为负数")


@dataclass(frozen=True)
class WaitAction:
    kind: str
    seconds: float

    def __post_init__(self) -> None:
        if self.kind != "wait":
            raise CanonicalActionError("WaitAction.kind 必须是 wait")
        if self.seconds < 0:
            raise CanonicalActionError("wait.seconds 不能为负数")


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
            raise CanonicalActionError("TerminateAction.kind 必须是 terminate")
        if self.status not in {"success", "failure"}:
            raise CanonicalActionError("terminate.status 必须是 success 或 failure")


@dataclass(frozen=True)
class SequenceAction:
    """One source runner action composed of multiple executable primitives.

    It remains one global action for depth/horizon counting because MyPCBench executes the whole
    script in one ``env.step`` and records only one post-action observation.
    """

    kind: str
    actions: Tuple[Any, ...]

    def __post_init__(self) -> None:
        if self.kind != "sequence":
            raise CanonicalActionError("SequenceAction.kind 必须是 sequence")
        if not self.actions:
            raise CanonicalActionError("sequence 至少包含一个 primitive")
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
            raise CanonicalActionError("sequence.actions 只能包含 typed canonical primitives")
        if any(isinstance(action, SequenceAction) for action in self.actions):
            raise CanonicalActionError("sequence 不允许递归嵌套")


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
    """Reproject pointer coordinates into a different screenshot frame.

    MyPCBench's frozen Qwen3.8 scaffold runs at 1280x800, while a small
    subset of source trajectories was recorded at 1024x768. Endpoint-aware
    scaling keeps the replayed pointer location and the injected native tool
    call consistent with the current screenshot.
    """

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
    """把 action 转为 JSON-friendly dict；tuple keys 显式转为 list。"""

    if isinstance(action, SequenceAction):
        return {"kind": "sequence", "actions": [action_to_dict(item) for item in action.actions]}
    data = asdict(action)
    if isinstance(action, HotkeyAction):
        data["keys"] = list(action.keys)
    if isinstance(action, ShellAction):
        data["commands"] = list(action.commands)
    return data


def action_from_dict(raw: Mapping[str, Any]) -> Action:
    """严格解析 canonical action，拒绝未知字段而不是静默丢弃。

    严格模式很重要：字段拼写错误如果被忽略，可能把一个合法动作变成另一个动作，
    并在真实桌面上产生不可恢复的状态差异。
    """

    data = dict(raw)
    kind = data.get("kind")
    if kind not in _ACTION_TYPES:
        raise CanonicalActionError("未知 canonical action kind: %r" % kind)

    action_type = _ACTION_TYPES[str(kind)]
    allowed = {field.name for field in fields(action_type)}
    unknown = sorted(set(data) - allowed)
    if unknown:
        raise CanonicalActionError("action 包含未知字段: %s" % ", ".join(unknown))

    if kind == "hotkey" and "keys" in data:
        if not isinstance(data["keys"], (list, tuple)):
            raise CanonicalActionError("hotkey.keys 必须是数组")
        data["keys"] = tuple(str(key) for key in data["keys"])
    if kind == "shell" and "commands" in data:
        if not isinstance(data["commands"], (list, tuple)):
            raise CanonicalActionError("shell.commands must be an array")
        data["commands"] = tuple(str(command) for command in data["commands"])
    if kind == "sequence" and "actions" in data:
        if not isinstance(data["actions"], (list, tuple)):
            raise CanonicalActionError("sequence.actions 必须是数组")
        data["actions"] = tuple(action_from_dict(item) for item in data["actions"])

    try:
        return action_type(**data)
    except TypeError as exc:
        raise CanonicalActionError("action 缺失字段或字段类型错误: %s" % exc) from exc
