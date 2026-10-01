"""Lossless normalization of MyPCBench ``traj.jsonl`` executed actions."""

from __future__ import annotations

import ast
import base64
import binascii
import hashlib
import json
import re
import shlex
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Tuple

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
from recovery.canonical.trajectory import CanonicalStep
from recovery.derived.layout import atomic_write_json, atomic_write_jsonl, sha256_file
from recovery.derived.validation import strict_bool


class NormalizationError(ValueError):
    """A source action cannot be represented losslessly and safely."""


@dataclass(frozen=True)
class NormalizationIssue:
    source_line: int
    source_turn_index: Optional[int]
    source_action: str
    code: str
    message: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "source_line": self.source_line,
            "source_turn_index": self.source_turn_index,
            "source_action": self.source_action,
            "code": self.code,
            "message": self.message,
        }


def _literal(node: ast.AST) -> Any:
    try:
        return ast.literal_eval(node)
    except (ValueError, TypeError, SyntaxError) as exc:
        raise NormalizationError("PyAutoGUI arguments must be literals") from exc


def _parse_call(source: str) -> Tuple[str, List[Any], Dict[str, Any]]:
    try:
        expression = ast.parse(source.strip(), mode="eval").body
    except SyntaxError as exc:
        raise NormalizationError("not a single valid Python expression") from exc
    if not isinstance(expression, ast.Call):
        raise NormalizationError("only a single PyAutoGUI call is accepted")
    function = expression.func
    if not (
        isinstance(function, ast.Attribute)
        and isinstance(function.value, ast.Name)
        and function.value.id == "pyautogui"
    ):
        raise NormalizationError("only pyautogui.<primitive>(...) is accepted; arbitrary code is not allowed")
    args = [_literal(node) for node in expression.args]
    kwargs: Dict[str, Any] = {}
    for keyword in expression.keywords:
        if keyword.arg is None or keyword.arg in kwargs:
            raise NormalizationError("**kwargs and duplicate arguments are not accepted")
        kwargs[keyword.arg] = _literal(keyword.value)
    return function.attr, args, kwargs


_QWEN_BASE64_TYPING_TEMPLATE = """import base64, time, pyautogui
_text = base64.b64decode('').decode('utf-8')
_text = _text.replace('\\r\\n', '\\n').replace('\\r', '\\n')
for _line_index, _line in enumerate(_text.split('\\n')):
    for _part_index, _part in enumerate(_line.split('\\t')):
        if _part:
            pyautogui.typewrite(_part, interval=0.01)
        if _part_index < len(_line.split('\\t')) - 1:
            pyautogui.press('tab')
    if _line_index < len(_text.split('\\n')) - 1:
        pyautogui.press('enter')"""
_QWEN_BASE64_TYPING_TEMPLATE_AST = ast.dump(
    ast.parse(_QWEN_BASE64_TYPING_TEMPLATE, mode="exec"), include_attributes=False
)


def _parse_qwen_base64_typing_macro(source: str) -> Optional[Action]:
    if "base64.b64decode" not in source:
        return None
    try:
        program = ast.parse(source, mode="exec")
    except SyntaxError as exc:
        raise NormalizationError("Qwen base64 typing macro has invalid syntax") from exc
    calls = [
        node
        for node in ast.walk(program)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id == "base64"
        and node.func.attr == "b64decode"
    ]
    if len(calls) != 1 or len(calls[0].args) != 1 or calls[0].keywords:
        raise NormalizationError("Qwen base64 typing macro structure does not match")
    payload_node = calls[0].args[0]
    if not isinstance(payload_node, ast.Constant) or not isinstance(payload_node.value, str):
        raise NormalizationError("Qwen base64 typing macro payload must be a string literal")
    payload = payload_node.value
    payload_node.value = ""
    if ast.dump(program, include_attributes=False) != _QWEN_BASE64_TYPING_TEMPLATE_AST:
        raise NormalizationError("Qwen base64 typing macro does not match the frozen template")
    try:
        text = base64.b64decode(payload, validate=True).decode("utf-8")
    except (binascii.Error, ValueError, UnicodeDecodeError) as exc:
        raise NormalizationError("Qwen base64 typing macro payload is invalid") from exc
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    primitives: List[Action] = []
    lines = text.split("\n")
    for line_index, line in enumerate(lines):
        parts = line.split("\t")
        for part_index, part in enumerate(parts):
            if part:
                primitives.append(TypeAction(kind="type", text=part))
            if part_index < len(parts) - 1:
                primitives.append(HotkeyAction(kind="hotkey", keys=("tab",)))
        if line_index < len(lines) - 1:
            primitives.append(HotkeyAction(kind="hotkey", keys=("enter",)))
    if not primitives:
        return NoOpAction(kind="no_op", reason="empty Qwen typing macro")
    if len(primitives) == 1:
        return primitives[0]
    return SequenceAction(kind="sequence", actions=tuple(primitives))


def _parse_time_sleep_program(source: str) -> Optional[WaitAction]:
    if "time.sleep" not in source:
        return None
    try:
        program = ast.parse(source, mode="exec")
    except SyntaxError as exc:
        raise NormalizationError("time.sleep program has invalid syntax") from exc
    if len(program.body) != 2:
        raise NormalizationError("time.sleep program may only contain import time and one sleep")
    import_node, call_node = program.body
    valid_import = (
        isinstance(import_node, ast.Import)
        and len(import_node.names) == 1
        and import_node.names[0].name == "time"
        and import_node.names[0].asname is None
    )
    valid_call = (
        isinstance(call_node, ast.Expr)
        and isinstance(call_node.value, ast.Call)
        and isinstance(call_node.value.func, ast.Attribute)
        and isinstance(call_node.value.func.value, ast.Name)
        and call_node.value.func.value.id == "time"
        and call_node.value.func.attr == "sleep"
        and len(call_node.value.args) == 1
        and not call_node.value.keywords
    )
    if not valid_import or not valid_call:
        raise NormalizationError("time.sleep program does not match the allowed fixed structure")
    seconds = _number(_literal(call_node.value.args[0]), "seconds")
    if seconds < 0:
        raise NormalizationError("time.sleep seconds must not be negative")
    return WaitAction(kind="wait", seconds=seconds)


def _number(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise NormalizationError("%s must be a number" % name)
    return float(value)


def _integer(value: Any, name: str) -> int:
    number = _number(value, name)
    if not number.is_integer():
        raise NormalizationError("%s must be an integer pixel" % name)
    return int(number)


def _bind_positionals(
    method: str,
    args: List[Any],
    kwargs: Dict[str, Any],
    positional_names: Tuple[str, ...],
) -> Dict[str, Any]:
    if len(args) > len(positional_names):
        raise NormalizationError(
            "%s has too many positional arguments; refusing to drop them silently: %d" % (method, len(args))
        )
    bound = dict(kwargs)
    for index, value in enumerate(args):
        name = positional_names[index]
        if name in bound:
            raise NormalizationError("%s argument %s given both positionally and by keyword" % (method, name))
        bound[name] = value
    return bound


class PyAutoGUINormalizer:
    """Stateful parser needed for cursor-relative ``scroll`` and ``dragTo``."""

    def __init__(self, frame_width: int = 1280, frame_height: int = 800) -> None:
        self.frame_width = frame_width
        self.frame_height = frame_height
        self.cursor: Optional[Tuple[int, int]] = None
        self.clamps: List[Tuple[Tuple[int, int], Tuple[int, int]]] = []

    def _clamp_point(self, x_px: int, y_px: int) -> Tuple[int, int]:
        clamped = (
            min(max(x_px, 0), self.frame_width - 1),
            min(max(y_px, 0), self.frame_height - 1),
        )
        if clamped != (x_px, y_px):
            self.clamps.append(((x_px, y_px), clamped))
        return clamped

    def normalize(self, source: str, *, wait_seconds: Optional[float]) -> Action:
        qwen_typing = _parse_qwen_base64_typing_macro(source)
        if qwen_typing is not None:
            return qwen_typing
        sleep_action = _parse_time_sleep_program(source)
        if sleep_action is not None:
            return sleep_action
        statements = [line.strip() for line in source.splitlines() if line.strip()]
        if len(statements) > 1 or ";" in source:
            try:
                program = ast.parse(source, mode="exec")
            except SyntaxError as exc:
                raise NormalizationError("PyAutoGUI program has invalid syntax") from exc
            body = list(program.body)
            if body and isinstance(body[0], ast.Import):
                import_node = body.pop(0)
                if not (
                    len(import_node.names) == 1
                    and import_node.names[0].name == "pyautogui"
                    and import_node.names[0].asname is None
                ):
                    raise NormalizationError("PyAutoGUI program only allows an optional import pyautogui")
            if not body or any(not isinstance(node, ast.Expr) for node in body):
                raise NormalizationError("PyAutoGUI program only allows expression calls")
            statements = [ast.unparse(node.value) for node in body]
        if len(statements) > 1:
            primitives = tuple(
                self.normalize(statement, wait_seconds=wait_seconds) for statement in statements
            )
            if len(primitives) == 2 and isinstance(primitives[0], MoveAction) and isinstance(
                primitives[1], ScrollAction
            ):
                return primitives[1]
            flattened: List[Action] = []
            for primitive in primitives:
                if isinstance(primitive, SequenceAction):
                    flattened.extend(primitive.actions)
                else:
                    flattened.append(primitive)
            return SequenceAction(kind="sequence", actions=tuple(flattened))
        if statements:
            source = statements[0]
        sentinel = source.strip().upper()
        if sentinel == "DONE":
            return TerminateAction(kind="terminate", status="success")
        if sentinel == "FAIL":
            return TerminateAction(kind="terminate", status="failure")
        if sentinel == "WAIT":
            if wait_seconds is None:
                raise NormalizationError("WAIT needs wait_seconds from the runner config")
            return WaitAction(kind="wait", seconds=wait_seconds)
        if sentinel == "NO_ACTIONS":
            return NoOpAction(kind="no_op", reason="NO_ACTIONS")
        if sentinel in {"PREDICT_CRASH", "PROTOCOL_ERROR", "INFRASTRUCTURE_ERROR"}:
            return TerminateAction(kind="terminate", status="failure", answer=sentinel)
        if sentinel == "TOOL_CALL":
            return NoOpAction(kind="no_op", reason="non-desktop tool call")

        method, args, kwargs = _parse_call(source)
        allowed_kwargs = {
            "click": {"x", "y", "button", "clicks", "interval", "duration"},
            "doubleClick": {"x", "y", "button", "interval", "duration"},
            "tripleClick": {"x", "y", "button", "interval", "duration"},
            "rightClick": {"x", "y", "duration"},
            "moveTo": {"x", "y", "duration"},
            "scroll": {"clicks", "x", "y"},
            "hscroll": {"clicks", "x", "y"},
            "write": {"message", "interval"},
            "typewrite": {"message", "interval"},
            "press": {"keys", "presses", "interval"},
            "hotkey": {"interval"},
            "keyDown": set(),
            "keyUp": set(),
            "mouseDown": {"button"},
            "mouseUp": {"button"},
            "sleep": set(),
            "dragTo": {"x", "y", "duration", "button"},
        }
        if method not in allowed_kwargs:
            raise NormalizationError("unsupported PyAutoGUI primitive: %s" % method)
        unknown = set(kwargs) - allowed_kwargs[method]
        if unknown:
            raise NormalizationError("%s has unknown arguments: %s" % (method, sorted(unknown)))

        if method in {"click", "doubleClick", "tripleClick", "rightClick"}:
            positional = (
                ("x", "y", "clicks", "interval", "button", "duration")
                if method == "click"
                else (
                    ("x", "y", "interval", "button", "duration")
                    if method in {"doubleClick", "tripleClick"}
                    else ("x", "y", "duration")
                )
            )
            bound = _bind_positionals(method, args, kwargs, positional)
            x_raw = bound.get("x")
            y_raw = bound.get("y")
            if x_raw is None or y_raw is None:
                if self.cursor is None:
                    raise NormalizationError("click without coordinates needs a known cursor state")
                x, y = self.cursor
            else:
                x, y = self._clamp_point(_integer(x_raw, "x"), _integer(y_raw, "y"))
            if _number(bound.get("interval", 0), "interval") != 0:
                raise NormalizationError("click with interval has no lossless canonical form yet")
            if _number(bound.get("duration", 0), "duration") != 0:
                raise NormalizationError("click with duration has no lossless canonical form yet")
            button = "right" if method == "rightClick" else str(bound.get("button", "left"))
            clicks = _integer(bound.get("clicks", 1), "clicks")
            if method == "doubleClick":
                clicks = 2
            elif method == "tripleClick":
                clicks = 3
            if clicks not in {1, 2, 3}:
                raise NormalizationError("clicks=%d cannot be mapped losslessly to a canonical click" % clicks)
            self.cursor = (x, y)
            click = ClickAction(
                kind="double_click" if clicks == 2 else "click",
                x_px=x,
                y_px=y,
                button=button,
                frame_width=self.frame_width,
                frame_height=self.frame_height,
            )
            return (
                SequenceAction(kind="sequence", actions=(click, click, click))
                if clicks == 3
                else click
            )

        if method == "moveTo":
            bound = _bind_positionals(method, args, kwargs, ("x", "y", "duration"))
            if not {"x", "y"}.issubset(bound):
                raise NormalizationError("moveTo is missing x/y")
            x, y = self._clamp_point(_integer(bound["x"], "x"), _integer(bound["y"], "y"))
            duration = _number(bound.get("duration", 0), "duration")
            if duration != 0:
                raise NormalizationError("moveTo with duration cannot be mapped losslessly to a canonical move")
            self.cursor = (x, y)
            return MoveAction(
                kind="move", x_px=x, y_px=y,
                frame_width=self.frame_width, frame_height=self.frame_height,
            )

        if method in {"scroll", "hscroll"}:
            bound = _bind_positionals(method, args, kwargs, ("clicks", "x", "y"))
            raw_delta = bound.get("clicks")
            if raw_delta is None:
                raise NormalizationError("%s is missing clicks" % method)
            delta = _integer(raw_delta, "clicks")
            if delta == 0 and bound.get("x") is None and bound.get("y") is None:
                return NoOpAction(kind="no_op", reason="%s(0)" % method)
            if (bound.get("x") is None) != (bound.get("y") is None):
                raise NormalizationError("%s x/y must be given together" % method)
            if bound.get("x") is not None:
                self.cursor = self._clamp_point(
                    _integer(bound["x"], "x"), _integer(bound["y"], "y")
                )
            if self.cursor is None:
                raise NormalizationError("%s needs a known cursor state or explicit x/y" % method)
            common = {
                "x_px": self.cursor[0],
                "y_px": self.cursor[1],
                "frame_width": self.frame_width,
                "frame_height": self.frame_height,
            }
            if method == "hscroll":
                return HorizontalScrollAction(
                    kind="horizontal_scroll",
                    delta_x=delta,
                    **common,
                )
            return ScrollAction(kind="scroll", delta_y=delta, **common)

        if method in {"write", "typewrite"}:
            bound = _bind_positionals(method, args, kwargs, ("message", "interval"))
            raw_text = bound.get("message")
            if not isinstance(raw_text, str):
                raise NormalizationError("%s is missing a string message" % method)
            return TypeAction(
                kind="type",
                text=raw_text,
                interval_s=_number(bound.get("interval", 0), "interval"),
            )

        if method == "press":
            bound = _bind_positionals(method, args, kwargs, ("keys", "presses", "interval"))
            raw_keys = bound.get("keys")
            keys = [raw_keys] if isinstance(raw_keys, str) else raw_keys
            if not isinstance(keys, (list, tuple)) or not keys or any(
                not isinstance(key, str) for key in keys
            ):
                raise NormalizationError("press keys must be a string or a non-empty array of strings")
            presses = _integer(bound.get("presses", 1), "presses")
            if presses < 1:
                raise NormalizationError("presses must be a positive integer")
            if _number(bound.get("interval", 0), "interval") != 0:
                raise NormalizationError("press with interval has no lossless canonical form yet")
            primitives = tuple(
                HotkeyAction(kind="hotkey", keys=(("space" if key == " " else key),))
                for _ in range(presses)
                for key in keys
            )
            return primitives[0] if len(primitives) == 1 else SequenceAction(
                kind="sequence", actions=primitives
            )

        if method == "hotkey":
            if len(args) == 1 and isinstance(args[0], (list, tuple)):
                args = list(args[0])
            if not args or any(not isinstance(key, str) for key in args):
                raise NormalizationError("hotkey arguments must be strings")
            if _number(kwargs.get("interval", 0), "interval") != 0:
                raise NormalizationError("hotkey with interval has no lossless canonical form yet")
            return HotkeyAction(kind="hotkey", keys=tuple(args))

        if method in {"keyDown", "keyUp"}:
            bound = _bind_positionals(method, args, kwargs, ("key",))
            key = bound.get("key")
            if not isinstance(key, str) or not key.strip():
                raise NormalizationError("%s needs a non-empty string key" % method)
            return KeyTransitionAction(
                kind="key_down" if method == "keyDown" else "key_up",
                key=key,
            )

        if method in {"mouseDown", "mouseUp"}:
            bound = _bind_positionals(method, args, kwargs, ("button",))
            return MouseButtonTransitionAction(
                kind="mouse_down" if method == "mouseDown" else "mouse_up",
                button=str(bound.get("button", "left")),
            )

        if method == "sleep":
            bound = _bind_positionals(method, args, kwargs, ("seconds",))
            if "seconds" not in bound:
                raise NormalizationError("sleep is missing seconds")
            seconds = _number(bound["seconds"], "seconds")
            if seconds < 0:
                raise NormalizationError("sleep seconds must not be negative")
            return WaitAction(kind="wait", seconds=seconds)

        if method == "dragTo":
            if self.cursor is None:
                raise NormalizationError("dragTo needs a known starting cursor state")
            bound = _bind_positionals(method, args, kwargs, ("x", "y", "duration"))
            if not {"x", "y"}.issubset(bound):
                raise NormalizationError("dragTo is missing x/y")
            end_x, end_y = self._clamp_point(
                _integer(bound["x"], "x"), _integer(bound["y"], "y")
            )
            action = DragAction(
                kind="drag",
                start_x_px=self.cursor[0],
                start_y_px=self.cursor[1],
                end_x_px=end_x,
                end_y_px=end_y,
                duration_s=_number(bound.get("duration", 0.5), "duration"),
                button=str(bound.get("button", "left")),
                frame_width=self.frame_width,
                frame_height=self.frame_height,
            )
            self.cursor = (end_x, end_y)
            return action
        raise AssertionError("unreachable")


def _hash_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _png_dimensions(path: Path) -> Tuple[int, int]:
    with path.open("rb") as handle:
        header = handle.read(24)
    if len(header) < 24 or header[:8] != b"\x89PNG\r\n\x1a\n" or header[12:16] != b"IHDR":
        raise NormalizationError("not a recognizable PNG screenshot: %s" % path)
    return struct.unpack(">II", header[16:24])


def _detect_task_frame(task_dir: Path) -> Tuple[int, int]:
    for path in sorted(task_dir.glob("step_*.png")):
        return _png_dimensions(path)
    raise NormalizationError("cannot auto-detect frame: task has no screenshot")


def _public_tool_messages(raw: Dict[str, Any]) -> List[Dict[str, Any]]:
    metadata = raw.get("agent_metadata")
    if isinstance(metadata, dict) and isinstance(metadata.get("tool_messages"), list):
        return [dict(item) for item in metadata["tool_messages"] if isinstance(item, dict)]
    messages: List[Dict[str, Any]] = []
    response = raw.get("response")
    if not isinstance(response, str):
        return messages
    try:
        response_object = json.loads(response)
    except json.JSONDecodeError:
        response_object = None
    if isinstance(response_object, dict) and isinstance(
        response_object.get("tool_messages"), list
    ):
        return [
            dict(item)
            for item in response_object["tool_messages"]
            if isinstance(item, dict)
        ]
    for line in response.splitlines():
        candidate = line.strip()
        if candidate.startswith("[tool] "):
            candidate = candidate[len("[tool] ") :]
        if not candidate.startswith("{"):
            continue
        try:
            item = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(item, dict) and item.get("type") in {
            "shell_call",
            "shell_call_output",
            "tool_use",
            "tool_result",
        }:
            messages.append(item)
    return messages


def _openai_tool_messages(raw: Dict[str, Any]) -> List[Dict[str, Any]]:
    response = raw.get("response")
    if not isinstance(response, str):
        return []
    try:
        payload = json.loads(response)
    except json.JSONDecodeError:
        return []
    if not isinstance(payload, dict) or not isinstance(payload.get("tool_calls"), list):
        return []
    messages = []
    for call in payload["tool_calls"]:
        function = call.get("function") if isinstance(call, dict) else None
        if not isinstance(function, dict) or function.get("name") != "bash":
            continue
        try:
            arguments = json.loads(str(function.get("arguments") or ""))
        except json.JSONDecodeError as exc:
            raise NormalizationError("bash tool arguments are invalid JSON") from exc
        messages.append(
            {
                "type": "tool_use",
                "id": str(call.get("id") or ""),
                "name": "bash",
                "input": arguments,
            }
        )
    return messages


def _xml_shell_tool_messages(raw: Dict[str, Any]) -> List[Dict[str, Any]]:
    response = raw.get("response")
    if not isinstance(response, str):
        return []
    messages = []
    for index, match in enumerate(
        re.finditer(r"<tool_call>(.*?)</tool_call>", response, re.DOTALL)
    ):
        body = match.group(1)
        if not re.search(r"<function=(?:bash|computer_use)>", body):
            continue
        command = re.search(
            r"<parameter=command>\s*(.*?)\s*</parameter>", body, re.DOTALL
        )
        if command:
            messages.append({
                "type": "tool_use",
                "id": f"recorded_xml_shell_{index}",
                "name": "bash",
                "input": {"command": command.group(1).strip()},
            })
    return messages


def _is_assistant_only_tool_call(raw: Dict[str, Any], source_agent: str) -> bool:
    response = raw.get("response")
    if source_agent != "qwen3_5_35b_a3b" or not isinstance(response, str):
        return False
    return not (
        _public_tool_messages(raw)
        or _openai_tool_messages(raw)
        or _xml_shell_tool_messages(raw)
    )


def _task_tool_results(task_dir: Path) -> Dict[str, str]:
    path = task_dir / "messages.json"
    if not path.is_file():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise NormalizationError("messages.json must contain an array")
    bash_ids = {
        str(call.get("id") or "")
        for message in payload
        if isinstance(message, dict) and message.get("role") == "assistant"
        for call in message.get("tool_calls", [])
        if isinstance(call, dict)
        and isinstance(call.get("function"), dict)
        and call["function"].get("name") == "bash"
    }
    results: Dict[str, str] = {}
    for message in payload:
        if not isinstance(message, dict) or message.get("role") != "tool":
            continue
        call_id = str(message.get("tool_call_id") or "")
        if call_id not in bash_ids:
            continue
        content = message.get("content")
        if not call_id or not isinstance(content, str):
            raise NormalizationError("tool message must contain tool_call_id and string content")
        if call_id in results and results[call_id] != content:
            raise NormalizationError(f"conflicting tool results for call id {call_id}")
        results[call_id] = content
    return results


def _anthropic_tool_messages(task_dir: Path, source_line: int) -> List[Dict[str, Any]]:
    path = task_dir / "messages.json"
    if not path.is_file():
        return []
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise NormalizationError("messages.json must contain an array")
    row_cursor = 1
    for index, message in enumerate(payload):
        if not isinstance(message, dict) or message.get("role") != "assistant":
            continue
        content = message.get("content") if isinstance(message.get("content"), list) else []
        computer_calls = [
            block for block in content
            if isinstance(block, dict) and block.get("type") == "tool_use"
            and block.get("name") == "computer"
        ]
        row_count = max(1, len(computer_calls))
        if row_cursor <= source_line < row_cursor + row_count:
            result = payload[index + 1] if index + 1 < len(payload) else {}
            result_content = result.get("content", []) if isinstance(result, dict) else []
            return [
                dict(block) for block in [*content, *result_content]
                if isinstance(block, dict)
            ]
        row_cursor += row_count
    return []


def _normalize_shell_tool_row(
    raw: Dict[str, Any], external_results: Optional[Mapping[str, str]] = None
) -> Tuple[Action, str]:
    messages = (
        _public_tool_messages(raw)
        or _openai_tool_messages(raw)
        or _xml_shell_tool_messages(raw)
    )
    results: Dict[str, Any] = dict(external_results or {})
    calls: List[Tuple[str, ShellAction]] = []
    for item in messages:
        item_type = item.get("type")
        if item_type == "shell_call_output":
            results[str(item.get("call_id") or "")] = item.get("output", [])
            continue
        if item_type == "tool_result":
            results[str(item.get("tool_use_id") or "")] = {
                "content": item.get("content", ""),
                "is_error": bool(item.get("is_error", False)),
            }
            continue
        if item_type == "shell_call":
            call_id = str(item.get("call_id") or "")
            payload = item.get("action")
            if not isinstance(payload, dict):
                raise NormalizationError("shell_call.action must be an object")
            commands = payload.get("commands")
            timeout_ms = payload.get("timeout_ms") or 120000
            max_output_length = payload.get("max_output_length") or 8192
        elif item_type == "tool_use" and item.get("name") == "bash":
            call_id = str(item.get("id") or "")
            payload = item.get("input")
            if not isinstance(payload, dict):
                raise NormalizationError("bash tool input must be an object")
            command = payload.get("command")
            commands = payload.get("commands", [command] if isinstance(command, str) else None)
            timeout_ms = payload.get("timeout_ms") or 120000
            max_output_length = payload.get("max_output_length") or 8192
        elif item_type == "tool_use" and item.get("name") in {
            "str_replace_editor", "str_replace_based_edit_tool"
        }:
            call_id = str(item.get("id") or "")
            payload = item.get("input")
            if not isinstance(payload, dict):
                raise NormalizationError("text editor input must be an object")
            commands = [_editor_replay_command(payload)]
            timeout_ms = 120000
            max_output_length = 8192
        else:
            continue
        if not isinstance(commands, list) or not commands or any(
            not isinstance(command, str) or not command.strip() for command in commands
        ):
            raise NormalizationError("captured shell call has no valid command list")
        if (
            isinstance(timeout_ms, bool)
            or not isinstance(timeout_ms, int)
            or not 1 <= timeout_ms <= 120000
        ):
            raise NormalizationError("captured shell timeout_ms is invalid")
        if (
            isinstance(max_output_length, bool)
            or not isinstance(max_output_length, int)
            or not 1 <= max_output_length <= 20000
        ):
            raise NormalizationError("captured shell max_output_length is invalid")
        calls.append(
            (
                call_id,
                ShellAction(
                    kind="shell",
                    commands=tuple(commands),
                    timeout_ms=timeout_ms,
                    max_output_length=max_output_length,
                ),
            )
        )
    if not calls:
        raise NormalizationError("TOOL_CALL row has no captured bash/shell call")
    action: Action = calls[0][1]
    if len(calls) > 1:
        action = SequenceAction(kind="sequence", actions=tuple(call[1] for call in calls))
    result = {
        "shell_results": [
            {"call_id": call_id, "result": results.get(call_id, "(result unavailable)")}
            for call_id, _action in calls
        ]
    }
    return action, json.dumps(result, ensure_ascii=False, sort_keys=True)


def _b64_text(value: str) -> str:
    return base64.b64encode(value.encode("utf-8")).decode("ascii")


def _editor_replay_command(payload: Dict[str, Any]) -> str:
    operation = payload.get("command")
    path = payload.get("path")
    if not isinstance(path, str) or not path.startswith("/"):
        raise NormalizationError("text editor path must be absolute")
    if operation == "view":
        return f"test -e {shlex.quote(path)}"
    if operation == "create" and isinstance(payload.get("file_text"), str):
        path64, text64 = _b64_text(path), _b64_text(payload["file_text"])
        return (
            "python3 -c \"import base64,pathlib;"
            f"p=pathlib.Path(base64.b64decode('{path64}').decode());"
            f"p.write_bytes(base64.b64decode('{text64}'))\""
        )
    if operation == "str_replace" and all(
        isinstance(payload.get(key), str) for key in ("old_str", "new_str")
    ):
        path64 = _b64_text(path)
        old64 = _b64_text(payload["old_str"])
        new64 = _b64_text(payload["new_str"])
        return (
            "python3 -c \"import base64,pathlib;"
            f"p=pathlib.Path(base64.b64decode('{path64}').decode());s=p.read_text();"
            f"o=base64.b64decode('{old64}').decode();n=base64.b64decode('{new64}').decode();"
            "assert s.count(o)==1;p.write_text(s.replace(o,n,1))\""
        )
    raise NormalizationError(f"unsupported text editor operation: {operation!r}")


def _task_provenance(
    task_dir: Path,
    output_dir: Path,
    task_config_source: Optional[Path] = None,
) -> Dict[str, Any]:
    task_id = task_dir.name
    batch_path = (
        task_config_source.resolve()
        if task_config_source is not None
        else task_dir.parent / "_tasks" / "batch.json"
    )
    if not batch_path.is_file():
        raise NormalizationError("missing task batch provenance: %s" % batch_path)
    try:
        batch = json.loads(batch_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        raise NormalizationError("invalid task batch JSON") from exc
    if isinstance(batch, dict):
        batch = [batch]
    if not isinstance(batch, list):
        raise NormalizationError("task batch top level must be an array or a single task object")
    matches = [item for item in batch if isinstance(item, dict) and item.get("id") == task_id]
    if len(matches) != 1:
        raise NormalizationError("task batch must contain exactly one task id=%s" % task_id)
    task_config = matches[0]
    instruction = task_config.get("instruction")
    if not isinstance(instruction, str) or not instruction.strip():
        raise NormalizationError("task config is missing a non-empty instruction")
    task_config_path = output_dir / "task_config.json"
    atomic_write_json(task_config_path, task_config)
    rubric_path = task_dir / "rubric_bundle.json"
    if not rubric_path.is_file():
        raise NormalizationError("missing PESR rubric_bundle.json")
    pre_command = task_config.get("pre_command", "")
    if not isinstance(pre_command, str):
        raise NormalizationError("task pre_command must be a string")
    return {
        "task_id": task_id,
        "instruction": instruction,
        "instruction_sha256": _hash_text(instruction),
        "pre_command_sha256": _hash_text(pre_command),
        "task_config_uri": str(task_config_path.resolve()),
        "task_config_sha256": sha256_file(task_config_path),
        "source_batch_uri": str(batch_path.resolve()),
        "source_batch_sha256": sha256_file(batch_path),
        "rubric_bundle_uri": str(rubric_path.resolve()),
        "rubric_bundle_sha256": sha256_file(rubric_path),
    }


def normalize_task_directory(
    task_dir: Path,
    output_dir: Path,
    *,
    trajectory_id: str,
    source_agent: str,
    wait_seconds: Optional[float],
    frame_width: Optional[int] = None,
    frame_height: Optional[int] = None,
    task_config_source: Optional[Path] = None,
) -> Dict[str, Any]:
    """Normalize one task and atomically write its canonical records/report."""

    traj_path = task_dir / "traj.jsonl"
    if not traj_path.is_file():
        raise NormalizationError("missing traj.jsonl: %s" % traj_path)
    if (frame_width is None) != (frame_height is None):
        raise NormalizationError("frame_width/frame_height must both be given or both auto-detected")
    if frame_width is None or frame_height is None:
        frame_width, frame_height = _detect_task_frame(task_dir)
    parser = PyAutoGUINormalizer(frame_width, frame_height)
    steps: List[CanonicalStep] = []
    issues: List[NormalizationIssue] = []
    per_turn_count: Dict[int, int] = {}
    previous_observation_sha = ""
    previous_observation_uri = ""
    source_records = 0
    provenance_issue: Optional[NormalizationIssue] = None
    try:
        task_provenance = _task_provenance(task_dir, output_dir, task_config_source)
    except NormalizationError as exc:
        task_provenance = {}
        provenance_issue = NormalizationIssue(0, None, "", "missing_task_provenance", str(exc))
        issues.append(provenance_issue)

    task_tool_results = _task_tool_results(task_dir)
    with traj_path.open(encoding="utf-8") as handle:
        for source_line, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            source_records += 1
            try:
                raw = json.loads(line)
            except json.JSONDecodeError as exc:
                issues.append(NormalizationIssue(source_line, None, "", "invalid_json", str(exc)))
                continue
            turn_raw = raw.get("step_num")
            turn_index = int(turn_raw) - 1 if isinstance(turn_raw, int) else None
            source_action = str(raw.get("action", ""))
            parser.clamps.clear()
            captured_tool_result = ""
            try:
                if source_action.strip().upper() == "TOOL_CALL":
                    native_messages = (
                        _anthropic_tool_messages(task_dir, source_line)
                        if source_agent == "claude_opus_4_8"
                        else []
                    )
                    if native_messages:
                        raw = dict(raw)
                        raw["agent_metadata"] = {"tool_messages": native_messages}
                    if _is_assistant_only_tool_call(raw, source_agent):
                        action = NoOpAction(kind="no_op", reason="assistant-only terminal row")
                    else:
                        action, captured_tool_result = _normalize_shell_tool_row(
                            raw, task_tool_results
                        )
                else:
                    action = parser.normalize(source_action, wait_seconds=wait_seconds)
            except (NormalizationError, ValueError, TypeError) as exc:
                if source_action.strip().upper() == "TOOL_CALL":
                    action = NoOpAction(kind="no_op", reason="non-desktop tool call")
                    issues.append(
                        NormalizationIssue(
                            source_line,
                            turn_index,
                            source_action,
                            "non_desktop_tool_call",
                            str(exc),
                        )
                    )
                else:
                    issues.append(
                        NormalizationIssue(
                            source_line,
                            turn_index,
                            source_action,
                            "unsupported_action",
                            str(exc),
                        )
                    )
                    continue

            if parser.clamps:
                issues.append(
                    NormalizationIssue(
                        source_line,
                        turn_index,
                        source_action,
                        "coordinate_clamped",
                        "; ".join(
                            "(%d, %d) -> (%d, %d), delta=(%d, %d)"
                            % (ox, oy, cx, cy, cx - ox, cy - oy)
                            for (ox, oy), (cx, cy) in parser.clamps
                        ),
                    )
                )

            if source_action.strip().upper() == "TOOL_CALL" and not isinstance(
                action, (ShellAction, SequenceAction)
            ) and not (
                isinstance(action, NoOpAction)
                and action.reason == "assistant-only terminal row"
            ) and not any(
                issue.source_line == source_line and issue.code == "non_desktop_tool_call"
                for issue in issues
            ):
                issues.append(
                    NormalizationIssue(
                        source_line,
                        turn_index,
                        source_action,
                        "non_desktop_tool_call",
                        "kept as a non-desktop tool step for human annotation; not a replayable desktop action",
                    )
                )

            if turn_index is None:
                issues.append(
                    NormalizationIssue(
                        source_line,
                        None,
                        source_action,
                        "missing_turn_index",
                        "step_num is missing or not an integer",
                    )
                )
                continue
            within_turn = per_turn_count.get(turn_index, 0)
            per_turn_count[turn_index] = within_turn + 1
            screenshot_name = str(raw.get("screenshot_file", ""))
            screenshot_path = task_dir / screenshot_name if screenshot_name else None
            if screenshot_path is not None and screenshot_path.is_file():
                observation_after_sha = sha256_file(screenshot_path)
                observation_after_uri = str(screenshot_path.resolve())
                try:
                    actual_frame = _png_dimensions(screenshot_path)
                    if actual_frame != (frame_width, frame_height):
                        issues.append(
                            NormalizationIssue(
                                source_line,
                                turn_index,
                                source_action,
                                "frame_mismatch",
                                "screenshot=%s, expected=%s"
                                % (actual_frame, (frame_width, frame_height)),
                            )
                        )
                except NormalizationError as exc:
                    issues.append(
                        NormalizationIssue(
                            source_line, turn_index, source_action, "invalid_screenshot", str(exc)
                        )
                    )
            else:
                shell_only = isinstance(action, ShellAction) or (
                    isinstance(action, SequenceAction)
                    and all(isinstance(item, ShellAction) for item in action.actions)
                )
                if isinstance(action, (NoOpAction, TerminateAction)) or shell_only:
                    observation_after_sha = previous_observation_sha
                    observation_after_uri = previous_observation_uri
                else:
                    observation_after_sha = ""
                    observation_after_uri = ""
                    issues.append(
                        NormalizationIssue(
                            source_line,
                            turn_index,
                            source_action,
                            "missing_screenshot",
                            "source row has no usable post-action screenshot",
                        )
                    )
            action_index = source_records - 1
            steps.append(
                CanonicalStep(
                    step_id=action_index,
                    turn_index=turn_index,
                    action_index_within_turn=within_turn,
                    action=action,
                    observation_before_sha256=previous_observation_sha,
                    observation_before_uri=previous_observation_uri,
                    observation_after_sha256=observation_after_sha,
                    observation_after_uri=observation_after_uri,
                    tool_result=(
                        captured_tool_result
                        or (
                            "done"
                            if strict_bool(raw.get("done", False), "raw.done")
                            else "ok"
                        )
                    ),
                    source_agent=source_agent,
                    source_step_id=turn_index,
                    source_action_timestamp=str(raw.get("action_timestamp", "")),
                    source_record_uri="%s#line=%d" % (traj_path.resolve(), source_line),
                )
            )
            previous_observation_sha = observation_after_sha
            previous_observation_uri = observation_after_uri

    output_dir.mkdir(parents=True, exist_ok=True)
    canonical_path = output_dir / "trajectory.jsonl"
    atomic_write_jsonl(canonical_path, (step.to_dict() for step in steps))
    annotation_complete = len(steps) == source_records and all(
        issue.code in {"missing_screenshot", "non_desktop_tool_call", "coordinate_clamped"}
        for issue in issues
    )
    report: Dict[str, Any] = {
        "trajectory_id": trajectory_id,
        "source_trajectory_uri": str(traj_path.resolve()),
        "source_trajectory_sha256": sha256_file(traj_path),
        "canonical_trajectory_uri": str(canonical_path.resolve()),
        "canonical_trajectory_sha256": sha256_file(canonical_path),
        "source_record_count": source_records,
        "canonical_action_count": len(steps),
        "turn_count": len(per_turn_count),
        "normalization_complete": not issues and len(steps) == source_records,
        "annotation_complete": annotation_complete,
        "case_eligible": not issues and len(steps) == source_records,
        "root_action_zero_eligible": bool(
            steps
            and steps[0].observation_before_sha256
            and steps[0].observation_before_uri
        ),
        "wait_seconds": wait_seconds,
        "frame": [frame_width, frame_height],
        "issues": [issue.to_dict() for issue in issues],
        "response_storage_policy": "raw_only_not_copied_to_canonical",
        "source_lines_digest": _hash_text("%d:%d" % (source_records, len(steps))),
        "task_provenance": task_provenance,
    }
    atomic_write_json(output_dir / "normalization_report.json", report)
    return report


def load_canonical_jsonl(path: Path, *, allow_gaps: bool = False) -> Tuple[CanonicalStep, ...]:
    """Load canonical steps; ``allow_gaps`` admits a repaired prefix with removed actions."""

    records = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                records.append(CanonicalStep.from_dict(json.loads(line)))
    actual = [record.action_index_global for record in records]
    if allow_gaps:
        if not actual or actual != sorted(set(actual)):
            raise NormalizationError("repaired canonical action indices must be strictly increasing and unique")
    elif actual != list(range(len(records))):
        raise NormalizationError("canonical action indices must be contiguous starting at 0")
    return tuple(records)
