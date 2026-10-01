"""OpenAI-compatible native tool-calling agent."""

from __future__ import annotations

import ast
import base64
import copy
import hashlib
import io
import json
import logging
import os
import re
import urllib.parse
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

from .agent_config import (
    EXECUTE_ALL_CALLS_IN_ORDER,
    MULTI_TOOL_POLICIES,
    ONE_INTERACTION_PLUS_COLLAPSED_WAITS,
    AgentConfig,
    load_agent_config,
)

logger = logging.getLogger("recovery.mypcbench.tool_agent")
REPO_ROOT = Path(os.environ.get("RECOVERY_REPO_ROOT", Path(__file__).resolve().parents[3]))

_MULTI_TOOL_POLICIES = MULTI_TOOL_POLICIES

# kimi-k3 rejects assistant messages with empty content, even with tool_calls.
_EMPTY_ASSISTANT_PLACEHOLDER = "(tool call)"

_FOLDED_SCREENSHOT_PLACEHOLDER = (
    "(screenshot from this step has been removed from the context to save space)"
)

TAKEOVER_PROMPT_CONDITIONS = frozenset({"unaware", "notified", "diagnosed", "hinted"})


def takeover_condition_prompt(
    condition: str,
    diagnosis: str = "",
    root_cause_action_index: Optional[int] = None,
    hint: str = "",
) -> str:
    """Return the intervention text for one RECOVERY takeover condition."""

    if condition not in TAKEOVER_PROMPT_CONDITIONS:
        raise ValueError(f"unknown takeover prompt condition: {condition!r}")
    diagnosis = diagnosis.strip()
    hint = hint.strip()
    if condition == "hinted":
        if not hint or diagnosis or root_cause_action_index is not None:
            raise ValueError("hinted takeover takes only a non-empty free-text hint")
        return hint
    if hint:
        raise ValueError("only the hinted takeover condition takes a hint")
    if condition == "unaware":
        if diagnosis or root_cause_action_index is not None:
            raise ValueError("unaware takeover must not receive root-cause information")
        return ""
    if condition == "notified":
        if diagnosis or root_cause_action_index is not None:
            raise ValueError("notified takeover must not receive root-cause information")
        return (
            "Takeover notice: The preceding trajectory includes one or more serious "
            "errors that caused the overall task to fail. Their locations and types are "
            "not provided. Identify and fix the errors, then complete the original task."
        )
    if not diagnosis or (
        isinstance(root_cause_action_index, bool)
        or not isinstance(root_cause_action_index, int)
        or root_cause_action_index < 0
    ):
        raise ValueError(
            "diagnosed takeover requires a public diagnosis and root-cause action index"
        )
    return (
        "Takeover diagnosis: The diagnosed root cause is recorded at prefix action "
        f"action_index_global={root_cause_action_index}. Root-cause evidence from the "
        f"human annotation rationale: {diagnosis}\n"
        "Fix the errors and complete the original task."
    )


class ToolCallError(ValueError):
    """Model tool call does not match the frozen schema."""


class BatchedToolCallError(ToolCallError):
    """Model returned multiple interactive actions under the conservative policy."""


@dataclass(frozen=True)
class ToolAgentProtocol:
    """Decoder behavior parameters."""

    agent_id: str
    coordinate_protocol: str
    system_prompt: str
    max_images_in_context: int = 3
    history_turns: int = 3
    image_fold_size: int = 10
    previous_action_log: bool = False
    temperature: Optional[float] = 0.0
    max_tokens: int = 2048
    tool_choice: str = "auto"
    multi_tool_policy: str = ONE_INTERACTION_PLUS_COLLAPSED_WAITS
    schema_repair_attempts: int = 2
    alternating_action_repeat_limit: int = 0
    stalled_state_step_limit: int = 0
    enable_bash: bool = False
    enable_shell: bool = False

    def __post_init__(self) -> None:
        if self.multi_tool_policy not in _MULTI_TOOL_POLICIES:
            raise ValueError(f"unknown multi_tool_policy: {self.multi_tool_policy}")


_SAFE_KEY = re.compile(r"^[A-Za-z0-9_+\-]{1,32}$")
_BUTTONS = {"left", "right", "middle"}
_MAX_SCROLL_NOTCHES = 30
_BASH_COMMAND_MAX_CHARS = 2000
_BASH_OUTPUT_CAP = 8192
_MAX_BASH_ROUNDS_PER_STEP = 16

_MAX_CONSECUTIVE_BASH_STEPS = 64
_SAFE_PYAUTOGUI_METHODS = {
    "click",
    "doubleClick",
    "dragTo",
    "hotkey",
    "keyDown",
    "keyUp",
    "middleClick",
    "mouseDown",
    "mouseUp",
    "moveTo",
    "press",
    "rightClick",
    "scroll",
    "tripleClick",
    "typewrite",
    "write",
}


def validate_pyautogui_program(code: str) -> str:
    """Validate that text-code agent output contains only literal PyAutoGUI calls."""

    if not isinstance(code, str):
        raise ToolCallError("PyAutoGUI program must be a string")
    if code in {"WAIT", "DONE", "FAIL"}:
        return code
    if not code.strip() or len(code) > 20000:
        raise ToolCallError("PyAutoGUI program is empty or too long")
    try:
        tree = ast.parse(code, mode="exec")
    except SyntaxError as exc:
        raise ToolCallError("PyAutoGUI program has invalid syntax") from exc
    if not 1 <= len(tree.body) <= 2000:
        raise ToolCallError("each step allows only 1--2000 PyAutoGUI calls")
    for statement in tree.body:
        if not isinstance(statement, ast.Expr) or not isinstance(statement.value, ast.Call):
            raise ToolCallError("only direct PyAutoGUI calls are allowed; no assignment, import or control flow")
        call = statement.value
        function = call.func
        if not (
            isinstance(function, ast.Attribute)
            and isinstance(function.value, ast.Name)
            and function.value.id == "pyautogui"
            and function.attr in _SAFE_PYAUTOGUI_METHODS
        ):
            raise ToolCallError("call is not in the PyAutoGUI allowlist")
        if any(keyword.arg is None for keyword in call.keywords):
            raise ToolCallError("**kwargs unpacking is not allowed")
        try:
            for argument in call.args:
                ast.literal_eval(argument)
            for keyword in call.keywords:
                ast.literal_eval(keyword.value)
        except (ValueError, TypeError) as exc:
            raise ToolCallError("PyAutoGUI arguments must be literals, not expressions") from exc
    return code


def _coordinate_pair_from_string(raw: str) -> Optional[tuple[float, float]]:
    text = raw.strip()
    try:
        parsed = json.loads(text if text.startswith("[") else f"[{text}]")
    except json.JSONDecodeError:
        return None
    if (
        isinstance(parsed, list)
        and len(parsed) == 2
        and all(
            isinstance(value, (int, float)) and not isinstance(value, bool)
            for value in parsed
        )
    ):
        return parsed[0], parsed[1]
    return None


def _tool(
    name: str,
    description: str,
    properties: Mapping[str, Any],
    required: Sequence[str],
) -> dict[str, Any]:
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": {
                "type": "object",
                "properties": dict(properties),
                "required": list(required),
                "additionalProperties": False,
            },
        },
    }


def build_computer_tools(
    x_maximum: int, y_maximum: Optional[int] = None, *, include_bash: bool = False
) -> list[dict[str, Any]]:
    """Build the frozen computer action schema."""

    y_maximum = x_maximum if y_maximum is None else y_maximum
    x_coordinate = {"type": "integer", "minimum": 0, "maximum": x_maximum}
    y_coordinate = {"type": "integer", "minimum": 0, "maximum": y_maximum}
    button = {"type": "string", "enum": sorted(_BUTTONS)}
    tools = [
        _tool(
            "click",
            "Click one screen location.",
            {"x": x_coordinate, "y": y_coordinate, "button": button},
            ["x", "y"],
        ),
        _tool(
            "double_click",
            "Double-click one screen location.",
            {"x": x_coordinate, "y": y_coordinate, "button": button},
            ["x", "y"],
        ),
        _tool(
            "write",
            "Type text into the focused field.",
            {
                "content": {"type": "string"},
                "clear_existing": {"type": "boolean"},
                "press_enter": {"type": "boolean"},
            },
            ["content"],
        ),
        _tool(
            "hotkey",
            "Press one key or a simultaneous key chord.",
            {
                "keys": {
                    "type": "array",
                    "items": {"type": "string"},
                    "minItems": 1,
                    "maxItems": 5,
                }
            },
            ["keys"],
        ),
        _tool(
            "scroll",
            "Scroll at a location. delta_y counts mouse wheel notches, NOT pixels: "
            "one notch moves well over a hundred pixels, so 1-2 is a small scroll and "
            "about 5 already reaches the end of a typical page. "
            "Positive delta_y scrolls up and negative scrolls down.",
            {
                "x": x_coordinate,
                "y": y_coordinate,
                "delta_y": {
                    "type": "integer",
                    "minimum": -_MAX_SCROLL_NOTCHES,
                    "maximum": _MAX_SCROLL_NOTCHES,
                },
            },
            ["x", "y", "delta_y"],
        ),
        _tool(
            "move", "Move the pointer.", {"x": x_coordinate, "y": y_coordinate}, ["x", "y"]
        ),
        _tool(
            "drag",
            "Drag from one location to another.",
            {
                "start_x": x_coordinate,
                "start_y": y_coordinate,
                "end_x": x_coordinate,
                "end_y": y_coordinate,
                "button": button,
                "duration_s": {"type": "number", "minimum": 0, "maximum": 10},
            },
            ["start_x", "start_y", "end_x", "end_y"],
        ),
        _tool(
            "wait",
            "Wait briefly for the interface to update.",
            {"seconds": {"type": "number", "minimum": 0, "maximum": 30}},
            [],
        ),
        _tool(
            "answer",
            "End the episode after success or when the task cannot be completed.",
            {
                "status": {"type": "string", "enum": ["success", "failure"]},
                "content": {"type": "string"},
            },
            ["status"],
        ),
    ]
    if include_bash:
        tools.append(
            _tool(
                "bash",
                "Run one shell command on the desktop VM as user `user`. The exit "
                "code, stdout and stderr come back as this tool call's result in "
                "the same turn, so you can chain commands or switch back to the "
                "GUI tools afterwards. Prefer it for read-only data work "
                "(ls, cat, find, sqlite3, python3, ...); use the GUI tools for "
                "whatever the user asked to see done in the visible environment. "
                "Return bash alone, not mixed with other tools.",
                {
                    "command": {
                        "type": "string",
                        "minLength": 1,
                        "maxLength": _BASH_COMMAND_MAX_CHARS,
                    }
                },
                ["command"],
            )
        )
    return tools


def build_shell_tool() -> dict[str, Any]:
    """Return the VM-shell function schema used in live and replay history."""

    return _tool(
        "bash",
        "Run one or more shell commands inside the same task VM. Results include "
        "stdout, stderr, and an exit status for every command.",
        {
            "commands": {
                "type": "array",
                "items": {"type": "string", "minLength": 1},
                "minItems": 1,
                "maxItems": 16,
            },
            "timeout_ms": {
                "type": "integer",
                "minimum": 1,
                "maximum": 120000,
            },
            "max_output_length": {
                "type": "integer",
                "minimum": 1,
                "maximum": 20000,
            },
        },
        ["commands"],
    )


def _number(args: Mapping[str, Any], name: str) -> float:
    value = args.get(name)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ToolCallError(f"{name} must be a number")
    return float(value)


def _boolean(args: Mapping[str, Any], name: str, default: bool = False) -> bool:
    value = args.get(name, default)
    if not isinstance(value, bool):
        raise ToolCallError(f"{name} must be a boolean")
    return value


class SafePyAutoGUICompiler:
    """Compile whitelisted tool calls into MyPCBench PyAutoGUI actions."""

    def __init__(self, screen_size: tuple[int, int], coordinate_protocol: str):
        self.width, self.height = screen_size
        if self.width <= 0 or self.height <= 0:
            raise ValueError("screen_size must be positive")
        if coordinate_protocol not in {"absolute_pixels", "normalized_0_1000"}:
            raise ValueError(f"unknown coordinate protocol: {coordinate_protocol}")
        self.coordinate_protocol = coordinate_protocol
        self.clamps: list[dict[str, Any]] = []

    def _coordinate(
        self, args: Mapping[str, Any], x_name: str, y_name: str
    ) -> tuple[int, int]:
        coordinate_args: Mapping[str, Any] = args
        raw_x = args.get(x_name)
        if y_name not in args and isinstance(raw_x, str):
            pair = _coordinate_pair_from_string(raw_x)
            if pair is not None:
                coordinate_args = {**args, x_name: pair[0], y_name: pair[1]}
                logger.warning(
                    "leniently parsed %s=%r as coordinate pair %s=%s, %s=%s (protocol %s)",
                    x_name,
                    raw_x,
                    x_name,
                    pair[0],
                    y_name,
                    pair[1],
                    self.coordinate_protocol,
                )
        x = _number(coordinate_args, x_name)
        y = _number(coordinate_args, y_name)
        if self.coordinate_protocol == "normalized_0_1000":
            if not 0 <= x <= 1000 or not 0 <= y <= 1000:
                raise ToolCallError("normalized coordinates must be within [0, 1000]")
            x = x * (self.width - 1) / 1000
            y = y * (self.height - 1) / 1000
        if not 0 <= x < self.width or not 0 <= y < self.height:
            raise ToolCallError(
                f"coordinate ({x}, {y}) is outside the {self.width}x{self.height} screen"
            )
        return round(x), round(y)

    @staticmethod
    def _button(args: Mapping[str, Any]) -> str:
        button = args.get("button", "left")
        if button not in _BUTTONS:
            raise ToolCallError(f"invalid mouse button: {button!r}")
        return str(button)

    @staticmethod
    def _reject_unknown(args: Mapping[str, Any], allowed: set[str]) -> None:
        unknown = set(args) - allowed
        if unknown:
            raise ToolCallError(f"tool arguments contain unknown fields: {sorted(unknown)!r}")

    def compile(self, name: str, args: Mapping[str, Any]) -> list[str]:
        if not isinstance(args, Mapping):
            raise ToolCallError("tool arguments must be a JSON object")

        if name in {"click", "double_click"}:
            self._reject_unknown(args, {"x", "y", "button"})
            x, y = self._coordinate(args, "x", "y")
            button = self._button(args)
            function = "click" if name == "click" else "doubleClick"
            return [f"pyautogui.{function}({x}, {y}, button={button!r})"]

        if name == "write":
            self._reject_unknown(args, {"content", "clear_existing", "press_enter"})
            content = args.get("content")
            if not isinstance(content, str):
                raise ToolCallError("write.content must be a string")
            if len(content) > 10000:
                raise ToolCallError("a single write exceeds 10000 characters")
            actions: list[str] = []
            if _boolean(args, "clear_existing"):
                actions.append("pyautogui.hotkey('ctrl', 'a')")
            actions.append(f"pyautogui.write({content!r}, interval=0.01)")
            if _boolean(args, "press_enter"):
                actions.append("pyautogui.press('enter')")
            return actions

        if name == "hotkey":
            self._reject_unknown(args, {"keys"})
            keys = args.get("keys")
            if not isinstance(keys, list) or not 1 <= len(keys) <= 5:
                raise ToolCallError("hotkey.keys must contain 1--5 keys")
            if not all(isinstance(key, str) and _SAFE_KEY.fullmatch(key) for key in keys):
                raise ToolCallError("hotkey contains an invalid key name")
            quoted = ", ".join(repr(key.lower()) for key in keys)
            if len(keys) == 1:
                return [f"pyautogui.press({quoted})"]
            return [f"pyautogui.hotkey({quoted})"]

        if name == "scroll":
            self._reject_unknown(args, {"x", "y", "delta_y"})
            x, y = self._coordinate(args, "x", "y")
            delta_y = int(_number(args, "delta_y"))
            if abs(delta_y) > _MAX_SCROLL_NOTCHES:
                clamped = _MAX_SCROLL_NOTCHES if delta_y > 0 else -_MAX_SCROLL_NOTCHES
                self.clamps.append(
                    {
                        "type": "scroll_clamp",
                        "reason": (
                            "scroll.delta_y is in mouse wheel notches, not pixels"
                        ),
                        "requested_delta_y": delta_y,
                        "executed_delta_y": clamped,
                    }
                )
                delta_y = clamped
            return [f"pyautogui.moveTo({x}, {y})", f"pyautogui.scroll({delta_y})"]

        if name == "move":
            self._reject_unknown(args, {"x", "y"})
            x, y = self._coordinate(args, "x", "y")
            return [f"pyautogui.moveTo({x}, {y})"]

        if name == "drag":
            self._reject_unknown(
                args,
                {"start_x", "start_y", "end_x", "end_y", "button", "duration_s"},
            )
            start_x, start_y = self._coordinate(args, "start_x", "start_y")
            end_x, end_y = self._coordinate(args, "end_x", "end_y")
            button = self._button(args)
            duration = _number(args, "duration_s") if "duration_s" in args else 0.5
            if not 0 <= duration <= 10:
                raise ToolCallError("drag.duration_s must be within [0, 10]")
            return [
                f"pyautogui.moveTo({start_x}, {start_y})",
                f"pyautogui.dragTo({end_x}, {end_y}, duration={duration:g}, button={button!r})",
            ]

        if name == "wait":
            self._reject_unknown(args, {"seconds"})
            seconds = _number(args, "seconds") if "seconds" in args else 2
            if not 0 <= seconds <= 30:
                raise ToolCallError("wait.seconds must be within [0, 30]")
            return ["WAIT"]

        if name == "answer":
            self._reject_unknown(args, {"status", "content"})
            status = args.get("status")
            if status not in {"success", "failure"}:
                raise ToolCallError("answer.status must be success or failure")
            return ["DONE" if status == "success" else "FAIL"]

        raise ToolCallError(f"unregistered tool: {name!r}")


def _tool_call_dict(call: Any, fallback_index: int) -> dict[str, Any]:
    if isinstance(call, Mapping):
        function = call.get("function", {})
        arguments = function.get("arguments", "{}")
        if isinstance(arguments, Mapping):
            arguments = json.dumps(arguments, ensure_ascii=False)
        return {
            "id": str(call.get("id") or f"call_{fallback_index}"),
            "type": "function",
            "function": {
                "name": function.get("name"),
                "arguments": arguments,
            },
        }
    function = getattr(call, "function", None)
    return {
        "id": str(getattr(call, "id", None) or f"call_{fallback_index}"),
        "type": "function",
        "function": {
            "name": getattr(function, "name", None),
            "arguments": getattr(function, "arguments", "{}"),
        },
    }


def _content_tool_calls(content: Optional[str]) -> list[dict[str, Any]]:
    if not content:
        return []
    candidates = re.findall(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", content, re.DOTALL)
    if not candidates and content.strip().startswith("{"):
        candidates = [content.strip()]
    calls = []
    for index, candidate in enumerate(candidates):
        try:
            parsed = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        name = parsed.get("name")
        arguments = parsed.get("arguments", {})
        if name and isinstance(arguments, Mapping):
            calls.append(
                {
                    "id": f"content_call_{index}",
                    "type": "function",
                    "function": {
                        "name": name,
                        "arguments": json.dumps(arguments, ensure_ascii=False),
                    },
                }
            )
    return calls


def _read_frozen_prompt(filename: str) -> str:
    path = REPO_ROOT / "prompts" / "agents" / filename
    if not path.is_file():
        raise RuntimeError(f"frozen prompt not found: {path}")
    return path.read_text(encoding="utf-8").strip()


MYPCBENCH_SHARED_BLOCK_FILE = "mypcbench_shared_block.txt"
MYPCBENCH_SHARED_BLOCK_BASH_FILE = "mypcbench_shared_block_bash.txt"


def compose_system_prompt(scaffold_filename: str, *, has_bash: bool = False) -> str:
    """Concatenate the scaffold prompt and the shared environment block."""
    shared_file = MYPCBENCH_SHARED_BLOCK_BASH_FILE if has_bash else MYPCBENCH_SHARED_BLOCK_FILE
    for filename in (scaffold_filename, shared_file):
        path = REPO_ROOT / "prompts" / "agents" / filename
        if not path.is_file():
            raise RuntimeError(f"frozen prompt not found: {path}")
    root = REPO_ROOT / "prompts" / "agents"
    scaffold = (root / scaffold_filename).read_text(encoding="utf-8")
    shared = (root / shared_file).read_text(encoding="utf-8")
    return (scaffold + shared).strip()


class NativeToolComputerAgent:
    """Visual tool-calling agent implementing MyPCBench's ``reset/predict`` contract."""

    def __init__(
        self,
        model: str,
        screen_size: tuple[int, int],
        protocol: ToolAgentProtocol,
        *,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        client: Any = None,
        env: Any = None,
        environment: Any = None,
    ):
        self.model = model
        self.screen_size = tuple(screen_size)
        self.protocol = protocol
        self.compiler = SafePyAutoGUICompiler(self.screen_size, protocol.coordinate_protocol)
        self._env = env
        if protocol.coordinate_protocol == "normalized_0_1000":
            self.tools = build_computer_tools(1000, 1000, include_bash=protocol.enable_bash)
        else:
            self.tools = build_computer_tools(
                self.screen_size[0] - 1,
                self.screen_size[1] - 1,
                include_bash=protocol.enable_bash,
            )
        if protocol.enable_shell:
            self.tools.append(build_shell_tool())
        if protocol.enable_bash and env is None:
            logger.warning(
                "enable_bash=True but the runner passed no env: bash calls will get "
                "an explicit error and fall back to the GUI path"
            )
        self._base_url = base_url or os.environ.get("OPENAI_BASE_URL")
        self._api_key = api_key or os.environ.get("OPENAI_API_KEY") or "EMPTY"
        self._client = client
        self._environment = environment
        self._turns: list[list[dict[str, Any]]] = []
        self.last_trajectory_tool_messages: list[dict[str, Any]] = []
        self.agent_metadata: dict[str, Any] = {}
        self._turn_actions: list[tuple[str, ...]] = []
        self._action_batches: list[tuple[str, ...]] = []
        self._state_actions: list[tuple[str, tuple[str, ...]]] = []
        accounting = os.environ.get("RECOVERY_BASH_ACCOUNTING", "internal")
        if accounting not in ("internal", "steps"):
            raise ValueError(
                f"RECOVERY_BASH_ACCOUNTING must be internal or steps, got {accounting!r}"
            )
        self._bash_accounting = accounting
        self._consecutive_bash_steps = 0
        self._takeover_instruction: Optional[str] = None
        self._takeover_prompt = ""

    @property
    def messages(self) -> list[dict[str, Any]]:
        """Full visible conversation with images removed."""

        flattened = [{"role": "system", "content": self.protocol.system_prompt}]
        for turn in self._turns:
            flattened.extend(turn)
        return flattened

    def _get_client(self) -> Any:
        if self._client is None:
            try:
                from openai import OpenAI
            except ImportError as exc:  # pragma: no cover
                raise RuntimeError("openai missing; install with `pip install -e '.[collection]'`") from exc
            if not self._base_url:
                raise RuntimeError("OPENAI_BASE_URL missing; cannot reach the local inference server")
            self._client = OpenAI(base_url=self._base_url, api_key=self._api_key)
        return self._client

    def reset(self, _logger: Any = None, vm_ip: Optional[str] = None) -> None:
        del vm_ip
        global logger
        if _logger is not None:
            logger = _logger
        self._turns = []
        self.last_trajectory_tool_messages = []
        self.agent_metadata = {}
        self._turn_actions = []
        self._action_batches = []
        self._state_actions = []
        self._consecutive_bash_steps = 0
        self._takeover_instruction = None
        self._takeover_prompt = ""

    def seed_native_history(
        self,
        instruction: str,
        native_history: Sequence[Mapping[str, Any]],
        *,
        condition: str = "unaware",
        diagnosis: str = "",
        root_cause_action_index: Optional[int] = None,
        hint: str = "",
    ) -> None:
        """Seed replay-rendered Qwen history and select the post-prefix prompt layer."""

        if not instruction.strip() or not native_history:
            raise ValueError("takeover instruction and native history must be non-empty")
        frozen = copy.deepcopy([dict(message) for message in native_history])
        for message in frozen:
            content = message.get("content")
            if message.get("role") != "user" or not isinstance(content, list):
                continue
            for part in content:
                if not isinstance(part, dict) or part.get("type") != "image_url":
                    continue
                image = part.get("image_url")
                if not isinstance(image, dict) or not isinstance(image.get("url"), str):
                    raise ValueError("native history contains an invalid image_url")
                image["url"] = self._resolve_history_image_url(image["url"])
        first = frozen.pop(0)
        if first.get("role") != "system" or first.get("content") != self.protocol.system_prompt:
            raise ValueError("native history system prompt does not match the live Qwen protocol")
        turns: list[list[dict[str, Any]]] = []
        current: list[dict[str, Any]] = []
        for message in frozen:
            if message.get("role") == "user" and current:
                turns.append(current)
                current = []
            current.append(message)
        if current:
            turns.append(current)
        task_marker = f"Task: {instruction}\n"
        for turn in turns:
            roles = [message.get("role") for message in turn]
            if not turn or roles[:2] != ["user", "assistant"] or any(
                role not in {"user", "assistant", "tool"} for role in roles
            ):
                raise ValueError("native history is not a sequence of complete Qwen tool turns")
            content = turn[0].get("content")
            text_parts = (
                [part.get("text", "") for part in content if isinstance(part, Mapping)]
                if isinstance(content, list)
                else []
            )
            if not any(task_marker in str(text) for text in text_parts):
                raise ValueError("native history does not preserve the takeover instruction")
            calls = turn[1].get("tool_calls")
            if not isinstance(calls, list) or not calls:
                raise ValueError("native history assistant turn has no tool calls")
            call_ids = [str(call.get("id")) for call in calls if isinstance(call, Mapping)]
            tool_ids = [
                str(message.get("tool_call_id"))
                for message in turn[2:]
                if message.get("role") == "tool"
            ]
            if len(call_ids) != len(calls) or len(set(call_ids)) != len(call_ids):
                raise ValueError("native history contains invalid or duplicate tool call IDs")
            if tool_ids != call_ids or len(turn) != 2 + len(calls):
                raise ValueError("native history tool results do not match assistant calls")
        self._turns = turns
        self._turn_actions = []
        for turn in turns:
            calls = [
                call
                for message in turn
                if message.get("role") == "assistant"
                for call in message.get("tool_calls", ())
            ]
            self._turn_actions.append(tuple(self._render_call(call) for call in calls))
        self._action_batches = list(self._turn_actions)
        self._state_actions = []
        self._takeover_instruction = instruction
        self._takeover_prompt = takeover_condition_prompt(
            condition, diagnosis, root_cause_action_index, hint
        )

    @staticmethod
    def _resolve_history_image_url(value: str) -> str:
        if value.startswith(("data:image/", "http://", "https://")):
            return value
        if value.startswith("file://"):
            parsed = urllib.parse.urlparse(value)
            path = Path(urllib.parse.unquote(parsed.path))
        else:
            path = Path(value)
        if not path.is_file():
            raise ValueError(f"native history screenshot does not exist: {value}")
        payload = path.read_bytes()
        if payload.startswith(b"\x89PNG\r\n\x1a\n"):
            media_type = "image/png"
        elif payload.startswith(b"\xff\xd8\xff"):
            media_type = "image/jpeg"
        else:
            raise ValueError(f"native history screenshot is not PNG/JPEG: {value}")
        return f"data:{media_type};base64," + base64.b64encode(payload).decode("ascii")

    def _execute_shell_call(self, arguments: Mapping[str, Any]) -> str:
        if not self.protocol.enable_shell:
            raise ToolCallError("bash is not enabled for this agent")
        commands = arguments.get("commands")
        if not isinstance(commands, list) or not 1 <= len(commands) <= 16 or any(
            not isinstance(command, str) or not command.strip() for command in commands
        ):
            raise ToolCallError("bash.commands must contain 1--16 non-empty strings")
        timeout_ms = arguments.get("timeout_ms", 120000)
        max_output_length = arguments.get("max_output_length", 8192)
        if (
            isinstance(timeout_ms, bool)
            or not isinstance(timeout_ms, int)
            or not 1 <= timeout_ms <= 120000
        ):
            raise ToolCallError("bash.timeout_ms must be in [1, 120000]")
        if (
            isinstance(max_output_length, bool)
            or not isinstance(max_output_length, int)
            or not 1 <= max_output_length <= 20000
        ):
            raise ToolCallError("bash.max_output_length must be in [1, 20000]")
        execute = getattr(self._environment, "_execute_command", None)
        if not callable(execute):
            raise RuntimeError("bash requires the active MyPCBench VM environment")
        outputs = []
        for command in commands:
            result = execute(command, shell=True)
            if not isinstance(result, Mapping):
                raise RuntimeError("VM shell returned a non-object result")
            stdout = str(result.get("output", ""))[:max_output_length]
            stderr = str(result.get("error", ""))[:max_output_length]
            returncode = result.get("returncode", -1)
            if isinstance(returncode, bool) or not isinstance(returncode, int):
                returncode = -1
            outputs.append(
                {
                    "stdout": stdout,
                    "stderr": stderr,
                    "outcome": {"type": "exit", "exit_code": returncode},
                }
            )
        return json.dumps({"outputs": outputs}, ensure_ascii=False)

    @staticmethod
    def _image_url(screenshot: bytes) -> str:
        if not isinstance(screenshot, (bytes, bytearray)):
            raise TypeError("obs['screenshot'] must be PNG/JPEG bytes")
        return "data:image/png;base64," + base64.b64encode(screenshot).decode("ascii")

    @staticmethod
    def _screenshot_fingerprint(screenshot: bytes) -> str:
        try:
            from PIL import Image

            with Image.open(io.BytesIO(screenshot)) as image:
                image = image.convert("L")
                if image.height > 64:
                    image = image.crop((0, 32, image.width, image.height))
                image = image.resize((64, 40))
                quantized = bytes(pixel // 32 for pixel in image.getdata())
            return hashlib.sha256(quantized).hexdigest()
        except Exception:
            return hashlib.sha256(bytes(screenshot)).hexdigest()

    def _folded_prefix(self, retained: int) -> int:
        images = self.protocol.max_images_in_context
        fold = max(1, self.protocol.image_fold_size)
        folded = 0
        while (retained - folded) > images:
            folded += fold
        return min(folded, retained)

    @staticmethod
    def _fold_turn(turn: list[dict[str, Any]]) -> list[dict[str, Any]]:
        folded: list[dict[str, Any]] = []
        for message in turn:
            content = message.get("content")
            if message.get("role") != "user" or not isinstance(content, list):
                folded.append(message)
                continue
            kept = [part for part in content if part.get("type") != "image_url"]
            if len(kept) == len(content):
                folded.append(message)
                continue
            kept.insert(0, {"type": "text", "text": _FOLDED_SCREENSHOT_PLACEHOLDER})
            folded.append({**message, "content": kept})
        return folded

    def _previous_actions_text(self, dropped: int) -> str:
        lines = [
            f"Step {index + 1}: {' + '.join(actions) if actions else '(no action)'}"
            for index, actions in enumerate(self._turn_actions[:dropped])
        ]
        return "\n".join(lines) if lines else "None"

    def _messages(
        self, instruction: str, screenshot: bytes
    ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": self.protocol.system_prompt}
        ]
        retained_turns = self._turns[-self.protocol.history_turns :] if self._turns else []
        dropped = len(self._turns) - len(retained_turns)
        folded = self._folded_prefix(len(retained_turns))
        for index, turn in enumerate(retained_turns):
            messages.extend(self._fold_turn(turn) if index < folded else turn)

        text = (
            f"Task: {instruction}\n"
            "Inspect the current trajectory logs and screenshot and emit the next action "
            "as a tool call."
        )
        if self._takeover_prompt:
            text += "\n\n" + self._takeover_prompt
        if self.protocol.previous_action_log and dropped > 0:
            text += f"\n\nPrevious actions:\n{self._previous_actions_text(dropped)}"
        user_message = {
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": self._image_url(screenshot)}},
                {"type": "text", "text": text},
            ],
        }
        messages.append(user_message)
        return messages, user_message

    @staticmethod
    def _assistant_message(content: Optional[str], calls: list[dict[str, Any]]) -> dict[str, Any]:
        if not (content or "").strip():
            content = _EMPTY_ASSISTANT_PLACEHOLDER
        message: dict[str, Any] = {"role": "assistant", "content": content}
        if calls:
            message["tool_calls"] = calls
        return message

    @staticmethod
    def _tool_message(call_id: str, content: str) -> dict[str, Any]:
        return {"role": "tool", "tool_call_id": call_id, "content": content}

    @staticmethod
    def _render_value(value: Any) -> str:
        if isinstance(value, str) and len(value) > 40:
            return repr(value[:40] + "…")
        return repr(value)

    @classmethod
    def _render_call(cls, call: Mapping[str, Any]) -> str:
        try:
            arguments = json.loads(call["function"].get("arguments") or "{}")
        except json.JSONDecodeError:
            arguments = None
        if not isinstance(arguments, Mapping):
            return f"{cls._call_name(call)}(...)"
        rendered = ", ".join(
            f"{name}={cls._render_value(value)}" for name, value in arguments.items()
        )
        return f"{cls._call_name(call)}({rendered})"

    @classmethod
    def _rejection_text(
        cls, error: str, calls: list[dict[str, Any]], *, batched: bool = False
    ) -> str:
        prefix = f"Rejected by the frozen RECOVERY tool schema: {error}."
        if not calls:
            return (
                f"{prefix} Reply with at least one tool call from the provided tools; "
                "plain text is not an action."
            )
        emitted = " + ".join(cls._render_call(call) for call in calls)
        if batched:
            keep = next(
                (call for call in calls if cls._call_name(call) != "wait"), calls[0]
            )
            return (
                f"{prefix} You emitted: {emitted}. Emit ONLY `{cls._render_call(keep)}` "
                "now, as a single tool call. Send each remaining action separately, "
                "after you have seen the screenshot that follows this one."
            )
        return (
            f"{prefix} You emitted: {emitted}. Re-send corrected tool calls; "
            "fix the reported problem and do not guess or omit required fields."
        )

    @staticmethod
    def _call_name(call: Mapping[str, Any]) -> str:
        return str(call["function"].get("name"))

    @classmethod
    def _collapse_waits(
        cls, calls: list[dict[str, Any]]
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        executed: list[dict[str, Any]] = []
        collapsed: list[dict[str, Any]] = []
        for call in calls:
            if (
                cls._call_name(call) == "wait"
                and executed
                and cls._call_name(executed[-1]) == "wait"
            ):
                collapsed.append(call)
                continue
            executed.append(call)
        return executed, collapsed

    def _decode_message(self, message: Any) -> tuple[
        Optional[str],
        list[dict[str, Any]],
        list[str],
        list[dict[str, Any]],
        Optional[dict[str, Any]],
        list[dict[str, Any]],
    ]:
        content = getattr(message, "content", None)
        raw_calls = list(getattr(message, "tool_calls", None) or [])
        calls = [_tool_call_dict(call, index) for index, call in enumerate(raw_calls)]
        if not calls:
            calls = _content_tool_calls(content)
        if not calls:
            raise ToolCallError("the model returned no parsable tool call")

        if self.protocol.multi_tool_policy == EXECUTE_ALL_CALLS_IN_ORDER:
            executed, collapsed = calls, []
        else:
            executed, collapsed = self._collapse_waits(calls)
        names = [self._call_name(call) for call in executed]
        if self.protocol.multi_tool_policy == ONE_INTERACTION_PLUS_COLLAPSED_WAITS:
            if sum(name != "wait" for name in names) > 1:
                raise BatchedToolCallError(
                    "this agent allows at most one interaction per screenshot"
                )
            if "bash" in names and len(executed) > 1:
                raise BatchedToolCallError("bash must be returned as the only tool call")
            if "answer" in names and len(executed) > 1:
                raise BatchedToolCallError(
                    "answer must be returned alone, not batched with other tool calls"
                )
        elif "answer" in names:
            if names.count("answer") > 1 or names[-1] != "answer":
                raise ToolCallError(
                    "an execute_all batch may contain at most one answer, and it must be last"
                )

        if self.protocol.enable_bash:
            bash_count = sum(name == "bash" for name in names)
            if bash_count > 1 or (bash_count == 1 and len(executed) > 1):
                raise BatchedToolCallError(
                    "bash must be returned as the only tool call: one command at a time, "
                    "not mixed with other actions"
                )

        actions: list[str] = []
        summaries: list[dict[str, Any]] = []
        clamps: list[dict[str, Any]] = []
        for call in executed:
            function = call["function"]
            name = function.get("name")
            try:
                arguments = json.loads(function.get("arguments") or "{}")
            except json.JSONDecodeError as exc:
                raise ToolCallError(f"{name} arguments are not valid JSON") from exc
            if str(name) == "bash" and self.protocol.enable_bash:
                command = arguments.get("command")
                if not isinstance(command, str) or not command.strip():
                    raise ToolCallError("bash.command must be a non-empty string")
                if len(command) > _BASH_COMMAND_MAX_CHARS:
                    raise ToolCallError(
                        f"bash.command exceeds the {_BASH_COMMAND_MAX_CHARS}-character limit"
                    )
                unknown = set(arguments) - {"command"}
                if unknown:
                    raise ToolCallError(
                        f"bash arguments contain unknown fields: {sorted(unknown)!r}"
                    )
                summaries.append(
                    {"name": "bash", "arguments": arguments, "compiled": []}
                )
                continue
            self.compiler.clamps.clear()
            if name == "bash":
                if not self.protocol.enable_shell:
                    raise ToolCallError("bash is not enabled for this agent")
                commands = arguments.get("commands")
                if not isinstance(commands, list) or not 1 <= len(commands) <= 16 or any(
                    not isinstance(command, str) or not command.strip()
                    for command in commands
                ):
                    raise ToolCallError("bash.commands must contain 1--16 non-empty strings")
                compiled = []
            else:
                compiled = self.compiler.compile(str(name), arguments)
            clamps.extend(
                {**clamp, "tool_call_id": call["id"]} for clamp in self.compiler.clamps
            )
            actions.extend(compiled)
            summaries.append(
                {
                    "tool_call_id": call["id"],
                    "name": name,
                    "arguments": arguments,
                    "compiled": compiled,
                }
            )

        normalization = None
        if collapsed:
            normalization = {
                "type": "wait_normalization",
                "reason": "collapsed consecutive wait tool calls into a single wait",
                "collapsed_tool_call_ids": [call["id"] for call in collapsed],
                "tool_calls": calls,
            }

        visible_content = content if content is None or isinstance(content, str) else str(content)
        return visible_content, calls, actions, summaries, normalization, clamps

    def _run_bash(
        self, call: Mapping[str, Any], arguments: Mapping[str, Any]
    ) -> tuple[str, dict[str, Any]]:
        command = str(arguments["command"])
        intervention: dict[str, Any] = {
            "type": "bash_round",
            "tool_call_id": call["id"],
            "command": command,
        }
        if self._env is None:
            intervention.update({"exit_code": None, "error": "no_env"})
            return (
                "Error: no VM environment is wired to this agent; the bash tool "
                "cannot execute anything. Continue with the GUI tools.",
                intervention,
            )
        try:
            result = self._env._execute_command(command, shell=True)
        except Exception as exc:  # noqa: BLE001
            intervention.update({"exit_code": None, "error": str(exc)})
            return f"Error: {exc}", intervention
        raw_stdout = str(result.get("output") or "")
        raw_stderr = str(result.get("error") or "")
        try:
            exit_code = int(result.get("returncode", -1))
        except (TypeError, ValueError):
            exit_code = -1
        stdout = raw_stdout[:_BASH_OUTPUT_CAP]
        stderr = raw_stderr[:_BASH_OUTPUT_CAP]
        truncated = len(raw_stdout) > _BASH_OUTPUT_CAP or len(raw_stderr) > _BASH_OUTPUT_CAP
        intervention.update(
            {
                "exit_code": exit_code,
                "stdout_chars": len(raw_stdout),
                "stderr_chars": len(raw_stderr),
                "truncated": truncated,
            }
        )
        note = f"\n(output truncated at {_BASH_OUTPUT_CAP} chars per stream)" if truncated else ""
        text = (
            f"exit code: {exit_code}\n"
            f"stdout:\n{stdout if stdout.strip() else '(empty)'}\n"
            f"stderr:\n{stderr if stderr.strip() else '(empty)'}{note}"
        )
        return text, intervention

    def _loop_reason(
        self, screenshot_digest: str, signature: tuple[str, ...]
    ) -> Optional[str]:
        if signature in {("DONE",), ("FAIL",), ("WAIT",)}:
            return None

        stall_limit = self.protocol.stalled_state_step_limit
        if stall_limit:
            stalled = 1
            for prior_digest, _ in reversed(self._state_actions):
                if prior_digest != screenshot_digest:
                    break
                stalled += 1
            if stalled >= stall_limit:
                return f"visual state unchanged for {stalled} consecutive steps"

        repeat_limit = self.protocol.alternating_action_repeat_limit
        if not repeat_limit:
            return None

        if (
            len(self._action_batches) >= repeat_limit - 1
            and all(
                prior_signature == signature
                for prior_signature in self._action_batches[-(repeat_limit - 1) :]
            )
            and self._state_actions
            and self._state_actions[-1][0] == screenshot_digest
        ):
            return (
                f"same action would occur {repeat_limit} consecutive times "
                "after an unchanged visual state"
            )

        candidate_history = [*self._action_batches, signature]
        if repeat_limit and len(candidate_history) >= 2 * repeat_limit:
            tail = candidate_history[-2 * repeat_limit :]
            first, second = tail[0], tail[1]
            if first != second and all(
                tail[index] == (first if index % 2 == 0 else second)
                for index in range(len(tail))
            ):
                return f"two-action cycle repeated {repeat_limit} times"
        return None

    def predict(self, instruction: str, obs: Mapping[str, Any]) -> tuple[str, list[str]]:
        if self._takeover_instruction is not None and instruction != self._takeover_instruction:
            raise ValueError("predict instruction differs from the seeded takeover task")
        self.last_trajectory_tool_messages = []
        self.agent_metadata = {}
        screenshot = obs.get("screenshot")
        messages, user_message = self._messages(instruction, screenshot)
        self._takeover_prompt = ""
        screenshot_digest = self._screenshot_fingerprint(bytes(screenshot))
        request_messages = list(messages)
        turn_messages: list[dict[str, Any]] = [user_message]
        interventions: list[dict[str, Any]] = []
        schema_repairs_left = self.protocol.schema_repair_attempts
        loop_repairs_left = (
            1
            if self.protocol.alternating_action_repeat_limit
            or self.protocol.stalled_state_step_limit
            else 0
        )
        bash_rounds_left = (
            _MAX_BASH_ROUNDS_PER_STEP
            if self.protocol.enable_bash and self._bash_accounting == "internal"
            else 0
        )

        while True:
            request_kwargs: dict[str, Any] = dict(
                model=self.model,
                messages=request_messages,
                tools=self.tools,
                tool_choice=self.protocol.tool_choice,
                max_tokens=self.protocol.max_tokens,
            )
            if self.protocol.temperature is not None:
                request_kwargs["temperature"] = self.protocol.temperature
            response = self._get_client().chat.completions.create(**request_kwargs)
            message = response.choices[0].message
            content = getattr(message, "content", None)
            raw_calls = list(getattr(message, "tool_calls", None) or [])
            calls = [_tool_call_dict(call, index) for index, call in enumerate(raw_calls)]
            if not calls:
                calls = _content_tool_calls(content)
            visible_content = content if content is None or isinstance(content, str) else str(content)

            try:
                (
                    visible_content,
                    calls,
                    actions,
                    summaries,
                    normalization,
                    clamps,
                ) = self._decode_message(message)
            except ToolCallError as exc:
                error = str(exc)
                assistant_message = self._assistant_message(visible_content, calls)
                rejection_text = self._rejection_text(
                    error, calls, batched=isinstance(exc, BatchedToolCallError)
                )
                rejection_messages = (
                    [self._tool_message(call["id"], rejection_text) for call in calls]
                    if calls
                    else [{"role": "user", "content": rejection_text}]
                )
                intervention = {
                    "type": "schema_repair",
                    "error": error,
                    "tool_calls": calls,
                }
                interventions.append(intervention)
                turn_messages.append(assistant_message)
                turn_messages.extend(rejection_messages)
                if schema_repairs_left > 0:
                    schema_repairs_left -= 1
                    request_messages.extend([assistant_message, *rejection_messages])
                    logger.warning("tool schema validation failed, running one bounded repair: %s", error)
                    continue

                abort = {"type": "INVALID_TOOL_CALL", "error": error}
                self._turns.append(turn_messages)
                self._turn_actions.append(())
                trajectory_response = {
                    "content": visible_content,
                    "tool_calls": calls,
                    "compiled_actions": [],
                    "interventions": interventions,
                    "abort": abort,
                }
                return json.dumps(trajectory_response, ensure_ascii=False), ["FAIL"]

            if normalization is not None:
                interventions.append(normalization)
            interventions.extend(clamps)

            if self.protocol.enable_bash and summaries and summaries[0]["name"] == "bash":
                if self._bash_accounting == "internal":
                    if bash_rounds_left <= 0:
                        turn_messages.append(self._assistant_message(visible_content, calls))
                        abort = {
                            "type": "BASH_BUDGET_ABORT",
                            "reason": (
                                f"more than {_MAX_BASH_ROUNDS_PER_STEP} bash rounds in "
                                "one step without a GUI action or answer"
                            ),
                        }
                        self._turns.append(turn_messages)
                        self._turn_actions.append(())
                        trajectory_response = {
                            "content": visible_content,
                            "tool_calls": calls,
                            "compiled_actions": summaries,
                            "interventions": interventions,
                            "abort": abort,
                        }
                        return json.dumps(trajectory_response, ensure_ascii=False), ["FAIL"]
                    bash_rounds_left -= 1
                bash_call = next(call for call in calls if self._call_name(call) == "bash")
                if (
                    self._bash_accounting == "steps"
                    and self._consecutive_bash_steps >= _MAX_CONSECUTIVE_BASH_STEPS
                ):
                    turn_messages.append(self._assistant_message(visible_content, calls))
                    abort = {
                        "type": "BASH_BUDGET_ABORT",
                        "reason": (
                            f"more than {_MAX_CONSECUTIVE_BASH_STEPS} consecutive "
                            "bash steps without a GUI action or answer"
                        ),
                    }
                    self._turns.append(turn_messages)
                    self._turn_actions.append(())
                    trajectory_response = {
                        "content": visible_content,
                        "tool_calls": calls,
                        "compiled_actions": summaries,
                        "interventions": interventions,
                        "abort": abort,
                    }
                    return json.dumps(trajectory_response, ensure_ascii=False), ["FAIL"]
                result_text, bash_intervention = self._run_bash(
                    bash_call, summaries[0]["arguments"]
                )
                interventions.append(bash_intervention)
                assistant_message = self._assistant_message(visible_content, calls)
                tool_reply = self._tool_message(bash_call["id"], result_text)
                turn_messages.extend([assistant_message, tool_reply])
                if self._bash_accounting == "steps":
                    self._consecutive_bash_steps += 1
                    self._turns.append(turn_messages)
                    self._turn_actions.append(())
                    logger.info(
                        "bash step (accounting=steps) #%d: %s",
                        self._consecutive_bash_steps,
                        str(summaries[0]["arguments"].get("command", ""))[:120],
                    )
                    trajectory_response = {
                        "content": visible_content,
                        "tool_calls": calls,
                        "compiled_actions": summaries,
                        "interventions": interventions,
                    }
                    return json.dumps(trajectory_response, ensure_ascii=False), []
                request_messages.extend([assistant_message, tool_reply])
                logger.info(
                    "bash round %d/%d: %s",
                    _MAX_BASH_ROUNDS_PER_STEP - bash_rounds_left,
                    _MAX_BASH_ROUNDS_PER_STEP,
                    str(summaries[0]["arguments"].get("command", ""))[:120],
                )
                continue

            shell_summaries = [item for item in summaries if item["name"] == "bash"]
            if self.protocol.enable_shell and shell_summaries:
                shell = shell_summaries[0]
                result_text = self._execute_shell_call(shell["arguments"])
                assistant_message = self._assistant_message(visible_content, calls)
                tool_message = self._tool_message(shell["tool_call_id"], result_text)
                turn_messages.extend([assistant_message, tool_message])
                self._turns.append(turn_messages)
                call = next(
                    item for item in calls if item["id"] == shell["tool_call_id"]
                )
                rendered = self._render_call(call)
                self._turn_actions.append((rendered,))
                self._action_batches.append((rendered,))
                self.last_trajectory_tool_messages = [
                    {
                        "type": "tool_use",
                        "id": shell["tool_call_id"],
                        "name": "bash",
                        "input": shell["arguments"],
                    },
                    {
                        "type": "tool_result",
                        "tool_use_id": shell["tool_call_id"],
                        "content": result_text,
                        "is_error": False,
                    },
                ]
                trajectory_response = {
                    "content": visible_content,
                    "tool_calls": calls,
                    "compiled_actions": summaries,
                    "tool_results": [json.loads(result_text)],
                    "interventions": interventions,
                }
                return json.dumps(trajectory_response, ensure_ascii=False), []

            signature = tuple(actions)
            loop_reason = self._loop_reason(screenshot_digest, signature)
            if loop_reason is not None:
                assistant_message = self._assistant_message(visible_content, calls)
                rejection = self._tool_message(
                    calls[0]["id"],
                    "Rejected by the RECOVERY loop guard: recent actions made no progress "
                    f"({loop_reason}). Choose one materially different action or answer with failure.",
                )
                interventions.append(
                    {
                        "type": "loop_repair",
                        "reason": loop_reason,
                        "tool_calls": calls,
                        "compiled_actions": summaries,
                    }
                )
                turn_messages.extend([assistant_message, rejection])
                if loop_repairs_left > 0:
                    loop_repairs_left -= 1
                    request_messages.extend([assistant_message, rejection])
                    logger.warning("repeated action loop detected, running one bounded replan: %s", loop_reason)
                    continue

                abort = {"type": "LOOP_ABORT", "reason": loop_reason}
                self._turns.append(turn_messages)
                self._turn_actions.append(tuple(actions))
                trajectory_response = {
                    "content": visible_content,
                    "tool_calls": calls,
                    "compiled_actions": summaries,
                    "interventions": interventions,
                    "abort": abort,
                }
                return json.dumps(trajectory_response, ensure_ascii=False), ["FAIL"]

            break

        assistant_message = self._assistant_message(visible_content, calls)
        collapsed_ids = (
            set(normalization["collapsed_tool_call_ids"]) if normalization else set()
        )
        clamped_by_id = {clamp["tool_call_id"]: clamp for clamp in clamps}

        def _tool_reply(call: Mapping[str, Any]) -> str:
            if call["id"] in collapsed_ids:
                return (
                    "Collapsed by the RECOVERY scaffold: consecutive wait calls run as a single wait."
                )
            clamp = clamped_by_id.get(call["id"])
            if clamp is not None:
                return (
                    "Clamped by the RECOVERY scaffold: delta_y is in mouse wheel notches, "
                    f"not pixels. You asked for {clamp['requested_delta_y']}; "
                    f"{clamp['executed_delta_y']} notches were executed "
                    f"(the maximum is {_MAX_SCROLL_NOTCHES}). "
                    "A few notches already scroll a whole page — use 1-2 for a small scroll."
                )
            return "Accepted for execution; inspect the next screenshot for the result."

        tool_messages = [self._tool_message(call["id"], _tool_reply(call)) for call in calls]
        turn_messages.extend([assistant_message, *tool_messages])
        self._turns.append(turn_messages)
        self._turn_actions.append(signature)
        self._action_batches.append(signature)
        self._state_actions.append((screenshot_digest, signature))
        self._consecutive_bash_steps = 0
        trajectory_response = {
            "content": visible_content,
            "tool_calls": calls,
            "compiled_actions": summaries,
            "interventions": interventions,
        }
        return json.dumps(trajectory_response, ensure_ascii=False), actions


def protocol_from_config(config: AgentConfig) -> ToolAgentProtocol:
    """Build the decoder protocol from a validated yaml config."""

    return ToolAgentProtocol(
        agent_id=config.agent_id,
        coordinate_protocol=config["coordinate_protocol"],
        temperature=config["temperature"],
        max_tokens=config["max_tokens"],
        max_images_in_context=config["max_images_in_context"],
        history_turns=config["history_turns"],
        image_fold_size=config["image_fold_size"],
        previous_action_log=config["previous_action_log"],
        tool_choice=config["tool_choice"],
        multi_tool_policy=config["multi_tool_policy"],
        schema_repair_attempts=config["schema_repair_attempts"],
        alternating_action_repeat_limit=config["alternating_action_repeat_limit"],
        stalled_state_step_limit=config["stalled_state_step_limit"],
        enable_bash=config["enable_bash"],
        enable_shell=bool(config.live.get("enable_shell", False)),
        system_prompt=compose_system_prompt(
            config["system_prompt_file"], has_bash=config["enable_bash"]
        ),
    )


def kimi_k3_protocol() -> ToolAgentProtocol:
    return protocol_from_config(load_agent_config("kimi_k3"))


def kimi_k3_cuabash_protocol() -> ToolAgentProtocol:
    return protocol_from_config(load_agent_config("kimi_k3_cuabash"))
