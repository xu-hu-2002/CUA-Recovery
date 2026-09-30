"""OpenAI-compatible 原生工具调用 agent。

本模块只把经过验证的结构化 tool call 编译成 PyAutoGUI；绝不执行模型返回的
任意 Python。Holo-3.1 使用 0--1000 归一化坐标，Qwen3.6 使用 DERAIL 明确定义的
1280x800 绝对像素 scaffold。后者不是 Qwen 官方 computer-use 协议。
"""

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

logger = logging.getLogger("derail.mypcbench.tool_agent")
REPO_ROOT = Path(os.environ.get("DERAIL_REPO_ROOT", Path(__file__).resolve().parents[3]))

# 取值域定义在 agent_config，那里是 yaml 的校验层；这里重新导出，保持既有
# `from .tool_agent import EXECUTE_ALL_CALLS_IN_ORDER` 的调用点不变。
_MULTI_TOOL_POLICIES = MULTI_TOOL_POLICIES

# kimi-k3 上游请求校验会拒绝 content 为空的 assistant 消息（"the message at
# position N with role 'assistant' must not be empty"），即使该消息携带
# tool_calls（2026-08-09 smoke 采集两条轨迹尾段 400 崩溃的根因）。推理模型
# 经常返回空可见 content + tool_calls，回放历史时统一补占位符；traj.jsonl
# 记录的是真实 visible_content，不受此处影响。
_EMPTY_ASSISTANT_PLACEHOLDER = "(tool call)"

# 历史 turn 的截图被折叠掉时留下的痕迹。措辞跟着上游 qwen35vl_agent 的
# collapse_text 走：模型必须知道那一步「有过截图」，否则会以为自己没看过。
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
    """Return the intervention text for one DERAIL takeover condition.

    The text is appended only to the first post-prefix user turn.  In
    particular, it is never written into the replay-derived history itself.
    ``hinted`` (recovery-data generation) returns the free-text ``hint`` verbatim,
    in the position the ``diagnosed`` text takes.
    """

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
    """模型 tool call 不符合冻结 schema。"""


class BatchedToolCallError(ToolCallError):
    """保守策略下，模型一次返回了多个交互动作。"""




@dataclass(frozen=True)
class ToolAgentProtocol:
    """decoder 的行为参数。

    正式路径上每个字段都由 ``configs/agents/<agent_id>.yaml`` 逐项提供（见
    :func:`protocol_from_config`），取值理由也写在那里 —— 这里的注释只解释字段
    含义。下面的 default 仅供测试直接构造使用；agent_config 要求 yaml 写全所有
    live 字段，正式路径落不到默认值上。
    """

    agent_id: str
    coordinate_protocol: str
    system_prompt: str
    # 保留图片的 turn 数上限（上游 qwen35vl_agent 的 image_max）。超出的历史 turn
    # 仍留在对话里，只是图片被换成占位文本。
    max_images_in_context: int = 3
    # 保留完整消息的 turn 数上限（上游的 history_n）。默认等于 max_images_in_context
    # 时行为与 2026-08-09 v1 采集一致：窗口外的 turn 直接丢掉。
    history_turns: int = 3
    # 折叠图片的粒度（上游的 fold_size）：一次多折 N 张，避免每步都重算前缀。
    image_fold_size: int = 10
    # 是否把窗口外 turn 的动作压成 `Previous actions:` 文本块附在当前 user 消息里
    # （上游 qwen35vl_agent.py:462 的做法）。关掉时窗口外的历史彻底消失。
    previous_action_log: bool = False
    # None = 请求体不带 temperature。kimi-k3（routify）是推理模型，显式传
    # temperature 会被 400 拒绝（2026-08-09 实测）；其余 scaffold 保持 0.0。
    temperature: Optional[float] = 0.0
    max_tokens: int = 2048
    tool_choice: str = "auto"
    # Holo 原生倾向于把 click→write、click→click、动作→wait 作为一个有序批次。
    # Qwen3.6 仍使用保守的单交互策略；不要在共享 decoder 里无条件放开。
    multi_tool_policy: str = ONE_INTERACTION_PLUS_COLLAPSED_WAITS
    # 2 而不是 1：一次修复后仍违规就整局判 FAIL，v1 smoke 里 115 次拒绝因此杀掉
    # 10 局。Holo 打批的习惯很顽固，第一次修复常常只从三个动作减到两个，第二次
    # 才收敛。所有 DERAIL scaffold 必须取同一个值，否则失败分布不可比。
    schema_repair_attempts: int = 2
    # 同一动作连发多少次、或两动作循环多少轮才算死循环。0 = 停用，
    # _loop_reason 在 limit 为 0 时直接短路。
    alternating_action_repeat_limit: int = 0
    # 连续多少步视觉状态不变就判定空转。与 alternating_action_repeat_limit 互补：
    # 后者盯的是"动作重复"，这一条盯的是"屏幕没反应"，动作各不相同也照样触发。
    # 同样 0 = 停用。
    stalled_state_step_limit: int = 0
    # cuabash 变体：是否给模型暴露 `bash` 工具。命令经 runner 传入的 env 引用
    # （VM 控制通道 /execute，shell=True）执行，stdout/stderr/exit code 以 tool
    # 消息回填、同一 predict 内继续对话 —— 对齐 gpt_5_5 shell_call 的
    # agent-internal 语义：bash 轮不消耗 runner 的 max_steps 预算。
    # 与 qwen_cuabash 的"bash 单独占一步"语义不同，横比时必须写明。
    enable_bash: bool = False
    # Qwen3.8 is evaluated as a hybrid computer-use agent.  Other DERAIL
    # general-model scaffolds keep their frozen GUI-only behavior.
    # 与 enable_bash 不是同一个东西：两者工具装配路径、记账语义、轨迹记录都不同
    # （见 predict 里的两条分流），qwen3_8_27b.yaml 同时声明二者即为此故。
    enable_shell: bool = False

    def __post_init__(self) -> None:
        if self.multi_tool_policy not in _MULTI_TOOL_POLICIES:
            raise ValueError(f"未知 multi_tool_policy：{self.multi_tool_policy}")


_SAFE_KEY = re.compile(r"^[A-Za-z0-9_+\-]{1,32}$")
_BUTTONS = {"left", "right", "middle"}
# scroll.delta_y 的单位是 pyautogui 的滚轮格数。旧 schema 只写 "delta_y" 不写单位，
# 上限是形同虚设的 10000，于是 Holo-3.1 按像素理解：2026-08-08 的 smoke10 里 107 次
# scroll 有 96 次 |delta| >= 150，每一次都把页面直接顶到页首或页尾。
#
# 单位确实是格数，不是像素 —— 走上游 scaffold 的 qwen3_5 自己发的就是 -3/-5/-10
# （1957 次 scroll 里 1657 次落在 |delta| <= 15），EvoCUA 发 -3，OpenCUA 发 -10。
# 所以这里修的是「schema 没说清单位」，不是改执行语义，改完仍与对照组同口径。
#
# 30 这个上限的依据：对 smoke20_evocua / smoke20_opencua / v1_qwen3_5 的截图做
# 逐像素竖直位移互相关，MyPCBench 各 app 页面的可滚动余量只有 ~420-465px，3 格就
# 已经滚到底（3 格 -> 414px，故一格 >= 138px）。15 格必定饱和，30 格留了一倍余量。
_MAX_SCROLL_NOTCHES = 30
# cuabash 变体的 bash 护栏。命令长度与输出截断防止上下文被单条命令打爆；
# 轮数上限防"一步内无限 bash"烧穿网关 RPM 配额（bash 轮不占 max_steps 预算，
# 没有轮数上限的话一个死循环模型可以在单个 predict 里永续请求）。烧满上限
# 仍无 GUI 动作/answer，按 BASH_BUDGET_ABORT 判 FAIL，interventions 保留全部
# bash_round 供审计。
# 上限取值沿革：初值 8。F5 kimi_k3_cuabash 正式全量（8 轮）实测 ~54% 任务被
# 熔断（116/214，其中 101 个死在第 1 步），每步轮数双峰分布（1-2 轮 vs 打满
# 8），打满≈熔断——8 对 kimi_k3 的长链 bash 风格偏紧。2026-08-23 放宽到 16，
# 先用 shard_1 同批 46 任务做 8 vs 16 配对 A/B，熔断率显著回落再全量补跑
# （见 results/COLLECTION_LOG.md 02:xx 立项条目）。
# 记账模式：DERAIL_BASH_ACCOUNTING=internal（默认）时 bash 轮在 predict 内部
# 循环、不占 runner 步数，由本上限兜底；=steps（统一记账，对齐上游
# openai_cuabash）时 bash 轮每轮返回、由 runner 计为 TOOL_CALL 步，预算主体
# 回到 max_steps，本上限不再介入（见 _MAX_CONSECUTIVE_BASH_STEPS）。
_BASH_COMMAND_MAX_CHARS = 2000
_BASH_OUTPUT_CAP = 8192
_MAX_BASH_ROUNDS_PER_STEP = 16

# 统一记账模式（DERAIL_BASH_ACCOUNTING=steps）的安全阀：bash 轮本身已消耗
# runner 步数（对齐上游 openai_cuabash 的 TOOL_CALL 轮语义），max_steps 就是
# 预算主体；这个上限只拦「连发 bash 始终不回 GUI」的病态循环，取值刻意远大于
# 任何正常侦查链（gpt_5_5 实测最长连发 54 轮且能回头）。
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
    """验证官方 text-code agent 的输出只含 literal PyAutoGUI 调用。

    OpenCUA 的官方协议输出 PyAutoGUI code block。MyPCBench 会直接交给 Python，
    因而这里必须在保留官方 prompt/parser 的同时补一层 AST 白名单。
    """

    if not isinstance(code, str):
        raise ToolCallError("PyAutoGUI program 必须是字符串")
    if code in {"WAIT", "DONE", "FAIL"}:
        return code
    if not code.strip() or len(code) > 20000:
        raise ToolCallError("PyAutoGUI program 为空或过长")
    try:
        tree = ast.parse(code, mode="exec")
    except SyntaxError as exc:
        raise ToolCallError("PyAutoGUI program 语法无效") from exc
    # EvoCUA 的官方 type parser 会把长文本展开成逐字符 press，因此上限需明显高于
    # 一般 GUI step；代码长度与 AST 白名单仍阻止资源滥用和任意执行。
    if not 1 <= len(tree.body) <= 2000:
        raise ToolCallError("每步只允许 1--2000 个 PyAutoGUI 调用")
    for statement in tree.body:
        if not isinstance(statement, ast.Expr) or not isinstance(statement.value, ast.Call):
            raise ToolCallError("只允许直接调用 PyAutoGUI，不允许赋值、import 或控制流")
        call = statement.value
        function = call.func
        if not (
            isinstance(function, ast.Attribute)
            and isinstance(function.value, ast.Name)
            and function.value.id == "pyautogui"
            and function.attr in _SAFE_PYAUTOGUI_METHODS
        ):
            raise ToolCallError("调用不在 PyAutoGUI 白名单中")
        if any(keyword.arg is None for keyword in call.keywords):
            raise ToolCallError("不允许 **kwargs 展开")
        try:
            for argument in call.args:
                ast.literal_eval(argument)
            for keyword in call.keywords:
                ast.literal_eval(keyword.value)
        except (ValueError, TypeError) as exc:
            raise ToolCallError("PyAutoGUI 参数必须是 literal，不能包含表达式") from exc
    return code


def _coordinate_pair_from_string(raw: str) -> Optional[tuple[float, float]]:
    """把 "[878, 160]" / "867, 163" 解析成坐标对，其余一律返回 None。

    只认「恰好两个数」。三个数、单个数、带单位、非数字都返回 None，让 schema
    repair 去处理 —— 猜错一个坐标比多花一轮 repair 代价大得多。
    """

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
    """构造冻结的动作 schema；坐标含义由 system prompt 明确说明。

    ``include_bash=True`` 追加 `bash` 工具（cuabash 变体）。bash 不经
    SafePyAutoGUICompiler —— 它在 predict 内直接执行并把结果作为 tool 消息
    回填，见 NativeToolComputerAgent 的 bash 分流。
    """

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
    """Return the Qwen3.8 VM-shell function schema used in live and replay history."""

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
        raise ToolCallError(f"{name} 必须是数值")
    return float(value)


def _boolean(args: Mapping[str, Any], name: str, default: bool = False) -> bool:
    value = args.get(name, default)
    if not isinstance(value, bool):
        raise ToolCallError(f"{name} 必须是 boolean")
    return value


class SafePyAutoGUICompiler:
    """将白名单 tool call 转成 MyPCBench 接受的 PyAutoGUI action。"""

    def __init__(self, screen_size: tuple[int, int], coordinate_protocol: str):
        self.width, self.height = screen_size
        if self.width <= 0 or self.height <= 0:
            raise ValueError("screen_size 必须为正数")
        if coordinate_protocol not in {"absolute_pixels", "normalized_0_1000"}:
            raise ValueError(f"未知坐标协议：{coordinate_protocol}")
        self.coordinate_protocol = coordinate_protocol
        # 本次 compile() 里发生的越界截断。调用方在每次 compile 前清空、compile
        # 后读取，用来把截断写进 trajectory 的 interventions 并回告模型。
        self.clamps: list[dict[str, Any]] = []

    def _coordinate(
        self, args: Mapping[str, Any], x_name: str, y_name: str
    ) -> tuple[int, int]:
        coordinate_args: Mapping[str, Any] = args
        raw_x = args.get(x_name)
        if y_name not in args and isinstance(raw_x, str):
            # vLLM 的 qwen3_coder tool parser 会把两个坐标压进 x 一个字段，y 整个
            # 丢掉。两种已观测到的形态：
            #   Qwen3.6  x="[878, 160]"   （JSON 数组）
            #   Holo-3.1 x="867, 163"     （裸的逗号对）
            # 两者都是无歧义的坐标对。只接受「恰好两个数」这一种形态，其余一律
            # 交给 schema repair；原始 arguments 仍原样写进 trajectory 以便审计。
            # 这里只补回被 parser 吃掉的 y，不改变任何坐标协议的取值范围或缩放。
            #
            # Holo 这一支的证据：2026-08-08 smoke10 的 45 次 schema_repair 里有 33
            # 次是裸逗号对，其中 32 次模型在重试轮给出的 x/y 与串里的两个数完全
            # 一致，说明按 (A, B) 解读就是模型本意。
            pair = _coordinate_pair_from_string(raw_x)
            if pair is not None:
                coordinate_args = {**args, x_name: pair[0], y_name: pair[1]}
                logger.warning(
                    "兼容解析 %s=%r 为坐标对 %s=%s, %s=%s（协议 %s）",
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
                raise ToolCallError("归一化坐标必须位于 [0, 1000]")
            # 1000 对应最右/下方可点击像素，而不是屏幕外的 width/height。
            x = x * (self.width - 1) / 1000
            y = y * (self.height - 1) / 1000
        if not 0 <= x < self.width or not 0 <= y < self.height:
            raise ToolCallError(
                f"坐标 ({x}, {y}) 越过 {self.width}x{self.height} 屏幕边界"
            )
        return round(x), round(y)

    @staticmethod
    def _button(args: Mapping[str, Any]) -> str:
        button = args.get("button", "left")
        if button not in _BUTTONS:
            raise ToolCallError(f"非法鼠标键：{button!r}")
        return str(button)

    @staticmethod
    def _reject_unknown(args: Mapping[str, Any], allowed: set[str]) -> None:
        unknown = set(args) - allowed
        if unknown:
            raise ToolCallError(f"tool arguments 包含未知字段：{sorted(unknown)!r}")

    def compile(self, name: str, args: Mapping[str, Any]) -> list[str]:
        if not isinstance(args, Mapping):
            raise ToolCallError("tool arguments 必须是 JSON object")

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
                raise ToolCallError("write.content 必须是字符串")
            if len(content) > 10000:
                raise ToolCallError("单次 write 超过 10000 字符")
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
                raise ToolCallError("hotkey.keys 必须包含 1--5 个按键")
            if not all(isinstance(key, str) and _SAFE_KEY.fullmatch(key) for key in keys):
                raise ToolCallError("hotkey 包含非法按键名")
            quoted = ", ".join(repr(key.lower()) for key in keys)
            if len(keys) == 1:
                return [f"pyautogui.press({quoted})"]
            return [f"pyautogui.hotkey({quoted})"]

        if name == "scroll":
            self._reject_unknown(args, {"x", "y", "delta_y"})
            x, y = self._coordinate(args, "x", "y")
            delta_y = int(_number(args, "delta_y"))
            if abs(delta_y) > _MAX_SCROLL_NOTCHES:
                # 截断而不是拒绝。拒绝会走 schema repair，而 repair 预算
                # （schema_repair_attempts=2）一旦耗尽 predict() 直接返回 FAIL：
                # 把「单位理解错」升级成「整局作废」，制造的正是我们想从数据里
                # 剔除的、与模型能力无关的失败。
                #
                # 截断也不是静默的：这一步会记进 interventions 供审计，并通过
                # tool 消息明确告诉模型「你的 delta 被当作像素了，实际执行的是 N
                # 格」，它下一步就能自己改口径。
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
                raise ToolCallError("drag.duration_s 必须位于 [0, 10]")
            return [
                f"pyautogui.moveTo({start_x}, {start_y})",
                f"pyautogui.dragTo({end_x}, {end_y}, duration={duration:g}, button={button!r})",
            ]

        if name == "wait":
            self._reject_unknown(args, {"seconds"})
            seconds = _number(args, "seconds") if "seconds" in args else 2
            if not 0 <= seconds <= 30:
                raise ToolCallError("wait.seconds 必须位于 [0, 30]")
            # MyPCBench 对 WAIT 有专门语义，避免让模型注入 sleep 代码。
            return ["WAIT"]

        if name == "answer":
            self._reject_unknown(args, {"status", "content"})
            status = args.get("status")
            if status not in {"success", "failure"}:
                raise ToolCallError("answer.status 必须是 success 或 failure")
            return ["DONE" if status == "success" else "FAIL"]

        raise ToolCallError(f"未注册的 tool：{name!r}")


def _tool_call_dict(call: Any, fallback_index: int) -> dict[str, Any]:
    """兼容 OpenAI SDK object 与测试时的普通字典。"""

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
    """兼容未开启 vLLM tool parser 时落在 content 中的 JSON。"""

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
        raise RuntimeError(f"找不到冻结 prompt：{path}")
    return path.read_text(encoding="utf-8").strip()


# 所有 agent 共享的那段 prompt 的唯一副本：完成纪律 + 环境块
# （persona + 17 应用端口表）。
#
# 这一块**与模型无关**，因此每个 agent 必须逐字节拿到同一份——它决定 agent 知不
# 知道 app 在 localhost:PORT。2026-08-10 的 smoke 实测过缺它的代价：EvoCUA 把
# Cheskepdia 当成公网站点，敲 cheskepdia.com 撞 Server Not Found 后 6 步就 FAIL。
#
# 以前这段文本在 prompts/agents/*_mypcbench_system.txt 里各存了一份（三份逐字节
# 相同），靠 dump_upstream_prompts.py 同步各自的尾部。现在只留这一个文件：
# 上游 build_mypcbench_context(has_bash=False) 生成，--check 负责检测漂移。
# 不在运行时 import agents.prompts，是因为 third_party/ 不进 Git——别人 clone
# DERAIL 下来必须仍然看得见、也跑得起 prompt。
MYPCBENCH_SHARED_BLOCK_FILE = "mypcbench_shared_block.txt"
# cuabash 变体（enable_bash: true）用这份：has_bash=True 的环境块带 sudo 密码与
# Python/LibreOffice CLI 提示，完成纪律保留 bash 从句。同样由 dump 脚本生成与
# 校验；两个变体都必须对拥有对应工具面的 agent 逐字节一致。
MYPCBENCH_SHARED_BLOCK_BASH_FILE = "mypcbench_shared_block_bash.txt"


def compose_system_prompt(scaffold_filename: str, *, has_bash: bool = False) -> str:
    """scaffold 段 + 共享块（按 agent 的 bash 工具面选变体）。

    两段都按原样拼接、不加分隔符：scaffold 文件以恰好一个换行结尾，环境块以
    ``\\n## Persona`` 开头，接起来正好是剥离之前的那两个换行。整体再 strip 一次，
    与旧的 `_read_frozen_prompt(整份文件)` 逐字节相同。这里不能用
    `_read_frozen_prompt` 读共享块——它会 strip 掉块首那个换行。
    """
    shared_file = MYPCBENCH_SHARED_BLOCK_BASH_FILE if has_bash else MYPCBENCH_SHARED_BLOCK_FILE
    for filename in (scaffold_filename, shared_file):
        path = REPO_ROOT / "prompts" / "agents" / filename
        if not path.is_file():
            raise RuntimeError(f"找不到冻结 prompt：{path}")
    root = REPO_ROOT / "prompts" / "agents"
    scaffold = (root / scaffold_filename).read_text(encoding="utf-8")
    shared = (root / shared_file).read_text(encoding="utf-8")
    return (scaffold + shared).strip()


class NativeToolComputerAgent:
    """满足 MyPCBench ``reset/predict`` contract 的视觉工具调用 agent。"""

    def __init__(
        self,
        model: str,
        screen_size: tuple[int, int],
        protocol: ToolAgentProtocol,
        *,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        client: Any = None,
        # env / environment 是同一个 VM 句柄的两个入口名：cuabash 分支传 env=，
        # 上游 qwen38 takeover 分支传 environment=。构造体分别存进 self._env 与
        # self._environment，两条 bash 分流各读各的，故不做归一。
        env: Any = None,
        environment: Any = None,
    ):
        self.model = model
        self.screen_size = tuple(screen_size)
        self.protocol = protocol
        self.compiler = SafePyAutoGUICompiler(self.screen_size, protocol.coordinate_protocol)
        # runner 从 run_mypcbench.get_agent(..., env=env) 一路传进来的 VM 控制句柄，
        # 仅 cuabash 变体的 bash 分流使用；GUI-only agent 持有引用但不触碰。
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
                "enable_bash=True 但 runner 没有传入 env：bash 调用会收到"
                "明确的错误提示并回退 GUI 路径"
            )
        self._base_url = base_url or os.environ.get("OPENAI_BASE_URL")
        self._api_key = api_key or os.environ.get("OPENAI_API_KEY") or "EMPTY"
        self._client = client
        self._environment = environment
        self._turns: list[list[dict[str, Any]]] = []
        self.last_trajectory_tool_messages: list[dict[str, Any]] = []
        self.agent_metadata: dict[str, Any] = {}
        # 与 self._turns 一一对应的动作文本，用来重建 `Previous actions:`。取的是
        # 编译后的 pyautogui 串而不是模型 content：kimi-k3 有 33% 的步返回空
        # content（2026-08-09 smoke 实测 204/625），拿 content 做日志会丢掉三分之一。
        self._turn_actions: list[tuple[str, ...]] = []
        self._action_batches: list[tuple[str, ...]] = []
        self._state_actions: list[tuple[str, tuple[str, ...]]] = []
        # bash 记账模式（见 _MAX_BASH_ROUNDS_PER_STEP 注释）：internal=步内循环
        # （默认，与 cap8/cap16 各批次一致）；steps=统一记账，bash 轮消耗步数。
        accounting = os.environ.get("DERAIL_BASH_ACCOUNTING", "internal")
        if accounting not in ("internal", "steps"):
            raise ValueError(
                f"DERAIL_BASH_ACCOUNTING 只接受 internal/steps，实际为 {accounting!r}"
            )
        self._bash_accounting = accounting
        # steps 模式专用：连续 bash 步计数（GUI 动作/answer 落地时清零），
        # 配合 _MAX_CONSECUTIVE_BASH_STEPS 拦病态循环。
        self._consecutive_bash_steps = 0
        self._takeover_instruction: Optional[str] = None
        self._takeover_prompt = ""

    @property
    def messages(self) -> list[dict[str, Any]]:
        """供 MyPCBench 在 episode 结束时保存去除图片后的完整可见对话。"""

        flattened = [{"role": "system", "content": self.protocol.system_prompt}]
        for turn in self._turns:
            flattened.extend(turn)
        return flattened

    def _get_client(self) -> Any:
        if self._client is None:
            try:
                from openai import OpenAI
            except ImportError as exc:  # pragma: no cover - 由 collection 环境触发
                raise RuntimeError("缺少 openai；请安装 `pip install -e '.[collection]'`") from exc
            if not self._base_url:
                raise RuntimeError("缺少 OPENAI_BASE_URL，无法连接本地推理服务")
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
        """Resolve replay screenshot paths before sending an OpenAI-compatible request."""

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
            raise RuntimeError("Qwen3.8 bash requires the active MyPCBench VM environment")
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
            raise TypeError("obs['screenshot'] 必须是 PNG/JPEG bytes")
        return "data:image/png;base64," + base64.b64encode(screenshot).decode("ascii")

    @staticmethod
    def _screenshot_fingerprint(screenshot: bytes) -> str:
        """Build a stable visual-state fingerprint for bounded loop detection.

        The GNOME top bar contains a changing clock, so byte hashes treat an
        otherwise identical desktop as new every minute.  For loop detection
        only, ignore that 32-pixel strip and quantize a small grayscale image.
        The original screenshot remains untouched in the model request and
        trajectory artifacts.
        """

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
            # Tests and defensive callers may supply non-image bytes. The
            # exact digest remains safe, though less tolerant of UI clocks.
            return hashlib.sha256(bytes(screenshot)).hexdigest()

    def _folded_prefix(self, retained: int) -> int:
        """How many of the retained turns lose their screenshot.

        Mirrors ``qwen35vl_agent._update_folding_state``: fold in blocks of
        ``image_fold_size`` until at most ``max_images_in_context`` images are
        left, so the prefix only grows and the cached prompt prefix stays
        stable between steps.  Folding in blocks means the count can undershoot
        the cap — that is upstream's behaviour, not a rounding bug.

        The cap counts *history* images only; the current screenshot is
        appended afterwards and is never folded.  Upstream's ``image_max``
        counts the current one too, so kimi_k3 at 20 sends 21 images where
        upstream sends 20.  Keeping the field's v1 meaning is worth more than
        that one screenshot: holo_3_1 / qwen3_6 sit at
        ``history_turns == max_images_in_context``, where any other reading
        would silently start folding turns that v1 sent intact.
        """

        images = self.protocol.max_images_in_context
        fold = max(1, self.protocol.image_fold_size)
        folded = 0
        while (retained - folded) > images:
            folded += fold
        return min(folded, retained)

    @staticmethod
    def _fold_turn(turn: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Drop the screenshot from a history turn, keeping the turn itself.

        The user/assistant/tool triple has to stay intact: an orphan ``tool``
        message whose ``tool_call_id`` no longer resolves is a 400 from the
        gateway, the same failure class as the empty assistant message.
        """

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
        """`Previous actions:` block for the turns that fell out of the window.

        Upstream keeps this log unbounded (qwen35vl_agent.py:462) — it is the
        only thing that survives a dropped turn, so truncating it would put the
        agent back where it started.
        """

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
        """把拒绝写成可执行的指令。

        Holo 的 execute-all 策略只会因参数/终止位置等问题进入这里；Qwen3.6 的
        保守策略仍可能拒绝多交互批次，此时回显原批次并点名保留第一个真实动作。
        """

        prefix = f"Rejected by the frozen DERAIL tool schema: {error}."
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
        """把连续的 wait 折叠成一个，返回 (执行, 折叠掉的)。

        协议要管的是"一屏一个交互动作"，wait 放在哪一侧从来不是协议关心的事。
        MyPCBench 逐个执行 actions（env.step 对 "WAIT" 就是 sleep），所以
        ["WAIT", click] 本来就正确表达了"先等界面稳定再点"。此前要求 wait 必须
        后置、且总数不超过 2，把 `wait+wait` 和 `wait+动作` 一律判为违规——v1
        smoke 里这两种形状占了全部 schema 拒绝的三分之一，全是形式噪声。

        这里只折叠重复 wait，不重排顺序、不丢掉唯一的 wait，被折叠的 call 由
        调用方记进 interventions，保持 scaffold 干预可审计。
        """

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
            raise ToolCallError("模型没有返回可解析的 tool call")

        if self.protocol.multi_tool_policy == EXECUTE_ALL_CALLS_IN_ORDER:
            # Holo runner：不重排、不截断，也不折叠 wait。返回列表会由 MyPCBench
            # runner 按顺序逐个 env.step，并在每个物理动作后保存截图。
            executed, collapsed = calls, []
        else:
            executed, collapsed = self._collapse_waits(calls)
        names = [self._call_name(call) for call in executed]
        if self.protocol.multi_tool_policy == ONE_INTERACTION_PLUS_COLLAPSED_WAITS:
            if sum(name != "wait" for name in names) > 1:
                raise BatchedToolCallError("当前 agent 每张截图最多一个交互动作")
            if "bash" in names and len(executed) > 1:
                raise BatchedToolCallError("bash must be returned as the only tool call")
            if "answer" in names and len(executed) > 1:
                raise BatchedToolCallError("answer 必须单独返回，不能与其他 tool call 同批")
        elif "answer" in names:
            # env.step(DONE/FAIL) 会令 runner 立即 break；若 answer 后还有 call，承诺的
            # “全部执行”便无法兑现。允许 [click, answer]，但拒绝 [answer, click]。
            if names.count("answer") > 1 or names[-1] != "answer":
                raise ToolCallError("execute_all 批次中的 answer 最多一个且必须位于末尾")

        # bash 必须单独成批：一次一条命令、不与 GUI 动作或 wait 混发。混批意味着
        # 「执行 shell 的同时挪动鼠标」，两种语义都没法兑现；one_interaction 的
        # 交互计数不认 bash（它不是屏幕交互），所以这里单独拦。
        if self.protocol.enable_bash:
            bash_count = sum(name == "bash" for name in names)
            if bash_count > 1 or (bash_count == 1 and len(executed) > 1):
                raise BatchedToolCallError(
                    "bash 必须单独一个 tool call 返回：一次一条命令，不与其他动作混发"
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
                raise ToolCallError(f"{name} arguments 不是合法 JSON") from exc
            if str(name) == "bash" and self.protocol.enable_bash:
                # bash 不经 SafePyAutoGUICompiler；参数校验在这里，执行在 predict
                # 的 bash 分流（结果以 tool 消息回填后继续对话）。
                command = arguments.get("command")
                if not isinstance(command, str) or not command.strip():
                    raise ToolCallError("bash.command 必须是非空字符串")
                if len(command) > _BASH_COMMAND_MAX_CHARS:
                    raise ToolCallError(
                        f"bash.command 超过 {_BASH_COMMAND_MAX_CHARS} 字符上限"
                    )
                unknown = set(arguments) - {"command"}
                if unknown:
                    raise ToolCallError(
                        f"bash arguments 包含未知字段：{sorted(unknown)!r}"
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
        """在 VM 里执行一条 shell 命令，返回 (回填模型的 tool 文本, intervention)。

        执行走 runner 传入的 env 控制句柄（``_execute_command(command,
        shell=True)``，与 qwen_cuabash / openai_cuabash 同一条通道）；HTTP 层
        自带 120s 超时与 3 次重试。任何执行层故障都转成文本回填给模型，让它
        自己换命令或转 GUI —— bash 故障绝不让 predict 崩掉。
        """

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
        except Exception as exc:  # noqa: BLE001 - 执行层故障回填给模型而不是崩溃
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

        # 空转检测：屏幕连续多少步没有任何变化。动作可以每步都不同——模型在一个
        # 滚到底的列表上换着花样操作、界面纹丝不动，同样算空转。
        stall_limit = self.protocol.stalled_state_step_limit
        if stall_limit:
            stalled = 1  # 当前这一步
            for prior_digest, _ in reversed(self._state_actions):
                if prior_digest != screenshot_digest:
                    break
                stalled += 1
            if stalled >= stall_limit:
                return f"visual state unchanged for {stalled} consecutive steps"

        repeat_limit = self.protocol.alternating_action_repeat_limit
        if not repeat_limit:
            return None

        # 这里曾经还有一条"同画面 + 同动作在整局里累计出现 3 次"的判据，已移除。
        # 它不要求连续，会把正常干活误判成死循环：一个列表逐条处理、每处理完一条
        # 回到看起来一样的列表页，画面和动作都重复，但这是任务本身的形状。剩下两
        # 条判据都要求连续，不会有这个问题。

        # 模型即将连续第 repeat_limit 次发出同一个动作，且上一步没让画面产生任何
        # 变化。要求"画面没变"是为了不误伤向导类界面——Next 按钮位置固定，但每次
        # 点击确实翻到了看得见的下一页。
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
        # The experimental intervention belongs only to the first post-prefix
        # request.  Schema-repair retries reuse request_messages below, while
        # later environment turns receive the ordinary task prompt.
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
        # cuabash 变体的 bash 轮预算。internal 模式：bash 不占 runner 步数，
        # 一步内的 bash 轮数必须有界（护栏理由见 _MAX_BASH_ROUNDS_PER_STEP）。
        # steps 模式：每轮 bash 直接返回、由 runner 计步，此预算不介入。
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
                    logger.warning("tool schema 校验失败，执行一次有界修复：%s", error)
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

            # cuabash bash 分流：decode 出的批恰为一条 bash（decode 已强制单独
            # 成批）。internal 模式：执行后把 stdout/stderr/exit code 作为 tool
            # 消息回填、同一 predict 内继续对话 —— 不落 signature/loop guard
            # （bash 轮不动屏幕），也不消耗 runner 的 max_steps 步数。
            # steps 模式（统一记账）：执行后把本轮作为完整一步返回给 runner，
            # runner 记 TOOL_CALL 轨迹行并计入 step_idx（对齐上游 openai_cuabash
            # 的 shell 轮语义），预算主体回到 max_steps。
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
                # steps 模式的安全阀在命令执行前拦（与 internal 的先检查语义一致）：
                # 第 N+1 轮连发请求直接判 FAIL，命令不落地。
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

            # 上游 qwen38 的 shell 分流。与上面 cuabash 那条同名工具 `bash`，但
            # 参数 schema 不同（commands 列表 vs command 单串）、summary 形状也
            # 不同（带 tool_call_id），故单独用 enable_shell 设门：否则 cuabash
            # 的 summary 落到这里会在 shell["tool_call_id"] 上 KeyError。
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
                    "Rejected by the DERAIL loop guard: recent actions made no progress "
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
                    logger.warning("检测到重复动作循环，执行一次有界重规划：%s", loop_reason)
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
        # 被折叠的 wait 仍然要有对应的 tool 消息：assistant 消息里保留模型原样发出的
        # 全部 tool call（审计需要），OpenAI 的对话 contract 要求每个都有回复。
        collapsed_ids = (
            set(normalization["collapsed_tool_call_ids"]) if normalization else set()
        )
        # 被截断的 scroll 要如实回告，否则模型只会看到「页面又滑到底了」而永远
        # 不知道自己把格数当成了像素。
        clamped_by_id = {clamp["tool_call_id"]: clamp for clamp in clamps}

        def _tool_reply(call: Mapping[str, Any]) -> str:
            if call["id"] in collapsed_ids:
                return (
                    "Collapsed by the DERAIL scaffold: consecutive wait calls run as a single wait."
                )
            clamp = clamped_by_id.get(call["id"])
            if clamp is not None:
                return (
                    "Clamped by the DERAIL scaffold: delta_y is in mouse wheel notches, "
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
        # GUI 动作/answer 落地：steps 模式的连发 bash 计数清零。
        self._consecutive_bash_steps = 0
        # 保存 provider 返回的可见 content，供 EAR judge 审查；这里不访问或注入
        # provider 私有 hidden reasoning。
        trajectory_response = {
            "content": visible_content,
            "tool_calls": calls,
            "compiled_actions": summaries,
            "interventions": interventions,
        }
        return json.dumps(trajectory_response, ensure_ascii=False), actions


def protocol_from_config(config: AgentConfig) -> ToolAgentProtocol:
    """把一份校验过的 yaml 配置装配成 decoder 协议。

    每个字段都来自 yaml，没有代码侧默认值 —— ToolAgentProtocol 上那些 default
    只服务于测试里直接构造的场景，正式路径上 agent_config 已经要求全部 live
    字段present，落不到默认值上。各字段的取值理由写在 yaml 里，那里现在是权威。
    """

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
        # 只有 qwen3_8_27b 的 live spec 声明了 enable_shell，其余 agent 缺键即 False。
        enable_shell=bool(config.live.get("enable_shell", False)),
        # has_bash 默认 False：GUI-only agent 拿到的 prompt 与不传该参数时逐字节相同。
        system_prompt=compose_system_prompt(
            config["system_prompt_file"], has_bash=config["enable_bash"]
        ),
    )


def qwen36_protocol() -> ToolAgentProtocol:
    return protocol_from_config(load_agent_config("qwen3_6_27b"))


def qwen38_protocol() -> ToolAgentProtocol:
    return protocol_from_config(load_agent_config("qwen3_8_27b"))


def holo31_protocol() -> ToolAgentProtocol:
    return protocol_from_config(load_agent_config("holo_3_1_35b_a3b"))


def kimi_k3_protocol() -> ToolAgentProtocol:
    return protocol_from_config(load_agent_config("kimi_k3"))


def kimi_k3_cuabash_protocol() -> ToolAgentProtocol:
    return protocol_from_config(load_agent_config("kimi_k3_cuabash"))
