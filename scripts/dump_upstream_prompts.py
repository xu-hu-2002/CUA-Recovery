#!/usr/bin/env python3
"""把 5 个复用上游 scaffold 的 agent 的 system prompt 导出成纯文本，供人工核对。

导出的文件是**只读快照**，不是真值：runtime 仍然从 third_party/ 里的上游代码
读取 prompt。改这些 .txt 不会影响任何一次 collection。之所以要导出，是因为
third_party/ 不进 Git，仓库里看不到这五家的 prompt 到底长什么样。

取值方式尽量走真实代码路径：
- 常量型（OpenAI / Claude / OpenCUA）直接 import 后按 runtime 的参数 format；
- 组装型（Qwen3.5 / EvoCUA）在 predict() 里才拼好，所以 monkeypatch call_llm
  截获真实 payload 的 system message，保证是逐字节的线上内容。

用法：
    python scripts/dump_upstream_prompts.py [--check]

--check 只比对不写盘，退出码非零表示快照与上游已经不一致（可进 CI）。
"""

from __future__ import annotations

import argparse
import contextlib
import datetime as _datetime
import importlib
import io
import json
import os
import sys
from pathlib import Path
from typing import Any, Callable

REPO_ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = REPO_ROOT / "prompts" / "agents" / "upstream_resolved"
JUDGE_OUT_DIR = REPO_ROOT / "prompts" / "judges"

# 与 run_mypcbench.py 的 argparse 默认值一致；collection 脚本没有覆盖它们。
CLIENT_PASSWORD = "password"
SCREEN_SIZE = (1280, 800)
# prompt 里嵌了 datetime.today()，为了让快照可复现必须钉死一个日期。
FROZEN_DATE_TEXT = "Monday, January 01, 2024"
FROZEN_DATE = _datetime.datetime(2024, 1, 1)


def _frozen_datetime() -> type:
    """替身 datetime 类，today()/now() 恒返回 FROZEN_DATE。

    给「跑上游真实代码再截获 payload」那几个 dumper 用：它们没法像常量型那样
    .format(CURRENT_DATE=...)，日期由上游模块内部的 datetime.today() 决定。
    """

    class _Frozen(_datetime.datetime):
        @classmethod
        def today(cls) -> _datetime.datetime:
            return FROZEN_DATE

        @classmethod
        def now(cls, tz: Any = None) -> _datetime.datetime:
            return FROZEN_DATE

    return _Frozen

if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))
# dump 出来的快照必须用 collection 实际使用的那组参数生成，所以这里和 factory
# 走同一个加载器，而不是各自抄一份常量。
from derail.mypcbench.agent_config import load_agent_config as _live_config  # noqa: E402

MYPCBENCH_HARNESS = REPO_ROOT / "third_party" / "MyPCBench" / "agent-harness"
EVOCUA_ROOT = REPO_ROOT / "third_party" / "EvoCUA"
OPENCUA_ROOT = REPO_ROOT / "third_party" / "OpenCUA-OSWorld"

# 所有 agent 共享的那段 prompt 的唯一副本 = COMPLETION_DISCIPLINE + 环境块，
# 两段都来自上游 agents/prompts.py，本脚本只负责渲染并保证不漂。
#
# 有两个变体：GUI-only（mypcbench_shared_block.txt，删掉 bash 从句、环境块不带
# CLI 提示）和 cuabash（mypcbench_shared_block_bash.txt，保留 bash 从句、环境块
# 带 sudo 密码与 Python/LibreOffice CLI 行，{CLIENT_PASSWORD} 在生成时渲染）。
#
# 以前环境块在每个 DERAIL scaffold 的 .txt 尾部各存一份（三份逐字节相同），本脚本
# 逐个同步它们的尾部；agent 一多，「有没有被同步」就成了要逐个记的事，kimi_k3 就
# 曾经漏在名单外。现在只有这一个文件：DERAIL scaffold 走
# tool_agent.compose_system_prompt()，EvoCUA/OpenCUA 走 factory._mypcbench_shared_block()，
# 新增 agent 不必再往任何名单里加一行。
#
# 用生成的纯文本文件而不是运行时 import agents.prompts：third_party/ 不进 Git，
# 别人 clone DERAIL 下来必须仍然看得见发给模型的原文。
DERAIL_PROMPT_DIR = REPO_ROOT / "prompts" / "agents"
SHARED_BLOCK_FILE = "mypcbench_shared_block.txt"
SHARED_BLOCK_BASH_FILE = "mypcbench_shared_block_bash.txt"
# 环境块的起始标记。prompt 是原样发给模型的，不能塞注释型分隔符，所以直接用
# 块本身的第一个标题当边界。scaffold .txt 里**不应该**再出现它。
CONTEXT_SENTINEL = "\n## Persona\n"


class _Captured(Exception):
    """从 call_llm 里把 payload 抛出来，避免真的发请求。"""

    def __init__(self, payload: Any) -> None:
        super().__init__("captured")
        self.payload = payload


@contextlib.contextmanager
def _sys_path(*roots: Path, purge: tuple[str, ...] = ()):
    """临时把上游仓库放到 sys.path 头部。

    EvoCUA 和 OpenCUA 用了同一个顶层包名 `mm_agents`，谁先 import 谁就会把对方
    遮住，所以切换仓库前必须把该前缀的模块从 sys.modules 里清掉。
    """

    original_path = list(sys.path)
    evicted = {
        name: module
        for name, module in sys.modules.items()
        if any(name == prefix or name.startswith(prefix + ".") for prefix in purge)
    }
    for name in evicted:
        del sys.modules[name]
    for root in roots:
        sys.path.insert(0, str(root))
    try:
        yield
    finally:
        sys.path[:] = original_path
        for name in list(sys.modules):
            if any(name == prefix or name.startswith(prefix + ".") for prefix in purge):
                del sys.modules[name]
        sys.modules.update(evicted)


def _blank_screenshot() -> bytes:
    """1280x800 纯色 PNG；prompt 内容与像素无关，只是让 predict() 能跑通。"""
    from PIL import Image

    buffer = io.BytesIO()
    Image.new("RGB", SCREEN_SIZE, (30, 30, 30)).save(buffer, format="PNG")
    return buffer.getvalue()


def _system_text(payload: Any) -> str:
    """从 OpenAI 风格 payload 里取出 system message 的纯文本。"""
    messages = payload.get("messages") if isinstance(payload, dict) else None
    if not messages:
        raise RuntimeError("payload 中没有 messages")
    first = messages[0]
    if first.get("role") != "system":
        raise RuntimeError("第一条消息不是 system，无法确定 prompt 边界")
    content = first.get("content")
    if isinstance(content, str):
        return content
    parts = [
        part.get("text", "")
        for part in content
        if isinstance(part, dict) and part.get("type") == "text"
    ]
    return "".join(parts)


# --------------------------------------------------------------------------
# 各 agent 的取值函数。返回 (prompt_text, 来源说明)
# --------------------------------------------------------------------------


def dump_gpt_5_5() -> tuple[str, str]:
    with _sys_path(MYPCBENCH_HARNESS):
        module = importlib.import_module("agents.openai_cuabash")
        # _build_primer 把 OPERATOR_PROMPT 的工具行换成 computer+shell，再接
        # GUI_WORKFLOW_HINT；OPERATOR_PROMPT 本身已经 = OPENAI_CUA_OPERATOR_PROMPT
        # + "\n\n" + MYPCBENCH_CONTEXT（openai_base.py:153）。
        primer = module._build_primer("cua_tools")
    text = primer.format(CLIENT_PASSWORD=CLIENT_PASSWORD, CURRENT_DATE=FROZEN_DATE_TEXT)
    return text, (
        "agents/openai_cuabash.py:_build_primer('cua_tools')，作为**第一条 user "
        "消息的文本块**注入（不是 system role），见 openai_cuabash.py:178-187"
    )


def dump_claude_sonnet_5() -> tuple[str, str]:
    with _sys_path(MYPCBENCH_HARNESS):
        prompts = importlib.import_module("agents.prompts")
        template = prompts.CLAUDE_CUA_SYSTEM_PROMPT
    text = template.format(CLIENT_PASSWORD=CLIENT_PASSWORD, CURRENT_DATE=FROZEN_DATE_TEXT)
    return text, (
        "agents/prompts.py:CLAUDE_CUA_SYSTEM_PROMPT，enable_computer=True 分支，"
        "见 claude_cuabash.py:264"
    )


def dump_qwen3_5_35b_a3b() -> tuple[str, str]:
    with _sys_path(MYPCBENCH_HARNESS):
        qwen_cua = importlib.import_module("agents.qwen_cua")
        vendored = importlib.import_module(
            "agents.vendored_paper_results.qwen35vl_agent"
        )

        captured: dict[str, Any] = {}

        def fake_call_llm(self: Any, payload: Any, model: Any = None) -> str:
            captured["payload"] = payload
            raise _Captured(payload)

        # 常量型 dumper 用 .format(CURRENT_DATE=FROZEN_DATE_TEXT) 冻结日期，但这条
        # 路径跑的是上游真实代码，qwen35vl_agent.py:444 里的 datetime.today() 会取
        # 当天。不冻结的话快照每天都变，--check 会天天误报 [stale]，漂移检测形同虚设。
        original_datetime = vendored.datetime
        vendored.datetime = _frozen_datetime()
        original = vendored.Qwen35VLAgent.call_llm
        vendored.Qwen35VLAgent.call_llm = fake_call_llm
        try:
            agent = qwen_cua.QwenOSWorldAgent(
                model="dump-only",
                screen_size=SCREEN_SIZE,
                client_password=CLIENT_PASSWORD,
                # 论文主实验用的是 qwen_cuabash，带 bash 工具。
                enable_bash=True,
                env=None,
            )
            try:
                agent.predict("DUMP", {"screenshot": _blank_screenshot()})
            except _Captured:
                pass
            except Exception as exc:  # noqa: BLE001 - 捕获后仍可能拿到 payload
                if "payload" not in captured:
                    raise RuntimeError(f"Qwen3.5 predict 未走到 call_llm：{exc}") from exc
        finally:
            vendored.Qwen35VLAgent.call_llm = original
            vendored.datetime = original_datetime

    return _system_text(captured["payload"]), (
        "vendored_paper_results/qwen35vl_agent.py 的 tools_def system prompt，"
        "外加 agents/qwen_cua.py:call_llm 注入的 build_mypcbench_context(has_bash=True) "
        "和 _BASH_TOOL_DESCRIPTION"
    )


def dump_evocua_32b() -> tuple[str, str]:
    with _sys_path(EVOCUA_ROOT, purge=("mm_agents",)):
        module = importlib.import_module("mm_agents.evocua.evocua_agent")

        captured: dict[str, Any] = {}

        def fake_call_llm(self: Any, payload: Any) -> str:
            captured["payload"] = payload
            raise _Captured(payload)

        original = module.EvoCUAAgent.call_llm
        module.EvoCUAAgent.call_llm = fake_call_llm
        original_sys = module.S2_SYSTEM_PROMPT
        # 必须和 factory._create_evocua 打一样的补丁，否则快照展示的是一份线上
        # 根本没发出去过的 prompt —— 快照的全部意义就是让人看见真正发了什么。
        module.S2_SYSTEM_PROMPT = (
            original_sys
            + "\n\n"
            + _shared_block().replace("{", "{{").replace("}", "}}")
        )
        try:
            # 与 factory._create_evocua 读同一份 yaml，快照才不会和线上漂移。
            # （改这里之前传的是 cot_level="s2"，EvoCUAAgent 没有这个参数，被
            # **kwargs 吞掉了，实际靠的是 prompt_style 的默认值。）
            config = _live_config("evocua_32b")
            agent = module.EvoCUAAgent(
                model="dump-only",
                prompt_style=config["prompt_style"],
                screen_size=SCREEN_SIZE,
                coordinate_type=config["coordinate_type"],
                password=CLIENT_PASSWORD,
                resize_factor=config["resize_factor"],
            )
            try:
                agent.predict("DUMP", {"screenshot": _blank_screenshot()})
            except _Captured:
                pass
            except Exception as exc:  # noqa: BLE001
                if "payload" not in captured:
                    raise RuntimeError(f"EvoCUA predict 未走到 call_llm：{exc}") from exc
        finally:
            module.EvoCUAAgent.call_llm = original
            module.S2_SYSTEM_PROMPT = original_sys

    return _system_text(captured["payload"]), (
        "mm_agents/evocua/prompts.py:S2_SYSTEM_PROMPT，tools_xml 由 "
        "build_s2_tools_def(S2_DESCRIPTION_PROMPT_TEMPLATE) 生成；"
        "coordinate_type='relative' 固定 resolution 为 1000x1000；"
        "尾部由 factory._create_evocua 追加 prompts/agents/mypcbench_shared_block.txt"
    )


def dump_opencua_72b() -> tuple[str, str]:
    # 与 factory._create_opencua 读同一份 yaml。
    cot_level = _live_config("opencua_72b")["cot_level"]
    with _sys_path(OPENCUA_ROOT, purge=("mm_agents",)):
        prompts = importlib.import_module("mm_agents.opencua.prompts")
        text = prompts.build_sys_prompt(
            level=cot_level, password=CLIENT_PASSWORD, use_random=False
        )
    # 与 factory._create_opencua 的 inner.system_prompt 拼法保持一致。
    text = text.rstrip() + "\n\n" + _shared_block().strip()
    return text, (
        f"mm_agents/opencua/prompts.py:build_sys_prompt(level={cot_level!r}, use_random=False)，"
        "见 opencua_agent.py:295；"
        "尾部由 factory._create_opencua 追加 prompts/agents/mypcbench_shared_block.txt"
    )


DUMPERS: dict[str, Callable[[], tuple[str, str]]] = {
    "gpt_5_5": dump_gpt_5_5,
    "qwen3_5_35b_a3b": dump_qwen3_5_35b_a3b,
    "claude_sonnet_5": dump_claude_sonnet_5,
    "evocua_32b": dump_evocua_32b,
    "opencua_72b": dump_opencua_72b,
}


# COMPLETION_DISCIPLINE 里唯一一处 bash 提法。prompts/agents/README.md 第 1 条：
# 对没有 shell 的 agent 提终端，小模型会当真，白白往 GUI 终端里敲命令。
_BASH_CLAUSE = "(different coordinates, different app, different bash command)"
_NO_BASH_CLAUSE = "(different coordinates, different app)"


def _context(has_bash: bool) -> str:
    """MyPCBench 的环境块。渲染后不含任何 format 占位符。"""

    with _sys_path(MYPCBENCH_HARNESS):
        prompts = importlib.import_module("agents.prompts")
        text = prompts.build_mypcbench_context(has_bash=has_bash)
    if has_bash:
        # has_bash=True 的模板里 sudo 密码行是 {CLIENT_PASSWORD} 占位符（上游用
        # f-string 双花括号转义保留的）。不能 .format —— 万一块里出现别的花括号
        # 会炸，逐字替换后下面的花括号校验兜底。
        text = text.replace("{CLIENT_PASSWORD}", CLIENT_PASSWORD)
    if "{" in text or "}" in text:
        raise RuntimeError("环境块出现未渲染的占位符，不能冻结进 .txt")
    return text


def _completion_discipline(keep_bash: bool = False) -> str:
    """上游 COMPLETION_DISCIPLINE；keep_bash=False 时返回 GUI-only 变体。

    这段在上游是死代码：只有 SYS_PROMPT_IN_SCREENSHOT_OUT_CODE 引用它，而那个
    常量全仓无人 import，所以 MyPCBench 自家 agent 也一个都没收到。2026-08-10
    的 smoke 实测了它缺席的样子——`preference_inference-f014` 六步就 FAIL，六条
    撞满步数上限的末动作是普通 click、没有写下最终答案（而判官读的正是最后一条
    回复）。DERAIL 决定对所有 agent 一并补上，因此它和环境块一起进共享块。

    注意这不是"环境描述"而是行为引导，README 记录过一次同体裁的失败 A/B；开启
    它是一次显式的协议决定，不是顺手优化。
    """

    with _sys_path(MYPCBENCH_HARNESS):
        prompts = importlib.import_module("agents.prompts")
        text = prompts.COMPLETION_DISCIPLINE
    if _BASH_CLAUSE not in text:
        raise RuntimeError(
            "上游 COMPLETION_DISCIPLINE 里找不到预期的 bash 从句；"
            "上游改过措辞，请重新确认 GUI-only 变体该怎么裁"
        )
    if not keep_bash:
        # 对没有 shell 的 agent 提终端，小模型会当真，白白往 GUI 终端里敲命令
        # （kimi_k3 的 F3 实测就是这么丢的 109 次）。cuabash 变体保留原从句。
        text = text.replace(_BASH_CLAUSE, _NO_BASH_CLAUSE)
    if "{" in text or "}" in text:
        raise RuntimeError("completion discipline 出现未渲染的占位符")
    return text


def _shared_block(has_bash: bool = False) -> str:
    """所有 agent 都拿到的共享块 = 完成纪律 + 环境描述。

    顺序对齐上游 SYS_PROMPT_IN_SCREENSHOT_OUT_CODE 的拼法：scaffold 之后先是
    COMPLETION_DISCIPLINE，再是 MYPCBENCH_CONTEXT。has_bash 两个分支的裁剪口径
    必须一致：要么都保留 bash 提法，要么都删。
    """
    return _completion_discipline(keep_bash=has_bash) + _context(has_bash)


def sync_derail_prompts(check: bool) -> list[str]:
    """维护两份共享环境块，并确保没有 scaffold 文件私藏一份副本。返回不一致的项。"""

    stale: list[str] = []
    for filename, has_bash in (
        (SHARED_BLOCK_FILE, False),
        (SHARED_BLOCK_BASH_FILE, True),
    ):
        context = _shared_block(has_bash)
        target = DERAIL_PROMPT_DIR / filename
        current = target.read_text(encoding="utf-8") if target.is_file() else ""
        if current == context:
            print(f"[ok]    {target.relative_to(REPO_ROOT)}")
        elif check:
            stale.append(filename)
            print(f"[stale] {target.relative_to(REPO_ROOT)}")
        else:
            target.write_text(context, encoding="utf-8")
            print(f"已写入共享块 -> {target.relative_to(REPO_ROOT)}")

    # 环境块只许出现在上面两个共享块文件里。scaffold 文件里再出现 `## Persona`
    # （两个变体共同的起始标记）就意味着有人手工粘了副本回去，那正是这次重构要
    # 消灭的东西——两份副本只会不声不响地漂开。
    for path in sorted(DERAIL_PROMPT_DIR.glob("*_mypcbench_system.txt")):
        if CONTEXT_SENTINEL in path.read_text(encoding="utf-8"):
            stale.append(path.name)
            print(f"[dup]   {path.relative_to(REPO_ROOT)} 内嵌了环境块副本，请删除")
        else:
            print(f"[ok]    {path.relative_to(REPO_ROOT)}（纯 scaffold）")
    return stale


def _render(agent_id: str, text: str, provenance: str) -> str:
    header = (
        f"# 由 scripts/dump_upstream_prompts.py 自动生成，请勿手工编辑。\n"
        f"# agent_id: {agent_id}\n"
        f"# 来源: {provenance}\n"
        f"# 渲染参数: CLIENT_PASSWORD={CLIENT_PASSWORD!r} "
        f"screen_size={SCREEN_SIZE} CURRENT_DATE={FROZEN_DATE_TEXT!r}\n"
        f"# 分隔线以下是逐字节的 prompt 正文。\n"
        f"{'-' * 78}\n"
    )
    return header + text + ("" if text.endswith("\n") else "\n")


def dump_judge_prompts() -> list[tuple[str, str, str]]:
    """判官侧的逐字 prompt：(文件名, 正文, 来源说明)。

    和 agent prompt 同样是只读快照 —— runtime 仍从 third_party 读。判官 prompt
    是评测协议本身，接管它等于 fork：上游一改我们不会知道，而且我们的
    Rubric %/Perfect % 与 leaderboard 的差距将无法归因。

    system 常量直接 import；两条 user 消息模板由函数在运行时拼装，这里用占位值
    调用真函数取回逐字结果，避免手抄出第二份副本（正是 prompts/judges/README.md
    要求避免的）。
    """
    with _sys_path(MYPCBENCH_HARNESS, MYPCBENCH_HARNESS / "utils"):
        judge = importlib.import_module("utils.osworld_full_traj_judge")

    placeholder_rubric = {
        "id": "<RUBRIC_ID>",
        "requirement": "<REQUIREMENT>",
        "verification": "<VERIFICATION>",
        "weight": "<WEIGHT>",
    }
    per_rubric = judge._user_text_for_rubric(
        "<TASK_INSTRUCTION>", placeholder_rubric, "<ACTION_HISTORY>", "<N_SCREENSHOTS>", "<N_STEPS>"
    )
    shared_prefix = judge._shared_prefix_text(
        "<TASK_INSTRUCTION>", "<ACTION_HISTORY>", "<N_SCREENSHOTS>", "<N_STEPS>"
    )
    rubric_suffix = judge._rubric_suffix_text(placeholder_rubric)
    return [
        (
            "mypcbench_full_traj_system",
            judge.FULL_TRAJ_JUDGMENT_SYSTEM,
            "osworld_full_traj_judge.py:FULL_TRAJ_JUDGMENT_SYSTEM —— 两条路径共用。"
            "OpenAI 走 messages[0] role=system，Gemini 走 GenerateContentConfig.system_instruction",
        ),
        (
            "mypcbench_full_traj_gemini_user",
            per_rubric,
            "osworld_full_traj_judge.py:_user_text_for_rubric() —— 仅 Gemini 路径（判官第 816 行）。"
            "整条 user 消息一次拼好，rubric 在图片之前。**本仓库当前不走这条路径**",
        ),
        (
            "mypcbench_full_traj_openai_user_prefix",
            shared_prefix,
            "osworld_full_traj_judge.py:_shared_prefix_text() —— 仅 OpenAI 路径（第 827 行）。"
            "user 消息的第 1 段，其后紧跟图片块；同一 task 的各 rubric 共享此前缀以命中 prompt 缓存",
        ),
        (
            "mypcbench_full_traj_openai_user_suffix",
            rubric_suffix,
            "osworld_full_traj_judge.py:_rubric_suffix_text() —— 仅 OpenAI 路径（第 711 行）。"
            "user 消息的第 3 段，接在图片块**之后**；rubric 放尾部才不会打断缓存前缀",
        ),
    ]


def _render_judge(name: str, text: str, provenance: str) -> str:
    header = (
        f"# 由 scripts/dump_upstream_prompts.py 自动生成，请勿手工编辑。\n"
        f"# prompt_id: {name}\n"
        f"# 来源: {provenance}\n"
        f"# 这是只读快照：runtime 仍从 third_party/ 的上游代码读取，改本文件不影响判分。\n"
        f"# 生效的判官超参数见 configs/judges/mypcbench_rubric.yaml。\n"
        f"# 分隔线以下是逐字节的 prompt 正文。\n"
        f"{'-' * 78}\n"
    )
    return header + text + ("" if text.endswith("\n") else "\n")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check",
        action="store_true",
        help="只比对不写盘；快照与上游不一致时返回 1。",
    )
    args = parser.parse_args()

    for root, marker in (
        (MYPCBENCH_HARNESS, "agents/prompts.py"),
        (EVOCUA_ROOT, "mm_agents/evocua/prompts.py"),
        (OPENCUA_ROOT, "mm_agents/opencua/prompts.py"),
    ):
        if not (root / marker).is_file():
            print(f"缺少上游源码：{root / marker}", file=sys.stderr)
            print("third_party/ 不进 Git，请先按 docs/ 的说明取回上游仓库。", file=sys.stderr)
            return 2

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    stale: list[str] = []
    for agent_id, dumper in DUMPERS.items():
        text, provenance = dumper()
        rendered = _render(agent_id, text, provenance)
        target = OUT_DIR / f"{agent_id}.txt"
        if args.check:
            current = target.read_text(encoding="utf-8") if target.is_file() else ""
            if current != rendered:
                stale.append(agent_id)
                print(f"[stale] {target.relative_to(REPO_ROOT)}")
            else:
                print(f"[ok]    {target.relative_to(REPO_ROOT)}")
            continue
        target.write_text(rendered, encoding="utf-8")
        print(f"已写入 {target.relative_to(REPO_ROOT)}（{len(text)} 字符）")

    JUDGE_OUT_DIR.mkdir(parents=True, exist_ok=True)
    for name, text, provenance in dump_judge_prompts():
        rendered = _render_judge(name, text, provenance)
        target = JUDGE_OUT_DIR / f"{name}.txt"
        if args.check:
            current = target.read_text(encoding="utf-8") if target.is_file() else ""
            if current != rendered:
                stale.append(name)
                print(f"[stale] {target.relative_to(REPO_ROOT)}")
            else:
                print(f"[ok]    {target.relative_to(REPO_ROOT)}")
            continue
        target.write_text(rendered, encoding="utf-8")
        print(f"已写入 {target.relative_to(REPO_ROOT)}（{len(text)} 字符）")

    stale.extend(sync_derail_prompts(check=args.check))

    if stale:
        print(
            "\n上游 prompt 已变化，请重新运行不带 --check 的本脚本并审查 diff。",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
