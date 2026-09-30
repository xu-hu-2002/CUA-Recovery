#!/usr/bin/env python3
"""Export upstream agent and judge prompts as read-only text snapshots; --check only compares."""

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

REPO_ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = REPO_ROOT / "prompts" / "agents" / "upstream_resolved"
JUDGE_OUT_DIR = REPO_ROOT / "prompts" / "judges"

CLIENT_PASSWORD = "password"
SCREEN_SIZE = (1280, 800)
FROZEN_DATE_TEXT = "Monday, January 01, 2024"
FROZEN_DATE = _datetime.datetime(2024, 1, 1)


def _frozen_datetime() -> type:
    """Stand-in datetime class whose today()/now() return FROZEN_DATE."""

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
from derail.mypcbench.agent_config import load_agent_config as _live_config  # noqa: E402

MYPCBENCH_HARNESS = REPO_ROOT / "third_party" / "MyPCBench" / "agent-harness"
EVOCUA_ROOT = REPO_ROOT / "third_party" / "EvoCUA"
OPENCUA_ROOT = REPO_ROOT / "third_party" / "OpenCUA-OSWorld"

DERAIL_PROMPT_DIR = REPO_ROOT / "prompts" / "agents"
SHARED_BLOCK_FILE = "mypcbench_shared_block.txt"
SHARED_BLOCK_BASH_FILE = "mypcbench_shared_block_bash.txt"
CONTEXT_SENTINEL = "\n## Persona\n"


class _Captured(Exception):
    """Raised from call_llm to capture the payload without sending a request."""

    def __init__(self, payload: Any) -> None:
        super().__init__("captured")
        self.payload = payload


@contextlib.contextmanager
def _sys_path(*roots: Path, purge: tuple[str, ...] = ()):
    """Temporarily put an upstream repo at the head of sys.path."""

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
    """Solid-color 1280x800 PNG so predict() can run."""
    from PIL import Image

    buffer = io.BytesIO()
    Image.new("RGB", SCREEN_SIZE, (30, 30, 30)).save(buffer, format="PNG")
    return buffer.getvalue()


def _system_text(payload: Any) -> str:
    """Plain text of the system message in an OpenAI-style payload."""
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


def dump_gpt_5_5() -> tuple[str, str]:
    with _sys_path(MYPCBENCH_HARNESS):
        module = importlib.import_module("agents.openai_cuabash")
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

        original_datetime = vendored.datetime
        vendored.datetime = _frozen_datetime()
        original = vendored.Qwen35VLAgent.call_llm
        vendored.Qwen35VLAgent.call_llm = fake_call_llm
        try:
            agent = qwen_cua.QwenOSWorldAgent(
                model="dump-only",
                screen_size=SCREEN_SIZE,
                client_password=CLIENT_PASSWORD,
                enable_bash=True,
                env=None,
            )
            try:
                agent.predict("DUMP", {"screenshot": _blank_screenshot()})
            except _Captured:
                pass
            except Exception as exc:  # noqa: BLE001
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
        module.S2_SYSTEM_PROMPT = (
            original_sys
            + "\n\n"
            + _shared_block().replace("{", "{{").replace("}", "}}")
        )
        try:
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
    cot_level = _live_config("opencua_72b")["cot_level"]
    with _sys_path(OPENCUA_ROOT, purge=("mm_agents",)):
        prompts = importlib.import_module("mm_agents.opencua.prompts")
        text = prompts.build_sys_prompt(
            level=cot_level, password=CLIENT_PASSWORD, use_random=False
        )
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


_BASH_CLAUSE = "(different coordinates, different app, different bash command)"
_NO_BASH_CLAUSE = "(different coordinates, different app)"


def _context(has_bash: bool) -> str:
    """MyPCBench environment block with no format placeholders left."""

    with _sys_path(MYPCBENCH_HARNESS):
        prompts = importlib.import_module("agents.prompts")
        text = prompts.build_mypcbench_context(has_bash=has_bash)
    if has_bash:
        text = text.replace("{CLIENT_PASSWORD}", CLIENT_PASSWORD)
    if "{" in text or "}" in text:
        raise RuntimeError("环境块出现未渲染的占位符，不能冻结进 .txt")
    return text


def _completion_discipline(keep_bash: bool = False) -> str:
    """Upstream COMPLETION_DISCIPLINE; the GUI-only variant when keep_bash is False."""

    with _sys_path(MYPCBENCH_HARNESS):
        prompts = importlib.import_module("agents.prompts")
        text = prompts.COMPLETION_DISCIPLINE
    if _BASH_CLAUSE not in text:
        raise RuntimeError(
            "上游 COMPLETION_DISCIPLINE 里找不到预期的 bash 从句；"
            "上游改过措辞，请重新确认 GUI-only 变体该怎么裁"
        )
    if not keep_bash:
        text = text.replace(_BASH_CLAUSE, _NO_BASH_CLAUSE)
    if "{" in text or "}" in text:
        raise RuntimeError("completion discipline 出现未渲染的占位符")
    return text


def _shared_block(has_bash: bool = False) -> str:
    """Shared block for every agent: completion discipline plus environment description."""
    return _completion_discipline(keep_bash=has_bash) + _context(has_bash)


def sync_derail_prompts(check: bool) -> list[str]:
    """Maintain the two shared environment blocks and return any mismatches."""

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

    for path in sorted(DERAIL_PROMPT_DIR.glob("*_mypcbench_system.txt")):
        if CONTEXT_SENTINEL in path.read_text(encoding="utf-8"):
            stale.append(path.name)
            print(f"[dup]   {path.relative_to(REPO_ROOT)} 内嵌了环境块副本，请删除")
        else:
            print(f"[ok]    {path.relative_to(REPO_ROOT)}（纯 scaffold）")
    return stale


def _render(agent_id: str, text: str, provenance: str) -> str:
    header = (
        f"# 由 scripts/collection/dump_upstream_prompts.py 自动生成，请勿手工编辑。\n"
        f"# agent_id: {agent_id}\n"
        f"# 来源: {provenance}\n"
        f"# 渲染参数: CLIENT_PASSWORD={CLIENT_PASSWORD!r} "
        f"screen_size={SCREEN_SIZE} CURRENT_DATE={FROZEN_DATE_TEXT!r}\n"
        f"# 分隔线以下是逐字节的 prompt 正文。\n"
        f"{'-' * 78}\n"
    )
    return header + text + ("" if text.endswith("\n") else "\n")


def dump_judge_prompts() -> list[tuple[str, str, str]]:
    """Verbatim judge prompts as (filename, text, source) tuples."""
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
        f"# 由 scripts/collection/dump_upstream_prompts.py 自动生成，请勿手工编辑。\n"
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
