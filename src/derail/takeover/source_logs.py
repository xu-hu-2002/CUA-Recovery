"""Read recorded execution evidence from canonical steps' source traj.jsonl rows."""

from __future__ import annotations

import json
import os
import re
import urllib.parse
from functools import lru_cache
from pathlib import Path
from typing import Any, Mapping

from derail.canonical.trajectory import CanonicalStep


_PRIVATE_KEYS = frozenset(
    {"analysis", "chain_of_thought", "reasoning", "reasoning_content", "thought"}
)


_THOUGHT_SECTION = re.compile(r"#+\s*Thought:.*?(?=\n#+\s|\Z)", re.DOTALL | re.IGNORECASE)
# openai_cuabash appends reasoning summaries to the visible response as "[reasoning] ..." lines.
_REASONING_SUMMARY_LINE = re.compile(r"(?m)^\[reasoning\] .*(?:\n|$)")


def _public_value(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {
            str(key): _public_value(item)
            for key, item in value.items()
            if str(key).lower() not in _PRIVATE_KEYS
        }
    if isinstance(value, (list, tuple)):
        return [_public_value(item) for item in value]
    return value


def sanitize_visible_response(value: Any) -> str:
    """Keep source actions and tool calls while removing private reasoning."""

    if not isinstance(value, str):
        return ""
    text = value.strip()
    if not text:
        return ""
    if text.startswith("{"):
        try:
            record = json.loads(text)
        except json.JSONDecodeError:
            record = None
        if isinstance(record, dict) and isinstance(record.get("content"), str):
            record["content"] = ""
            return json.dumps(record, ensure_ascii=False)
    lowered = text.lower()
    if "</think>" in lowered:
        text = text[lowered.rfind("</think>") + len("</think>") :]
    elif "<think>" in lowered:
        text = text[: lowered.find("<think>")]
    return _THOUGHT_SECTION.sub("", text).strip()


# Backward-compatible private alias for callers/tests written before this became
# the shared public sanitizer used by Judge evidence construction.
_visible_response = sanitize_visible_response


def _path_mappings() -> Mapping[str, Any]:
    mappings_file = os.environ.get("DERAIL_PATH_REMAP_FILE", "").strip()
    mappings_text = os.environ.get("DERAIL_PATH_REMAP_JSON", "").strip()
    if mappings_file:
        mappings_text = Path(mappings_file).read_text(encoding="utf-8")
    if not mappings_text:
        return {}
    mappings = json.loads(mappings_text)
    if not isinstance(mappings, Mapping):
        raise ValueError("takeover path mappings must be a JSON object")
    return mappings


def _source_location(value: str) -> tuple[Path, int]:
    if "#line=" not in value:
        raise ValueError(f"source_record_uri has no line binding: {value}")
    path_text, line_text = value.rsplit("#line=", 1)
    if path_text.startswith("file://"):
        path_text = urllib.parse.unquote(urllib.parse.urlparse(path_text).path)
    try:
        line_number = int(line_text)
    except ValueError as exc:
        raise ValueError(f"invalid source_record_uri line: {value}") from exc
    if line_number <= 0:
        raise ValueError(f"source_record_uri line must be positive: {value}")
    path = Path(path_text).expanduser().resolve()
    replacement = _path_mappings().get(str(path))
    if replacement:
        path = Path(str(replacement)).expanduser().resolve()
    return path, line_number


@lru_cache(maxsize=256)
def _claude_native_turn(
    messages_path: Path, source_line: int, allow_unbound: bool = False
) -> Mapping[str, Any] | None:
    """Bind a traj row to its recorded Anthropic assistant/result turn.

    ``allow_unbound`` 用于可能没有原生 assistant 消息的合成终止行；这类行返回
    None，其余无法绑定的行仍然报错。
    """

    payload = json.loads(messages_path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError(f"Claude messages.json must contain an array: {messages_path}")
    row_cursor = 1
    for assistant_index, item in enumerate(payload):
        if not isinstance(item, Mapping) or item.get("role") != "assistant":
            continue
        content = item.get("content")
        computer_calls = [
            block for block in content or []
            if isinstance(block, Mapping)
            and block.get("type") == "tool_use"
            and block.get("name") == "computer"
        ]
        # Bash/editor calls execute inside one predict() and produce one TOOL_CALL
        # traj row. A batched computer response is expanded by the runner into one
        # traj row per GUI action.
        row_count = max(1, len(computer_calls))
        group_end = row_cursor + row_count - 1
        if row_cursor <= source_line <= group_end:
            result = None
            if assistant_index + 1 < len(payload):
                candidate = payload[assistant_index + 1]
                if isinstance(candidate, Mapping) and candidate.get("role") == "user":
                    result = candidate
            return {
                "initial_user": payload[0] if row_cursor == 1 else None,
                # A repaired prefix may drop the first recorded turn; the task message
                # still opens the injected conversation.
                "task_user": payload[0],
                "assistant": item,
                "result_user": result,
                "source_row_start": row_cursor,
                "source_row_end": group_end,
                "is_group_end": source_line == group_end,
            }
        row_cursor = group_end + 1
    if allow_unbound:
        return None
    raise ValueError("Claude traj row has no matching native assistant action")


def _claude_control_row(raw: Mapping[str, Any], step: CanonicalStep) -> bool:
    """True for the synthetic end-of-run row a stalled rollout appends.

    harness 在源运行 stall 结束时会补写一行 ``action="FAIL"``、``done=true`` 的
    终止行（step_num 与前一行重复）；它没有 Anthropic assistant 消息，被 canonical
    化为轨迹的 termination 事件。
    """

    return raw.get("done") is True and getattr(step.action, "kind", "") == "terminate"


def _read_jsonl_line(path: Path, line_number: int, uri: str) -> Mapping[str, Any]:
    try:
        with path.open(encoding="utf-8") as handle:
            raw_line = next(
                (line for index, line in enumerate(handle, start=1) if index == line_number),
                "",
            )
        raw = json.loads(raw_line)
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read source trajectory row: {uri}") from exc
    if not isinstance(raw, Mapping):
        raise ValueError(f"source trajectory row is not an object: {uri}")
    return raw


def read_source_row(step: CanonicalStep) -> tuple[Path, int, Mapping[str, Any]]:
    """Return ``(traj.jsonl path, 1-based line, row)`` bound to one canonical step."""

    path, line_number = _source_location(step.source_record_uri)
    raw = _read_jsonl_line(path, line_number, step.source_record_uri)
    if step.source_step_id is not None and raw.get("step_num") != step.source_step_id + 1:
        raise ValueError("canonical/source trajectory step binding mismatch")
    return path, line_number, raw


def load_trajectory_log(step: CanonicalStep, *, strip_reasoning: bool = True) -> str:
    """Return the recorded source output that accompanies one injected history step.

    The canonical action and replay result are represented separately as the
    target agent's native assistant/tool messages.  This supplement carries
    the remaining useful traj.jsonl fields, especially for tool-only turns and
    rows whose original screenshot is missing.  Every target renderer reads
    source text only through here, so ``strip_reasoning`` (paper history
    H = (o_i, a_i)) removes private reasoning for all agents at once; ``False``
    passes the row through as recorded.
    """

    if not step.source_record_uri:
        return ""
    path, line_number, raw = read_source_row(step)
    public = _public_value if strip_reasoning else (lambda value: value)
    metadata = raw.get("agent_metadata")
    tool_messages = metadata.get("tool_messages") if isinstance(metadata, Mapping) else None
    record = {
        "step_num": raw.get("step_num"),
        "reward": raw.get("reward"),
        "done": raw.get("done"),
        "info": public(raw.get("info", {})),
        "source_screenshot_file": raw.get("screenshot_file", ""),
        "source_screenshot_available": bool(raw.get("screenshot_file")),
        "visible_response": (
            _REASONING_SUMMARY_LINE.sub("", sanitize_visible_response(raw.get("response"))).strip()
            if strip_reasoning
            else str(raw.get("response") or "")
        ),
        "source_record_uri": step.source_record_uri,
    }
    if not strip_reasoning and raw.get("reasoning"):
        record["reasoning"] = raw["reasoning"]
    messages_path = path.parent / "messages.json"
    if "claude" in step.source_agent.lower() and messages_path.is_file():
        turn = _claude_native_turn(
            messages_path, line_number, allow_unbound=_claude_control_row(raw, step)
        )
        if turn is None:
            record["control_flow_row"] = True
        else:
            # Opus 4.8 thinking blocks hold only a provider signature (empty text); they stay
            # because the Messages API rejects a tool-use turn whose thinking block is gone.
            record["anthropic_native_turn"] = turn
    if metadata:
        record["agent_metadata"] = public(metadata)
    if tool_messages:
        record["tool_messages"] = public(tool_messages)
    return json.dumps(record, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


# Compatibility alias for callers outside this repository.
load_public_trajectory_log = load_trajectory_log
