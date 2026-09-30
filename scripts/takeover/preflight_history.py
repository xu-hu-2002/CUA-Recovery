#!/usr/bin/env python3
"""Validate takeover history shape and, when requested, its live token budget."""

from __future__ import annotations

import argparse
import base64
import io
import json
import os
import sys
import time
import urllib.error
import urllib.request
from collections import Counter
from functools import lru_cache
from pathlib import Path
from typing import Any

from jsonschema import Draft202012Validator
from PIL import Image


REPOSITORY = Path(__file__).resolve().parents[2]
SRC = REPOSITORY / "src"
HARNESS = REPOSITORY / "third_party" / "MyPCBench" / "agent-harness"
for entry in (str(SRC), str(HARNESS)):
    if entry not in sys.path:
        sys.path.insert(0, entry)

from derail.adapters import HistoryStep, create_native_history_adapter  # noqa: E402
from derail.adapters.qwen35 import QWEN35_AGENT_IDS  # noqa: E402
from derail.derived.layout import atomic_write_json, sha256_file, sha256_json  # noqa: E402
from derail.mypcbench.agent_config import (  # noqa: E402
    DERAIL_TOOL_AGENT,
    load_agent_config,
    load_config,
    resolve_checkpoint,
)
from derail.mypcbench.tool_agent import (  # noqa: E402
    NativeToolComputerAgent,
    protocol_from_config,
)
from derail.mypcbench.qwen35_takeover import wrap_qwen35_takeover_target  # noqa: E402
from derail.mypcbench.claude_takeover import wrap_claude_takeover_target  # noqa: E402
from derail.takeover.diagnosis import load_human_diagnosis_evidence  # noqa: E402
from derail.takeover.protocol import (  # noqa: E402
    DEFAULT_CONFIG_PATH,
    ProtocolExclusion,
    load_takeover_config,
    load_takeover_steps,
    takeover_prefix,
)
from derail.takeover.source_logs import load_trajectory_log  # noqa: E402


_STATE_HISTORY_ROLES = {
    "qwen35vl_snapshot_state_public_response": "qwen35_state",
    "opencua_observations_actions_cots_action_history": "opencua_state",
}


class _PreflightEnvironment:
    """Prevent token preflight from silently executing a live shell call."""

    def _execute_command(self, *_args: Any, **_kwargs: Any) -> Any:
        raise RuntimeError("token preflight must not execute VM commands")


PREFLIGHT_ENVIRONMENT = _PreflightEnvironment()


def _validate_state_history_messages(
    history_format: str, messages: list[dict[str, Any]], step: Any
) -> int:
    role = _STATE_HISTORY_ROLES[history_format]
    if role == "qwen35_state":
        expected = 1 if step.action_index_within_turn == 0 else 0
        if len(messages) != expected:
            raise ValueError("Qwen 3.5 state record count differs from turn boundary")
        if not messages:
            return 0
        required = {"observation_image_url", "observation_sha256", "action", "response"}
        state = messages[0]
    else:
        expected_roles = ["system", role] if step.step_id == 0 else [role]
        if [message.get("role") for message in messages] != expected_roles:
            raise ValueError("OpenCUA state record sequence is invalid")
        required = {"observation_image_url", "action"}
        state = messages[-1]
    if state.get("role") != role or any(not state.get(key) for key in required):
        raise ValueError(f"{role} record is incomplete")
    return 1


def _validate_claude_step_messages(messages: list[dict[str, Any]], action: Any) -> int:
    """Validate one recorded Anthropic step without rejecting native termination."""

    calls = [
        block
        for message in messages
        if message.get("role") == "assistant"
        for block in message.get("content", [])
        if isinstance(block, dict) and block.get("type") == "tool_use"
    ]
    result_ids = {
        str(block["tool_use_id"])
        for message in messages
        if message.get("role") == "user"
        for block in message.get("content", [])
        if isinstance(block, dict) and block.get("type") == "tool_result"
    }
    call_ids = {str(call["id"]) for call in calls}
    if calls and call_ids == result_ids:
        return len(calls)
    if not calls and not result_ids and getattr(action, "kind", "") == "terminate":
        return 0
    raise ValueError("Claude tool_use/tool_result IDs do not match")


def _validate_openai_step_messages(messages: list[dict[str, Any]], action: Any) -> int:
    """Every Responses computer/shell call is answered by its output item, in order."""

    calls = [m["call_id"] for m in messages if m.get("type") in {"computer_call", "shell_call"}]
    outputs = [
        m["call_id"]
        for m in messages
        if m.get("type") in {"computer_call_output", "shell_call_output"}
    ]
    if calls != outputs or len(set(calls)) != len(calls):
        raise ValueError("GPT call/output items do not pair")
    if not calls and getattr(action, "kind", "") != "terminate":
        raise ValueError("GPT renderer emitted no call for a non-terminal action")
    return len(calls)


def _validate_evocua_step_messages(
    adapter: Any,
    messages: list[dict[str, Any]],
    schemas: dict[str, Any],
) -> list[dict[str, Any]]:
    """Validate EvoCUA's inline S2 calls without inventing OpenAI call IDs."""

    calls = adapter.extract_tool_calls(messages)
    if not calls:
        raise ValueError("EvoCUA renderer emitted no inline target tool call")
    for call in calls:
        function = call["function"]
        name = str(function["name"])
        if name not in schemas:
            raise ValueError(f"renderer emitted unknown target tool: {name}")
        Draft202012Validator(schemas[name]).validate(
            json.loads(function["arguments"])
        )
    return calls


def _task_instruction(task_path: Path) -> str:
    task = json.loads(task_path.read_text(encoding="utf-8"))
    instruction = task.get("instruction", "")
    if not instruction:
        task_map = task.get("task", {})
        instruction = task_map.get("en", "") if isinstance(task_map, dict) else str(task_map)
    if not isinstance(instruction, str) or not instruction.strip():
        raise RuntimeError(f"task has no instruction: {task_path}")
    return instruction


@lru_cache(maxsize=256)
def _surrogate_screenshot(seed: int) -> str:
    """Return a distinct 1280x800 PNG with the same vision-token geometry."""

    color = ((37 * seed + 17) % 256, (67 * seed + 29) % 256, (97 * seed + 43) % 256)
    image = Image.new("RGB", (1280, 800), color)
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode("ascii")


def _tokenize_endpoints(base_url: str) -> tuple[str, ...]:
    """Return supported vLLM tokenizer routes without discarding the API prefix."""

    base = base_url.rstrip("/")
    endpoints = [base + "/tokenize"]
    if base.endswith("/v1"):
        endpoints.append(base[:-3] + "/tokenize")
    return tuple(endpoints)


def _http_error_detail(exc: urllib.error.HTTPError) -> str:
    """Return a bounded response body without leaking request headers."""

    try:
        body = exc.read().decode("utf-8", errors="replace").strip()
    except Exception:
        body = ""
    return f"; response={body[:500]}" if body else ""


def _runtime_model_name(target_agent: str, configured_model: str) -> str:
    """Match preflight model routing to the live rollout's runtime alias."""

    if target_agent == "opencua_72b":
        return os.environ.get("OPENCUA_MODEL", configured_model)
    if target_agent == "evocua_32b":
        return os.environ.get("EVOCUA_MODEL", configured_model)
    return configured_model


def _tokenize_request(
    base_url: str,
    *,
    model: str,
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]],
    timeout: float,
) -> tuple[int, int]:
    urls = _tokenize_endpoints(base_url)
    payload = json.dumps(
        {
            "model": model,
            "messages": messages,
            "tools": tools,
            "add_generation_prompt": True,
        },
        ensure_ascii=False,
    ).encode("utf-8")
    headers = {"Content-Type": "application/json", "Accept": "application/json"}
    api_key = os.environ.get("OPENAI_API_KEY")
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    result: Any = None
    for index, url in enumerate(urls):
        request = urllib.request.Request(url, data=payload, headers=headers, method="POST")
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                result = json.load(response)
            break
        except urllib.error.HTTPError as exc:
            if exc.code == 404 and index + 1 < len(urls):
                continue
            detail = _http_error_detail(exc)
            raise RuntimeError(
                f"cannot tokenize takeover history via {url}: {exc}{detail}"
            ) from exc
        except Exception as exc:
            raise RuntimeError(f"cannot tokenize takeover history via {url}: {exc}") from exc
    count = result.get("count") if isinstance(result, dict) else None
    max_model_len = result.get("max_model_len") if isinstance(result, dict) else None
    if (
        isinstance(count, bool)
        or not isinstance(count, int)
        or count <= 0
        or isinstance(max_model_len, bool)
        or not isinstance(max_model_len, int)
        or max_model_len <= 0
    ):
        raise RuntimeError("vLLM /tokenize returned invalid count/max_model_len")
    return count, max_model_len


def _chat_completions_url(base_url: str) -> str:
    return base_url.rstrip("/") + "/chat/completions"


def _chat_usage_request(
    base_url: str,
    *,
    model: str,
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]],
    timeout: float,
    retries: int = 3,
) -> int:
    """Return the prompt count reported by the target Chat Completions route."""

    payload: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "max_completion_tokens": 1,
    }
    if tools:
        payload["tools"] = tools
        payload["tool_choice"] = "auto"
    headers = {"Content-Type": "application/json", "Accept": "application/json"}
    api_key = os.environ.get("OPENAI_API_KEY")
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    request = urllib.request.Request(
        _chat_completions_url(base_url),
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers=headers,
        method="POST",
    )
    result: Any = None
    for attempt in range(1, retries + 1):
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                result = json.load(response)
            break
        except urllib.error.HTTPError as exc:
            if exc.code < 500 and exc.code != 429:
                raise RuntimeError(
                    f"cannot count takeover history via Chat Completions: {exc}"
                    f"{_http_error_detail(exc)}"
                ) from exc
            last_error: BaseException = exc
        except (TimeoutError, urllib.error.URLError) as exc:
            last_error = exc
        if attempt == retries:
            raise RuntimeError(
                f"Chat Completions token count failed after {retries} attempts: {last_error}"
            ) from last_error
        time.sleep(min(2 ** (attempt - 1), 8))
    usage = result.get("usage") if isinstance(result, dict) else None
    count = usage.get("prompt_tokens") if isinstance(usage, dict) else None
    if isinstance(count, bool) or not isinstance(count, int) or count <= 0:
        raise RuntimeError("Chat Completions response has no positive usage.prompt_tokens")
    return count


def _anthropic_usage_request(
    base_url: str, *, request_payload: dict[str, Any], timeout: float, retries: int = 3
) -> int:
    """Return the input count reported by the target Anthropic Messages route."""

    payload = dict(request_payload)
    betas = payload.pop("betas", [])
    payload["max_tokens"] = 1
    headers = {
        "Content-Type": "application/json",
        "Accept": "application/json",
        "anthropic-version": "2023-06-01",
        "x-api-key": os.environ.get("ANTHROPIC_API_KEY", ""),
    }
    if betas:
        headers["anthropic-beta"] = ",".join(str(value) for value in betas)
    request = urllib.request.Request(
        base_url.rstrip("/") + "/v1/messages",
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers=headers,
        method="POST",
    )
    last_error: BaseException | None = None
    for attempt in range(1, retries + 1):
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                result = json.load(response)
            break
        except urllib.error.HTTPError as exc:
            detail = _http_error_detail(exc)
            transient = exc.code in {408, 429} or exc.code >= 500 or (
                exc.code == 400
                and any(marker in detail for marker in ("AllModelsFailed", "invoke_error"))
            )
            if not transient:
                raise RuntimeError(
                    f"cannot count Anthropic takeover history: {exc}{detail}"
                ) from exc
            last_error = RuntimeError(f"{exc}{detail}")
        except (TimeoutError, urllib.error.URLError) as exc:
            last_error = exc
        if attempt == retries:
            raise RuntimeError(
                f"Anthropic token count failed after {retries} attempts: {last_error}"
            ) from last_error
        time.sleep(min(2 ** (attempt - 1), 8))
    usage = result.get("usage") if isinstance(result, dict) else None
    count = usage.get("input_tokens") if isinstance(usage, dict) else None
    if isinstance(count, bool) or not isinstance(count, int) or count <= 0:
        raise RuntimeError("Anthropic response has no positive usage.input_tokens")
    return count


def _available_depths(value: str) -> tuple[int, ...]:
    """Parse the selection TSV field; an empty field means no usable depth."""

    return tuple(int(item) for item in value.split(",") if item)


def _parse_selection_row(row: str, line_number: int) -> tuple[str, ...]:
    fields = tuple(row.split("\t"))
    if len(fields) != 8:
        raise RuntimeError(f"selection row {line_number} has {len(fields)} fields")
    return fields


def _active_shard_conditions(
    rows: list[tuple[str, ...]],
    depths: list[int],
    conditions: list[str],
    excluded_ids: frozenset[str],
    *,
    shard_count: int,
    shard_offset: int,
    shard_workers: int,
) -> dict[tuple[int, int], tuple[str, ...]]:
    active: dict[tuple[int, int], list[str]] = {}
    active_shards = range(shard_offset, shard_offset + shard_workers)
    job_index = 0
    for depth in depths:
        for condition in conditions:
            for line_number, row in enumerate(rows, start=1):
                if row[0] in excluded_ids or depth not in _available_depths(row[7]):
                    continue
                if job_index % shard_count in active_shards:
                    active.setdefault((line_number, depth), []).append(condition)
                job_index += 1
    return {key: tuple(values) for key, values in active.items()}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selection-list", type=Path, required=True)
    parser.add_argument("--target-agent", required=True)
    parser.add_argument("--depths", nargs="+", type=int, required=True)
    parser.add_argument(
        "--conditions",
        nargs="+",
        choices=("unaware", "notified", "diagnosed"),
        default=("unaware", "notified", "diagnosed"),
    )
    parser.add_argument(
        "--tokenize-base-url",
        default="",
        help="OpenAI-compatible /v1 base URL; enables live token-budget preflight",
    )
    parser.add_argument("--tokenize-timeout", type=float, default=60.0)
    parser.add_argument(
        "--tokenize-mode",
        choices=("vllm", "chat_usage", "anthropic_usage"),
        default="vllm",
    )
    parser.add_argument("--context-cap", type=int, default=0)
    parser.add_argument("--shard-count", type=int, default=1)
    parser.add_argument("--shard-offset", type=int, default=0)
    parser.add_argument("--shard-workers", type=int, default=1)
    parser.add_argument(
        "--overflow-policy",
        choices=("fail", "exclude"),
        default="fail",
    )
    parser.add_argument("--token-cache", type=Path)
    parser.add_argument(
        "--exclude-trajectory-id",
        action="append",
        default=[],
        help="Exact trajectory ID to omit from this recovery preflight; repeatable",
    )
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--takeover-config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--prefix-source", choices=("repaired", "original"))
    parser.add_argument("--repaired-prefix-dir", type=Path)
    parser.add_argument("--on-missing-repaired", choices=("error", "skip", "fallback_original"))
    args = parser.parse_args()
    config = load_takeover_config(args.takeover_config)
    prefix_config = dict(config["prefix"])
    if args.prefix_source:
        prefix_config["source"] = args.prefix_source
    if args.on_missing_repaired:
        prefix_config["on_missing_repaired"] = args.on_missing_repaired
    strip_reasoning = bool(config["history"]["strip_reasoning"])
    if args.tokenize_timeout <= 0:
        raise RuntimeError("tokenize-timeout must be positive")
    if args.shard_count <= 0 or args.shard_workers <= 0:
        raise RuntimeError("shard-count and shard-workers must be positive")
    if args.shard_offset < 0 or args.shard_offset + args.shard_workers > args.shard_count:
        raise RuntimeError("active shard range must fit inside shard-count")
    if args.tokenize_mode in {"chat_usage", "anthropic_usage"} and args.context_cap <= 0:
        raise RuntimeError("usage token preflight requires a positive context-cap")

    selection = args.selection_list.resolve()
    excluded_trajectory_ids = frozenset(args.exclude_trajectory_id)
    requested_depths = frozenset(args.depths)
    probe_kwargs = (
        {"system_prompt": "preflight"}
        if args.target_agent in {"claude_opus_4_8", "gpt_5_5"}
        else {}
    )
    probe = create_native_history_adapter(args.target_agent, "preflight", **probe_kwargs)
    history_tools = getattr(probe, "history_tool_definitions", probe.tool_definitions)()
    schemas = {
        item["function"]["name"]: item["function"]["parameters"]
        for item in history_tools
    }
    history_format = probe.capabilities.history_format
    state_history = history_format in _STATE_HISTORY_ROLES
    counts: Counter[str] = Counter()
    checked_prefixes = checked_steps = checked_calls = 0
    maximum: dict[str, Any] = {"tool_calls": 0, "trajectory_id": "", "depth": None}
    token_checks = 0
    token_cache_hits = 0
    token_cache: dict[str, int] = {}
    if args.token_cache and args.token_cache.is_file():
        cached = json.loads(args.token_cache.read_text(encoding="utf-8"))
        token_cache = {key: int(value) for key, value in cached.items()}
    token_overflows: list[dict[str, Any]] = []
    prefix_exclusions: list[dict[str, Any]] = []
    token_measurements: list[dict[str, Any]] = []
    maximum_context: dict[str, Any] = {
        "prompt_tokens": 0,
        "prompt_plus_completion_tokens": 0,
        "trajectory_id": "",
        "depth": None,
        "condition": "",
    }
    observed_max_model_len: int | None = None
    target_model = ""
    requested_max_tokens = 0
    live_agent: Any = None
    qwen35_live = False
    opencua_live = False
    evocua_live = False
    claude_live = False
    if args.tokenize_base_url:
        live_agent_id = "kimi_k3_cuabash" if args.target_agent == "kimi_k3" else args.target_agent
        config = (
            None
            if args.target_agent == "claude_opus_4_8"
            else load_config(live_agent_id)
        )
        os.environ.setdefault("DERAIL_REPO_ROOT", str(REPOSITORY))
        if args.target_agent in QWEN35_AGENT_IDS:
            from run_mypcbench import get_agent

            requested = os.environ.get("MYPCBENCH_QWEN_MAX_TOKENS", "")
            if not requested.isdigit() or int(requested) <= 0:
                raise RuntimeError("Qwen 3.5 token preflight requires MYPCBENCH_QWEN_MAX_TOKENS")
            if os.environ.get("MYPCBENCH_QWEN_CONTEXT_POLICY") != "tokenize_oldest_first_v1":
                raise RuntimeError("Qwen 3.5 token preflight requires tokenize_oldest_first_v1")
            target_model = resolve_checkpoint(config)
            requested_max_tokens = int(requested)
            os.environ["OPENAI_BASE_URL"] = args.tokenize_base_url
            upstream = get_agent(
                agent_type=str(config.document["agent_type"]),
                model=target_model,
                screen_size=(1280, 800),
                client_password="password",
            )
            live_agent = wrap_qwen35_takeover_target(upstream, args.target_agent)
            qwen35_live = True
        elif args.target_agent == "opencua_72b":
            from run_mypcbench import get_agent

            configured_model = str(config.document.get("checkpoint") or "")
            target_model = _runtime_model_name(args.target_agent, configured_model)
            if not target_model:
                raise RuntimeError("OpenCUA target config has no checkpoint")
            requested_max_tokens = int(
                os.environ.get("OPENCUA_MAX_TOKENS_OVERRIDE") or config["max_tokens"]
            )
            if not 1 <= requested_max_tokens <= int(config["max_tokens"]):
                raise RuntimeError(
                    "OPENCUA_MAX_TOKENS_OVERRIDE must be within the frozen budget"
                )
            os.environ["OPENAI_BASE_URL"] = args.tokenize_base_url
            live_agent = get_agent(
                agent_type=str(config.document["agent_type"]),
                model=target_model,
                screen_size=(1280, 800),
                client_password="password",
            )
            if not callable(getattr(live_agent, "preflight_next_request", None)):
                raise RuntimeError("OpenCUA target has no native request preflight API")
            opencua_live = True
        elif args.target_agent == "evocua_32b":
            from run_mypcbench import get_agent

            configured_model = str(config.document.get("checkpoint") or "")
            target_model = _runtime_model_name(args.target_agent, configured_model)
            if not target_model:
                raise RuntimeError("EvoCUA target config has no checkpoint")
            requested_max_tokens = int(config["max_tokens"])
            os.environ["OPENAI_BASE_URL"] = args.tokenize_base_url
            live_agent = get_agent(
                agent_type=str(config.document["agent_type"]),
                model=target_model,
                screen_size=(1280, 800),
                client_password="password",
            )
            if not callable(getattr(live_agent, "preflight_next_request", None)):
                raise RuntimeError("EvoCUA target has no native request preflight API")
            evocua_live = True
        elif args.target_agent == "claude_opus_4_8":
            from run_mypcbench import get_agent

            target_model = os.environ.get("CLAUDE_OPUS_4_8_MODEL", "claude-opus-4-8")
            upstream = get_agent(
                agent_type="claude_cuabash",
                model=target_model,
                screen_size=(1280, 800),
                client_password="password",
                env=PREFLIGHT_ENVIRONMENT,
            )
            live_agent = wrap_claude_takeover_target(upstream, args.target_agent)
            requested_max_tokens = int(upstream.max_tokens)
            claude_live = True
        elif config is None or config.scaffold != DERAIL_TOOL_AGENT:
            raise RuntimeError(
                "live token preflight currently requires a derail_tool_agent target"
            )
        else:
            target_model = str(
                config.document.get("checkpoint") or config.document.get("model") or ""
            )
            if not target_model:
                raise RuntimeError("target agent config has no checkpoint/model")
            requested_max_tokens = int(config["max_tokens"])
            live_agent = NativeToolComputerAgent(
                target_model,
                (1280, 800),
                protocol_from_config(config),
                base_url=args.tokenize_base_url,
                env=PREFLIGHT_ENVIRONMENT,
            )

    selection_rows = selection.read_text(encoding="utf-8").splitlines()
    parsed_rows = [
        _parse_selection_row(row, line_number)
        for line_number, row in enumerate(selection_rows, start=1)
    ]
    active_conditions = _active_shard_conditions(
        parsed_rows,
        args.depths,
        args.conditions,
        excluded_trajectory_ids,
        shard_count=args.shard_count,
        shard_offset=args.shard_offset,
        shard_workers=args.shard_workers,
    )
    for line_number, fields in enumerate(parsed_rows, start=1):
        trajectory_id, canonical, normalization_report, task_json, annotation, root, _last, depths = fields
        if trajectory_id in excluded_trajectory_ids:
            continue
        instruction = _task_instruction(Path(task_json))
        try:
            steps, prefix_info = load_takeover_steps(
                Path(canonical), trajectory_id, int(root), prefix_config, args.repaired_prefix_dir
            )
        except ProtocolExclusion as exc:
            prefix_exclusions.extend(
                {"trajectory_id": trajectory_id, "depth": depth, "condition": condition,
                 "reason": exc.reason, "detail": exc.detail}
                for (row_number, depth), conditions in sorted(active_conditions.items())
                if row_number == line_number
                for condition in conditions
            )
            continue
        source_agent = steps[0].source_agent
        adapter_kwargs: dict[str, Any] = {}
        if args.target_agent == "evocua_32b":
            if source_agent != "evocua_32b" and not config["history"]["evocua_cross_agent"]:
                raise RuntimeError("evocua_32b cross-agent history is disabled by the config")
            adapter_kwargs["source_agent"] = source_agent
        elif args.target_agent == "gpt_5_5":
            adapter_kwargs["system_prompt"] = "preflight"
        if opencua_live:
            adapter_kwargs["system_prompt"] = live_agent.native_history_system_prompt
        elif args.target_agent == "claude_opus_4_8":
            adapter_kwargs["system_prompt"] = (
                live_agent.native_history_system_prompt if claude_live else "preflight"
            )
        diagnosis = None
        if live_agent is not None and "diagnosed" in args.conditions:
            report_payload = json.loads(Path(normalization_report).read_text(encoding="utf-8"))
            diagnosis = load_human_diagnosis_evidence(
                Path(annotation),
                expected_trajectory_id=trajectory_id,
                expected_source_trajectory_sha256=str(
                    report_payload.get("source_trajectory_sha256", "")
                ),
                maximum_action_index=steps[-1].action_index_global,
            )
        for depth in _available_depths(depths):
            if depth not in requested_depths:
                continue
            prefix_conditions = active_conditions.get((line_number, depth), ())
            if not prefix_conditions:
                continue
            prefix_end = int(root) + depth
            prefix = takeover_prefix(steps, int(root), depth)
            adapter = create_native_history_adapter(
                args.target_agent, instruction, **adapter_kwargs
            )
            boundary_step_id = max(
                (
                    step.step_id
                    for step in prefix
                    if step.action.kind not in adapter.capabilities.silent_action_kinds
                ),
                default=-1,
            )
            prefix_calls = 0
            all_call_ids: set[str] = set()
            native_history: list[dict[str, Any]] = []
            for step in prefix:
                try:
                    messages = adapter.render_step(
                        HistoryStep(
                            step_id=step.step_id,
                            turn_index=step.turn_index or 0,
                            action_index_within_turn=step.action_index_within_turn,
                            observation_image_url=_surrogate_screenshot(step.step_id),
                            observation_after_image_url=(
                                _surrogate_screenshot(step.step_id + 1)
                                if history_format
                                in {"anthropic_recorded_messages", "openai_responses_input_items"}
                                else ""
                            ),
                            observation_sha256="0" * 64,
                            action=step.action,
                            tool_result=step.tool_result,
                            trajectory_log=load_trajectory_log(
                                step, strip_reasoning=strip_reasoning
                            ),
                        )
                    )
                    native_history.extend(messages)
                    if state_history:
                        rendered_records = _validate_state_history_messages(
                            history_format, messages, step
                        )
                        checked_steps += 1
                        prefix_calls += rendered_records
                        continue
                    if history_format == "anthropic_recorded_messages":
                        if not messages:
                            if step.step_id == boundary_step_id:
                                raise ValueError(
                                    "Claude prefix boundary renders no native message"
                                )
                            checked_steps += 1
                            continue
                        call_count = _validate_claude_step_messages(messages, step.action)
                        prefix_calls += call_count
                        checked_calls += call_count
                        checked_steps += 1
                        continue
                    if history_format == "openai_responses_input_items":
                        call_count = _validate_openai_step_messages(messages, step.action)
                        prefix_calls += call_count
                        checked_calls += call_count
                        checked_steps += 1
                        continue
                    if history_format == "evocua_s2_raw_response_turns":
                        calls = _validate_evocua_step_messages(
                            adapter, messages, schemas
                        )
                        for call in calls:
                            counts[str(call["function"]["name"])] += 1
                        checked_steps += 1
                        checked_calls += len(calls)
                        prefix_calls += len(calls)
                        continue
                    calls = [
                        call
                        for message in messages
                        if message.get("role") == "assistant"
                        for call in message.get("tool_calls", [])
                    ]
                    if not calls:
                        raise ValueError("renderer emitted no target tool call")
                    step_ids = {str(call["id"]) for call in calls}
                    if len(step_ids) != len(calls) or all_call_ids.intersection(step_ids):
                        raise ValueError("renderer emitted duplicate tool call IDs")
                    all_call_ids.update(step_ids)
                    tool_result_ids = {
                        str(message["tool_call_id"])
                        for message in messages
                        if message.get("role") == "tool"
                    }
                    if tool_result_ids != step_ids:
                        raise ValueError("tool results do not match assistant tool calls")
                    for call in calls:
                        function = call["function"]
                        name = str(function["name"])
                        if name not in schemas:
                            raise ValueError(f"renderer emitted unknown target tool: {name}")
                        Draft202012Validator(schemas[name]).validate(
                            json.loads(function["arguments"])
                        )
                        counts[name] += 1
                except Exception as exc:
                    raise RuntimeError(
                        f"native history preflight failed: trajectory={trajectory_id} "
                        f"depth={depth} step={step.step_id} action={step.action!r}"
                    ) from exc
                checked_steps += 1
                checked_calls += len(calls)
                prefix_calls += len(calls)
            checked_prefixes += 1
            if prefix_calls > maximum["tool_calls"]:
                maximum = {
                    "tool_calls": prefix_calls,
                    "trajectory_id": trajectory_id,
                    "depth": depth,
                }
            if live_agent is not None:
                for condition in prefix_conditions:
                    live_agent.reset()
                    seed_kwargs: dict[str, Any] = {"condition": condition}
                    if condition == "diagnosed":
                        if diagnosis is None:
                            raise RuntimeError("diagnosed token preflight has no diagnosis")
                        seed_kwargs.update(
                            diagnosis=diagnosis.evidence,
                            root_cause_action_index=diagnosis.root_cause_action_index,
                        )
                    live_agent.seed_native_history(
                        instruction, native_history, **seed_kwargs
                    )
                    current_image = _surrogate_screenshot(prefix_end + 100)
                    if qwen35_live:
                        metadata = live_agent.preflight_next_request(
                            instruction,
                            {"screenshot": base64.b64decode(current_image.split(",", 1)[1])},
                        )
                        prompt_tokens = int(metadata["final_prompt_tokens"])
                        max_model_len = int(metadata["max_model_len"])
                        token_checks += 1
                        if observed_max_model_len is None:
                            observed_max_model_len = max_model_len
                        elif observed_max_model_len != max_model_len:
                            raise RuntimeError(
                                "Qwen 3.5 /tokenize max_model_len changed during preflight"
                            )
                        total_tokens = prompt_tokens + requested_max_tokens
                        context_row = {
                            "prompt_tokens": prompt_tokens,
                            "prompt_plus_completion_tokens": total_tokens,
                            "trajectory_id": trajectory_id,
                            "depth": depth,
                            "condition": condition,
                        }
                        if total_tokens > maximum_context["prompt_plus_completion_tokens"]:
                            maximum_context = context_row
                        if total_tokens > max_model_len:
                            token_overflows.append(context_row)
                        continue
                    current_observation = {
                        "screenshot": base64.b64decode(current_image.split(",", 1)[1])
                    }
                    if opencua_live or evocua_live or claude_live:
                        request_payload = live_agent.preflight_next_request(
                            instruction, current_observation
                        )
                        request_messages = request_payload["messages"]
                        request_tools = request_payload.get("tools", [])
                    else:
                        request_messages, _current = live_agent._messages(
                            instruction, current_observation["screenshot"]
                        )
                        request_tools = live_agent.tools
                    if args.tokenize_mode in {"chat_usage", "anthropic_usage"}:
                        cache_key = sha256_json(
                            [target_model, request_messages, request_tools]
                        )
                        prompt_tokens = token_cache.get(cache_key, 0)
                        if prompt_tokens > 0:
                            token_cache_hits += 1
                        else:
                            if args.tokenize_mode == "anthropic_usage":
                                prompt_tokens = _anthropic_usage_request(
                                    args.tokenize_base_url,
                                    request_payload=request_payload,
                                    timeout=args.tokenize_timeout,
                                )
                            else:
                                prompt_tokens = _chat_usage_request(
                                    args.tokenize_base_url,
                                    model=target_model,
                                    messages=request_messages,
                                    tools=request_tools,
                                    timeout=args.tokenize_timeout,
                                )
                            token_cache[cache_key] = prompt_tokens
                            if args.token_cache:
                                atomic_write_json(args.token_cache.resolve(), token_cache)
                        max_model_len = args.context_cap
                    else:
                        prompt_tokens, max_model_len = _tokenize_request(
                            args.tokenize_base_url,
                            model=target_model,
                            messages=request_messages,
                            tools=request_tools,
                            timeout=args.tokenize_timeout,
                        )
                    token_checks += 1
                    if observed_max_model_len is None:
                        observed_max_model_len = max_model_len
                    elif observed_max_model_len != max_model_len:
                        raise RuntimeError(
                            "vLLM /tokenize max_model_len changed during preflight"
                        )
                    total_tokens = prompt_tokens + requested_max_tokens
                    context_row = {
                        "prompt_tokens": prompt_tokens,
                        "prompt_plus_completion_tokens": total_tokens,
                        "trajectory_id": trajectory_id,
                        "depth": depth,
                        "condition": condition,
                    }
                    token_measurements.append(context_row)
                    if total_tokens > maximum_context["prompt_plus_completion_tokens"]:
                        maximum_context = context_row
                    if total_tokens > max_model_len:
                        token_overflows.append(context_row)

    if checked_prefixes == 0 and not prefix_exclusions:
        raise RuntimeError("no selected prefix matched the requested depths")
    report = {
        "status": (
            "partial" if token_overflows and args.overflow_policy == "exclude"
            else "failed" if token_overflows
            else "passed"
        ),
        "target_agent": args.target_agent,
        "requested_depths": sorted(requested_depths),
        "shard": {
            "count": args.shard_count,
            "offset": args.shard_offset,
            "workers": args.shard_workers,
        },
        "excluded_trajectory_ids": sorted(excluded_trajectory_ids),
        "takeover_config_uri": config["config_uri"],
        "takeover_config_sha256": config["config_sha256"],
        "prefix": {**prefix_config, "repaired_prefix_dir": str(args.repaired_prefix_dir or "")},
        "strip_reasoning": strip_reasoning,
        "prefix_exclusions": prefix_exclusions,
        "selection_list_uri": str(selection),
        "selection_list_sha256": sha256_file(selection),
        "checked_prefixes": checked_prefixes,
        "checked_steps": checked_steps,
        "checked_tool_calls": checked_calls,
        "tool_call_kinds": dict(sorted(counts.items())),
        "maximum_tool_calls_in_prefix": maximum,
        "token_preflight": {
            "status": (
                "partial"
                if token_overflows and args.overflow_policy == "exclude"
                else "failed"
                if token_overflows
                else "passed"
                if live_agent is not None
                else "not_requested"
            ),
            "endpoint": args.tokenize_base_url or None,
            "mode": args.tokenize_mode if args.tokenize_base_url else None,
            "experiment_context_cap": args.context_cap or None,
            "model": target_model or None,
            "checked_requests": token_checks,
            "cache_hits": token_cache_hits,
            "cache_entries": len(token_cache),
            "requested_max_tokens": requested_max_tokens or None,
            "max_model_len": observed_max_model_len,
            "maximum": maximum_context if token_checks else None,
            "measurements": token_measurements,
            "overflows": token_overflows,
            "overflow_policy": args.overflow_policy,
        },
    }
    atomic_write_json(args.report.resolve(), report)
    if token_overflows and args.overflow_policy == "fail":
        first = token_overflows[0]
        raise RuntimeError(
            "takeover token preflight failed before VM start: "
            f"trajectory={first['trajectory_id']} depth={first['depth']} "
            f"condition={first['condition']} prompt={first['prompt_tokens']} + "
            f"completion={requested_max_tokens} > max_model_len={observed_max_model_len}"
        )
    verdict = "partial" if token_overflows else "passed"
    print(
        f"native-history-preflight={verdict} prefixes={checked_prefixes} "
        f"steps={checked_steps} tool_calls={checked_calls} token_requests={token_checks} "
        f"overflows={len(token_overflows)} "
        f"report={args.report.resolve()}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
