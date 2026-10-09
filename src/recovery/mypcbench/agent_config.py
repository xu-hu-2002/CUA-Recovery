"""Load and strictly validate ``configs/agents/*.yaml``."""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, FrozenSet, Mapping, Optional, Tuple

REPO_ROOT = Path(os.environ.get("RECOVERY_REPO_ROOT", Path(__file__).resolve().parents[3]))

ONE_INTERACTION_PLUS_COLLAPSED_WAITS = "one_interaction_plus_collapsed_waits"
EXECUTE_ALL_CALLS_IN_ORDER = "execute_all_calls_in_order"
MULTI_TOOL_POLICIES = frozenset(
    {ONE_INTERACTION_PLUS_COLLAPSED_WAITS, EXECUTE_ALL_CALLS_IN_ORDER}
)

RUNNER_AUTHORITATIVE_UPSTREAM_GUARD_PLUS_ONE = "runner_authoritative_upstream_guard_plus_one"
STEP_BUDGET_POLICIES = frozenset({RUNNER_AUTHORITATIVE_UPSTREAM_GUARD_PLUS_ONE})

RECOVERY_TOOL_AGENT = "recovery_tool_agent"
UPSTREAM_OFFICIAL = "upstream_official"
UPSTREAM_RUNNER = "upstream_runner"
SCAFFOLDS = frozenset({RECOVERY_TOOL_AGENT, UPSTREAM_OFFICIAL, UPSTREAM_RUNNER})

LOCAL_VLLM = "local_vllm"
HOSTED_API = "hosted_api"

AGENT_ID_BY_TYPE: Mapping[str, str] = {
    "recovery_kimi_k3_cuabash": "kimi_k3_cuabash",
    "recovery_evocua": "evocua_32b",
    "recovery_opencua": "opencua_72b",
}

_ENV_REFERENCE = re.compile(r"\$\{(\w+)\}")


class AgentConfigError(RuntimeError):
    """Agent yaml has a missing, mistyped, or out-of-range field."""


@dataclass(frozen=True)
class _Field:
    kind: type
    choices: Optional[FrozenSet[str]] = None
    minimum: Optional[float] = None
    nullable: bool = False


_COMMON_REQUIRED: Mapping[str, _Field] = {
    "agent_id": _Field(str),
    "family": _Field(str),
    "scaffold": _Field(str, choices=SCAFFOLDS),
    "agent_type": _Field(str),
    "status": _Field(str),
    "eta_step_seconds": _Field(float, minimum=0.1),
    "eta_step_seconds_source": _Field(str),
}

_LOCAL_REQUIRED: Mapping[str, _Field] = {
    "checkpoint": _Field(str),
    "revision": _Field(str),
    "tensor_parallel_size": _Field(int, minimum=1),
}

_HOSTED_REQUIRED: Mapping[str, _Field] = {
    "model": _Field(str),
    "num_vms": _Field(int, minimum=1),
}

_COMMON_DOC: FrozenSet[str] = frozenset(
    {
        "api_gateway",
        "api_snapshot",
        "anthropic_beta",
        "checkpoint",
        "computer_tool_type",
        "context_policy",
        "coordinate_protocol",
        "endpoint_contract",
        "generation_budget_source",
        "history_renderer",
        "live_adapter",
        "model",
        "notes",
        "observed_rpm_limit",
        "output_format",
        "probe_history_window",
        "prompt_caching",
        "revision",
        "runner_requires_env",
        "scaffold_origin",
        "scroll_unit_understanding",
        "temperature",
        "tokenizer_image_policy",
        "tool_turn_ratio_observed",
        "upstream_agent_commit",
    }
)


@dataclass(frozen=True)
class _AgentSpec:
    scaffold: str
    serving: str
    live: Mapping[str, _Field] = field(default_factory=dict)
    doc: FrozenSet[str] = frozenset()


_TOOL_AGENT_LIVE: Mapping[str, _Field] = {
    "coordinate_protocol": _Field(str, choices=frozenset({"absolute_pixels", "normalized_0_1000"})),
    "system_prompt_file": _Field(str),
    "temperature": _Field(float, minimum=0.0, nullable=True),
    "max_tokens": _Field(int, minimum=1),
    "max_images_in_context": _Field(int, minimum=1),
    "history_turns": _Field(int, minimum=1),
    "image_fold_size": _Field(int, minimum=1),
    "previous_action_log": _Field(bool),
    "tool_choice": _Field(str, choices=frozenset({"auto", "required", "none"})),
    "schema_repair_attempts": _Field(int, minimum=0),
    "multi_tool_policy": _Field(str, choices=MULTI_TOOL_POLICIES),
    "alternating_action_repeat_limit": _Field(int, minimum=0),
    "stalled_state_step_limit": _Field(int, minimum=0),
    "enable_bash": _Field(bool),
}

_UPSTREAM_COMMON_LIVE: Mapping[str, _Field] = {
    "max_tokens": _Field(int, minimum=1),
    "top_p": _Field(float, minimum=0.0),
    "temperature": _Field(float, minimum=0.0),
    "step_budget_policy": _Field(str, choices=STEP_BUDGET_POLICIES),
    "upstream_max_steps_fallback": _Field(int, minimum=1),
}

_SPECS: Mapping[str, _AgentSpec] = {
    "kimi_k3_cuabash": _AgentSpec(RECOVERY_TOOL_AGENT, HOSTED_API, live=_TOOL_AGENT_LIVE),
    "evocua_32b": _AgentSpec(
        UPSTREAM_OFFICIAL,
        LOCAL_VLLM,
        live={
            **_UPSTREAM_COMMON_LIVE,
            "prompt_style": _Field(str, choices=frozenset({"S1", "S2"})),
            "coordinate_type": _Field(str, choices=frozenset({"relative", "absolute"})),
            "resize_factor": _Field(int, minimum=1),
            "max_history_turns": _Field(int, minimum=1),
        },
    ),
    "opencua_72b": _AgentSpec(
        UPSTREAM_OFFICIAL,
        LOCAL_VLLM,
        live={
            **_UPSTREAM_COMMON_LIVE,
            "history_type": _Field(str),
            "coordinate_type": _Field(str),
            "cot_level": _Field(str),
            "max_image_history_length": _Field(int, minimum=1),
            "use_old_sys_prompt": _Field(bool),
        },
    ),
    "qwen3_5_35b_a3b": _AgentSpec(UPSTREAM_RUNNER, LOCAL_VLLM),
    "rerail_35b_a3b": _AgentSpec(UPSTREAM_RUNNER, LOCAL_VLLM),
    "gpt_5_5": _AgentSpec(UPSTREAM_RUNNER, HOSTED_API),
    "claude_opus_4_8": _AgentSpec(UPSTREAM_RUNNER, HOSTED_API),
}


@dataclass(frozen=True)
class AgentConfig:
    """A validated agent config."""

    agent_id: str
    scaffold: str
    serving: str
    path: Path
    live: Mapping[str, Any]
    document: Mapping[str, Any]

    def __getitem__(self, name: str) -> Any:
        try:
            return self.live[name]
        except KeyError:
            raise KeyError(
                f"{name!r} is not a live field of {self.agent_id}; "
                f"available fields: {sorted(self.live)}"
            ) from None


def config_dir() -> Path:
    return REPO_ROOT / "configs" / "agents"


def all_agent_ids() -> Tuple[str, ...]:
    """All agent config ids, sorted by filename."""

    return tuple(sorted(path.stem for path in config_dir().glob("*.yaml")))


def agent_id_for_type(agent_type: str) -> str:
    """Map the runner's ``--agent-type`` back to a yaml agent_id."""

    try:
        agent_id = AGENT_ID_BY_TYPE[agent_type]
    except KeyError:
        raise ValueError(f"unknown RECOVERY MyPCBench agent_type: {agent_type!r}") from None
    declared = os.environ.get("RECOVERY_AGENT_ID")
    if declared and declared != agent_id:
        raise AgentConfigError(
            f"RECOVERY_AGENT_ID={declared!r} does not match {agent_id!r} for "
            f"agent_type={agent_type!r}; resolve_agent() in scripts/collection/collect_trajectories.sh "
            "and agent_config.AGENT_ID_BY_TYPE have drifted and must be fixed together"
        )
    return agent_id


def _coerce(agent_id: str, name: str, value: Any, spec: _Field) -> Any:
    where = f"{name} in {agent_id}.yaml"
    if value is None:
        if spec.nullable:
            return None
        raise AgentConfigError(f"{where} must not be null")
    # bool is a subclass of int.
    if spec.kind is bool:
        if not isinstance(value, bool):
            raise AgentConfigError(f"{where} must be true/false, got {value!r}")
    elif spec.kind is int:
        if isinstance(value, bool) or not isinstance(value, int):
            raise AgentConfigError(f"{where} must be an integer, got {value!r}")
    elif spec.kind is float:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise AgentConfigError(f"{where} must be a number, got {value!r}")
        value = float(value)
    elif spec.kind is str:
        if not isinstance(value, str) or not value.strip():
            raise AgentConfigError(f"{where} must be a non-empty string, got {value!r}")
    else:  # pragma: no cover
        raise AgentConfigError(f"{where} has an uncheckable spec type: {spec.kind!r}")

    if spec.choices is not None and value not in spec.choices:
        raise AgentConfigError(
            f"{where} must be one of {sorted(spec.choices)}, got {value!r}"
        )
    if spec.minimum is not None and value < spec.minimum:
        raise AgentConfigError(f"{where} must be at least {spec.minimum}, got {value!r}")
    return value


def _required_fields(spec: _AgentSpec) -> Dict[str, _Field]:
    serving = _LOCAL_REQUIRED if spec.serving == LOCAL_VLLM else _HOSTED_REQUIRED
    return {**_COMMON_REQUIRED, **serving}


def load_config(agent_id: str) -> AgentConfig:
    """Load and validate any agent config, including upstream_runner ones."""

    spec = _SPECS.get(agent_id)
    if spec is None:
        raise AgentConfigError(
            f"no spec defined for {agent_id!r}; a new agent config must also register "
            "its scaffold and serving in agent_config._SPECS"
        )

    try:
        import yaml
    except ImportError as exc:  # pragma: no cover
        raise AgentConfigError("PyYAML missing; reinstall dependencies with `pip install -e .`") from exc

    path = config_dir() / f"{agent_id}.yaml"
    if not path.is_file():
        raise AgentConfigError(f"agent config not found: {path}")
    document = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(document, dict):
        raise AgentConfigError(f"{path} is not a YAML mapping")

    required = _required_fields(spec)
    if _ENV_REFERENCE.fullmatch(str(document.get("checkpoint", ""))):
        required.pop("revision")
    missing = [name for name in required if name not in document]
    if missing:
        raise AgentConfigError(f"{agent_id}.yaml is missing required fields: {missing}")
    if document["agent_id"] != agent_id:
        raise AgentConfigError(
            f"agent_id={document['agent_id']!r} in {path} does not match the file name"
        )
    if document["scaffold"] != spec.scaffold:
        raise AgentConfigError(
            f"scaffold={document['scaffold']!r} in {agent_id}.yaml does not match the spec "
            f"{spec.scaffold!r}; changing the scaffold requires updating the agent_config spec"
        )

    for name, rule in required.items():
        _coerce(agent_id, name, document[name], rule)

    allowed = set(required) | set(spec.live) | set(spec.doc) | _COMMON_DOC
    forbidden = sorted(
        name
        for name in ({"num_vms", "tensor_parallel_size"} - set(required))
        if name in document
    )
    if forbidden:
        hint = (
            "local serving concurrency is derived from GPU count / tensor_parallel_size"
            if spec.serving == LOCAL_VLLM
            else "hosted APIs use no local GPUs, so tensor_parallel_size does not apply"
        )
        raise AgentConfigError(f"{agent_id}.yaml must not contain {forbidden}; {hint}")
    unknown = sorted(set(document) - allowed - {"num_vms", "tensor_parallel_size", "served_model_name"})
    if unknown:
        raise AgentConfigError(
            f"{agent_id}.yaml has unknown fields: {unknown}; "
            "misspelled live field names are rejected as unknown keys, see agent_config._SPECS"
        )
    absent = sorted(set(spec.live) - set(document))
    if absent:
        raise AgentConfigError(f"{agent_id}.yaml is missing live fields: {absent}")

    live = {
        name: _coerce(agent_id, name, document[name], rule)
        for name, rule in spec.live.items()
    }
    if "history_turns" in live and live["history_turns"] < live["max_images_in_context"]:
        raise AgentConfigError(
            f"history_turns({live['history_turns']}) in {agent_id}.yaml must not be less than "
            f"max_images_in_context({live['max_images_in_context']})"
        )
    return AgentConfig(
        agent_id=agent_id,
        scaffold=document["scaffold"],
        serving=spec.serving,
        path=path,
        live=live,
        document=document,
    )


def resolve_checkpoint(config: AgentConfig) -> str:
    """The checkpoint, with a ``${VAR}`` value read from the environment."""

    value = str(config.document.get("checkpoint") or "")
    match = _ENV_REFERENCE.fullmatch(value)
    if match is None:
        return value
    resolved = os.environ.get(match.group(1), "").strip()
    if not resolved:
        raise AgentConfigError(
            f"checkpoint in {config.agent_id}.yaml comes from environment variable {match.group(1)}, which is not set"
        )
    return resolved


def load_agent_config(agent_id: str) -> AgentConfig:
    """Load the execution config of a RECOVERY factory agent."""

    config = load_config(agent_id)
    if config.scaffold == UPSTREAM_RUNNER:
        raise AgentConfigError(
            f"{agent_id} uses a built-in MyPCBench agent"
            f" (--agent-type {config.document['agent_type']}) that RECOVERY cannot construct; "
            "its yaml is documentation only, do not load it here"
        )
    return config
