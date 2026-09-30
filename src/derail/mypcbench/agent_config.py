"""``configs/agents/*.yaml`` —— agent 配置的唯一权威来源。

在此之前这些 yaml 是纯文档：运行时没有任何代码读它们，值硬编码在 factory.py 和
tool_agent.py 里。代价在 df8ab8a 被兑现过 —— holo 的 yaml 写着 temperature 0.8 /
tool_choice required，实际跑的是 0.0 / auto，漂移了整整一个版本没人发现。

现在方向反过来：yaml 是权威，代码不持有默认值。每份 config 都逐字段严格校验 ——
未知键、缺键、类型错、枚举外的取值一律 hard fail，不做静默回退。论文实际使用的
那组数值由 ``tests/test_agent_config.py`` 的金标快照钉住。

三类 scaffold（每份 yaml 都必须显式声明 ``scaffold``，见 :data:`SCAFFOLDS`）：

``derail_tool_agent``
    DERAIL 自己的冻结 schema scaffold，经 SafePyAutoGUICompiler。行为参数全部
    是 live，由 :func:`~derail.mypcbench.tool_agent.protocol_from_config` 装配。

``upstream_official``
    上游官方 agent 实现，但由 DERAIL factory 构造，所以采样/历史等参数仍是 live。
    不经过 SafePyAutoGUICompiler。

``upstream_runner``
    MyPCBench 内置 agent，DERAIL 构造不到，yaml 是纯文档。对这类调用
    :func:`load_agent_config` 会明确报错，而不是返回一份"看起来生效了"的配置；
    只有 :func:`load_config` 会读它们（校验身份字段与并发字段）。
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, FrozenSet, Mapping, Optional, Tuple

REPO_ROOT = Path(os.environ.get("DERAIL_REPO_ROOT", Path(__file__).resolve().parents[3]))

# multi_tool_policy 的取值域。定义在这里而不是 tool_agent 里，是为了让校验层
# 不依赖 decoder；tool_agent 从本模块 import 并重新导出，方向只有一条。
ONE_INTERACTION_PLUS_COLLAPSED_WAITS = "one_interaction_plus_collapsed_waits"
EXECUTE_ALL_CALLS_IN_ORDER = "execute_all_calls_in_order"
MULTI_TOOL_POLICIES = frozenset(
    {ONE_INTERACTION_PLUS_COLLAPSED_WAITS, EXECUTE_ALL_CALLS_IN_ORDER}
)

# 目前唯一实现的步数策略；语义见 factory._upstream_step_budget()。
RUNNER_AUTHORITATIVE_UPSTREAM_GUARD_PLUS_ONE = "runner_authoritative_upstream_guard_plus_one"
STEP_BUDGET_POLICIES = frozenset({RUNNER_AUTHORITATIVE_UPSTREAM_GUARD_PLUS_ONE})

DERAIL_TOOL_AGENT = "derail_tool_agent"
UPSTREAM_OFFICIAL = "upstream_official"
UPSTREAM_RUNNER = "upstream_runner"
SCAFFOLDS = frozenset({DERAIL_TOOL_AGENT, UPSTREAM_OFFICIAL, UPSTREAM_RUNNER})

# serving 决定身份字段和并发字段各要哪一组，见 _AgentSpec。
LOCAL_VLLM = "local_vllm"
HOSTED_API = "hosted_api"

# agent_type 由 01_collect_trajectories.sh 的 resolve_agent() 传给官方 runner，
# agent_id 才是 yaml 的文件名。DERAIL_AGENT_ID 只用来交叉核对这张表和 bash 那边
# 有没有漂移，不参与选文件（见 agent_id_for_type）。
AGENT_ID_BY_TYPE: Mapping[str, str] = {
    "derail_qwen36": "qwen3_6_27b",
    "derail_qwen38": "qwen3_8_27b",
    "derail_holo31": "holo_3_1_35b_a3b",
    "derail_kimi_k3": "kimi_k3",
    "derail_kimi_k3_cuabash": "kimi_k3_cuabash",
    "derail_evocua": "evocua_32b",
    "derail_opencua": "opencua_72b",
}

class AgentConfigError(RuntimeError):
    """agent yaml 缺字段、类型不对、或取值不在允许范围内。"""


@dataclass(frozen=True)
class _Field:
    """一个字段的校验规则。

    ``minimum`` 是闭区间下界；``choices`` 只对 str 字段有意义；``nullable`` 允许
    显式写 null（语义由使用方定义，例如 temperature 的 null = 请求体省略该字段）。
    """

    kind: type
    choices: Optional[FrozenSet[str]] = None
    minimum: Optional[float] = None
    nullable: bool = False


# 每份 config 都必须有的字段，与 scaffold 无关。
#
# eta_step_seconds 是 01_collect_all.sh 预估时长用的单步耗时；以前是那个脚本里的
# 一张 case 表，只覆盖 5 个开源 agent，其余 5 个会被当成"未知 agent_id"打死。
# 它是 agent 的属性不是脚本的属性，所以住在这里；_source 强制写清楚数字的来历，
# 免得借用值和实测值在预估表里长得一模一样。
_COMMON_REQUIRED: Mapping[str, _Field] = {
    "agent_id": _Field(str),
    "family": _Field(str),
    "scaffold": _Field(str, choices=SCAFFOLDS),
    "agent_type": _Field(str),
    "status": _Field(str),
    "eta_step_seconds": _Field(float, minimum=0.1),
    "eta_step_seconds_source": _Field(str),
}

# 本地 vLLM serving 的 agent：身份是权重快照，并发度由显卡数除以 TP 尺寸算出来，
# 不写死在 yaml 里（见 scripts/lib/collection_config.sh 的 agent_vm_count）。
_LOCAL_REQUIRED: Mapping[str, _Field] = {
    "checkpoint": _Field(str),
    "revision": _Field(str),
    "tensor_parallel_size": _Field(int, minimum=1),
}

# 托管 API 的 agent：没有权重快照，也不占卡，所以并发度是配额/花费的选择，只能
# 显式声明。
_HOSTED_REQUIRED: Mapping[str, _Field] = {
    "model": _Field(str),
    "num_vms": _Field(int, minimum=1),
}

# 允许出现、但代码不读的说明性字段。列白名单而不是放任，是为了让拼错的 live
# 字段名（resize_factr）被当成未知键抓住，而不是被当成注释放过。
#
# 这一套对所有 agent 共用：以前每个 spec 各带一份，于是 holo 能写
# history_renderer、qwen3_6 能写 scaffold_origin，两份同族 config 不能互相复制。
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
    # 会真正传给 agent 的字段。upstream_runner 的 agent 这里是空的 —— 它们的
    # yaml 改任何一行都不会改变行为。
    live: Mapping[str, _Field] = field(default_factory=dict)
    doc: FrozenSet[str] = frozenset()


# —— derail_tool_agent ——
# 这些字段逐个进 ToolAgentProtocol；coordinate_protocol 走 SafePyAutoGUICompiler
# 的枚举，和 upstream_official 那两份 config 里的同名字段不是一套词汇表。
_TOOL_AGENT_LIVE: Mapping[str, _Field] = {
    "coordinate_protocol": _Field(str, choices=frozenset({"absolute_pixels", "normalized_0_1000"})),
    "system_prompt_file": _Field(str),
    # null = 请求体省略 temperature。kimi-k3 是推理模型，显式传会被 400 拒绝。
    "temperature": _Field(float, minimum=0.0, nullable=True),
    "max_tokens": _Field(int, minimum=1),
    "max_images_in_context": _Field(int, minimum=1),
    # 上游 qwen35vl_agent 的 history_n / fold_size / Previous actions 三件套。
    "history_turns": _Field(int, minimum=1),
    "image_fold_size": _Field(int, minimum=1),
    "previous_action_log": _Field(bool),
    "tool_choice": _Field(str, choices=frozenset({"auto", "required", "none"})),
    "schema_repair_attempts": _Field(int, minimum=0),
    "multi_tool_policy": _Field(str, choices=MULTI_TOOL_POLICIES),
    "alternating_action_repeat_limit": _Field(int, minimum=0),
    "stalled_state_step_limit": _Field(int, minimum=0),
    # cuabash 变体开关：true 时 scaffold 暴露 bash 工具（agent-internal 执行，
    # 不占 runner 步数）并改用 has_bash=True 的共享环境块。GUI-only 的四个
    # agent 必须显式写 false，工具面在 yaml 里可见。
    "enable_bash": _Field(bool),
}

_QWEN38_TOOL_AGENT_LIVE: Mapping[str, _Field] = {
    **_TOOL_AGENT_LIVE,
    "enable_shell": _Field(bool),
}

# —— upstream_official 共用 ——
# 采样三项以前是 EVOCUA_/OPENCUA_ 开头的环境变量，没写进 env.example.yaml，属于
# 隐藏后门：run manifest 里看不出实际用了什么。现在只认 yaml。
_UPSTREAM_COMMON_LIVE: Mapping[str, _Field] = {
    "max_tokens": _Field(int, minimum=1),
    "top_p": _Field(float, minimum=0.0),
    "temperature": _Field(float, minimum=0.0),
    "step_budget_policy": _Field(str, choices=STEP_BUDGET_POLICIES),
    # 仅当 runner 没有导出 DERAIL_AGENT_MAX_STEPS 时才会用到。runner 始终是权威。
    "upstream_max_steps_fallback": _Field(int, minimum=1),
}

_SPECS: Mapping[str, _AgentSpec] = {
    "qwen3_6_27b": _AgentSpec(DERAIL_TOOL_AGENT, LOCAL_VLLM, live=_TOOL_AGENT_LIVE),
    "qwen3_8_27b": _AgentSpec(
        DERAIL_TOOL_AGENT, LOCAL_VLLM, live=_QWEN38_TOOL_AGENT_LIVE
    ),
    "holo_3_1_35b_a3b": _AgentSpec(DERAIL_TOOL_AGENT, LOCAL_VLLM, live=_TOOL_AGENT_LIVE),
    "kimi_k3": _AgentSpec(DERAIL_TOOL_AGENT, HOSTED_API, live=_TOOL_AGENT_LIVE),
    # kimi_k3 的 cuabash 对照组：同一条 derail_tool_agent 装配路径，仅
    # enable_bash 翻转（+ 对应 prompt/共享块变体）。与 GUI-only 的 kimi_k3
    # 构成"工具面 vs 能力"的可比实验组，跨组对比必须写明 bash 语义。
    "kimi_k3_cuabash": _AgentSpec(DERAIL_TOOL_AGENT, HOSTED_API, live=_TOOL_AGENT_LIVE),
    "evocua_32b": _AgentSpec(
        UPSTREAM_OFFICIAL,
        LOCAL_VLLM,
        live={
            **_UPSTREAM_COMMON_LIVE,
            # 上游 evocua_agent.py:64 自己也 assert 这两个取值域；这里提前拦，
            # 免得错误要等到 VM 已经起来才暴露。
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
            # OpenCUA 不 vendored 在仓库里（采集脚本按需 clone），拿不到它的取值域，
            # 所以这三项只校验非空字符串，实际取值由金标快照钉住。
            "history_type": _Field(str),
            "coordinate_type": _Field(str),
            "cot_level": _Field(str),
            "max_image_history_length": _Field(int, minimum=1),
            "use_old_sys_prompt": _Field(bool),
        },
    ),
    "qwen3_5_35b_a3b": _AgentSpec(UPSTREAM_RUNNER, LOCAL_VLLM),
    "gpt_5_5": _AgentSpec(UPSTREAM_RUNNER, HOSTED_API),
    "gpt_5_6_luna": _AgentSpec(UPSTREAM_RUNNER, HOSTED_API),
    "claude_sonnet_5": _AgentSpec(UPSTREAM_RUNNER, HOSTED_API),
    "claude_opus_4_8": _AgentSpec(UPSTREAM_RUNNER, HOSTED_API),
}


@dataclass(frozen=True)
class AgentConfig:
    """一份校验通过的 agent 配置。

    ``live`` 是会真正传给 agent 的字段（upstream_runner 的 agent 为空）；
    ``document`` 是整份 yaml，供 manifest 记录和排错使用。
    """

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
                f"{name!r} 不是 {self.agent_id} 的 live 字段；"
                f"可用字段：{sorted(self.live)}"
            ) from None


def config_dir() -> Path:
    return REPO_ROOT / "configs" / "agents"


def all_agent_ids() -> Tuple[str, ...]:
    """仓库里所有 agent config 的 agent_id，按文件名排序。"""

    return tuple(sorted(path.stem for path in config_dir().glob("*.yaml")))


def agent_id_for_type(agent_type: str) -> str:
    """把官方 runner 的 ``--agent-type`` 翻回 yaml 的文件名。

    agent_id 由本模块的表决定，不由环境变量决定 —— 否则忘了 export 就会静悄悄
    换一份配置。``DERAIL_AGENT_ID`` 只做交叉核对：它由采集脚本按 resolve_agent()
    的结果导出，对不上就说明 bash 和 Python 两张表漂移了。
    """

    try:
        agent_id = AGENT_ID_BY_TYPE[agent_type]
    except KeyError:
        raise ValueError(f"未知 DERAIL MyPCBench agent_type：{agent_type!r}") from None
    declared = os.environ.get("DERAIL_AGENT_ID")
    if declared and declared != agent_id:
        raise AgentConfigError(
            f"DERAIL_AGENT_ID={declared!r} 与 agent_type={agent_type!r} 对应的 "
            f"{agent_id!r} 不一致；01_collect_trajectories.sh 的 resolve_agent() "
            "和 agent_config.AGENT_ID_BY_TYPE 已经漂移，必须同时修"
        )
    return agent_id


def _coerce(agent_id: str, name: str, value: Any, spec: _Field) -> Any:
    where = f"{agent_id}.yaml 的 {name}"
    if value is None:
        if spec.nullable:
            return None
        raise AgentConfigError(f"{where} 不能是 null")
    # bool 是 int 的子类，不先挡住的话 `temperature: true` 会被当成 1.0 收下。
    if spec.kind is bool:
        if not isinstance(value, bool):
            raise AgentConfigError(f"{where} 必须是 true/false，实际是 {value!r}")
    elif spec.kind is int:
        if isinstance(value, bool) or not isinstance(value, int):
            raise AgentConfigError(f"{where} 必须是整数，实际是 {value!r}")
    elif spec.kind is float:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise AgentConfigError(f"{where} 必须是数值，实际是 {value!r}")
        value = float(value)
    elif spec.kind is str:
        if not isinstance(value, str) or not value.strip():
            raise AgentConfigError(f"{where} 必须是非空字符串，实际是 {value!r}")
    else:  # pragma: no cover - spec 表写错才会到这里
        raise AgentConfigError(f"{where} 的 spec 类型无法校验：{spec.kind!r}")

    if spec.choices is not None and value not in spec.choices:
        raise AgentConfigError(
            f"{where} 只能取 {sorted(spec.choices)} 之一，实际是 {value!r}"
        )
    if spec.minimum is not None and value < spec.minimum:
        raise AgentConfigError(f"{where} 不能小于 {spec.minimum}，实际是 {value!r}")
    return value


def _required_fields(spec: _AgentSpec) -> Dict[str, _Field]:
    serving = _LOCAL_REQUIRED if spec.serving == LOCAL_VLLM else _HOSTED_REQUIRED
    return {**_COMMON_REQUIRED, **serving}


def load_config(agent_id: str) -> AgentConfig:
    """读取并校验任意一份 agent config，包括 yaml 纯文档的那几个。

    :func:`load_agent_config` 是它的收窄版本，会拒绝 ``upstream_runner``。需要
    scaffold、并发度、预估步长这类"每个 agent 都有"的字段时用这个。
    """

    spec = _SPECS.get(agent_id)
    if spec is None:
        raise AgentConfigError(
            f"没有为 {agent_id!r} 定义 spec；新增 agent config 必须同时在 "
            "agent_config._SPECS 里登记 scaffold 与 serving"
        )

    try:
        import yaml
    except ImportError as exc:  # pragma: no cover - 依赖缺失时触发
        raise AgentConfigError("缺少 PyYAML；请重新安装依赖 `pip install -e .`") from exc

    path = config_dir() / f"{agent_id}.yaml"
    if not path.is_file():
        raise AgentConfigError(f"找不到 agent 配置：{path}")
    document = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(document, dict):
        raise AgentConfigError(f"{path} 不是一个 YAML 映射")

    required = _required_fields(spec)
    missing = [name for name in required if name not in document]
    if missing:
        raise AgentConfigError(f"{agent_id}.yaml 缺少必填字段：{missing}")
    if document["agent_id"] != agent_id:
        raise AgentConfigError(
            f"{path} 里的 agent_id={document['agent_id']!r} 与文件名不一致"
        )
    if document["scaffold"] != spec.scaffold:
        raise AgentConfigError(
            f"{agent_id}.yaml 的 scaffold={document['scaffold']!r} 与 spec 声明的 "
            f"{spec.scaffold!r} 不一致；换 scaffold 必须同时改 agent_config 的 spec"
        )

    for name, rule in required.items():
        _coerce(agent_id, name, document[name], rule)

    allowed = set(required) | set(spec.live) | set(spec.doc) | _COMMON_DOC
    # 本地 serving 的 agent 不许写 num_vms：并发度由显卡数推出来，写在 yaml 里
    # 的那个数字换台机器就是错的（TP8 模型曾因此以 4 个 VM 起跑）。
    forbidden = sorted(
        name
        for name in ({"num_vms", "tensor_parallel_size"} - set(required))
        if name in document
    )
    if forbidden:
        hint = (
            "本地 serving 的并发度由 GPU 数 / tensor_parallel_size 推导"
            if spec.serving == LOCAL_VLLM
            else "托管 API 不占卡，没有 tensor_parallel_size 可言"
        )
        raise AgentConfigError(f"{agent_id}.yaml 不该出现 {forbidden}；{hint}")
    unknown = sorted(set(document) - allowed - {"num_vms", "tensor_parallel_size", "served_model_name"})
    if unknown:
        raise AgentConfigError(
            f"{agent_id}.yaml 出现未知字段：{unknown}；"
            "live 字段名拼错会被当成未知键拦下，请对照 agent_config._SPECS"
        )
    absent = sorted(set(spec.live) - set(document))
    if absent:
        raise AgentConfigError(f"{agent_id}.yaml 缺少 live 字段：{absent}")

    live = {
        name: _coerce(agent_id, name, document[name], rule)
        for name, rule in spec.live.items()
    }
    # 折叠只发生在保留下来的 turn 里，所以图片上限不能超过 turn 上限 —— 写反了不会
    # 报错，只会静默退化成「窗口 = history_turns，一张都不折」。
    if "history_turns" in live and live["history_turns"] < live["max_images_in_context"]:
        raise AgentConfigError(
            f"{agent_id}.yaml 的 history_turns({live['history_turns']}) 不能小于 "
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


def load_agent_config(agent_id: str) -> AgentConfig:
    """读取一个 DERAIL factory agent 的执行配置。

    对 ``upstream_runner`` 的 agent 明确报错，而不是返回一份 live 为空、看起来
    却"配置生效了"的对象。
    """

    config = load_config(agent_id)
    if config.scaffold == UPSTREAM_RUNNER:
        raise AgentConfigError(
            f"{agent_id} 走的是 MyPCBench 内置 agent"
            f"（--agent-type {config.document['agent_type']}），DERAIL 构造不到它，"
            "它的 yaml 仍然是纯文档；不要在这里加载"
        )
    return config
