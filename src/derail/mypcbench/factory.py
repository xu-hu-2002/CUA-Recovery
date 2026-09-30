"""创建可被 MyPCBench 官方 runner 调用的 DERAIL agents。"""

from __future__ import annotations

import importlib
import base64
import logging
import os
import sys
import urllib.parse
from io import BytesIO
from pathlib import Path
from types import MethodType
from typing import Any, Mapping, Optional, Sequence

from PIL import Image

from derail.adapters.evocua import EvoCUAS2Adapter

from .agent_config import AgentConfig, agent_id_for_type, load_agent_config
from .tool_agent import (
    MYPCBENCH_SHARED_BLOCK_FILE,
    NativeToolComputerAgent,
    takeover_condition_prompt,
    protocol_from_config,
    validate_pyautogui_program,
)

REPO_ROOT = Path(__file__).resolve().parents[3]


def _mypcbench_shared_block() -> str:
    """MyPCBench 共享块（完成纪律 + 环境描述），与 derail_tool_agent 读同一个文件。

    环境描述与模型无关，因此必须逐字节一致；动作协议块才是各家自己的。这两件事
    在 upstream_official 的 scaffold 里是粘在一起的，所以只能追加、不能替换：
    把 MyPCBench 的 system prompt 整份换给 EvoCUA，它的 _parse_response_s2()
    读不到 <tool_call> XML，一个动作都解析不出来 —— 而 runner 对空动作零容忍
    （run_mypcbench.py 的 abort_no_actions），整条 episode 会当场中止。
    """
    path = REPO_ROOT / "prompts" / "agents" / MYPCBENCH_SHARED_BLOCK_FILE
    if not path.is_file():
        raise RuntimeError(f"找不到 MyPCBench 共享块：{path}")
    return path.read_text(encoding="utf-8").strip()


def _positive_int(name: str, default: int) -> int:
    value = int(os.environ.get(name, default))
    if value <= 0:
        raise ValueError(f"{name} 必须是正整数")
    return value


def _upstream_step_budget(config: AgentConfig) -> int:
    """Place an upstream guard one turn beyond the runner-owned hard limit.

    MyPCBench already executes at most ``DERAIL_AGENT_MAX_STEPS`` predictions.
    EvoCUA and OpenCUA independently replace the Nth valid action with ``FAIL``
    when their internal guard is also N.  N+1 prevents that boundary override
    without allowing the runner to execute an extra action.

    The runner stays authoritative: ``upstream_max_steps_fallback`` from the
    yaml only applies when the collection script did not export the limit.
    """

    runner_max_steps = _positive_int(
        "DERAIL_AGENT_MAX_STEPS", config["upstream_max_steps_fallback"]
    )
    upstream_guard = runner_max_steps + 1
    logging.getLogger("derail.mypcbench.factory").info(
        "step budget policy %s: runner_max_steps=%d, upstream_guard_max_steps=%d",
        config["step_budget_policy"],
        runner_max_steps,
        upstream_guard,
    )
    return upstream_guard


def _external_root(env_name: str, default_relative: str, required_file: str) -> Path:
    root = Path(os.environ.get(env_name, REPO_ROOT / default_relative)).expanduser().resolve()
    if not (root / required_file).is_file():
        raise RuntimeError(
            f"找不到官方 adapter source：{root / required_file}；"
            "请先运行 collection 脚本自动 clone 冻结版本"
        )
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    return root


class _OfficialMyPCBenchAdapter:
    """收窄官方 agent 的返回 contract，并在执行前验证生成代码。"""

    def __init__(self, inner: Any, *, reset_accepts_vm_ip: bool, returns_cot: bool):
        self.inner = inner
        self.reset_accepts_vm_ip = reset_accepts_vm_ip
        self.returns_cot = returns_cot

    def reset(self, _logger: Any = None, vm_ip: Optional[str] = None) -> None:
        if self.reset_accepts_vm_ip:
            self.inner.reset(_logger, vm_ip)
        else:
            self.inner.reset(_logger)

    def predict(self, instruction: str, obs: Any):
        result = self.inner.predict(instruction, obs)
        if self.returns_cot:
            response, actions, _structured_cot = result
        else:
            response, actions = result
        return response, [validate_pyautogui_program(action) for action in actions]


def _history_image_bytes(value: str) -> bytes:
    if value.startswith("data:image/"):
        try:
            return base64.b64decode(value.split(",", 1)[1], validate=True)
        except (IndexError, ValueError) as exc:
            raise ValueError("invalid data URL in OpenCUA history") from exc
    parsed = urllib.parse.urlparse(value) if value.startswith("file://") else None
    path = Path(urllib.parse.unquote(parsed.path)) if parsed else Path(value)
    if not path.is_file():
        raise ValueError(f"OpenCUA history screenshot does not exist: {value}")
    return path.read_bytes()


class _OpenCUAMyPCBenchAdapter(_OfficialMyPCBenchAdapter):
    """Expose exact state restoration for upstream OpenCUA action-history mode."""

    def __init__(self, inner: Any):
        super().__init__(inner, reset_accepts_vm_ip=False, returns_cot=True)
        self._takeover_prompt = ""

    @property
    def native_history_system_prompt(self) -> str:
        return str(self.inner.system_prompt)

    def reset(self, _logger: Any = None, vm_ip: Optional[str] = None) -> None:
        super().reset(_logger, vm_ip)
        self._takeover_prompt = ""

    def seed_native_history(self, instruction: str, history: Any, **kwargs: Any) -> None:
        records = [dict(item) for item in history]
        first = records.pop(0) if records else {}
        if first != {"role": "system", "content": self.native_history_system_prompt}:
            raise ValueError("OpenCUA history system prompt does not match the live agent")
        if any(record.get("role") != "opencua_state" for record in records):
            raise ValueError("OpenCUA history contains a non-native state record")
        self.inner.observations = [
            {"screenshot": _history_image_bytes(str(record["observation_image_url"]))}
            for record in records
        ]
        self.inner.actions = [str(record["action"]) for record in records]
        self.inner.cots = [
            dict(record.get("cot", {"action": record["action"]}))
            for record in records
        ]
        self._takeover_prompt = takeover_condition_prompt(
            str(kwargs.get("condition", "unaware")),
            str(kwargs.get("diagnosis", "")),
            kwargs.get("root_cause_action_index"),
            str(kwargs.get("hint", "")),
        )

    def preflight_next_request(self, instruction: str, obs: Any) -> dict[str, Any]:
        """Capture the exact upstream chat payload without issuing inference."""

        class PayloadCaptured(BaseException):
            pass

        payload: dict[str, Any] = {}
        original_call = self.inner.call_llm
        takeover_prompt = self._takeover_prompt

        def capture(request: Any, _model: str) -> str:
            payload.update(dict(request))
            raise PayloadCaptured

        prompt = instruction
        if takeover_prompt:
            prompt = f"{instruction}\n\n{takeover_prompt}"
        self.inner.call_llm = capture
        try:
            self.inner.predict(prompt, obs)
        except PayloadCaptured:
            pass
        finally:
            self.inner.call_llm = original_call
            self._takeover_prompt = takeover_prompt
        if not payload:
            raise RuntimeError("OpenCUA preflight did not capture a request payload")
        return payload

    def predict(self, instruction: str, obs: Any):
        prompt = instruction
        if self._takeover_prompt:
            prompt = f"{instruction}\n\n{self._takeover_prompt}"
            self._takeover_prompt = ""
        return super().predict(prompt, obs)

class _EvoCUATakeoverAdapter(_OfficialMyPCBenchAdapter):
    """Restore the exact EvoCUA S2 self-takeover state."""

    def __init__(self, inner: Any, *, reset_accepts_vm_ip: bool, returns_cot: bool):
        super().__init__(
            inner, reset_accepts_vm_ip=reset_accepts_vm_ip, returns_cot=returns_cot
        )
        self._takeover_prompt = ""

    def reset(self, _logger: Any = None, vm_ip: Optional[str] = None) -> None:
        super().reset(_logger, vm_ip)
        self._takeover_prompt = ""

    def seed_native_history(
        self,
        instruction: str,
        native_history: Sequence[Mapping[str, Any]],
        *,
        condition: str = "unaware",
        **kwargs: Any,
    ) -> None:
        if not instruction.strip() or not native_history:
            raise ValueError("takeover instruction and native history must be non-empty")
        system, *turns = native_history
        expected = {"role": "system", "content": EvoCUAS2Adapter.system_prompt()}
        if system != expected:
            raise ValueError("native history system prompt does not match the live S2 protocol")
        pairs = list(zip(turns[0::2], turns[1::2]))
        if len(turns) % 2 or any(
            user.get("role") != "user" or assistant.get("role") != "assistant"
            for user, assistant in pairs
        ):
            raise ValueError("native history is not a sequence of complete S2 turns")
        self.inner.screenshots, self.inner.responses, self.inner.actions = [], [], []
        for user, assistant in pairs:
            self._seed_evocua_turn(user, assistant)
        self._takeover_prompt = takeover_condition_prompt(
            condition,
            str(kwargs.get("diagnosis", "")),
            kwargs.get("root_cause_action_index"),
            str(kwargs.get("hint", "")),
        )

    def predict(self, instruction: str, obs: Any):
        prompt = instruction
        if self._takeover_prompt:
            prompt = f"{instruction}\n\n{self._takeover_prompt}"
            self._takeover_prompt = ""
        return super().predict(prompt, obs)

    def preflight_next_request(self, instruction: str, obs: Any) -> dict[str, Any]:
        """Capture the exact next EvoCUA request without consuming live state."""

        class PayloadCaptured(BaseException):
            pass

        payload: dict[str, Any] = {}
        original_call = self.inner.call_llm
        takeover_prompt = self._takeover_prompt
        snapshots = {
            name: list(getattr(self.inner, name))
            for name in ("screenshots", "responses", "actions")
        }

        def capture(request: Any) -> str:
            payload.update(dict(request))
            raise PayloadCaptured

        prompt = instruction
        if takeover_prompt:
            prompt = f"{instruction}\n\n{takeover_prompt}"
        self.inner.call_llm = capture
        try:
            self.inner.predict(prompt, obs)
        except PayloadCaptured:
            pass
        finally:
            self.inner.call_llm = original_call
            self._takeover_prompt = takeover_prompt
            for name, values in snapshots.items():
                setattr(self.inner, name, values)
        if not payload:
            raise RuntimeError("EvoCUA preflight did not capture a request payload")
        return payload

    def _seed_evocua_turn(self, user: Mapping[str, Any], assistant: Mapping[str, Any]) -> None:
        url = str(user["content"][0]["image_url"]["url"])
        payload = base64.b64decode(
            NativeToolComputerAgent._resolve_history_image_url(url).split(",", 1)[1]
        )
        module = load_evocua_upstream()
        processed, processed_width, processed_height = module.process_image(
            payload, factor=self.inner.resize_factor
        )
        with Image.open(BytesIO(payload)) as image:
            width, height = image.size
        response = str(assistant["content"][0]["text"])
        action, _ = self.inner._parse_response_s2(
            response, processed_width, processed_height, width, height
        )
        self.inner.screenshots.append(processed)
        self.inner.responses.append(response)
        self.inner.actions.append(action)


def load_evocua_upstream() -> Any:
    """Import pinned EvoCUA with the shared DERAIL prompt block applied."""
    _external_root("DERAIL_EVOCUA_ROOT", "third_party/EvoCUA", "mm_agents/evocua/evocua_agent.py")
    module = importlib.import_module("mm_agents.evocua.evocua_agent")
    if not getattr(module, "_DERAIL_ENV_BLOCK_APPLIED", False):
        escaped = _mypcbench_shared_block().replace("{", "{{").replace("}", "}}")
        module.S2_SYSTEM_PROMPT = module.S2_SYSTEM_PROMPT + "\n\n" + escaped
        module._DERAIL_ENV_BLOCK_APPLIED = True
    return module


def _create_evocua(
    config: AgentConfig, model: str, screen_size: tuple[int, int], password: str
) -> Any:
    module = load_evocua_upstream()
    # 只影响当前 agent child；MyPCBench 在此之前已经把真正的 OPENAI_API_KEY
    # 注入 VM 内的应用服务，所以模型 endpoint 可以使用独立 key。
    if os.environ.get("EVOCUA_API_KEY"):
        os.environ["OPENAI_API_KEY"] = os.environ["EVOCUA_API_KEY"]
    inner = module.EvoCUAAgent(
        model=model,
        max_tokens=config["max_tokens"],
        top_p=config["top_p"],
        temperature=config["temperature"],
        max_steps=_upstream_step_budget(config),
        prompt_style=config["prompt_style"],
        max_history_turns=config["max_history_turns"],
        screen_size=screen_size,
        coordinate_type=config["coordinate_type"],
        password=password,
        resize_factor=config["resize_factor"],
    )
    return _EvoCUATakeoverAdapter(inner, reset_accepts_vm_ip=True, returns_cot=False)


def _create_opencua(
    config: AgentConfig, model: str, screen_size: tuple[int, int], password: str
) -> Any:
    _external_root(
        "DERAIL_OPENCUA_OSWORLD_ROOT",
        "third_party/OpenCUA-OSWorld",
        "mm_agents/opencua/opencua_agent.py",
    )
    module = importlib.import_module("mm_agents.opencua.opencua_agent")
    # 上游当前 reset() 使用 logging 却未 import；只补模块依赖，不改变 agent 行为。
    module.logging = logging
    max_tokens = int(os.environ.get("OPENCUA_MAX_TOKENS_OVERRIDE") or config["max_tokens"])
    if not 1 <= max_tokens <= int(config["max_tokens"]):
        raise ValueError("OPENCUA_MAX_TOKENS_OVERRIDE must be within the frozen budget")
    inner = module.OpenCUAAgent(
        model=model,
        history_type=config["history_type"],
        max_steps=_upstream_step_budget(config),
        max_image_history_length=config["max_image_history_length"],
        # platform / action_space / observation_type 是 MyPCBench 这套 harness 的
        # 固有属性，不是 agent 的可选项，所以不进 yaml。
        platform="ubuntu",
        max_tokens=max_tokens,
        top_p=config["top_p"],
        temperature=config["temperature"],
        action_space="pyautogui",
        observation_type="screenshot",
        cot_level=config["cot_level"],
        screen_size=screen_size,
        coordinate_type=config["coordinate_type"],
        use_old_sys_prompt=config["use_old_sys_prompt"],
        password=password,
    )
    # OpenCUA 把 system prompt 挂在实例属性上（opencua_agent.py:287-295），
    # 所以不用像 EvoCUA 那样 patch 模块常量，也不经过 .format()，无需转义。
    inner.system_prompt = inner.system_prompt.rstrip() + "\n\n" + _mypcbench_shared_block()

    def local_call_llm(self: Any, payload: Any, ignored_model: str) -> str:
        del ignored_model
        try:
            from openai import OpenAI
        except ImportError as exc:  # pragma: no cover - collection 环境触发
            raise RuntimeError("缺少 openai；请安装 `pip install -e '.[collection]'`") from exc
        base_url = os.environ.get("OPENAI_BASE_URL")
        if not base_url:
            raise RuntimeError("OpenCUA 缺少 OPENAI_BASE_URL")
        client = OpenAI(
            base_url=base_url,
            api_key=os.environ.get("OPENCUA_API_KEY")
            or os.environ.get("OPENAI_API_KEY")
            or "EMPTY",
        )
        response = client.chat.completions.create(**payload)
        return response.choices[0].message.content or ""

    inner.call_llm = MethodType(local_call_llm, inner)
    return _OpenCUAMyPCBenchAdapter(inner)


def create_mypcbench_agent(
    agent_type: str,
    model: str,
    screen_size: tuple[int, int],
    client_password: str,
    **kwargs: Any,
) -> Any:
    """DERAIL plugin factory；参数形状与 MyPCBench ``get_agent`` 一致。

    行为参数一律来自 ``configs/agents/<agent_id>.yaml``。官方 runner 只传
    ``agent_type``，所以先经 :func:`agent_id_for_type` 翻回 agent_id 再加载。
    ``kwargs`` 里的 ``env``（VM 控制句柄，run_mypcbench.py:911 注入）由需要直接
    执行 VM shell 的变体使用：cuabash 的 bash 分流、qwen38 的 shell/takeover
    路径。GUI-only agent 收到也不触碰。
    """

    env = kwargs.get("env")
    agent_id = agent_id_for_type(agent_type)

    def configured() -> AgentConfig:
        config = load_agent_config(agent_id)
        logging.getLogger("derail.mypcbench.factory").info(
            "agent config: %s -> %s", agent_type, config.path
        )
        return config

    if agent_type == "derail_qwen36":
        return NativeToolComputerAgent(
            model,
            screen_size,
            protocol_from_config(configured()),
            api_key=os.environ.get("QWEN36_API_KEY"),
        )
    if agent_type == "derail_qwen38":
        return NativeToolComputerAgent(
            model,
            screen_size,
            protocol_from_config(configured()),
            api_key=os.environ.get("QWEN38_API_KEY"),
            environment=kwargs.get("env"),
        )
    if agent_type == "derail_holo31":
        return NativeToolComputerAgent(
            model,
            screen_size,
            protocol_from_config(configured()),
            api_key=os.environ.get("HOLO31_API_KEY"),
        )
    if agent_type == "derail_kimi_k3":
        # routify 网关直连（OPENAI_BASE_URL/OPENAI_API_KEY 由 tool_agent 默认读取），
        # 无本地 serving；KIMI_K3_API_KEY 预留显式覆盖。
        return NativeToolComputerAgent(
            model,
            screen_size,
            protocol_from_config(configured()),
            api_key=os.environ.get("KIMI_K3_API_KEY"),
        )
    if agent_type == "derail_kimi_k3_cuabash":
        # kimi_k3 的 GUI+bash 对照组：装配路径与 GUI-only 版逐项相同，仅
        # enable_bash=true（工具 schema + prompt/共享块变体由 protocol 驱动）。
        # env 由官方 runner 注入，bash 分流经它执行 VM shell。
        return NativeToolComputerAgent(
            model,
            screen_size,
            protocol_from_config(configured()),
            api_key=os.environ.get("KIMI_K3_API_KEY"),
            env=env,
        )
    if agent_type == "derail_evocua":
        return _create_evocua(configured(), model, screen_size, client_password)
    if agent_type == "derail_opencua":
        return _create_opencua(configured(), model, screen_size, client_password)
    # agent_id_for_type 已经挡下未知 agent_type；走到这里说明 AGENT_ID_BY_TYPE
    # 加了新条目却忘了在上面接线。
    raise ValueError(f"agent_type {agent_type!r} 已注册 agent_id 但没有构造分支")
