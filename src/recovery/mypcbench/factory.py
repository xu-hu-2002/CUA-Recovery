"""RECOVERY agents callable by the MyPCBench runner."""

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

from recovery.adapters.evocua import EvoCUAS2Adapter

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
    runner_max_steps = _positive_int(
        "RECOVERY_AGENT_MAX_STEPS", config["upstream_max_steps_fallback"]
    )
    upstream_guard = runner_max_steps + 1
    logging.getLogger("recovery.mypcbench.factory").info(
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
    """Import pinned EvoCUA with the shared RECOVERY prompt block applied."""
    _external_root("RECOVERY_EVOCUA_ROOT", "third_party/EvoCUA", "mm_agents/evocua/evocua_agent.py")
    module = importlib.import_module("mm_agents.evocua.evocua_agent")
    if not getattr(module, "_RECOVERY_ENV_BLOCK_APPLIED", False):
        escaped = _mypcbench_shared_block().replace("{", "{{").replace("}", "}}")
        module.S2_SYSTEM_PROMPT = module.S2_SYSTEM_PROMPT + "\n\n" + escaped
        module._RECOVERY_ENV_BLOCK_APPLIED = True
    return module


def _create_evocua(
    config: AgentConfig, model: str, screen_size: tuple[int, int], password: str
) -> Any:
    module = load_evocua_upstream()
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
        "RECOVERY_OPENCUA_OSWORLD_ROOT",
        "third_party/OpenCUA-OSWorld",
        "mm_agents/opencua/opencua_agent.py",
    )
    module = importlib.import_module("mm_agents.opencua.opencua_agent")
    # Upstream reset() uses logging without importing it.
    module.logging = logging
    max_tokens = int(os.environ.get("OPENCUA_MAX_TOKENS_OVERRIDE") or config["max_tokens"])
    if not 1 <= max_tokens <= int(config["max_tokens"]):
        raise ValueError("OPENCUA_MAX_TOKENS_OVERRIDE must be within the frozen budget")
    inner = module.OpenCUAAgent(
        model=model,
        history_type=config["history_type"],
        max_steps=_upstream_step_budget(config),
        max_image_history_length=config["max_image_history_length"],
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
    inner.system_prompt = inner.system_prompt.rstrip() + "\n\n" + _mypcbench_shared_block()

    def local_call_llm(self: Any, payload: Any, ignored_model: str) -> str:
        del ignored_model
        try:
            from openai import OpenAI
        except ImportError as exc:  # pragma: no cover
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
    """RECOVERY plugin factory with the same signature as MyPCBench ``get_agent``."""

    env = kwargs.get("env")
    agent_id = agent_id_for_type(agent_type)

    def configured() -> AgentConfig:
        config = load_agent_config(agent_id)
        logging.getLogger("recovery.mypcbench.factory").info(
            "agent config: %s -> %s", agent_type, config.path
        )
        return config

    if agent_type == "recovery_kimi_k3":
        return NativeToolComputerAgent(
            model,
            screen_size,
            protocol_from_config(configured()),
            api_key=os.environ.get("KIMI_K3_API_KEY"),
        )
    if agent_type == "recovery_kimi_k3_cuabash":
        return NativeToolComputerAgent(
            model,
            screen_size,
            protocol_from_config(configured()),
            api_key=os.environ.get("KIMI_K3_API_KEY"),
            env=env,
        )
    if agent_type == "recovery_evocua":
        return _create_evocua(configured(), model, screen_size, client_password)
    if agent_type == "recovery_opencua":
        return _create_opencua(configured(), model, screen_size, client_password)
    raise ValueError(f"agent_type {agent_type!r} 已注册 agent_id 但没有构造分支")
