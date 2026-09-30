"""源 rollout 的环境初始化与每步状态指纹（论文 02:6、03:25）。

- reset 之后：在 guest 里跑 ``disabled_nondeterminism_sources`` 的命令（关自动更新、
  通知守护），失败按 ``determinism_on_error`` 报错或告警。
- 每次 step 之后：跑 ``state_probes``（经 ``build_state_probe_commands`` 渲染成 guest
  命令；sqlite 走 infra/snapshot/changelog_replay.py 的 digest，剔除易变列），由
  ``probe_state_fingerprint`` 组成 :class:`StateFingerprint`，追加到任务目录的
  ``state_probe_file``。接管回放（derail.mypcbench.takeover_agent）读同一份 environment
  config、用同一对函数，按 ``traj_index`` 对齐比对。每行的
  ``traj_index`` 是该动作在 traj.jsonl 里的 0-based 行号（runner 在 step 返回后才写
  那一行，所以就是当时文件的行数）；reset 行为 -1。bash 工具轮不经过 env.step，
  其状态体现在下一次 step 的指纹里。

命令与开关都在 configs/environments/<env>.yaml。官方 runner 不改：
``site_hook/sitecustomize.py`` 在 run_mypcbench.py 进程启动时调 :func:`install_for_runner`，
把 :func:`install` 包到 ``MyPCBenchEnv.reset/step`` 上。
"""

from __future__ import annotations

import json
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Mapping, Optional

import yaml

from derail.replay.mypcbench import build_state_probe_commands, probe_state_fingerprint
from derail.replay.verification import ReplayVerificationError, StateFingerprint

REPOSITORY = Path(__file__).resolve().parents[3]
ENV_CONFIG_VAR = "DERAIL_ENVIRONMENT_CONFIG"
ON_ERROR = frozenset({"fail", "warn"})


@dataclass(frozen=True)
class EnvironmentHooks:
    determinism_commands: Mapping[str, str]
    determinism_on_error: str
    probe_enabled: bool
    probe_commands: Mapping[str, str]
    probe_on_error: str
    probe_file: str
    record_output: bool
    output_max_chars: int

    @classmethod
    def from_config(cls, path: Path) -> "EnvironmentHooks":
        cfg = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
        hooks = cls(
            determinism_commands=dict(cfg.get("disabled_nondeterminism_sources") or {}),
            determinism_on_error=cfg.get("determinism_on_error", "fail"),
            probe_enabled=bool(cfg.get("state_probe_enabled", False)),
            probe_commands=(
                build_state_probe_commands(cfg["state_probes"], REPOSITORY)
                if cfg.get("state_probes")
                else {}
            ),
            probe_on_error=cfg.get("state_probe_on_error", "fail"),
            probe_file=cfg.get("state_probe_file", "state_probes.jsonl"),
            record_output=bool(cfg.get("state_probe_record_output", False)),
            output_max_chars=int(cfg.get("state_probe_output_max_chars", 4000)),
        )
        if {hooks.determinism_on_error, hooks.probe_on_error} - ON_ERROR:
            raise ValueError(f"{path}: *_on_error 只能是 {sorted(ON_ERROR)}")
        if hooks.probe_enabled and not hooks.probe_commands:
            raise ValueError(f"{path}: state_probe_enabled=true 但没有 state_probes")
        return hooks


def fingerprint(
    execute_shell: Callable[[str], Any], commands: Mapping[str, str]
) -> tuple[StateFingerprint, Dict[str, str]]:
    """与接管回放同一规则（``probe_state_fingerprint``），另返回各探针原始输出。"""

    outputs: Dict[str, str] = {}
    return probe_state_fingerprint(execute_shell, commands, outputs=outputs), outputs


def run_determinism(execute_shell: Callable[[str], Any], hooks: EnvironmentHooks) -> Dict[str, Any]:
    results: Dict[str, Any] = {}
    for name, command in hooks.determinism_commands.items():
        response = execute_shell(command)
        response = response if isinstance(response, Mapping) else {}
        code = response.get("returncode")
        results[name] = {"returncode": code}
        if code != 0:
            results[name]["error"] = str(response.get("error", ""))[:500]
            message = "determinism command %s failed (returncode=%s)" % (name, code)
            if hooks.determinism_on_error == "fail":
                raise RuntimeError(message)
            print("[DERAIL env] warning: " + message, file=sys.stderr)
    return results


def _traj_lines(probe_path: Path) -> int:
    traj = probe_path.parent / "traj.jsonl"
    if not traj.is_file():
        return 0
    with traj.open("rb") as handle:
        return sum(1 for _ in handle)


def _record(env: Any, hooks: EnvironmentHooks, kind: str, action: Any, extra: Mapping[str, Any]) -> None:
    path: Optional[Path] = getattr(env, "_derail_probe_path", None)
    if path is None:
        return
    record: Dict[str, Any] = {
        "kind": kind,
        "traj_index": -1 if kind == "reset" else _traj_lines(path),
        "env_step": int(getattr(env, "_step_no", 0)),
        "ts": time.time(),
        **extra,
    }
    if action is not None:
        record["action"] = str(action)[:500]
    if hooks.probe_enabled:
        try:
            state, outputs = fingerprint(env._execute_shell, hooks.probe_commands)
            record["state_fingerprint"] = state.to_dict()
            if hooks.record_output:
                record["probe_output"] = {k: v[: hooks.output_max_chars] for k, v in outputs.items()}
        except ReplayVerificationError as exc:
            if hooks.probe_on_error == "fail":
                raise
            record["probe_error"] = str(exc)[:500]
            print("[DERAIL env] warning: %s" % exc, file=sys.stderr)
    with path.open("w" if kind == "reset" else "a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def install(env_cls: type, hooks: EnvironmentHooks, result_root: Optional[Path] = None) -> None:
    """包装 ``env_cls.reset/step``。探针文件放在 ``env.derail_probe_dir``（调用方可预先设），
    否则放在 ``result_root/<task id>``（官方 runner 的任务目录布局）。"""

    if getattr(env_cls, "_derail_hooks_installed", False):
        return
    original_reset, original_step = env_cls.reset, env_cls.step

    def reset(self: Any, task_config: Optional[Dict] = None, *args: Any, **kwargs: Any) -> Any:
        obs = original_reset(self, task_config, *args, **kwargs)
        task_id = str((task_config or {}).get("id", ""))
        directory = getattr(self, "derail_probe_dir", None) or (
            Path(result_root) / task_id if result_root and task_id else None
        )
        self._derail_probe_path = Path(directory) / hooks.probe_file if directory else None
        if self._derail_probe_path is not None:
            self._derail_probe_path.parent.mkdir(parents=True, exist_ok=True)
        determinism = run_determinism(self._execute_shell, hooks)
        _record(self, hooks, "reset", None, {"task_id": task_id, "determinism": determinism})
        return obs

    def step(self: Any, action: Any, *args: Any, **kwargs: Any) -> Any:
        result = original_step(self, action, *args, **kwargs)
        _record(self, hooks, "step", action, {})
        return result

    env_cls.reset, env_cls.step = reset, step
    env_cls._derail_hooks_installed = True


def _argv_value(flag: str) -> Optional[str]:
    argv = sys.argv[1:]
    for index, item in enumerate(argv):
        if item == flag and index + 1 < len(argv):
            return argv[index + 1]
        if item.startswith(flag + "="):
            return item.split("=", 1)[1]
    return None


def install_for_runner() -> None:
    """在官方 run_mypcbench.py 进程里装钩子（由 site_hook/sitecustomize.py 调用）。"""

    hooks = EnvironmentHooks.from_config(Path(os.environ[ENV_CONFIG_VAR]))
    harness_dir = str(Path(sys.argv[0]).resolve().parent)
    if harness_dir not in sys.path:
        sys.path.insert(0, harness_dir)
    import env as mypcbench_env  # type: ignore  # 官方 agent-harness/env.py

    result_dir = _argv_value("--result_dir")
    install(mypcbench_env.MyPCBenchEnv, hooks, Path(result_dir) if result_dir else None)
