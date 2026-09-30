"""Real MyPCBench VM backend for canonical prefix replay."""

from __future__ import annotations

import base64
import hashlib
import json
import os
import re
import shlex
import tempfile
import uuid
from pathlib import Path
from typing import Any, Callable, Dict, Mapping, Optional

from derail.canonical.actions import (
    Action,
    SequenceAction,
    ShellAction,
    TerminateAction,
    WaitAction,
    action_to_dict,
    reframe_action,
)
from derail.derived.layout import sha256_file
from derail.replay.executor import compile_pyautogui
from derail.replay.verification import ReplayVerificationError, StateFingerprint


_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_SQLITE_DIGEST_PROGRAM = (
    "import glob, json, os, sys\n"
    "sys.path.insert(0, sys.argv[1])\n"
    "import derail_state_digest as d\n"
    "rules = d.load_volatile_rules(os.path.join(sys.argv[1], 'volatile.json'))\n"
    "paths = [p for p in sorted(glob.glob(sys.argv[2]))"
    " if not os.path.islink(p) and os.path.getsize(p) > 0]\n"
    "print(json.dumps({os.path.basename(p): d.digest(p, rules=rules) for p in paths},"
    " sort_keys=True))\n"
)


def build_state_probe_commands(
    probes: Mapping[str, Mapping[str, Any]], repository: Path
) -> Dict[str, str]:
    """Render configured state probes into in-VM shell commands."""

    commands: Dict[str, str] = {}
    for name, spec in sorted(probes.items()):
        kind = spec.get("kind")
        if kind == "shell":
            commands[name] = str(spec["command"])
        elif kind == "sqlite_digest":
            def payload(key: str) -> str:
                path = Path(str(spec[key]))
                path = path if path.is_absolute() else repository / path
                return base64.b64encode(path.read_bytes()).decode("ascii")

            commands[name] = (
                'd=$(mktemp -d) && '
                "printf %%s %s | base64 -d > \"$d/derail_state_digest.py\" && "
                "printf %%s %s | base64 -d > \"$d/volatile.json\" && "
                "python3 -c %s \"$d\" %s; s=$?; rm -rf \"$d\"; exit $s"
                % (
                    payload("script"),
                    payload("volatile_columns"),
                    shlex.quote(_SQLITE_DIGEST_PROGRAM),
                    shlex.quote(str(spec["databases_glob"])),
                )
            )
        else:
            raise ReplayVerificationError("unknown state probe kind for %s: %r" % (name, kind))
    if not commands:
        raise ReplayVerificationError("real VM replay 至少需要一个非截图 state probe")
    return commands


def probe_state_fingerprint(
    execute_shell: Callable[[str], Any],
    commands: Mapping[str, str],
    screenshot_sha256: str = "",
    outputs: Optional[Dict[str, str]] = None,
) -> StateFingerprint:
    """Run the probes in the VM; a JSON ``{key: sha256}`` output expands to ``probe/key``."""

    components: Dict[str, str] = {}
    for name, command in sorted(commands.items()):
        output = str(
            MyPCBenchVMReplayBackend._checked_shell(execute_shell, command, "state probe %s" % name)
        )
        if outputs is not None:
            outputs[name] = output
        try:
            parsed = json.loads(output)
        except ValueError:
            parsed = None
        if (
            isinstance(parsed, dict)
            and parsed
            and all(isinstance(value, str) and _SHA256.match(value) for value in parsed.values())
        ):
            components.update({"%s/%s" % (name, key): value for key, value in parsed.items()})
        else:
            components[name] = hashlib.sha256(
                output.encode("utf-8", errors="replace")
            ).hexdigest()
    return StateFingerprint(components=components, screenshot_sha256=screenshot_sha256)


class MyPCBenchVMReplayBackend:
    execution_mode = "vm"

    def __init__(
        self,
        *,
        env_factory: Callable[[], Any],
        task_config: Mapping[str, Any],
        state_probe_commands: Mapping[str, str],
        evidence_dir: Path,
        sleep_after_s: float = 1.0,
    ) -> None:
        if not state_probe_commands:
            raise ReplayVerificationError("real VM replay 至少需要一个非截图 state probe")
        self._env_factory = env_factory
        self._task_config = dict(task_config)
        self._state_probe_commands = dict(state_probe_commands)
        self._sleep_after_s = sleep_after_s
        self._evidence_dir = evidence_dir.resolve()
        self._env: Optional[Any] = None
        self._last_screenshot_sha256 = ""
        self._last_screenshot_uri = ""
        self._action_count = 0
        self._session_id = ""
        self._bound_attempt_id = ""
        self._bound_state_sha256 = ""
        self._initial_state_alignment: Dict[str, Any] = {}

    @property
    def replay_session_id(self) -> str:
        return self._session_id

    @property
    def initial_state_alignment(self) -> Mapping[str, Any]:
        return dict(self._initial_state_alignment)

    def _refresh_seeded_firefox_tabs(self, env: Any) -> None:
        execute_shell = getattr(env, "_execute_shell", None)
        execute_pyautogui = getattr(env, "_execute_pyautogui", None)
        if not callable(execute_shell) or not callable(execute_pyautogui):
            raise ReplayVerificationError("VM cannot align the seeded Firefox tabs")
        focus = self._focus_seeded_firefox_window(execute_shell)
        if not isinstance(focus, Mapping) or focus.get("returncode") != 0:
            detail = ""
            if isinstance(focus, Mapping):
                detail = str(focus.get("error") or focus.get("output") or "").strip()
            suffix = ": %s" % detail[-500:] if detail else ""
            raise ReplayVerificationError("cannot focus the seeded Firefox window%s" % suffix)
        refresh = execute_pyautogui(
            "\nfor _ in range(3):"
            "\n    pyautogui.hotkey('ctrl', 'r')"
            "\n    time.sleep(0.5)"
            "\n    pyautogui.hotkey('ctrl', 'tab')"
            "\n    time.sleep(0.2)"
            "\ntime.sleep(5)"
        )
        if not isinstance(refresh, Mapping) or refresh.get("returncode") != 0:
            raise ReplayVerificationError("cannot refresh the seeded Firefox tabs")
        self._initial_state_alignment = {
            "kind": "refresh_seeded_firefox_tabs_after_app_warmup",
            "tab_count": 3,
            "preserves_active_tab": True,
            "focus_strategy": "wmctrl_readiness_retry_v2",
        }

    @staticmethod
    def _focus_seeded_firefox_window(execute_shell: Callable[[str], Any]) -> Any:
        return execute_shell(
            "sh -lc '"
            "auths="
            "$(find /run/user -maxdepth 3 -type f -name Xauthority 2>/dev/null; "
            "printf \"%s\\n\" /home/oai/.Xauthority /home/user/.Xauthority); "
            "for attempt in $(seq 1 60); do "
            "window_id=; selected_auth=; "
            "for auth in $auths; do [ -r \"$auth\" ] || continue; "
            "window_id=$(DISPLAY=:0 XAUTHORITY=\"$auth\" wmctrl -lx 2>/dev/null | "
            "awk '\"'\"'tolower($0) ~ /(firefox|navigator)/ {print $1; exit}'\"'\"'); "
            "if [ -z \"$window_id\" ]; then "
            "window_id=$(DISPLAY=:0 XAUTHORITY=\"$auth\" wmctrl -l 2>/dev/null | "
            "awk '\"'\"'tolower($0) ~ /mozilla firefox/ {print $1; exit}'\"'\"'); fi; "
            "[ -n \"$window_id\" ] && { selected_auth=$auth; break; }; done; "
            "if [ -n \"$window_id\" ]; then "
            "DISPLAY=:0 XAUTHORITY=\"$selected_auth\" wmctrl -ia \"$window_id\" && exit 0; "
            "command -v xdotool >/dev/null && DISPLAY=:0 XAUTHORITY=\"$selected_auth\" xdotool windowactivate --sync \"$window_id\" && exit 0; "
            "fi; "
            "if [ \"$attempt\" -eq 15 ] && ! pgrep -x firefox >/dev/null; then "
            "nohup firefox >/tmp/derail-firefox-start.log 2>&1 & "
            "fi; "
            "sleep 1; "
            "done; "
            "echo \"Firefox focus timeout; auths=$(printf %s \\\"$auths\\\" | tr \\\"\\n\\\" ,); processes=$(pgrep -a firefox | head -3); "
            "windows=$(for auth in $auths; do DISPLAY=:0 XAUTHORITY=\\\"$auth\\\" wmctrl -lx 2>&1; done | head -8)\" >&2; exit 1'"
        )

    def restore_snapshot(self, snapshot_uri: str, snapshot_sha256: str) -> None:
        path = Path(snapshot_uri).resolve()
        if not path.is_file():
            raise ReplayVerificationError("qcow2 snapshot 不存在: %s" % path)
        if sha256_file(path) != snapshot_sha256:
            raise ReplayVerificationError("qcow2 snapshot SHA-256 不匹配")
        env = self._env_factory()
        configured = getattr(env, "qcow2_path", None)
        if configured is not None and Path(configured).resolve() != path:
            raise ReplayVerificationError("env_factory 使用的 qcow2 与 replay plan 不一致")
        reset_config = dict(self._task_config)
        pre_command = reset_config.pop("pre_command", "")
        observation = env.reset(task_config=reset_config)
        if pre_command:
            execute_shell = getattr(env, "_execute_shell", None)
            if not callable(execute_shell):
                raise ReplayVerificationError("task pre_command 需要 in-VM shell 接口")
            self._checked_shell(execute_shell, pre_command, "task_pre_command")
            get_obs = getattr(env, "_get_obs", None)
            if not callable(get_obs):
                raise ReplayVerificationError("pre_command 后无法重新获取 VM observation")
            observation = get_obs()
        if hasattr(env, "config"):
            env.config = dict(self._task_config)
        self._refresh_seeded_firefox_tabs(env)
        self._env = env
        self._action_count = 0
        self._session_id = uuid.uuid4().hex
        self._bound_attempt_id = ""
        self._bound_state_sha256 = ""
        self._capture_screenshot(observation, "initial.png")

    def adopt_current_snapshot(
        self,
        snapshot_uri: str,
        snapshot_sha256: str,
        *,
        hash_preverified: bool = False,
    ) -> None:
        """Adopt the fresh task VM that the outer MyPCBench runner just reset."""

        path = Path(snapshot_uri).resolve()
        if not path.is_file():
            raise ReplayVerificationError("qcow2 snapshot 不存在: %s" % path)
        if not hash_preverified and sha256_file(path) != snapshot_sha256:
            raise ReplayVerificationError("qcow2 snapshot SHA-256 不匹配")
        env = self._env_factory()
        configured = getattr(env, "qcow2_path", None)
        if configured is None or Path(configured).resolve() != path:
            raise ReplayVerificationError("current VM qcow2 does not match the replay plan")
        if dict(getattr(env, "config", {})) != self._task_config:
            raise ReplayVerificationError("current VM task config does not match the replay plan")
        if list(getattr(env, "action_history", [])):
            raise ReplayVerificationError("current VM is not a fresh zero-action task state")
        self._refresh_seeded_firefox_tabs(env)
        self._env = env
        self._action_count = 0
        self._session_id = uuid.uuid4().hex
        self._bound_attempt_id = ""
        self._bound_state_sha256 = ""
        self._capture_screenshot(env._get_obs(), "initial.png")

    def assert_plan_provenance(self, plan: Any) -> None:
        try:
            task_config = json.loads(Path(plan.task_config_uri).read_text(encoding="utf-8"))
            state_probes = json.loads(
                Path(plan.state_probe_config_uri).read_text(encoding="utf-8")
            )
        except (OSError, json.JSONDecodeError) as exc:
            raise ReplayVerificationError("replay plan provenance JSON 无法读取") from exc
        if task_config != self._task_config:
            raise ReplayVerificationError("backend task_config 与 replay plan 不一致")
        if state_probes != self._state_probe_commands:
            raise ReplayVerificationError("backend state probes 与 replay plan 不一致")

    def _capture_screenshot(self, observation: Mapping[str, Any], filename: str) -> None:
        screenshot = observation.get("screenshot")
        if not isinstance(screenshot, bytes) or not screenshot.startswith(b"\x89PNG"):
            raise ReplayVerificationError("VM replay 未返回有效 PNG observation")
        self._evidence_dir.mkdir(parents=True, exist_ok=True)
        target = self._evidence_dir / filename
        fd, temporary = tempfile.mkstemp(prefix=".%s." % filename, dir=str(self._evidence_dir))
        try:
            with os.fdopen(fd, "wb") as handle:
                handle.write(screenshot)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, target)
        except BaseException:
            try:
                os.unlink(temporary)
            except OSError:
                pass
            raise
        self._last_screenshot_sha256 = hashlib.sha256(screenshot).hexdigest()
        self._last_screenshot_uri = str(target.resolve())

    def execute(self, action: Action) -> Mapping[str, Any]:
        if self._env is None:
            raise ReplayVerificationError("必须先 restore MyPCBench VM")
        source_action = action
        action = reframe_action(
            action,
            int(getattr(self._env, "screen_width", 1280)),
            int(getattr(self._env, "screen_height", 800)),
        )
        before_sha256 = self._last_screenshot_sha256
        before_uri = self._last_screenshot_uri
        primitives = action.actions if isinstance(action, SequenceAction) else (action,)
        shell_results = []
        saw_control_flow = False
        for primitive in primitives:
            if isinstance(primitive, ShellAction):
                execute_shell = getattr(self._env, "_execute_shell", None)
                if not callable(execute_shell):
                    raise ReplayVerificationError(
                        "shell prefix requires the in-VM shell interface"
                    )
                outputs = []
                for command in primitive.commands:
                    response = execute_shell(command)
                    if not isinstance(response, Mapping):
                        raise ReplayVerificationError("VM shell returned an invalid result")
                    returncode = response.get("returncode")
                    if isinstance(returncode, bool) or not isinstance(returncode, int):
                        raise ReplayVerificationError(
                            "VM shell result has no integer returncode"
                        )
                    outputs.append(
                        {
                            "stdout": str(response.get("output", ""))[
                                : primitive.max_output_length
                            ],
                            "stderr": str(response.get("error", ""))[
                                : primitive.max_output_length
                            ],
                            "outcome": {"type": "exit", "exit_code": returncode},
                        }
                    )
                shell_results.append(
                    {"call_id": "replay_%d" % len(shell_results), "result": outputs}
                )
                continue
            if isinstance(primitive, TerminateAction):
                saw_control_flow = True
                continue
            source = "WAIT" if isinstance(primitive, WaitAction) else compile_pyautogui(primitive)
            _observation, _reward, done, info = self._env.step(
                source, self._sleep_after_s
            )
            if done:
                raise ReplayVerificationError("VM 在 prefix replay 中意外终止: %s" % info)

        observation = self._env._get_obs()
        self._capture_screenshot(
            observation, "after_action_%06d.png" % self._action_count
        )
        if shell_results:
            result = json.dumps(
                {"shell_results": shell_results}, ensure_ascii=False, sort_keys=True
            )
        elif saw_control_flow:
            result = "control_flow_only"
        else:
            result = "ok"
        self._action_count += 1
        record = {
            "result": result,
            "observation_before_uri": before_uri,
            "observation_before_sha256": before_sha256,
            "observation_after_uri": self._last_screenshot_uri,
            "observation_after_sha256": self._last_screenshot_sha256,
        }
        if action != source_action:
            record["coordinate_reprojection"] = {
                "source_action": action_to_dict(source_action),
                "executed_action": action_to_dict(action),
            }
        return record

    def fingerprint(self) -> StateFingerprint:
        if self._env is None:
            raise ReplayVerificationError("必须先 restore MyPCBench VM")
        execute_shell = getattr(self._env, "_execute_shell", None)
        if not callable(execute_shell):
            raise ReplayVerificationError("MyPCBench env 缺少 in-VM state probe 接口")
        return probe_state_fingerprint(
            execute_shell, self._state_probe_commands, self._last_screenshot_sha256
        )

    @staticmethod
    def _checked_shell(execute_shell: Callable[[str], Any], command: str, label: str) -> Any:
        response = execute_shell(command)
        if not isinstance(response, Mapping):
            raise ReplayVerificationError("%s 返回格式无效" % label)
        returncode = response.get("returncode")
        if isinstance(returncode, bool) or not isinstance(returncode, int):
            raise ReplayVerificationError("%s 缺少整数 returncode" % label)
        if returncode != 0:
            raise ReplayVerificationError(
                "%s 失败(returncode=%d): %s"
                % (label, returncode, response.get("error", ""))
            )
        return response.get("output", "")

    def bind_verification(self, attempt_id: str, state_sha256: str) -> None:
        self._bound_attempt_id = attempt_id
        self._bound_state_sha256 = state_sha256

    def assert_replay_binding(self, verification: Any) -> None:
        if verification.attempt_id != self._bound_attempt_id:
            raise ReplayVerificationError("takeover environment 未绑定该 replay attempt")
        if verification.vm_session_id != self._session_id:
            raise ReplayVerificationError("takeover environment/replay VM session 不一致")
        if self.fingerprint().sha256 != self._bound_state_sha256:
            raise ReplayVerificationError("takeover 前 VM state 已偏离 verified replay state")

    def observe(self) -> Mapping[str, Any]:
        if self._env is None:
            raise ReplayVerificationError("VM 尚未 restore")
        observation = self._env._get_obs()
        self._capture_screenshot(observation, "takeover_observation.png")
        return observation

    def close(self) -> None:
        if self._env is None:
            return
        close = getattr(self._env, "close", None)
        if callable(close):
            close()
        self._env = None
