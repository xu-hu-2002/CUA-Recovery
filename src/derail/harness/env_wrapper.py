from __future__ import annotations

from typing import Any, Callable, Dict, List, Mapping, Optional, Tuple

from derail.harness.control_client import DerailControlClient
from derail.harness.trace_builder import ActionRecord, TraceBuilder

TERMINAL = {"DONE": "declared_complete", "FAIL": "declared_infeasible"}


class DerailEnvWrapper:
    def __init__(
        self,
        env: Any,
        control: DerailControlClient,
        builder_factory: Callable[[], TraceBuilder],
        a11y_provider: Optional[Callable[[], Optional[str]]] = None,
        screenshot_sink: Optional[Callable[[int, bytes], None]] = None,
    ) -> None:
        self._env = env
        self._control = control
        self._builder_factory = builder_factory
        self._a11y_provider = a11y_provider
        self._screenshot_sink = screenshot_sink
        self.builder: Optional[TraceBuilder] = None
        self._action_index = -1
        self._last_seq: Dict[str, int] = {}
        self._flags = {"declared_complete": False, "declared_infeasible": False}
        self._pending_thought: Optional[str] = None

    def __getattr__(self, name: str) -> Any:
        return getattr(self._env, name)

    def note_thought(self, thought: Optional[str]) -> None:
        self._pending_thought = thought

    def reset(self, task_config: Optional[Dict] = None, soft: bool = False) -> Dict[str, Any]:
        obs = self._env.reset(task_config=task_config, soft=soft)
        self.builder = self._builder_factory()
        self._control.set_cursor(-1)
        self._last_seq = self._control.latest_seq()
        self.builder.start(self._last_seq)
        self._action_index = -1
        self._flags = {"declared_complete": False, "declared_infeasible": False}
        return obs

    def _begin(self) -> int:
        self._action_index += 1
        self._control.set_cursor(self._action_index)
        return self._action_index

    def _ledgers(self, action_index: int) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        rows = self._control.changelog(self._last_seq)
        for row in rows:
            self._last_seq[row["db"]] = max(self._last_seq.get(row["db"], 0), int(row["seq"]))
        delta = [row for row in rows if int(row.get("action_index", -1)) == action_index]
        records = self._control.observations(action_index)
        return delta, records

    def _record(
        self, action_index: int, record: ActionRecord, obs: Optional[Mapping[str, Any]]
    ) -> None:
        assert self.builder is not None, "reset() must run before actions"
        delta, records = self._ledgers(action_index)
        screenshot = obs.get("screenshot") if obs else None
        if isinstance(screenshot, (bytes, bytearray)) and self._screenshot_sink is not None:
            self._screenshot_sink(action_index, bytes(screenshot))
        a11y = None
        if obs and obs.get("accessibility_tree"):
            a11y = str(obs["accessibility_tree"])
        elif self._a11y_provider is not None:
            a11y = self._a11y_provider()
        self.builder.record_step(
            action_index,
            record,
            delta=delta,
            trace_records=records,
            screenshot=screenshot if isinstance(screenshot, (bytes, bytearray)) else None,
            a11y_text=a11y,
        )

    def step(self, action: Any, pause: float = 2.0) -> Tuple[Dict, float, bool, Dict]:
        index = self._begin()
        thought, self._pending_thought = self._pending_thought, None
        obs, reward, done, info = self._env.step(action, pause)
        if isinstance(action, str) and action.strip().upper() in TERMINAL:
            self._flags[TERMINAL[action.strip().upper()]] = True
            modality = "control"
        else:
            modality = "gui"
        self._record(index, ActionRecord(raw=action, modality=modality, thought=thought), obs)
        return obs, reward, done, info

    def _execute_shell(self, command: str) -> Dict:
        index = self._begin()
        thought, self._pending_thought = self._pending_thought, None
        result = self._env._execute_shell(command)
        output = str(result.get("output", ""))[:4000] if isinstance(result, Mapping) else None
        self._record(
            index,
            ActionRecord(raw=command, modality="cli", thought=thought, shell_output=output),
            None,
        )
        return result

    def finish(
        self,
        *,
        step_budget_reached: bool,
        final_verifier: Optional[bool] = None,
        verifier_kind: Optional[str] = None,
        provenance: Optional[Mapping[str, Any]] = None,
    ) -> Dict[str, Any]:
        assert self.builder is not None, "reset() must run before finish"
        return self.builder.finish(
            declared_complete=self._flags["declared_complete"],
            declared_infeasible=self._flags["declared_infeasible"],
            budget_exhausted=step_budget_reached and not any(self._flags.values()),
            final_verifier=final_verifier,
            verifier_kind=verifier_kind,
            provenance=provenance,
        )

    @property
    def actions_taken(self) -> int:
        return self._action_index + 1
