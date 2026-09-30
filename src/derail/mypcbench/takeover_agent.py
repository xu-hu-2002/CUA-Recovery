"""Agent wrapper that replays a failed prefix before the target's first turn."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from derail.adapters import HistoryStep, create_native_history_adapter
from derail.canonical.trajectory import CanonicalStep
from derail.derived.layout import atomic_write_json
from derail.replay.mypcbench import MyPCBenchVMReplayBackend
from derail.replay.verification import compare_state_fingerprints
from derail.rollout.state_probe import EnvironmentHooks, run_determinism
from derail.takeover.diagnosis import HumanDiagnosisEvidence
from derail.takeover.history import REASONING_POLICIES, build_native_history
from derail.takeover.protocol import (
    ProtocolExclusion,
    recorded_state_fingerprint,
    takeover_prefix,
)
from derail.takeover.source_logs import load_trajectory_log


class PrefixTakeoverAgent:
    """Replay the (repaired) prefix through ``root+depth``, verify, inject history, delegate.

    MyPCBench calls ``reset`` before its normal react loop.  Replay therefore
    happens lazily on the first ``predict`` so the prefix and all target steps
    share exactly the same VM session and task reset.  ``replay_verification`` is the
    ``replay_verification`` section of the takeover config; when enabled the replayed
    state must match the fingerprint recorded by the source rollout at the same step.
    ``environment_hooks`` is the takeover config's ``environment_config`` (the source
    rollout's): its determinism commands run before the replay, and its state probes and
    ``state_probe_file`` define both sides of the comparison.  ``hinted`` injects ``hint``
    verbatim as the takeover prompt, where ``diagnosed`` puts the human diagnosis.
    """

    def __init__(
        self,
        *,
        target_agent: Any,
        environment: Any,
        task_config: Mapping[str, Any],
        canonical_steps: Sequence[CanonicalStep],
        diagnosis: HumanDiagnosisEvidence,
        condition: str,
        source_agent: str,
        target_agent_id: str,
        qcow2_path: Path,
        qcow2_sha256: str,
        artifact_dir: Path,
        qcow2_hash_preverified: bool = False,
        depth: int = 0,
        replay_sleep_after_s: float = 1.0,
        strip_reasoning: bool = True,
        evocua_cross_agent: bool = True,
        replay_verification: Mapping[str, Any] | None = None,
        prefix_info: Mapping[str, Any] | None = None,
        environment_hooks: EnvironmentHooks | None = None,
        hint: str = "",
    ) -> None:
        if condition not in {"unaware", "notified", "diagnosed", "hinted"}:
            raise ValueError(f"unknown takeover condition: {condition!r}")
        if (condition == "hinted") != bool(hint.strip()):
            raise ValueError("a takeover hint is given exactly for the hinted condition")
        if replay_verification and replay_verification.get("enabled") and not (
            environment_hooks and environment_hooks.probe_commands
        ):
            raise ValueError("replay verification needs the environment config's state probes")
        if not canonical_steps:
            raise ValueError("canonical trajectory is empty")
        root = diagnosis.root_cause_action_index
        if isinstance(depth, bool) or not isinstance(depth, int) or depth < 0:
            raise ValueError("takeover depth must be a non-negative integer")
        prefix_end = root + depth
        # A repaired prefix may skip removed actions before the root cause.
        prefix = takeover_prefix(canonical_steps, root, depth)
        observed_sources = {step.source_agent for step in prefix}
        if observed_sources != {source_agent}:
            raise ValueError(
                f"source agent mismatch: requested {source_agent!r}, "
                f"prefix contains {sorted(observed_sources)!r}"
            )
        if (
            target_agent_id == "evocua_32b"
            and source_agent != target_agent_id
            and not evocua_cross_agent
        ):
            raise ValueError(
                "evocua_32b cross-agent history is disabled (history.evocua_cross_agent)"
            )
        if not callable(getattr(target_agent, "seed_native_history", None)):
            raise TypeError(
                f"target agent {target_agent_id!r} has no native-history injection API"
            )

        self._target = target_agent
        self._environment = environment
        self._task_config = dict(task_config)
        self._prefix = prefix
        self._steps = tuple(canonical_steps)
        self._strip_reasoning = strip_reasoning
        self._replay_verification = dict(replay_verification or {})
        self._environment_hooks = environment_hooks
        self._hint = hint
        self._prefix_info = dict(prefix_info or {})
        self._diagnosis = diagnosis
        self._depth = depth
        self._prefix_end_action_index = prefix_end
        self._condition = condition
        self._source_agent = source_agent
        self._target_agent_id = target_agent_id
        self._qcow2_path = qcow2_path.expanduser().resolve()
        self._qcow2_sha256 = qcow2_sha256
        self._qcow2_hash_preverified = qcow2_hash_preverified
        self._artifact_dir = artifact_dir.resolve()
        self._replay_sleep_after_s = replay_sleep_after_s
        self._started = False
        self._takeover_metadata: dict[str, Any] = {}

    def reset(self, logger: Any = None, vm_ip: Any = None) -> None:
        del vm_ip
        try:
            self._target.reset(logger)
        except TypeError:
            self._target.reset()
        self._started = False
        self._takeover_metadata = {}

    @property
    def messages(self) -> Any:
        return getattr(self._target, "messages", None)

    @property
    def total_usage(self) -> Any:
        return getattr(self._target, "total_usage", None)

    @property
    def last_trajectory_tool_messages(self) -> Any:
        return getattr(self._target, "last_trajectory_tool_messages", None)

    @property
    def agent_metadata(self) -> dict[str, Any]:
        raw = getattr(self._target, "agent_metadata", None)
        target_metadata = dict(raw) if isinstance(raw, Mapping) else {}
        if self._takeover_metadata:
            target_metadata["takeover"] = dict(self._takeover_metadata)
        return target_metadata

    def _start_takeover(self, instruction: str) -> Mapping[str, Any]:
        expected_instruction = str(self._task_config.get("instruction", ""))
        if instruction != expected_instruction:
            raise ValueError("runner instruction differs from the canonical task instruction")

        replay_dir = self._artifact_dir / "prefix_replay"
        verify = bool(self._replay_verification.get("enabled"))
        backend = MyPCBenchVMReplayBackend(
            env_factory=lambda: self._environment,
            task_config=self._task_config,
            state_probe_commands=(
                self._environment_hooks.probe_commands
                if verify
                else {"takeover_runner": "printf takeover-ready"}
            ),
            evidence_dir=replay_dir,
            sleep_after_s=self._replay_sleep_after_s,
        )
        # run_single_example has just reset this exact pinned image and applied
        # the task config. Adopt that verified zero-action state instead of
        # paying for a redundant second QEMU boot.
        backend.adopt_current_snapshot(
            str(self._qcow2_path),
            self._qcow2_sha256,
            hash_preverified=self._qcow2_hash_preverified,
        )
        # Same guest setup as the source rollout (auto-updates / notification daemons off).
        determinism = (
            run_determinism(
                lambda command: self._environment._execute_shell(command),
                self._environment_hooks,
            )
            if self._environment_hooks
            else {}
        )
        atomic_write_json(
            self._artifact_dir / "prefix_replay_setup.json",
            {**dict(backend.initial_state_alignment), "determinism": determinism},
        )

        history_steps = []
        replay_log = []
        for step in self._prefix:
            record = dict(backend.execute(step.action))
            replay_log.append(
                {"action_index_global": step.action_index_global, **record}
            )
            history_steps.append(
                HistoryStep(
                    step_id=step.action_index_global,
                    turn_index=step.turn_index or 0,
                    action_index_within_turn=step.action_index_within_turn,
                    observation_image_url=str(record["observation_before_uri"]),
                    observation_after_image_url=str(record["observation_after_uri"]),
                    observation_sha256=str(record["observation_before_sha256"]),
                    action=step.action,
                    tool_result=str(record["result"]),
                    trajectory_log=load_trajectory_log(
                        step, strip_reasoning=self._strip_reasoning
                    ),
                )
            )
        atomic_write_json(self._artifact_dir / "prefix_replay_log.json", replay_log)
        state_verification = self._verify_replayed_state(backend) if verify else None

        adapter_kwargs: dict[str, Any] = {}
        system_prompt = getattr(self._target, "native_history_system_prompt", "")
        if system_prompt:
            adapter_kwargs["system_prompt"] = system_prompt
        if self._target_agent_id == "evocua_32b":
            adapter_kwargs["source_agent"] = self._source_agent
        adapter = create_native_history_adapter(
            self._target_agent_id, instruction, **adapter_kwargs
        )
        native_history = build_native_history(adapter, history_steps)
        encoded = json.dumps(
            native_history, ensure_ascii=False, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
        history_sha256 = hashlib.sha256(encoded).hexdigest()

        seed_kwargs: dict[str, Any] = {"condition": self._condition}
        if self._condition == "diagnosed":
            seed_kwargs.update(
                diagnosis=self._diagnosis.evidence,
                root_cause_action_index=self._diagnosis.root_cause_action_index,
            )
        elif self._condition == "hinted":
            seed_kwargs.update(hint=self._hint)
        self._target.seed_native_history(instruction, native_history, **seed_kwargs)

        atomic_write_json(
            self._artifact_dir / "native_history.json",
            {
                "source_agent": self._source_agent,
                "target_agent": self._target_agent_id,
                "condition": self._condition,
                "root_cause_action_index": self._diagnosis.root_cause_action_index,
                "depth": self._depth,
                "prefix_end_action_index": self._prefix_end_action_index,
                "action_indices": [step.action_index_global for step in self._prefix],
                "messages_sha256": history_sha256,
                "messages": native_history,
                "private_reasoning_policy": REASONING_POLICIES[self._strip_reasoning],
            },
        )
        self._takeover_metadata = {
            "source_agent": self._source_agent,
            "target_agent": self._target_agent_id,
            "condition": self._condition,
            "depth": self._depth,
            "prefix_end_action_index": self._prefix_end_action_index,
            "first_target_action_index": self._prefix_end_action_index + 1,
            "native_history_sha256": history_sha256,
            "human_annotation": self._diagnosis.to_manifest_dict(),
            "source_judge_authority": "human_annotation",
            "diagnosis_visible_to_target": self._condition == "diagnosed",
            **({"takeover_hint": self._hint} if self._condition == "hinted" else {}),
            "executed_action_indices": [step.action_index_global for step in self._prefix],
            **self._prefix_info,
            "replay_state_verification": (
                state_verification["status"] if state_verification else "disabled"
            ),
        }
        atomic_write_json(
            self._artifact_dir / "takeover_manifest.json", self._takeover_metadata
        )
        self._started = True
        return backend.observe()

    def _verify_replayed_state(self, backend: MyPCBenchVMReplayBackend) -> dict[str, Any]:
        """Gate the takeover on the replayed state matching the recorded one (C:29-31)."""

        config = self._replay_verification
        expected, missing_reason = recorded_state_fingerprint(
            self._steps, self._prefix_end_action_index, self._environment_hooks.probe_file
        )
        observed = backend.fingerprint()
        if expected is None:
            record: dict[str, Any] = {
                "status": "expected_missing",
                "reason": missing_reason,
                "observed": observed.to_dict(),
            }
            reject = config["on_missing_expected"] == "reject"
            exclusion = "replay_unverifiable"
        else:
            record = compare_state_fingerprints(
                expected, observed, config.get("ignore_components", ())
            )
            record["status"] = "match" if record["state_fingerprint_match"] else "mismatch"
            reject = record["status"] == "mismatch" and config["on_mismatch"] == "reject"
            exclusion = "replay_mismatch"
        record.update(
            prefix_end_action_index=self._prefix_end_action_index,
            accepted=not reject,
            vm_session_id=backend.replay_session_id,
        )
        atomic_write_json(self._artifact_dir / "prefix_state_verification.json", record)
        if reject:
            atomic_write_json(
                self._artifact_dir / "protocol_exclusion.json",
                {"reason": exclusion, "detail": record["status"], **self._prefix_info},
            )
            raise ProtocolExclusion(exclusion, record.get("reason") or record["status"])
        return record

    def predict(self, instruction: str, observation: Mapping[str, Any]):
        if not self._started:
            observation = self._start_takeover(instruction)
        return self._target.predict(instruction, observation)
