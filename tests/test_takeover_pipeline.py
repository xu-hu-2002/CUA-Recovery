"""Annotation-bound takeover orchestration tests (no live VM or model)."""

import hashlib
import json
import os
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

from derail.canonical.actions import ClickAction, SequenceAction, ShellAction
from derail.canonical.trajectory import CanonicalStep
from derail.mypcbench.takeover_agent import PrefixTakeoverAgent
from derail.replay.mypcbench import MyPCBenchVMReplayBackend
from derail.replay.verification import ReplayVerificationError
from derail.takeover.diagnosis import (
    HumanDiagnosisEvidence,
    load_human_diagnosis_evidence,
)
from derail.takeover.selection import select_takeover_failures
from derail.takeover.protocol import load_takeover_config
from derail.takeover.source_logs import load_public_trajectory_log


def _load_stage_module():
    import importlib.util

    path = Path(__file__).resolve().parents[1] / "scripts/takeover/stage_inputs.py"
    spec = importlib.util.spec_from_file_location("stage_takeover_inputs", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class ProductionBundleDefaultsTests(unittest.TestCase):
    def test_portable_bundle_requires_claude_native_messages_sidecar(self):
        stage = _load_stage_module()
        with tempfile.TemporaryDirectory() as directory:
            repo = Path(directory)
            source_dir = repo / "raw/claude/task"
            source_dir.mkdir(parents=True)
            trajectory_source = source_dir / "traj.jsonl"
            trajectory_source.write_text("{}\n", encoding="utf-8")
            messages = source_dir / "messages.json"
            messages.write_text("[]", encoding="utf-8")
            canonical = repo / "build/canonical/task/trajectory.jsonl"
            canonical.parent.mkdir(parents=True)
            canonical.write_text(
                json.dumps({
                    "source_agent": "claude_opus_4_8",
                    "source_record_uri": f"{trajectory_source}#line=1",
                }) + "\n",
                encoding="utf-8",
            )
            supporting = []
            for name in ("normalization_report.json", "task_config.json", "annotation.json", "rubric.json"):
                path = repo / name
                path.write_text("{}", encoding="utf-8")
                supporting.append(path)
            selection = repo / "selection.json"
            selection.write_text(json.dumps({"included": [{
                "canonical_trajectory_uri": str(canonical),
                "normalization_report_uri": str(supporting[0]),
                "task_config_uri": str(supporting[1]),
                "annotation_uri": str(supporting[2]),
                "rubric_review_uri": str(supporting[3]),
            }]}), encoding="utf-8")
            config = {"experiments": [{
                "selection_manifest": "selection.json",
                "build_dir": "build",
            }]}
            repaired = repo / "build/canonical_repaired/task/trajectory.jsonl"
            repaired.parent.mkdir(parents=True)
            repaired.write_text(canonical.read_text(encoding="utf-8"), encoding="utf-8")
            (source_dir / "state_probes.jsonl").write_text("{}\n", encoding="utf-8")

            required, missing = stage.selected_files(config, repo, load_takeover_config())

            self.assertIn(messages.resolve(), required)
            self.assertIn(repaired.resolve(), required)
            self.assertIn((source_dir / "state_probes.jsonl").resolve(), required)
            self.assertFalse(missing)

    def test_takeover_preflight_uses_the_same_external_shard_range_as_workers(self):
        repo = Path(__file__).resolve().parents[1]
        source = (repo / "scripts/takeover/run.sh").read_text()
        self.assertIn('--shard-count "$SHARD_COUNT"', source)
        self.assertIn('--shard-offset "$SHARD_OFFSET"', source)
        self.assertIn('--shard-workers "$NUM_WORKERS"', source)
        self.assertIn("--overflow-policy exclude", source)
        self.assertIn("PROTOCOL_EXCLUSION:", source)
        self.assertIn('reason=%s report=%s', source)
        self.assertIn('"token_overflow"', source)
        self.assertIn('--takeover-config "$TAKEOVER_CONFIG"', source)

    def test_evocua_takeover_preflight_has_no_native_request_api(self):
        repo = Path(__file__).resolve().parents[1]
        preflight = (repo / "scripts/takeover/preflight_history.py").read_text()
        self.assertIn('args.target_agent == "evocua_32b"', preflight)
        self.assertIn('"EVOCUA_MODEL"', preflight)
        self.assertIn("EvoCUA target has no native request preflight API", preflight)


class TakeoverDiagnosisTests(unittest.TestCase):
    def test_diagnosis_is_only_the_human_evidence_block_and_is_hash_bound(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "label.json"
            raw = {
                "annotation_id": "traj-1__human",
                "trajectory_id": "traj-1",
                "annotator_id": "human",
                "annotator_role": "human",
                "root_cause_action_index": 3,
                "source_trajectory_sha256": "a" * 64,
                "rationale": (
                    "Summary that must not enter the diagnosed prompt.\n\n"
                    "Error type evidence:\n- wrong_target [Action #3]: clicked Alice"
                ),
            }
            path.write_text(json.dumps(raw), encoding="utf-8")
            result = load_human_diagnosis_evidence(
                path,
                expected_trajectory_id="traj-1",
                expected_source_trajectory_sha256="a" * 64,
                maximum_action_index=4,
            )
            self.assertEqual(
                result.evidence, "- wrong_target [Action #3]: clicked Alice"
            )
            self.assertNotIn("Summary", result.evidence)
            self.assertEqual(result.annotation_sha256, hashlib.sha256(path.read_bytes()).hexdigest())

    def test_annotation_binding_mismatch_fails(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "label.json"
            path.write_text(
                json.dumps(
                    {
                        "annotation_id": "a",
                        "trajectory_id": "other",
                        "annotator_id": "h",
                        "annotator_role": "human",
                        "root_cause_action_index": 0,
                        "source_trajectory_sha256": "b" * 64,
                        "rationale": "Error type evidence:\n- evidence",
                    }
                ),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "does not match"):
                load_human_diagnosis_evidence(path, expected_trajectory_id="traj")


class PublicTrajectoryLogTests(unittest.TestCase):
    def test_strips_source_reasoning_and_keeps_actions(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "traj.jsonl"
            path.write_text(
                json.dumps(
                    {
                        "step_num": 1,
                        "action": "WAIT",
                        "response": (
                            "private visual reasoning and plan\n</think>\n\n"
                            "Action: wait for the application"
                        ),
                        "reward": 0.0,
                        "done": False,
                        "info": {"status": "running", "reasoning": "private metadata"},
                        "agent_metadata": {
                            "tool_messages": [{"name": "bash", "content": "public output"}],
                            "analysis": "private adapter analysis",
                        },
                        "screenshot_file": "",
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            step = CanonicalStep(
                step_id=0,
                action=ClickAction(kind="click", x_px=10, y_px=20),
                observation_before_sha256="a" * 64,
                source_agent="evocua_32b",
                source_step_id=0,
                source_record_uri=f"{path.resolve()}#line=1",
            )
            rendered = load_public_trajectory_log(step)
            self.assertNotIn("private visual reasoning", rendered)
            self.assertNotIn("private metadata", rendered)
            self.assertNotIn("private adapter analysis", rendered)
            self.assertNotIn("</think>", rendered)
            self.assertIn("Action: wait for the application", rendered)
            self.assertIn("public output", rendered)
            self.assertIn('"source_screenshot_available":false', rendered)

    def test_explicit_path_mapping_makes_a_staged_source_record_portable(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "traj.jsonl"
            path.write_text('{"step_num":1,"response":"Action: wait"}\n', encoding="utf-8")
            missing = "/original/host/collection/traj.jsonl"
            step = CanonicalStep(
                step_id=0,
                action=ClickAction(kind="click", x_px=10, y_px=20),
                observation_before_sha256="a" * 64,
                source_agent="kimi_k3",
                source_step_id=0,
                source_record_uri=f"{missing}#line=1",
            )
            with patch.dict(
                "os.environ", {"DERAIL_PATH_REMAP_JSON": json.dumps({missing: str(path)})}
            ):
                self.assertIn("Action: wait", load_public_trajectory_log(step))

    def test_path_mapping_file_avoids_large_environment_values(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / "traj.jsonl"
            path.write_text('{"step_num":1,"response":"Action: wait"}\n', encoding="utf-8")
            mappings_file = root / "mappings.json"
            missing = "/original/host/collection/traj.jsonl"
            mappings_file.write_text(json.dumps({missing: str(path)}), encoding="utf-8")
            step = CanonicalStep(
                step_id=0,
                action=ClickAction(kind="click", x_px=10, y_px=20),
                observation_before_sha256="a" * 64,
                source_agent="kimi_k3",
                source_step_id=0,
                source_record_uri=f"{missing}#line=1",
            )
            with patch.dict(
                "os.environ",
                {"DERAIL_PATH_REMAP_FILE": str(mappings_file)},
                clear=True,
            ):
                self.assertIn("Action: wait", load_public_trajectory_log(step))


class TakeoverSelectionTests(unittest.TestCase):
    def test_cross_annotator_wrong_rollout_does_not_veto_a_failure(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            build = root / "build"
            labels = root / "labels"
            (labels / "rubric_scores").mkdir(parents=True)
            (labels / "rollout_flags").mkdir()
            (build / "build_manifest.json").parent.mkdir(parents=True, exist_ok=True)
            (build / "build_manifest.json").write_text(
                json.dumps({"normalization_reports": []}), encoding="utf-8"
            )

            for trajectory_id in (
                "valid-failure",
                "cross-annotator-conflict",
                "wrong-rollout-only",
                "success",
            ):
                canonical = build / "canonical" / trajectory_id
                canonical.mkdir(parents=True)
                step = CanonicalStep(
                    step_id=0,
                    action=ClickAction(kind="click", x_px=10, y_px=20),
                    observation_before_sha256="a" * 64,
                    source_agent="evocua_32b",
                )
                trajectory_path = canonical / "trajectory.jsonl"
                trajectory_path.write_text(json.dumps(step.to_dict()) + "\n")
                source_sha = hashlib.sha256((trajectory_id + "-source").encode()).hexdigest()
                (canonical / "normalization_report.json").write_text(
                    json.dumps(
                        {
                            "trajectory_id": trajectory_id,
                            "source_trajectory_sha256": source_sha,
                            "canonical_trajectory_sha256": hashlib.sha256(
                                trajectory_path.read_bytes()
                            ).hexdigest(),
                        }
                    )
                )
                (canonical / "task_config.json").write_text(
                    json.dumps({"id": trajectory_id, "instruction": "Finish"})
                )
                review_name = f"{trajectory_id}__alice.json"
                (labels / "rubric_scores" / review_name).write_text(
                    json.dumps(
                        {
                            "trajectory_id": trajectory_id,
                            "source_trajectory_sha256": source_sha,
                            "reviewer_id": "alice",
                            "reviewer_role": "human",
                            "task_success": trajectory_id == "success",
                        }
                    )
                )
                if trajectory_id in {"valid-failure", "cross-annotator-conflict"}:
                    (labels / review_name).write_text(
                        json.dumps(
                            {
                                "annotation_id": review_name[:-5],
                                "trajectory_id": trajectory_id,
                                "source_trajectory_sha256": source_sha,
                                "annotator_id": "alice",
                                "annotator_role": "human",
                                "root_cause_action_index": 0,
                                "error_horizon_actions": 0,
                                "identifiable_at_action_index": 0,
                                "rationale": "Error type evidence:\n- [Action #0]: wrong value",
                            }
                        )
                    )
                if trajectory_id in {"cross-annotator-conflict", "wrong-rollout-only"}:
                    (labels / "rollout_flags" / f"{trajectory_id}__bob.json").write_text(
                        json.dumps(
                            {
                                "trajectory_id": trajectory_id,
                                "source_trajectory_sha256": source_sha,
                                "annotator_id": "bob",
                                "annotator_role": "human",
                                "rollout_status": "needs_rerun",
                                "reason": "wrong_rollout",
                            }
                        )
                    )

            rejected_id = "hash-mismatched-source"
            (labels / f"{rejected_id}__alice.json").write_text(
                json.dumps(
                    {
                        "trajectory_id": rejected_id,
                        "annotator_id": "alice",
                        "annotator_role": "human",
                    }
                ),
                encoding="utf-8",
            )
            (build / "build_manifest.json").write_text(
                json.dumps(
                    {
                        "normalization_reports": [
                            {
                                "trajectory_id": rejected_id,
                                "normalization_complete": False,
                                "uri": "",
                                "rejection_reason": "source trajectory sha256 mismatch",
                            }
                        ]
                    }
                ),
                encoding="utf-8",
            )

            result = select_takeover_failures(
                build_dir=build,
                human_labels_dir=labels,
                source_agent="evocua_32b",
                annotator_id="alice",
                depths=(0, 5),
            )
            self.assertEqual(
                [item["trajectory_id"] for item in result["included"]],
                ["cross-annotator-conflict", "valid-failure"],
            )
            self.assertEqual(result["excluded_reason_counts"]["wrong_rollout"], 1)
            self.assertEqual(
                result["excluded_reason_counts"]["no_human_failure_annotation"], 1
            )
            self.assertEqual(
                result["excluded_reason_counts"]["canonicalization_rejected"], 1
            )
            conflict = result["included"][0]
            self.assertEqual(conflict["available_depths"], [0])
            self.assertEqual(conflict["unavailable_depths"], [5])
            self.assertEqual(
                conflict["ignored_cross_annotator_wrong_rollout_flags"][0][
                    "annotator_id"
                ],
                "bob",
            )
            frozen = select_takeover_failures(
                build_dir=build,
                human_labels_dir=labels,
                source_agent="evocua_32b",
                annotator_id="alice",
                trajectory_id_allowlist=("valid-failure",),
                depths=(0, 5),
            )
            self.assertEqual(
                [item["trajectory_id"] for item in frozen["included"]],
                ["valid-failure"],
            )
            self.assertEqual(
                frozen["selection_policy"]["trajectory_id_allowlist"],
                ["valid-failure"],
            )


class TakeoverAgentTests(unittest.TestCase):
    def test_first_predict_replays_through_root_and_seeds_diagnosed_history(self):
        class Target:
            agent_metadata = {}
            last_trajectory_tool_messages = []
            total_usage = {}
            messages = []

            def reset(self, _logger=None):
                pass

            def seed_native_history(self, instruction, history, **kwargs):
                self.seeded = (instruction, history, kwargs)
                self.messages = history

            def predict(self, instruction, observation):
                self.predicted = (instruction, observation)
                return "next", ["WAIT"]

        class Replay:
            def __init__(self, **_kwargs):
                self.executed = []
                self.initial_state_alignment = {"kind": "test_alignment"}

            def adopt_current_snapshot(self, *_args, **_kwargs):
                pass

            def execute(self, action):
                index = len(self.executed)
                self.executed.append(action)
                return {
                    "result": "ok",
                    "observation_before_uri": f"/tmp/before-{index}.png",
                    "observation_before_sha256": str(index) * 64,
                    "observation_after_uri": f"/tmp/after-{index}.png",
                    "observation_after_sha256": str(index + 1) * 64,
                }

            def observe(self):
                return {"screenshot": b"current"}

        steps = tuple(
            CanonicalStep(
                step_id=index,
                action=ClickAction(kind="click", x_px=10 + index, y_px=20),
                observation_before_sha256="a" * 64,
                source_agent="evocua_32b",
            )
            for index in range(3)
        )
        diagnosis = HumanDiagnosisEvidence(
            annotation_id="ann",
            trajectory_id="traj",
            annotator_id="human",
            root_cause_action_index=1,
            evidence="[Action #1]: selected the wrong item",
            annotation_uri="/tmp/ann.json",
            annotation_sha256="a" * 64,
            source_trajectory_sha256="b" * 64,
        )
        with tempfile.TemporaryDirectory() as directory, patch(
            "derail.mypcbench.takeover_agent.MyPCBenchVMReplayBackend", Replay
        ):
            target = Target()
            agent = PrefixTakeoverAgent(
                target_agent=target,
                environment=object(),
                task_config={"instruction": "Finish"},
                canonical_steps=steps,
                diagnosis=diagnosis,
                condition="diagnosed",
                source_agent="evocua_32b",
                target_agent_id="kimi_k3",
                qcow2_path=Path("/tmp/base.qcow2"),
                qcow2_sha256="c" * 64,
                artifact_dir=Path(directory),
                depth=1,
            )
            response, actions = agent.predict("Finish", {"screenshot": b"stale"})
            self.assertEqual((response, actions), ("next", ["WAIT"]))
            _instruction, history, kwargs = target.seeded
            self.assertEqual(kwargs["condition"], "diagnosed")
            self.assertEqual(kwargs["root_cause_action_index"], 1)
            self.assertEqual(kwargs["diagnosis"], diagnosis.evidence)
            assistant_turns = [m for m in history if m["role"] == "assistant"]
            self.assertEqual(len(assistant_turns), 3)
            manifest = json.loads((Path(directory) / "takeover_manifest.json").read_text())
            self.assertEqual(manifest["depth"], 1)
            self.assertEqual(manifest["prefix_end_action_index"], 2)
            self.assertEqual(manifest["first_target_action_index"], 3)


class HybridReplayTests(unittest.TestCase):
    def test_seeded_firefox_focus_uses_window_class_with_retries(self):
        class Environment:
            def __init__(self):
                self.commands = []

            def _execute_shell(self, command):
                self.commands.append(command)
                return {"returncode": 0, "output": "", "error": ""}

            def _execute_pyautogui(self, command):
                return {"returncode": 0, "output": "", "error": ""}

        with tempfile.TemporaryDirectory() as directory:
            backend = MyPCBenchVMReplayBackend(
                env_factory=lambda: None,
                task_config={},
                state_probe_commands={"x": "true"},
                evidence_dir=Path(directory),
            )
            env = Environment()
            backend._refresh_seeded_firefox_tabs(env)
            self.assertIn("wmctrl -lx", env.commands[0])
            self.assertIn("/(firefox|navigator)/", env.commands[0])
            self.assertIn("find /run/user", env.commands[0])
            self.assertIn("/home/oai/.Xauthority", env.commands[0])
            self.assertIn("for attempt in $(seq 1 60)", env.commands[0])
            self.assertIn("xdotool windowactivate --sync", env.commands[0])
            self.assertIn("nohup firefox", env.commands[0])
            self.assertEqual(
                backend.initial_state_alignment["focus_strategy"],
                "wmctrl_readiness_retry_v2",
            )

    def test_seeded_firefox_focus_failure_keeps_diagnostics(self):
        class Environment:
            def _execute_shell(self, _command):
                return {"returncode": 1, "output": "", "error": "no X11 windows"}

            def _execute_pyautogui(self, _command):
                raise AssertionError("refresh must not run without a focused window")

        with tempfile.TemporaryDirectory() as directory:
            backend = MyPCBenchVMReplayBackend(
                env_factory=lambda: None,
                task_config={},
                state_probe_commands={"x": "true"},
                evidence_dir=Path(directory),
            )
            with self.assertRaisesRegex(ReplayVerificationError, "no X11 windows"):
                backend._refresh_seeded_firefox_tabs(Environment())

    def test_mixed_sequence_executes_shell_and_gui_in_order(self):
        class Environment:
            def __init__(self):
                self.events = []

            def _execute_shell(self, command):
                self.events.append(("shell", command))
                return {"returncode": 0, "output": "ok", "error": ""}

            def step(self, action, pause):
                self.events.append(("gui", action, pause))
                return {"screenshot": b"\x89PNG\r\n\x1a\npost"}, 0.0, False, {}

            def _get_obs(self):
                return {"screenshot": b"\x89PNG\r\n\x1a\npost"}

        with tempfile.TemporaryDirectory() as directory:
            backend = MyPCBenchVMReplayBackend(
                env_factory=lambda: None,
                task_config={},
                state_probe_commands={"x": "true"},
                evidence_dir=Path(directory),
            )
            env = Environment()
            backend._env = env
            backend._last_screenshot_sha256 = "a" * 64
            backend._last_screenshot_uri = "/tmp/before.png"
            record = backend.execute(
                SequenceAction(
                    kind="sequence",
                    actions=(
                        ShellAction(kind="shell", commands=("pwd",)),
                        ClickAction(kind="click", x_px=10, y_px=20),
                    ),
                )
            )
            self.assertEqual(env.events[0], ("shell", "pwd"))
            self.assertEqual(env.events[1][0], "gui")
            self.assertIn("shell_results", json.loads(record["result"]))


if __name__ == "__main__":
    unittest.main()
