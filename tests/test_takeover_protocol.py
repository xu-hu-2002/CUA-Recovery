"""Paper takeover protocol: repaired prefix, replayed-state gate, history policy, GPT/EvoCUA."""

import base64
import json
from dataclasses import replace
import tempfile
import unittest
from io import BytesIO
from pathlib import Path
from unittest.mock import patch

from PIL import Image

from derail.adapters import HistoryStep, create_native_history_adapter
from derail.canonical.actions import ClickAction, SequenceAction, ShellAction, TypeAction
from derail.canonical.trajectory import CanonicalStep
from derail.mypcbench.openai_takeover import OpenAITakeoverTarget
from derail.mypcbench.takeover_agent import PrefixTakeoverAgent
from derail.replay.verification import StateFingerprint, compare_state_fingerprints
from derail.rollout.state_probe import EnvironmentHooks
from derail.takeover.diagnosis import HumanDiagnosisEvidence
from derail.takeover.history import build_native_history
from derail.takeover.protocol import (
    ProtocolExclusion,
    load_takeover_config,
    load_takeover_steps,
    recorded_state_fingerprint,
    takeover_prefix,
)
from derail.takeover.source_logs import load_trajectory_log

A, B = "a" * 64, "b" * 64


def _png() -> str:
    buffer = BytesIO()
    Image.new("RGB", (1280, 800), (10, 10, 10)).save(buffer, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode("ascii")


def _write_steps(path: Path, indices, traj: Path, x_offset: int = 0) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = [
        CanonicalStep(
            step_id=index,
            action=ClickAction(kind="click", x_px=10 + index + x_offset, y_px=20),
            observation_before_sha256="0" * 64,
            source_agent="kimi_k3",
            source_record_uri=f"{traj}#line={index + 1}",
        ).to_dict()
        for index in indices
    ]
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


class ConfigTests(unittest.TestCase):
    def test_defaults_follow_the_paper(self):
        config = load_takeover_config()
        self.assertEqual(config["repeats"], 3)
        self.assertEqual(config["max_steps"], 100)
        self.assertEqual(config["timeout_seconds"], 3600)
        self.assertEqual(config["depths"], [0, 5, 10, 15, 20, 25])
        self.assertEqual(config["conditions"], ["unaware"])
        self.assertEqual(config["prefix"]["source"], "repaired")
        self.assertEqual(config["prefix"]["on_missing_repaired"], "error")
        self.assertTrue(config["history"]["strip_reasoning"])
        self.assertEqual(config["history"]["openai_context_screenshots"], 20)
        self.assertTrue(config["replay_verification"]["enabled"])
        self.assertEqual(config["replay_verification"]["on_mismatch"], "reject")


class RepairedPrefixTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        root = Path(self.tmp.name)
        self.traj = root / "raw/traj.jsonl"
        self.traj.parent.mkdir(parents=True)
        self.traj.write_text(
            "".join(json.dumps({"step_num": i + 1}) + "\n" for i in range(6)), encoding="utf-8"
        )
        self.canonical = root / "build/canonical/t1/trajectory.jsonl"
        _write_steps(self.canonical, range(6), self.traj)
        self.repaired = root / "build/canonical_repaired/t1/trajectory.jsonl"
        self.prefix = dict(load_takeover_config()["prefix"])

    def tearDown(self):
        self.tmp.cleanup()

    def test_repaired_prefix_keeps_original_indices_with_gaps(self):
        _write_steps(self.repaired, [0, 2, 3, 4, 5], self.traj)
        steps, info = load_takeover_steps(self.canonical, "t1", 3, self.prefix)
        self.assertEqual(info["prefix_source"], "repaired")
        self.assertEqual(info["removed_action_indices"], [1])
        prefix = takeover_prefix(steps, 3, 1)
        self.assertEqual([step.action_index_global for step in prefix], [0, 2, 3, 4])
        with self.assertRaisesRegex(ValueError, "unavailable"):
            takeover_prefix(steps, 3, 5)

    def test_repair_must_not_touch_the_root_cause_or_later(self):
        _write_steps(self.repaired, [0, 1, 2], self.traj)
        with open(self.repaired, "a", encoding="utf-8") as handle:
            other = CanonicalStep(
                step_id=3, action=ClickAction(kind="click", x_px=99, y_px=99),
                observation_before_sha256="0" * 64, source_agent="kimi_k3",
                source_record_uri=f"{self.traj}#line=4",
            )
            handle.write(json.dumps(other.to_dict()) + "\n")
        with self.assertRaisesRegex(ValueError, "root cause or later"):
            load_takeover_steps(self.canonical, "t1", 3, self.prefix)

    def test_missing_repaired_prefix_policies(self):
        with self.assertRaises(FileNotFoundError):
            load_takeover_steps(self.canonical, "t1", 3, self.prefix)
        with self.assertRaises(ProtocolExclusion) as caught:
            load_takeover_steps(
                self.canonical, "t1", 3, {**self.prefix, "on_missing_repaired": "skip"}
            )
        self.assertEqual(caught.exception.reason, "repaired_prefix_missing")
        steps, info = load_takeover_steps(
            self.canonical, "t1", 3, {**self.prefix, "on_missing_repaired": "fallback_original"}
        )
        self.assertEqual(info["prefix_source"], "original_fallback")
        self.assertEqual(len(steps), 6)


class ReplayStateTests(unittest.TestCase):
    def test_recorded_fingerprint_aligns_by_traj_index_and_turn_boundary(self):
        with tempfile.TemporaryDirectory() as directory:
            traj = Path(directory) / "traj.jsonl"
            first = StateFingerprint(components={"app_databases/a.sqlite": A}).to_dict()
            second = StateFingerprint(components={"app_databases/a.sqlite": B}).to_dict()
            # Rows 2 and 3 are two actions of one model turn (same step_num, one row each);
            # canonical steps 2 and 3 split the third row.
            traj.write_text(
                "".join(json.dumps({"step_num": n}) + "\n" for n in (1, 2, 2)), encoding="utf-8"
            )
            steps = [
                CanonicalStep(step_id=i, action=ClickAction(kind="click", x_px=1, y_px=1),
                              observation_before_sha256="0" * 64,
                              source_record_uri=f"{traj}#line={line}")
                for i, line in ((0, 1), (1, 2), (2, 3), (3, 3))
            ]
            probe_file = "state_probes.jsonl"
            (Path(directory) / probe_file).write_text("".join(json.dumps(r) + "\n" for r in (
                {"kind": "reset", "traj_index": -1, "state_fingerprint": second},
                {"kind": "step", "traj_index": 0, "state_fingerprint": second},
                {"kind": "step", "traj_index": 0, "state_fingerprint": first},  # last one wins
                {"kind": "step", "traj_index": 1, "state_fingerprint": second},
            )), encoding="utf-8")
            expected, reason = recorded_state_fingerprint(steps, 0, probe_file)
            self.assertEqual((expected.components, reason), (first["components"], ""))
            self.assertEqual(
                recorded_state_fingerprint(steps, 1, probe_file)[0].components,
                second["components"],
            )
            self.assertEqual(
                recorded_state_fingerprint(steps, 2, probe_file),
                (None, "takeover_point_inside_recorded_turn"),
            )
            self.assertEqual(
                recorded_state_fingerprint(steps, 3, probe_file)[1],
                "source_state_fingerprint_missing",
            )

    def test_component_comparison_tolerates_configured_components(self):
        expected = StateFingerprint(components={"db/a": A, "db/clock": A})
        observed = StateFingerprint(components={"db/a": A, "db/clock": B})
        self.assertFalse(compare_state_fingerprints(expected, observed)["state_fingerprint_match"])
        result = compare_state_fingerprints(expected, observed, ["db/clock"])
        self.assertTrue(result["state_fingerprint_match"])
        self.assertEqual(result["tolerated_components"], ["db/clock"])

    def _agent(self, directory: Path, observed: str):
        traj = directory / "traj.jsonl"
        recorded = StateFingerprint(components={"app_databases/a.sqlite": A}).to_dict()
        traj.write_text(
            "".join(
                json.dumps({"step_num": i + 1, "response": "Action: click"}) + "\n"
                for i in range(3)
            ),
            encoding="utf-8",
        )
        hooks = replace(
            EnvironmentHooks.from_config(Path(load_takeover_config()["environment_config"])),
            determinism_commands={},
        )
        (directory / hooks.probe_file).write_text(
            "".join(json.dumps({"traj_index": i, "state_fingerprint": recorded}) + "\n"
                    for i in range(3)),
            encoding="utf-8",
        )
        steps = tuple(
            CanonicalStep(step_id=i, action=ClickAction(kind="click", x_px=10 + i, y_px=20),
                          observation_before_sha256="0" * 64, source_agent="kimi_k3",
                          source_step_id=i, source_record_uri=f"{traj}#line={i + 1}")
            for i in range(3)
        )

        class Target:
            agent_metadata, messages = {}, []

            def reset(self, _logger=None):
                pass

            def seed_native_history(self, instruction, history, **kwargs):
                self.seeded = history

            def predict(self, instruction, observation):
                return "next", ["WAIT"]

        class Replay:
            replay_session_id = "session"
            initial_state_alignment = {}

            def __init__(self, **kwargs):
                self.probes = kwargs["state_probe_commands"]

            def adopt_current_snapshot(self, *_args, **_kwargs):
                pass

            def execute(self, _action):
                return {"result": "ok", "observation_before_uri": _png(),
                        "observation_before_sha256": A, "observation_after_uri": _png(),
                        "observation_after_sha256": A}

            def fingerprint(self):
                return StateFingerprint(components={"app_databases/a.sqlite": observed})

            def observe(self):
                return {"screenshot": b"current"}

        diagnosis = HumanDiagnosisEvidence(
            annotation_id="ann", trajectory_id="t", annotator_id="h", root_cause_action_index=1,
            evidence="[Action #1]: wrong", annotation_uri="/tmp/a.json",
            annotation_sha256=A, source_trajectory_sha256=B,
        )
        target = Target()
        agent = PrefixTakeoverAgent(
            target_agent=target, environment=object(), task_config={"instruction": "Finish"},
            canonical_steps=steps, diagnosis=diagnosis, condition="unaware",
            source_agent="kimi_k3", target_agent_id="qwen3_8_27b",
            qcow2_path=Path("/tmp/base.qcow2"), qcow2_sha256="c" * 64,
            artifact_dir=directory, depth=1,
            replay_verification=load_takeover_config()["replay_verification"],
            prefix_info={"prefix_source": "repaired"},
            environment_hooks=hooks,
        )
        return agent, target, Replay

    def test_mismatching_replay_is_rejected_before_takeover(self):
        with tempfile.TemporaryDirectory() as directory:
            agent, target, replay = self._agent(Path(directory), B)
            with patch("derail.mypcbench.takeover_agent.MyPCBenchVMReplayBackend", replay):
                with self.assertRaises(ProtocolExclusion) as caught:
                    agent.predict("Finish", {"screenshot": b"stale"})
            self.assertEqual(caught.exception.reason, "replay_mismatch")
            self.assertFalse(hasattr(target, "seeded"))
            exclusion = json.loads((Path(directory) / "protocol_exclusion.json").read_text())
            self.assertEqual(exclusion["reason"], "replay_mismatch")
            verification = json.loads(
                (Path(directory) / "prefix_state_verification.json").read_text()
            )
            self.assertEqual(verification["mismatched_components"], ["app_databases/a.sqlite"])

    def test_matching_replay_takes_over_with_real_state_probe(self):
        with tempfile.TemporaryDirectory() as directory:
            agent, target, replay = self._agent(Path(directory), A)
            with patch("derail.mypcbench.takeover_agent.MyPCBenchVMReplayBackend", replay):
                self.assertEqual(agent.predict("Finish", {"screenshot": b"x"}), ("next", ["WAIT"]))
            manifest = json.loads((Path(directory) / "takeover_manifest.json").read_text())
            self.assertEqual(manifest["replay_state_verification"], "match")
            self.assertEqual(manifest["prefix_source"], "repaired")
            self.assertTrue(target.seeded)


class StateProbeRoundTripTests(unittest.TestCase):
    """One per-step state record, written by collection, read by takeover and judge."""

    def test_collected_probes_feed_the_replay_gate_and_the_judge_final_state(self):
        import importlib.util

        from derail.rollout import state_probe

        class Env:
            def __init__(self):
                self._step_no, self.state = 0, 0

            def reset(self, task_config=None):
                return {}

            def step(self, action, pause=2.0):
                self._step_no += 1
                self.state += 1
                return {}, 0.0, False, {}

            def _execute_shell(self, command):
                return {"returncode": 0, "output": "orders.status=v%d" % self.state}

        hooks = replace(
            EnvironmentHooks.from_config(Path(load_takeover_config()["environment_config"])),
            determinism_commands={},
            probe_commands={"db": "probe"},
        )
        with tempfile.TemporaryDirectory() as root:
            env_cls = type("Env", (Env,), {})
            state_probe.install(env_cls, hooks, Path(root))
            env = env_cls()
            env.reset(task_config={"id": "task"})
            task = Path(root) / "task"
            traj = task / "traj.jsonl"
            for step_num in (1, 2, 2):  # the last two rows are one two-action turn
                env.step("pyautogui.click(1, 1)")
                with traj.open("a") as handle:  # the runner writes the row after env.step
                    handle.write(json.dumps({"step_num": step_num}) + "\n")
            records = [json.loads(line) for line in (task / hooks.probe_file).open()]
            steps = [
                CanonicalStep(step_id=i, action=ClickAction(kind="click", x_px=1, y_px=1),
                              observation_before_sha256="0" * 64,
                              source_record_uri=f"{traj}#line={i + 1}")
                for i in range(3)
            ]

            expected, reason = recorded_state_fingerprint(steps, 1, hooks.probe_file)
            self.assertEqual(reason, "")
            self.assertEqual(expected.to_dict(), records[2]["state_fingerprint"])

            spec = importlib.util.spec_from_file_location(
                "bundle_prefix_roundtrip",
                Path(__file__).resolve().parents[1] / "scripts/14_takeover_bundle_prefix.py",
            )
            bundle = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(bundle)
            final = bundle._final_state(task, hooks.probe_file, 1000)
            self.assertEqual(final["traj_index"], 2)
            self.assertIn("orders.status=v3", final["text"])
            self.assertIn(records[-1]["state_fingerprint"]["sha256"], final["text"])
            self.assertIn("[truncated", bundle._final_state(task, hooks.probe_file, 5)["text"])


class HistoryPolicyTests(unittest.TestCase):
    def test_strip_reasoning_is_the_single_switch(self):
        with tempfile.TemporaryDirectory() as directory:
            traj = Path(directory) / "traj.jsonl"
            traj.write_text(json.dumps({
                "step_num": 1,
                "response": "Clicked it.\n[reasoning] private plan\n[tool] computer_call {}",
                "agent_metadata": {"reasoning_content": "hidden"},
            }) + "\n", encoding="utf-8")
            step = CanonicalStep(step_id=0, action=ClickAction(kind="click", x_px=1, y_px=1),
                                 observation_before_sha256="0" * 64, source_step_id=0,
                                 source_record_uri=f"{traj}#line=1")
            stripped = load_trajectory_log(step)
            self.assertNotIn("private plan", stripped)
            self.assertNotIn("hidden", stripped)
            self.assertIn("[tool] computer_call", stripped)
            kept = load_trajectory_log(step, strip_reasoning=False)
            self.assertIn("private plan", kept)
            self.assertIn("hidden", kept)


class CrossAgentHistoryTests(unittest.TestCase):
    def test_gpt_renders_and_seeds_responses_items(self):
        image = _png()
        steps = [
            HistoryStep(step_id=0, observation_image_url=image, observation_after_image_url=image,
                        action=SequenceAction(kind="sequence", actions=(
                            ClickAction(kind="click", x_px=5, y_px=6),
                            TypeAction(kind="type", text="hi", press_enter=True),
                        ))),
            HistoryStep(step_id=2, observation_image_url=image, observation_after_image_url=image,
                        action=ShellAction(kind="shell", commands=("ls",)),
                        tool_result=json.dumps({"shell_results": [{"result": [
                            {"stdout": "a", "stderr": "", "outcome": {"type": "exit",
                                                                     "exit_code": 0}}]}]})),
        ]

        class Upstream:
            _history: list = []
            pending_items: list = []
            _operator_prompt = "PRIMER {CLIENT_PASSWORD} {CURRENT_DATE}"
            client_password = "pw"

            def predict(self, instruction, observation):
                return "ok", []

        target = OpenAITakeoverTarget(Upstream(), context_screenshots=20)
        adapter = create_native_history_adapter(
            "gpt_5_5", "Do it", system_prompt=target.native_history_system_prompt
        )
        history = build_native_history(adapter, steps)
        self.assertEqual(
            [item.get("type", item.get("role")) for item in history],
            ["user", "computer_call", "computer_call_output", "shell_call", "shell_call_output"],
        )
        self.assertEqual(
            [a["type"] for a in history[1]["actions"]], ["click", "type", "keypress"]
        )
        target.seed_native_history("Do it", history, condition="notified")
        self.assertEqual(target._zdr_keep_images, 20)
        self.assertTrue(target._zdr_stateless)
        self.assertEqual(target._history[4]["output"][0]["stdout"], "a")
        target.predict("Do it", {"screenshot": b"png"})
        self.assertIn("Takeover notice:", target.pending_items[0]["content"][0]["text"])

    def test_evocua_takes_over_a_foreign_history_in_its_s2_grammar(self):
        from derail.mypcbench.factory import create_mypcbench_agent

        adapter = create_native_history_adapter("evocua_32b", "open calc", source_agent="kimi_k3")
        history = build_native_history(adapter, [
            HistoryStep(step_id=0, observation_image_url=_png(),
                        action=ClickAction(kind="click", x_px=640, y_px=400)),
        ])
        self.assertTrue(history[2]["content"][0]["text"].startswith("Action: "))
        agent = create_mypcbench_agent("derail_evocua", "m", (1280, 800), "pw")
        agent.reset()
        agent.seed_native_history("open calc", history, condition="unaware")
        self.assertTrue(agent.inner.actions[0])
        with self.assertRaises(ValueError):
            build_native_history(
                create_native_history_adapter("evocua_32b", "x", source_agent="kimi_k3"),
                [HistoryStep(step_id=0, observation_image_url=_png(),
                             action=ShellAction(kind="shell", commands=("ls",)))],
            )


if __name__ == "__main__":
    unittest.main()
