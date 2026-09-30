"""Resume/retry and takeover dashboard regression tests."""

from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import tempfile
import unittest
from unittest import mock
from pathlib import Path
from types import SimpleNamespace


REPOSITORY = Path(__file__).resolve().parents[1]


def _load_script(name: str, relative: str):
    spec = importlib.util.spec_from_file_location(name, REPOSITORY / relative)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


ROLLOUT = _load_script("takeover_rollout_controls", "scripts/takeover/run_rollout.py")


class TakeoverResumeTests(unittest.TestCase):
    def test_skip_completed_accepts_only_success_marker(self):
        with tempfile.TemporaryDirectory() as directory:
            result = Path(directory) / "result.txt"
            self.assertFalse(ROLLOUT._result_is_completed(result))
            for value, expected in (
                ("1.0\n", True),
                ("1\n", True),
                ("0.0\n", False),
                ("", False),
                ("not-a-number\n", False),
            ):
                result.write_text(value, encoding="utf-8")
                self.assertEqual(ROLLOUT._result_is_completed(result), expected)

    def test_retry_failed_only_accepts_only_zero_marker(self):
        with tempfile.TemporaryDirectory() as directory:
            result = Path(directory) / "result.txt"
            self.assertFalse(ROLLOUT._result_is_failed(result))
            result.write_text("0.0\n", encoding="utf-8")
            self.assertTrue(ROLLOUT._result_is_failed(result))
            result.write_text("1.0\n", encoding="utf-8")
            self.assertFalse(ROLLOUT._result_is_failed(result))

    def test_zero_result_returns_nonzero_for_job_retry(self):
        self.assertEqual(ROLLOUT._result_exit_code(1.0), 0)
        self.assertEqual(ROLLOUT._result_exit_code(0.0), 1)
        self.assertEqual(ROLLOUT._result_exit_code(None), 1)

    def test_kimi_uses_cuabash_live_config(self):
        self.assertEqual(ROLLOUT._live_target_agent_id("kimi_k3"), "kimi_k3_cuabash")
        self.assertEqual(ROLLOUT._live_target_agent_id("opencua_72b"), "opencua_72b")

    def test_claude_uses_validated_upstream_runner_config(self):
        config = ROLLOUT.load_config("claude_opus_4_8")
        self.assertEqual(config.scaffold, "upstream_runner")
        self.assertEqual(config.document["agent_type"], "claude_cuabash")

    def test_runtime_model_name_uses_live_opencua_alias(self):
        with mock.patch.dict("os.environ", {"OPENCUA_MODEL": "opencua-72b"}):
            self.assertEqual(
                ROLLOUT._runtime_model_name("opencua_72b", "xlangai/OpenCUA-72B"),
                "opencua-72b",
            )

    def test_runtime_model_name_uses_live_evocua_alias(self):
        with mock.patch.dict("os.environ", {"EVOCUA_MODEL": "EvoCUA"}):
            self.assertEqual(
                ROLLOUT._runtime_model_name(
                    "evocua_32b", "meituan/EvoCUA-32B-20260105"
                ),
                "EvoCUA",
            )

    def test_runtime_model_name_preserves_unmapped_target(self):
        self.assertEqual(ROLLOUT._runtime_model_name("other", "configured"), "configured")

    def test_live_cuabash_requires_the_active_environment(self):
        environment = object()
        target = SimpleNamespace(
            protocol=SimpleNamespace(enable_bash=True),
            _env=environment,
        )
        ROLLOUT._assert_bash_environment_bound(target, environment)
        with self.assertRaisesRegex(RuntimeError, "active VM environment"):
            ROLLOUT._assert_bash_environment_bound(target, object())

    def test_non_cuabash_target_does_not_require_env_binding(self):
        target = SimpleNamespace(protocol=SimpleNamespace(enable_bash=False))
        ROLLOUT._assert_bash_environment_bound(target, object())

    def test_artifact_contract_rejects_predict_crash_and_missing_evidence(self):
        with tempfile.TemporaryDirectory() as directory:
            result_dir = Path(directory)
            (result_dir / "result.txt").write_text("1.0\n", encoding="utf-8")
            (result_dir / "traj.jsonl").write_text("PREDICT_CRASH\n", encoding="utf-8")
            self.assertEqual(ROLLOUT._enforce_takeover_artifacts(result_dir, 1.0), 1)
            self.assertEqual((result_dir / "result.txt").read_text(), "0.0\n")
            self.assertIn("PREDICT_CRASH", (result_dir / "incomplete.txt").read_text())

    def test_artifact_contract_accepts_complete_episode(self):
        with tempfile.TemporaryDirectory() as directory:
            result_dir = Path(directory)
            for name in (
                "native_history.json",
                "prefix_replay_log.json",
                "takeover_manifest.json",
                "rubric_bundle.json",
            ):
                (result_dir / name).write_text("{}\n", encoding="utf-8")
            (result_dir / "traj.jsonl").write_text('{"action":"DONE"}\n', encoding="utf-8")
            (result_dir / "step_1.png").write_bytes(b"png")
            self.assertEqual(ROLLOUT._enforce_takeover_artifacts(result_dir, 1.0), 0)


class TakeoverRolloutShellTests(unittest.TestCase):
    def test_help_documents_multi_depth_and_condition_options_in_chinese(self):
        completed = subprocess.run(
            ["bash", str(REPOSITORY / "scripts/rock/run_takeover.sh"), "--help"],
            check=True,
            capture_output=True,
            text=True,
        )
        self.assertIn("--depth 0/5/10/15/20/25", completed.stdout)
        self.assertIn("--prompt-condition unaware/notified/diagnosed", completed.stdout)
        self.assertIn("根因后立即接管", completed.stdout)
        self.assertIn("笛卡尔积", completed.stdout)


class TakeoverDashboardTests(unittest.TestCase):
    def _run_dashboard(self, root: Path) -> str:
        env = dict(os.environ)
        env.update(
            {
                "OUTPUT_ROOT": str(root),
                "DEPTHS": "0",
                "CONDITIONS": "unaware",
                "RUN_JUDGE": "1",
            }
        )
        completed = subprocess.run(
            ["bash", str(REPOSITORY / "scripts/takeover/watch_takeover.sh"), "--once"],
            check=True,
            capture_output=True,
            text=True,
            env=env,
        )
        return completed.stdout

    def test_dashboard_separates_runner_and_agent_outcomes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            selection_rows = []
            outcomes = (
                ("done", 1.0, "DONE"),
                ("agent-fail", 1.0, "FAIL"),
                ("timeout", 1.0, "pyautogui.click(1, 1)"),
                ("predict-crash", 0.0, "PREDICT_CRASH"),
            )
            group = root / "depth_0" / "unaware"
            group.mkdir(parents=True)
            for index, (task_id, result, action) in enumerate(outcomes):
                selection_rows.append(
                    "\t".join(
                        (
                            task_id,
                            f"canonical-{index}",
                            f"report-{index}",
                            f"task-{index}",
                            f"annotation-{index}",
                            "0",
                            "0",
                            "0",
                        )
                    )
                )
                task = group / task_id
                task.mkdir()
                (task / "takeover_launch.json").write_text(
                    json.dumps({"trajectory_id": task_id}), encoding="utf-8"
                )
                (task / "result.txt").write_text(f"{result}\n", encoding="utf-8")
                (task / "traj.jsonl").write_text(
                    json.dumps({"step_num": 1, "action": action}) + "\n",
                    encoding="utf-8",
                )
            (group / "done" / "rubric_judge_result.json").write_text(
                json.dumps({"result": 100}), encoding="utf-8"
            )
            (root / "selection.tsv").write_text(
                "\n".join(selection_rows) + "\n", encoding="utf-8"
            )

            output = self._run_dashboard(root)
            self.assertIn(
                "outcomes DONE 1   agent FAIL 1   timeout 1   predict crash 1",
                output,
            )
            self.assertIn("quality  runner error 1   judged perfect 1/1", output)
            self.assertNotIn(" fail 1", output)

    def test_dashboard_shows_active_retry_with_preserved_zero_marker(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            group = root / "depth_0" / "unaware"
            task = group / "retrying"
            task.mkdir(parents=True)
            (root / "selection.tsv").write_text(
                "retrying\tcanonical\treport\ttask\tannotation\t0\t0\t0\n",
                encoding="utf-8",
            )
            result = task / "result.txt"
            result.write_text("0.0\n", encoding="utf-8")
            os.utime(result, (1, 1))
            (task / "takeover_launch.json").write_text(
                json.dumps({"trajectory_id": "retrying"}), encoding="utf-8"
            )

            output = self._run_dashboard(root)
            self.assertIn("running 1", output)
            self.assertIn("retrying", output)


class TakeoverJudgeShellTests(unittest.TestCase):
    def test_prepare_only_accepts_multiple_depths_and_conditions(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for depth in (0, 5):
                for condition in ("unaware", "diagnosed"):
                    task = root / f"depth_{depth}" / condition / "case"
                    task.mkdir(parents=True)
                    (task / "result.txt").write_text("1.0\n", encoding="utf-8")
                    (task / "rubric_bundle.json").write_text("{}\n", encoding="utf-8")

            env = dict(os.environ, OUTPUT_ROOT=str(root))
            completed = subprocess.run(
                [
                    "bash",
                    str(REPOSITORY / "scripts/judge/run_takeover_judge.sh"),
                    "--source-agent",
                    "source",
                    "--takeover-agent",
                    "target",
                    "--depth",
                    "0/5",
                    "--prompt-condition",
                    "unaware/diagonosis",
                    "--prepare-only",
                ],
                check=True,
                capture_output=True,
                text=True,
                env=env,
            )

            self.assertIn("2 depth(s) x 2 condition(s) = 4 cell(s)", completed.stdout)
            for depth in (0, 5):
                for condition in ("unaware", "diagnosed"):
                    self.assertTrue((root / f"_judge_d{depth}_{condition}_ok" / "case").is_symlink())
                    self.assertIn(f"source_to_target_d{depth}_{condition}.csv", completed.stdout)
            self.assertIn("batch complete", completed.stdout)

    def test_prepare_only_stages_only_runner_clean_episodes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            cell = root / "depth_5" / "notified"
            for task_id, result in (("ok", "1.0\n"), ("runner-error", "0.0\n"),
                                    ("protocol-excluded", "1.0\n")):
                task = cell / task_id
                task.mkdir(parents=True)
                (task / "result.txt").write_text(result, encoding="utf-8")
                (task / "rubric_bundle.json").write_text("{}\n", encoding="utf-8")
            (cell / "protocol-excluded" / "protocol_exclusion.json").write_text(
                '{"reason": "replay_mismatch"}', encoding="utf-8")

            env = dict(os.environ, OUTPUT_ROOT=str(root))
            completed = subprocess.run(
                [
                    "bash",
                    str(REPOSITORY / "scripts/judge/run_takeover_judge.sh"),
                    "--source-agent",
                    "source",
                    "--takeover-agent",
                    "target",
                    "--depth",
                    "5",
                    "--prompt-condition",
                    "notified",
                    "--prepare-only",
                ],
                check=True,
                capture_output=True,
                text=True,
                env=env,
            )

            staging = root / "_judge_d5_notified_ok"
            self.assertTrue((staging / "ok").is_symlink())
            self.assertFalse((staging / "runner-error").exists())
            self.assertFalse((staging / "protocol-excluded").exists())
            self.assertIn("runner-clean=1 excluded=0 judged=0 pending=1", completed.stdout)
            self.assertIn("source_to_target_d5_notified.csv", completed.stdout)
            self.assertIn("prepare-only: no API calls made", completed.stdout)

    def test_prepare_only_judges_every_repeat(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for repeat in (1, 2):
                task = root / f"repeat_{repeat}" / "depth_0" / "unaware" / "case"
                task.mkdir(parents=True)
                (task / "result.txt").write_text("1.0\n", encoding="utf-8")
                (task / "rubric_bundle.json").write_text("{}\n", encoding="utf-8")

            completed = subprocess.run(
                ["bash", str(REPOSITORY / "scripts/judge/run_takeover_judge.sh"),
                 "--source-agent", "source", "--takeover-agent", "target",
                 "--depth", "0", "--condition", "unaware", "--prepare-only"],
                check=True, capture_output=True, text=True,
                env=dict(os.environ, OUTPUT_ROOT=str(root)),
            )

            for repeat in (1, 2):
                staging = root / f"repeat_{repeat}" / "_judge_d0_unaware_ok"
                self.assertTrue((staging / "case").is_symlink())
                self.assertIn(f"source_to_target_r{repeat}_d0_unaware.csv", completed.stdout)
            self.assertIn("repeats complete: 2", completed.stdout)

    def test_prepare_only_rejects_existing_different_judge(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            task = root / "depth_0" / "unaware" / "case"
            task.mkdir(parents=True)
            (task / "result.txt").write_text("1.0\n", encoding="utf-8")
            (task / "rubric_bundle.json").write_text("{}\n", encoding="utf-8")
            (task / "osworld_full_traj_result.json").write_text(
                json.dumps({"model": "different-judge"}), encoding="utf-8"
            )

            env = dict(os.environ, OUTPUT_ROOT=str(root))
            completed = subprocess.run(
                [
                    "bash",
                    str(REPOSITORY / "scripts/judge/run_takeover_judge.sh"),
                    "--source-agent",
                    "source",
                    "--takeover-agent",
                    "target",
                    "--depth",
                    "0",
                    "--condition",
                    "unaware",
                    "--prepare-only",
                ],
                check=False,
                capture_output=True,
                text=True,
                env=env,
            )

            self.assertNotEqual(completed.returncode, 0)
            self.assertIn("already contains judge model", completed.stderr)


if __name__ == "__main__":
    unittest.main()
