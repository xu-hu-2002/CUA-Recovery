"""Regression tests for decoupled takeover rubric judging."""

from __future__ import annotations

import csv
import importlib.util
import json
import os
import shutil
import subprocess
import tempfile
import textwrap
import unittest
from itertools import product
from pathlib import Path


REPOSITORY = Path(__file__).resolve().parents[1]
LAUNCHER = REPOSITORY / "scripts" / "judge" / "run_takeover_judge.sh"
RUN_JUDGE = REPOSITORY / "scripts" / "judge" / "run_judge.sh"
CSV_SCRIPT = REPOSITORY / "scripts" / "judge" / "rubric_csv.py"
ARCHIVE_SCRIPT = REPOSITORY / "scripts" / "judge" / "archive.py"
SELECTION_SCRIPT = REPOSITORY / "scripts" / "judge" / "takeover_judge_selection.py"
REGISTRY_SCRIPT = REPOSITORY / "scripts" / "judge" / "judge_model_registry.py"
REGISTRY_CONFIG = REPOSITORY / "configs" / "judges" / "model_registry.json"
PREFIX_SCRIPT = REPOSITORY / "scripts" / "takeover" / "bundle_prefix.py"
JUDGE_WRAPPER = REPOSITORY / "scripts" / "judge" / "full_traj_judge.py"
HARD_TIMEOUT = REPOSITORY / "scripts" / "judge" / "run_with_timeout.py"
JUDGE_CONFIG = REPOSITORY / "configs" / "judges" / "default.yaml"


def _test_env(**overrides):
    env = dict(os.environ)
    for key in (
        "SOURCE_AGENT", "TAKEOVER_AGENT", "TARGET_AGENT", "DEPTH", "CONDITION",
        "RUN_TAG", "JUDGE_MODEL", "MAX_IMAGES", "REASONING_EFFORT",
        "MAX_COMPLETION_TOKENS", "CONCURRENCY", "CSV_LABEL", "CSV_OUT_DIR",
        "JUDGE_ARCHIVE_ROOT", "OUTPUT_ROOT",
        "DERAIL_RUN_JUDGE_MODEL", "DERAIL_RUN_JUDGE_MAX_IMAGES",
    ):
        env.pop(key, None)
    return dict(env, PYTHONDONTWRITEBYTECODE="1", **overrides)


def _human_provenance(task: Path, digest: str = "a" * 64):
    (task / "takeover_launch.json").write_text(
        json.dumps({
            "trajectory_id": "trajectory-a",
            "human_annotation": {
                "annotator_id": "human",
                "source_trajectory_sha256": "a" * 64,
                "root_cause_action_index": 3,
            },
            "human_source_judge": {
                "authority": "human_annotation",
                "failure_eligible": True,
                "trajectory_id": "trajectory-a",
                "source_trajectory_sha256": digest,
                "reviewer_id": "human",
                "root_cause_action_index": 3,
            },
        }),
        encoding="utf-8",
    )


STUB_JUDGE = textwrap.dedent("""\
    import json
    import os
    import sys
    from pathlib import Path

    model = os.environ["MYPCBENCH_RUBRIC_JUDGE_MODEL"]
    max_images = os.environ["MYPCBENCH_OSWORLD_JUDGE_MAX_IMAGES"]
    with Path("events.jsonl").open("a") as log:
        log.write(json.dumps(["judge", model, max_images, sys.argv[1:]]) + "\\n")
    if os.environ.get("STUB_FAIL") == "judge":
        sys.exit(23)
    assert model in {
        "gpt-5.6-terra",
        "claude-opus-4-8",
        "openai.gpt-5.5",
    } and 0 < int(max_images) <= 200
    staging = Path(sys.argv[sys.argv.index("--result_dir") + 1])
    (staging / "scores.json").write_text("{}\\n")
    if os.environ.get("STUB_FAIL") == "csv":
        sys.exit(0)
    for task in staging.iterdir():
        if task.is_dir():
            (task / "osworld_full_traj_result.json").write_text(json.dumps({
                "model": model, "score": 100, "passed": True,
                "rubric_results": [{"success": True, "score": 1, "weight": 1.0}],
            }))
    """)


def _load_csv_module():
    spec = importlib.util.spec_from_file_location("takeover_rubric_csv", CSV_SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class TakeoverJudgeTests(unittest.TestCase):
    def _episode(self, cell: Path, task_id: str, *, result: float) -> Path:
        task = cell / task_id
        task.mkdir(parents=True)
        (task / "result.txt").write_text(f"{result}\n", encoding="utf-8")
        (task / "rubric_bundle.json").write_text(
            json.dumps({"grading_manifest": {"category": "email"}}),
            encoding="utf-8",
        )
        return task

    def test_prepare_only_uses_terra_and_excludes_runner_errors(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            cell = root / "depth_0" / "unaware"
            accepted = self._episode(cell, "accepted", result=1.0)
            self._episode(cell, "infra-error", result=0.0)
            env = _test_env(OUTPUT_ROOT=str(root))
            completed = subprocess.run(
                [
                    "bash",
                    str(LAUNCHER),
                    "--source-agent",
                    "opencua_72b",
                    "--takeover-agent",
                    "opencua_72b",
                    "--depth",
                    "0",
                    "--prompt-condition",
                    "unaware",
                    "--prepare-only",
                ],
                check=True,
                capture_output=True,
                text=True,
                env=env,
            )
            self.assertIn("runner-clean=1", completed.stdout)
            self.assertIn("model=gpt-5.6-terra", completed.stdout)
            self.assertIn("max_images=50", completed.stdout)
            staging = root / "_judge_d0_unaware_ok"
            self.assertEqual((staging / "accepted").resolve(), accepted.resolve())
            self.assertFalse((staging / "infra-error").exists())

    def test_launcher_preserves_explicit_gateway(self):
        text = LAUNCHER.read_text(encoding="utf-8")
        self.assertIn('if [[ -z "${OPENAI_BASE_URL:-}" ]]', text)
        self.assertNotIn("inherited gateway base_url", text)

    def test_csv_builder_writes_rubric_columns(self):
        module = _load_csv_module()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            cell = root / "depth_10" / "notified"
            task = self._episode(cell, "task-a", result=1.0)
            (task / "osworld_full_traj_result.json").write_text(
                json.dumps(
                    {
                        "score": 100,
                        "passed": True,
                        "rubric_results": [
                            {"success": True, "score": 1, "weight": 0.999},
                            {"success": False, "score": 0, "weight": 0.001},
                        ],
                    }
                ),
                encoding="utf-8",
            )
            row = module.rubric_row(task)
            self.assertIsNotNone(row)
            out = root / "summary.csv"
            module.write_csv([row], out)
            with out.open(newline="", encoding="utf-8") as handle:
                written = next(csv.DictReader(handle))
            self.assertEqual(written["rubric_score"], "99.9")
            self.assertEqual(written["perfect_pass"], "0")
            self.assertEqual(written["R2"], "0")
            self.assertEqual(written["R2_weight"], "0.001")

    def test_archive_requires_and_preserves_human_source_judge(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            cell = root / "cell"
            task = self._episode(cell, "task-a", result=1.0)
            _human_provenance(task)
            (task / "osworld_full_traj_result.json").write_text(
                json.dumps({"score": 100, "passed": True}), encoding="utf-8"
            )
            scores, csv_path = root / "scores.json", root / "aggregate.csv"
            scores.write_text("{}\n", encoding="utf-8")
            csv_path.write_text("task_id\ntask-a\n", encoding="utf-8")
            destination = root / "judge"
            subprocess.run(
                [
                    "python3", str(ARCHIVE_SCRIPT), "--cell", str(cell),
                    "--scores", str(scores), "--csv", str(csv_path),
                    "--archive-root", str(destination), "--judge-model", "gpt-5.6-terra",
                    "--source-agent", "source", "--target-agent", "target",
                    "--condition", "unaware", "--depth", "0",
                ],
                check=True,
                capture_output=True,
                text=True,
            )
            archived = destination / "gpt-5.6-terra/source/target/unaware/d0"
            manifest = json.loads((archived / "archive_manifest.json").read_text())
            self.assertEqual(manifest["source_judge_authority"], "human_annotation")
            self.assertTrue((archived / "judge_records.json").is_file())

    def test_archive_rejects_mismatched_human_source_judge(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            cell = root / "cell"
            task = self._episode(cell, "task-a", result=1.0)
            _human_provenance(task, digest="b" * 64)
            (task / "osworld_full_traj_result.json").write_text(
                json.dumps({"score": 100, "passed": True}), encoding="utf-8"
            )
            scores, csv_path = root / "scores.json", root / "aggregate.csv"
            scores.write_text("{}\n", encoding="utf-8")
            csv_path.write_text("task_id\ntask-a\n", encoding="utf-8")
            completed = subprocess.run(
                [
                    "python3", str(ARCHIVE_SCRIPT), "--cell", str(cell),
                    "--scores", str(scores), "--csv", str(csv_path),
                    "--archive-root", str(root / "judge"),
                    "--judge-model", "gpt-5.6-terra", "--source-agent", "source",
                    "--target-agent", "target", "--condition", "unaware", "--depth", "0",
                ],
                capture_output=True,
                text=True,
            )
            self.assertNotEqual(completed.returncode, 0)
            self.assertIn("invalid human source-judge provenance", completed.stderr)

    def _shell_fixture(self):
        temporary = tempfile.TemporaryDirectory(prefix="takeover judge ")
        self.addCleanup(temporary.cleanup)
        root = Path(temporary.name).resolve()
        for script in (LAUNCHER, RUN_JUDGE, CSV_SCRIPT, ARCHIVE_SCRIPT,
                       SELECTION_SCRIPT, REGISTRY_SCRIPT, REGISTRY_CONFIG, PREFIX_SCRIPT,
                       JUDGE_WRAPPER, HARD_TIMEOUT, JUDGE_CONFIG):
            destination = root / script.relative_to(REPOSITORY)
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(script, destination)
        shutil.copytree(REPOSITORY / "src/derail", root / "src/derail")
        shutil.copytree(REPOSITORY / "configs/agents", root / "configs/agents")
        shutil.copytree(REPOSITORY / "configs/environments", root / "configs/environments")
        shutil.copytree(REPOSITORY / "configs/takeover", root / "configs/takeover")
        judge = root / "third_party/MyPCBench/agent-harness/judge_results.py"
        judge.parent.mkdir(parents=True)
        judge.write_text(STUB_JUDGE, encoding="utf-8")
        (root / "credentials.env").write_text("", encoding="utf-8")
        env = _test_env(
            OUTPUT_ROOT=str(root / "outputs"), CSV_OUT_DIR=str(root / "csv"),
            DERAIL_CRED_ENV=str(root / "credentials.env"),
            OPENAI_API_KEY="local-stub", OPENAI_BASE_URL="http://127.0.0.1:1",
            STUB_FAIL="",
        )
        task = self._episode(root / "outputs/depth_0/unaware", "task-a", result=1.0)
        _human_provenance(task)
        return root, env

    def _launch(self, root, env, *args):
        return subprocess.run(
            ["bash", str(root / LAUNCHER.relative_to(REPOSITORY)),
             "--source-agent", "source", "--takeover-agent", "target",
             "--depth", "0", "--condition", "unaware", *args],
            cwd=root, env=env, capture_output=True, text=True, timeout=30,
        )

    def _run_inner(self, root, env):
        return subprocess.run(
            ["bash", str(root / RUN_JUDGE.relative_to(REPOSITORY)),
             str(root / "outputs/depth_0/unaware")],
            cwd=root, env=env, capture_output=True, text=True, timeout=30,
        )

    def _events(self, root):
        log = root / "events.jsonl"
        return [json.loads(line) for line in log.read_text().splitlines()] if log.exists() else []

    def _archive(self, root):
        return root / "archive/gpt-5.6-terra/source/target/unaware/d0"

    def test_outer_rejects_images_over_50_from_cli_and_environment(self):
        root, env = self._shell_fixture()
        for value, use_env in product(("51", "200", "18446744073709551617"), (False, True)):
            with self.subTest(value=value, use_env=use_env):
                args = () if use_env else ("--max-images", value)
                child_env = dict(env, MAX_IMAGES=value) if use_env else env
                result = self._launch(root, child_env, *args, "--prepare-only")
                self.assertNotEqual(result.returncode, 0)
                self.assertIn("must be <= 50", result.stderr)
        self.assertEqual(self._events(root), [])
        self.assertFalse((root / "outputs/_judge_d0_unaware_ok").exists())

    def test_inner_defaults_to_50(self):
        root, env = self._shell_fixture()
        result = self._run_inner(root, env)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(self._events(root)[0][:3], ["judge", "gpt-5.6-terra", "50"])

    def test_inner_rejects_images_over_50(self):
        root, env = self._shell_fixture()
        for value in ("51", "200", "18446744073709551617"):
            with self.subTest(value=value):
                result = self._run_inner(root, dict(env, DERAIL_RUN_JUDGE_MAX_IMAGES=value))
                self.assertNotEqual(result.returncode, 0)
                self.assertIn("must be <= 50", result.stderr)
        self.assertEqual(self._events(root), [])

    def test_both_layers_reject_invalid_image_counts(self):
        root, env = self._shell_fixture()
        for value in ("0", "-1", "1.5", "abc", "050"):
            with self.subTest(value=value):
                outer = self._launch(root, env, "--max-images", value)
                inner = self._run_inner(root, dict(env, DERAIL_RUN_JUDGE_MAX_IMAGES=value))
                self.assertNotEqual(outer.returncode, 0)
                self.assertNotEqual(inner.returncode, 0)
        self.assertEqual(self._events(root), [])

    def test_both_layers_reject_every_model_but_the_configured_judge(self):
        root, env = self._shell_fixture()
        for value in ("claude-opus-4-8", "openai.gpt-5.5", "gpt-5.6-sol",
                      "gpt-4o", "../gpt-5.6-terra"):
            with self.subTest(value=value):
                outer = self._launch(root, env, "--judge-model", value, "--prepare-only")
                inner = self._run_inner(root, dict(env, DERAIL_RUN_JUDGE_MODEL=value))
                self.assertNotEqual(outer.returncode, 0)
                self.assertNotEqual(inner.returncode, 0)
                self.assertIn("default.yaml", outer.stderr)
                self.assertIn("default.yaml", inner.stderr)
        self.assertEqual(self._events(root), [])

    def test_explicit_override_still_applies_per_model_image_limits(self):
        root, env = self._shell_fixture()
        env = dict(env, ALLOW_CONFIG_OVERRIDE="1")
        opus = self._run_inner(root, dict(env, DERAIL_RUN_JUDGE_MODEL="claude-opus-4-8"))
        self.assertEqual(opus.returncode, 0, opus.stderr)
        self.assertEqual(self._events(root)[0][:3], ["judge", "claude-opus-4-8", "200"])
        rejected = self._launch(
            root, env, "--judge-model", "claude-opus-4-8", "--max-images", "201",
            "--prepare-only",
        )
        self.assertNotEqual(rejected.returncode, 0)
        self.assertIn("must be <= 200", rejected.stderr)

    def test_judge_may_not_grade_its_own_source(self):
        root, env = self._shell_fixture()
        manifest = root / "outputs/depth_0/unaware/task-a/takeover_manifest.json"
        manifest.write_text(json.dumps({"source_agent": "gpt_5_5",
                                        "target_agent": "gpt_5_5",
                                        "target_model": "gpt-5.6-sol"}), encoding="utf-8")
        inner = self._run_inner(root, env)
        self.assertNotEqual(inner.returncode, 0)
        self.assertIn("shares a source", inner.stderr)
        self.assertEqual(self._events(root), [])

    def test_agent_ids_cannot_escape_archive_tree(self):
        root, env = self._shell_fixture()
        for selector in ("--source-agent", "--takeover-agent"):
            with self.subTest(selector=selector):
                result = self._launch(root, env, selector, "..", "--prepare-only")
                self.assertNotEqual(result.returncode, 0)
                self.assertIn("set a valid", result.stderr)
        self.assertEqual(self._events(root), [])

    def test_prepare_only_never_judges_or_archives(self):
        root, env = self._shell_fixture()
        result = self._launch(
            root, env, "--prepare-only", "--archive-root", str(root / "archive"),
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(self._events(root), [])
        self.assertFalse((root / "archive").exists())
        self.assertFalse((root / "csv").exists())
        self.assertTrue((root / "outputs/_judge_d0_unaware_ok/task-a").is_symlink())

    def test_archive_writes_manifest(self):
        root, env = self._shell_fixture()
        result = self._launch(root, env, "--archive-root", str(root / "archive"))
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual([event[0] for event in self._events(root)], ["judge"])
        self.assertEqual(self._events(root)[0][:3], ["judge", "gpt-5.6-terra", "50"])
        self.assertTrue((self._archive(root) / "archive_manifest.json").is_file())

    def test_environment_relative_archive_root(self):
        root, env = self._shell_fixture()
        env.update(JUDGE_ARCHIVE_ROOT="archive")
        result = self._launch(root, env)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual([event[0] for event in self._events(root)], ["judge"])
        self.assertTrue((self._archive(root) / "archive_manifest.json").is_file())

    def test_batch_archives_each_cell(self):
        root, env = self._shell_fixture()
        for depth, condition in ((0, "notified"), (5, "unaware"), (5, "notified")):
            task = self._episode(root / f"outputs/depth_{depth}/{condition}", "task-a", result=1.0)
            _human_provenance(task)
        result = self._launch(
            root, env, "--depth", "0/5", "--condition", "unaware/notified",
            "--archive-root", str(root / "archive"), "--max-images", "25",
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        events = self._events(root)
        self.assertEqual([event[0] for event in events], ["judge"] * 4)
        for depth, condition in product((0, 5), ("unaware", "notified")):
            self.assertTrue((root / f"archive/gpt-5.6-terra/source/target/{condition}/d{depth}"
                             / "archive_manifest.json").is_file())
        self.assertTrue(all(event[2] == "25" for event in events if event[0] == "judge"))

    def test_batch_prepare_only_has_no_external_calls(self):
        root, env = self._shell_fixture()
        self._episode(root / "outputs/depth_5/unaware", "task-a", result=1.0)
        result = self._launch(
            root, env, "--depth", "0/5", "--archive-root", str(root / "archive"),
            "--prepare-only",
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(self._events(root), [])
        self.assertFalse((root / "archive").exists())
        self.assertTrue((root / "outputs/_judge_d5_unaware_ok/task-a").is_symlink())

    def test_judge_and_csv_failure_stop_before_archive(self):
        for phase in ("judge", "csv"):
            with self.subTest(phase=phase):
                root, env = self._shell_fixture()
                result = self._launch(
                    root, dict(env, STUB_FAIL=phase),
                    "--archive-root", str(root / "archive"),
                )
                self.assertNotEqual(result.returncode, 0)
                self.assertEqual([event[0] for event in self._events(root)], ["judge"])
                self.assertFalse((root / "archive").exists())
                if phase == "judge":
                    self.assertEqual(result.returncode, 23)

    def test_invalid_human_provenance_stops_before_archive(self):
        root, env = self._shell_fixture()
        _human_provenance(root / "outputs/depth_0/unaware/task-a", digest="b" * 64)
        result = self._launch(root, env, "--archive-root", str(root / "archive"))
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("invalid human source-judge provenance", result.stderr)
        self.assertEqual([event[0] for event in self._events(root)], ["judge"])
        self.assertFalse((root / "archive").exists())

    def test_help_documents_limits(self):
        result = subprocess.run(
            ["bash", str(LAUNCHER), "--help"], env=_test_env(),
            capture_output=True, text=True, check=True,
        )
        for text in ("model_registry.json", "registered model image limit", "without API calls"):
            self.assertIn(text, result.stdout)



if __name__ == "__main__":
    unittest.main()
