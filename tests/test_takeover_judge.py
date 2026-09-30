"""Regression tests for decoupled takeover rubric judging."""

from __future__ import annotations

import csv
import importlib.util
import json
import os
import shutil
import subprocess
import sys
import tempfile
import textwrap
import unittest
from itertools import product
from pathlib import Path
from unittest import mock


REPOSITORY = Path(__file__).resolve().parents[1]
LAUNCHER = REPOSITORY / "artifacts" / "takeover" / "run_takeover_judge.sh"
RUN_JUDGE = REPOSITORY / "scripts" / "run_judge.sh"
CSV_SCRIPT = REPOSITORY / "scripts" / "12_takeover_rubric_csv.py"
ARCHIVE_SCRIPT = REPOSITORY / "scripts" / "13_archive_takeover_judge.py"
SHIP_SCRIPT = REPOSITORY / "scripts" / "14_ship_takeover_judge_archive.py"
SELECTION_SCRIPT = REPOSITORY / "scripts" / "takeover_judge_selection.py"
REGISTRY_SCRIPT = REPOSITORY / "scripts" / "judge_model_registry.py"
REGISTRY_CONFIG = REPOSITORY / "configs" / "judges" / "routify_model_registry.json"
PREFIX_SCRIPT = REPOSITORY / "scripts" / "14_takeover_bundle_prefix.py"
JUDGE_WRAPPER = REPOSITORY / "scripts" / "30_full_traj_judge.py"
HARD_TIMEOUT = REPOSITORY / "scripts" / "31_run_with_timeout.py"
JUDGE_CONFIG = REPOSITORY / "configs" / "judges" / "default.yaml"
OSS_ROOT = "oss://example-bucket/derail/judge/takeover/failure_prefix_v1/"
os.environ["JUDGE_OSS_ROOT"] = OSS_ROOT


def _test_env(**overrides):
    env = dict(os.environ)
    for key in (
        "SOURCE_AGENT", "TAKEOVER_AGENT", "TARGET_AGENT", "DEPTH", "CONDITION",
        "RUN_TAG", "JUDGE_MODEL", "MAX_IMAGES", "REASONING_EFFORT",
        "MAX_COMPLETION_TOKENS", "CONCURRENCY", "CSV_LABEL", "CSV_OUT_DIR",
        "JUDGE_ARCHIVE_ROOT", "SHIP_TO_OSS", "OUTPUT_ROOT", "OSSUTIL",
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
        "openai.gpt-5.6-luna",
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

# OSSUTIL points to Python, which runs local cp/ls scripts without chmod or a network client.
STUB_OSSUTIL = textwrap.dedent("""\
    import json
    import os
    import sys
    from pathlib import Path

    command = Path(sys.argv[0]).name
    with Path("events.jsonl").open("a") as log:
        log.write(json.dumps([command, *sys.argv[1:]]) + "\\n")
    phase = "ls" if command == "ls" else "archive" if "-r" in sys.argv else "ledger"
    if os.environ.get("STUB_FAIL") == phase:
        sys.exit("stub ossutil failure: " + phase)
    remote = Path("remote.json")
    if phase == "archive":
        source = Path(sys.argv[-2])
        assert (source / "archive_manifest.json").is_file(), "upload before archive"
        remote.write_text(json.dumps({
            str(path.relative_to(source)): path.stat().st_size
            for path in source.rglob("*") if path.is_file()
        }))
    elif phase == "ledger":
        ledger = Path(sys.argv[-2]).read_text()
        assert json.loads(ledger)["object_parity"]
        Path("remote_ledger.json").write_text(ledger)
    else:
        sizes = json.loads(remote.read_text())
        if os.environ.get("STUB_PARITY_FAIL") == "1":
            sizes.pop("scores.json")
        destination = sys.argv[-1].rstrip("/") + "/"
        sizes[""] = 0
        for name, size in sizes.items():
            print(f"2026-09-11 00:00:00 +0000 CST {size} Standard HASH {destination}{name}")
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
        self.assertNotIn("inherited routify base_url", text)

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
            # ρ and V recomputed per criterion: the judge's rounded 100/passed is not a pass.
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
        for script in (LAUNCHER, RUN_JUDGE, CSV_SCRIPT, ARCHIVE_SCRIPT, SHIP_SCRIPT,
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
        for command in ("cp", "ls"):
            (root / command).write_text(STUB_OSSUTIL, encoding="utf-8")
        (root / "ossutil stub").symlink_to(sys.executable)
        (root / "credentials.env").write_text("", encoding="utf-8")
        env = _test_env(
            OUTPUT_ROOT=str(root / "outputs"), CSV_OUT_DIR=str(root / "csv"),
            DERAIL_CRED_ENV=str(root / "credentials.env"),
            OPENAI_API_KEY="local-stub", OPENAI_BASE_URL="http://127.0.0.1:1",
            OSSUTIL=str(root / "ossutil stub"), STUB_FAIL="", STUB_PARITY_FAIL="0",
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
        """One judge on the formal path: configs/judges/default.yaml (gpt-5.6-terra)."""
        root, env = self._shell_fixture()
        for value in ("claude-opus-4-8", "openai.gpt-5.5", "openai.gpt-5.6-luna",
                      "anthropic.claude-sonnet-4-6", "gpt-4o", "../gpt-5.6-terra"):
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
        manifest.write_text(json.dumps({"source_agent": "gpt_5_6_luna",
                                        "target_agent": "gpt_5_6_luna"}), encoding="utf-8")
        inner = self._run_inner(root, env)
        self.assertNotEqual(inner.returncode, 0)
        self.assertIn("shares a source", inner.stderr)
        self.assertEqual(self._events(root), [])

    def test_ship_requires_archive_even_in_prepare_only(self):
        root, env = self._shell_fixture()
        for use_env, prepare in product((False, True), ((), ("--prepare-only",))):
            with self.subTest(use_env=use_env, prepare=prepare):
                args = () if use_env else ("--ship",)
                child_env = dict(env, SHIP_TO_OSS="1") if use_env else env
                result = self._launch(root, child_env, *args, *prepare)
                self.assertNotEqual(result.returncode, 0)
                self.assertIn("--ship requires --archive-root", result.stderr)
        self.assertEqual(self._events(root), [])

    def test_ship_rejects_invalid_opt_in(self):
        root, env = self._shell_fixture()
        result = self._launch(root, dict(env, SHIP_TO_OSS="yes"), "--prepare-only")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("SHIP_TO_OSS must be 0 or 1", result.stderr)
        self.assertEqual(self._events(root), [])

    def test_agent_ids_cannot_escape_archive_tree(self):
        root, env = self._shell_fixture()
        for selector in ("--source-agent", "--takeover-agent"):
            with self.subTest(selector=selector):
                result = self._launch(root, env, selector, "..", "--prepare-only")
                self.assertNotEqual(result.returncode, 0)
                self.assertIn("set a valid", result.stderr)
        self.assertEqual(self._events(root), [])

    def test_prepare_only_with_ship_never_judges_archives_or_uploads(self):
        root, env = self._shell_fixture()
        for args, overrides in ((("--ship",), {}), ((), {"SHIP_TO_OSS": "1"})):
            with self.subTest(args=args):
                result = self._launch(
                    root, dict(env, **overrides), "--prepare-only", *args,
                    "--archive-root", str(root / "archive"),
                )
                self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(self._events(root), [])
        self.assertFalse((root / "archive").exists())
        self.assertFalse((root / "csv").exists())
        self.assertTrue((root / "outputs/_judge_d0_unaware_ok/task-a").is_symlink())

    def test_archive_without_opt_in_never_uploads(self):
        root, env = self._shell_fixture()
        result = self._launch(root, env, "--archive-root", str(root / "archive"))
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual([event[0] for event in self._events(root)], ["judge"])
        self.assertEqual(self._events(root)[0][:3], ["judge", "gpt-5.6-terra", "50"])
        self.assertTrue((self._archive(root) / "archive_manifest.json").is_file())
        self.assertFalse((self._archive(root) / "ship_ledger.json").exists())

    def test_ship_uploads_after_archive_and_publishes_parity_ledger(self):
        root, env = self._shell_fixture()
        result = self._launch(root, env, "--archive-root", str(root / "archive"), "--ship")
        self.assertEqual(result.returncode, 0, result.stderr)
        archive = self._archive(root)
        destination = OSS_ROOT + "gpt-5.6-terra/source/target/unaware/d0/"
        events = self._events(root)
        self.assertEqual([event[0] for event in events], ["judge", "cp", "ls", "cp"])
        self.assertEqual(events[1], ["cp", "-r", "-f", f"{archive}/", destination])
        self.assertEqual(events[2], ["ls", destination])
        self.assertEqual(events[3], ["cp", "-f", str(archive / "ship_ledger.json"), destination])
        ledger = json.loads((archive / "ship_ledger.json").read_text())
        self.assertTrue(ledger["object_parity"])
        self.assertEqual(ledger["local_object_count"], 5)
        self.assertEqual(ledger["remote_object_count"], 5)
        self.assertEqual(ledger, json.loads((root / "remote_ledger.json").read_text()))
        record = json.loads((archive / "judge_records.json").read_text())[0]
        self.assertEqual(record["launch"]["human_source_judge"]["authority"], "human_annotation")

    def test_environment_opt_in_and_relative_archive_root(self):
        root, env = self._shell_fixture()
        env.update(SHIP_TO_OSS="1", JUDGE_ARCHIVE_ROOT="archive")
        result = self._launch(root, env)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual([event[0] for event in self._events(root)], ["judge", "cp", "ls", "cp"])
        self.assertTrue((self._archive(root) / "ship_ledger.json").is_file())

    def test_batch_propagates_ship_for_each_cell(self):
        root, env = self._shell_fixture()
        for depth, condition in ((0, "notified"), (5, "unaware"), (5, "notified")):
            task = self._episode(root / f"outputs/depth_{depth}/{condition}", "task-a", result=1.0)
            _human_provenance(task)
        result = self._launch(
            root, env, "--depth", "0/5", "--condition", "unaware/notified",
            "--archive-root", str(root / "archive"), "--ship", "--max-images", "25",
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        events = self._events(root)
        self.assertEqual([event[0] for event in events], ["judge", "cp", "ls", "cp"] * 4)
        destinations = [event[-1] for event in events if event[0] == "ls"]
        expected = [OSS_ROOT + f"gpt-5.6-terra/source/target/{condition}/d{depth}/"
                    for depth in (0, 5) for condition in ("unaware", "notified")]
        self.assertEqual(destinations, expected)
        self.assertTrue(all(event[2] == "25" for event in events if event[0] == "judge"))

    def test_batch_prepare_only_with_ship_has_no_external_calls(self):
        root, env = self._shell_fixture()
        self._episode(root / "outputs/depth_5/unaware", "task-a", result=1.0)
        result = self._launch(
            root, env, "--depth", "0/5", "--archive-root", str(root / "archive"),
            "--ship", "--prepare-only",
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(self._events(root), [])
        self.assertFalse((root / "archive").exists())
        self.assertTrue((root / "outputs/_judge_d5_unaware_ok/task-a").is_symlink())

    def test_judge_and_csv_failure_stop_before_upload(self):
        for phase in ("judge", "csv"):
            with self.subTest(phase=phase):
                root, env = self._shell_fixture()
                result = self._launch(
                    root, dict(env, STUB_FAIL=phase), "--ship",
                    "--archive-root", str(root / "archive"),
                )
                self.assertNotEqual(result.returncode, 0)
                self.assertEqual([event[0] for event in self._events(root)], ["judge"])
                self.assertFalse((root / "archive").exists())
                if phase == "judge":
                    self.assertEqual(result.returncode, 23)

    def test_invalid_human_provenance_stops_before_upload(self):
        root, env = self._shell_fixture()
        _human_provenance(root / "outputs/depth_0/unaware/task-a", digest="b" * 64)
        result = self._launch(root, env, "--ship", "--archive-root", str(root / "archive"))
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("invalid human source-judge provenance", result.stderr)
        self.assertEqual([event[0] for event in self._events(root)], ["judge"])
        self.assertFalse((root / "archive").exists())

    def test_ossutil_failures_propagate_to_launcher(self):
        for phase, commands in (("archive", ["judge", "cp"]),
                                ("ls", ["judge", "cp", "ls"]),
                                ("ledger", ["judge", "cp", "ls", "cp"])):
            with self.subTest(phase=phase):
                root, env = self._shell_fixture()
                result = self._launch(
                    root, dict(env, STUB_FAIL=phase), "--ship",
                    "--archive-root", str(root / "archive"),
                )
                self.assertNotEqual(result.returncode, 0)
                self.assertIn("stub ossutil failure: " + phase, result.stderr)
                self.assertEqual([event[0] for event in self._events(root)], commands)
                self.assertTrue((self._archive(root) / "archive_manifest.json").is_file())
                self.assertFalse((root / "remote_ledger.json").exists())

    def test_parity_failure_keeps_local_ledger_without_publishing_it(self):
        root, env = self._shell_fixture()
        result = self._launch(
            root, dict(env, STUB_PARITY_FAIL="1"), "--ship",
            "--archive-root", str(root / "archive"),
        )
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("judge OSS object parity failed", result.stderr)
        self.assertEqual([event[0] for event in self._events(root)], ["judge", "cp", "ls"])
        ledger = json.loads((self._archive(root) / "ship_ledger.json").read_text())
        self.assertFalse(ledger["object_parity"])
        self.assertEqual(ledger["missing_remote"], ["scores.json"])
        self.assertFalse((root / "remote_ledger.json").exists())

    def test_help_documents_limits_and_shipping(self):
        result = subprocess.run(
            ["bash", str(LAUNCHER), "--help"], env=_test_env(),
            capture_output=True, text=True, check=True,
        )
        for text in ("routify_model_registry.json", "registered model image limit", "--ship",
                     "requires --archive-root", "without API calls or uploads", "OSSUTIL", "JUDGE_OSS_ROOT"):
            self.assertIn(text, result.stdout)


class TakeoverJudgeShippingTests(unittest.TestCase):
    def setUp(self):
        spec = importlib.util.spec_from_file_location("ship_takeover_judge", SHIP_SCRIPT)
        self.module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.module)
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.source = Path(temporary.name).resolve()
        self.destination = OSS_ROOT + "gpt-5.6-terra/source/target/unaware/d0/"

    def _listing(self, files):
        return "\n".join(
            f"2026-09-11 00:00:00 +0000 CST {size} Standard HASH {self.destination}{name}"
            for name, size in files.items()
        )

    def test_parse_empty_listing(self):
        self.assertEqual(self.module.parse_listing("", self.destination), {})

    def test_listing_excludes_root_marker_and_ship_ledger(self):
        output = self._listing({"": 0, "scores.json": 3, "ship_ledger.json": 99})
        output += "\nObject Number is: 3\n" + self._listing({"other.json": 1}).replace(
            "/d0/", "/d5/"
        )
        self.assertEqual(self.module.parse_listing(output, self.destination), {"scores.json": 3})
        self.assertEqual(self.module.parse_listing(output, self.destination.rstrip("/")),
                         {"scores.json": 3})

    def test_empty_archive_is_rejected_before_subprocess(self):
        with mock.patch.object(self.module, "run") as run:
            with self.assertRaisesRegex(ValueError, "archive is empty"):
                self.module.ship(self.source, self.destination, "stub")
        run.assert_not_called()

    def test_ledger_only_archive_is_rejected(self):
        (self.source / "ship_ledger.json").write_text("{}")
        with mock.patch.object(self.module, "run") as run:
            with self.assertRaisesRegex(ValueError, "archive is empty"):
                self.module.ship(self.source, self.destination, "stub")
        run.assert_not_called()

    def test_destination_outside_frozen_root_is_rejected(self):
        (self.source / "scores.json").write_text("{}\n")
        with mock.patch.object(self.module, "run") as run:
            with self.assertRaisesRegex(ValueError, "destination must be under"):
                self.module.ship(self.source, "oss://example-bucket/other/", "stub")
        run.assert_not_called()

    def test_empty_remote_listing_does_not_report_parity(self):
        (self.source / "scores.json").write_text("{}\n")
        with mock.patch.object(self.module, "run", side_effect=["", ""]):
            ledger = self.module.ship(self.source, self.destination, "stub")
        self.assertFalse(ledger["object_parity"])
        self.assertEqual(ledger["missing_remote"], ["scores.json"])
        self.assertEqual(ledger["remote_object_count"], 0)

    def test_parity_detects_missing_unexpected_and_size_mismatch(self):
        (self.source / "scores.json").write_text("{}\n")
        cases = [({}, ["scores.json"], [], []),
                 ({"scores.json": 3, "extra.json": 1}, [], ["extra.json"], []),
                 ({"scores.json": 4}, [], [], ["scores.json"])]
        for files, missing, unexpected, mismatches in cases:
            responses = ["", self._listing(files)]
            with self.subTest(files=files), mock.patch.object(self.module, "run", side_effect=responses):
                ledger = self.module.ship(self.source, self.destination, "stub")
                self.assertFalse(ledger["object_parity"])
                self.assertEqual(ledger["missing_remote"], missing)
                self.assertEqual(ledger["unexpected_remote"], unexpected)
                self.assertEqual(ledger["size_mismatches"], mismatches)

    def test_successful_parity_excludes_existing_ledger_on_both_sides(self):
        (self.source / "scores.json").write_text("{}\n")
        (self.source / "ship_ledger.json").write_text("{}\n")
        output = self._listing({"": 0, "scores.json": 3, "ship_ledger.json": 100})
        with mock.patch.object(self.module, "run", side_effect=["", output]) as run:
            ledger = self.module.ship(self.source, self.destination, "stub")
        self.assertTrue(ledger["object_parity"])
        self.assertEqual(ledger["local_object_count"], 1)
        self.assertEqual(ledger["remote_object_count"], 1)
        self.assertEqual(self.module.local_files(self.source), {"scores.json": 3})
        self.assertEqual(run.call_count, 2)

    def test_main_uses_ossutil_environment_and_publishes_ledger(self):
        (self.source / "scores.json").write_text("{}\n")
        argv = [str(SHIP_SCRIPT), "--source", str(self.source), "--destination", self.destination]
        responses = ["", self._listing({"scores.json": 3}), ""]
        with mock.patch.dict(os.environ, {"OSSUTIL": "/local/ossutil stub"}), mock.patch.object(sys, "argv", argv):
            with mock.patch.object(self.module, "run", side_effect=responses) as run, mock.patch("builtins.print"):
                self.assertEqual(self.module.main(), 0)
        self.assertTrue(all(call.args[0][0] == "/local/ossutil stub" for call in run.call_args_list))
        self.assertEqual(run.call_args_list[-1].args[0], [
            "/local/ossutil stub", "cp", "-f", str(self.source / "ship_ledger.json"), self.destination,
        ])
        self.assertTrue(json.loads((self.source / "ship_ledger.json").read_text())["object_parity"])

    def test_subprocess_failure_is_not_swallowed(self):
        failure = subprocess.CompletedProcess(["stub"], 17, stdout="", stderr="copy failed")
        with mock.patch.object(self.module.subprocess, "run", return_value=failure):
            with self.assertRaisesRegex(RuntimeError, "copy failed"):
                self.module.run(["stub", "cp"])


if __name__ == "__main__":
    unittest.main()
