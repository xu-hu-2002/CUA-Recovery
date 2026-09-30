"""Clean-start 采集协议：统一预算 / K / 任务来源 / 环境钩子（论文 02:6、02:12、05:15）。"""

from __future__ import annotations

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

import yaml

from derail.rollout import state_probe
from derail.rollout.tasks import load_source_tasks

REPOSITORY = Path(__file__).resolve().parents[1]
RUNTIME = yaml.safe_load((REPOSITORY / "configs/collection/mypcbench_runtime.yaml").read_text())
# 各 scaffold 里「上下文保留几张截图」的旋钮名。
IMAGE_KNOBS = ("max_images_in_context", "max_history_turns", "max_image_history_length")


class RuntimeProtocolTests(unittest.TestCase):
    def test_defaults_follow_the_paper(self) -> None:
        self.assertEqual(
            (RUNTIME["max_steps"], RUNTIME["task_timeout"], RUNTIME["repeats"], RUNTIME["context_images"]),
            (150, 3600, 3, 20),
        )

    def test_every_agent_keeps_the_same_k_and_budget(self) -> None:
        for path in sorted((REPOSITORY / "configs/agents").glob("*.yaml")):
            config = yaml.safe_load(path.read_text())
            for knob in IMAGE_KNOBS:
                if knob in config:
                    self.assertEqual(config[knob], RUNTIME["context_images"], f"{path.name}:{knob}")
            self.assertNotIn("max_steps", config, path.name)
            self.assertNotIn("task_timeout", config, path.name)

    def test_rock_driver_defaults_come_from_the_runtime_config(self) -> None:
        path = REPOSITORY / "scripts/rock/derail_rock_driver.py"
        spec = importlib.util.spec_from_file_location("derail_rock_driver_protocol", path)
        driver = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(driver)
        if not any(driver.os.environ.get(k) for k in ("MAX_STEPS", "TASK_TIMEOUT", "REPEATS")):
            self.assertEqual(
                (driver.MAX_STEPS, driver.TASK_TIMEOUT, driver.REPEATS),
                (str(RUNTIME["max_steps"]), str(RUNTIME["task_timeout"]), str(RUNTIME["repeats"])),
            )
        self.assertEqual(driver._runtime_default("context_images"), str(RUNTIME["context_images"]))


class TaskSourceTests(unittest.TestCase):
    def test_rerail_rows_become_graded_runner_tasks(self) -> None:
        rows = [
            {"id": "gen-a#v0", "instruction": "do x", "task_id": "gen-a", "variant": 0,
             "origin": "composed", "bucket": "2-3", "world_variant": "base"},
            {"id": "gen-b#v0", "instruction": "do y", "task_id": "gen-b", "variant": 0,
             "origin": "composed", "bucket": "1", "world_variant": "hz-1"},
            {"id": "gen-c#v0", "instruction": "do z", "task_id": "gen-c", "variant": 0,
             "origin": "seed", "bucket": "1", "world_variant": "base"},
        ]
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            tasks_file = root / "rollout_tasks.json"
            tasks_file.write_text(json.dumps(rows))
            (root / "rubrics").mkdir()
            (root / "rubrics" / "gen-a.json").write_text(json.dumps(
                {"task_id": "gen-a", "type": "llm_judge",
                 "rubrics": [{"rubric": "x is done", "weight": 1}]}))
            spec = yaml.safe_load((REPOSITORY / "configs/collection/sources.yaml").read_text())
            for source in spec["sources"]:
                if source["source_benchmark"] == "rerail_workflows":
                    source["rubrics_dir"] = str(root / "rubrics")
            config = root / "sources.yaml"
            config.write_text(yaml.safe_dump(spec))
            with self.assertRaises(ValueError):  # gen-c has no composed rubric
                load_source_tasks("rerail_workflows", tasks_file, config)
            for source in spec["sources"]:
                source["on_missing_rubric"] = "skip"
            config.write_text(yaml.safe_dump(spec))
            tasks = load_source_tasks("rerail_workflows", tasks_file, config)
        self.assertEqual([task["id"] for task in tasks], ["gen-a#v0"])
        self.assertEqual(tasks[0]["instruction"], "do x")
        self.assertEqual(tasks[0]["workflow_id"], "gen-a")
        self.assertEqual(tasks[0]["grading"]["rubrics"][0]["rubric"], "x is done")


class _FakeEnv:
    def __init__(self, fail_probe: bool = False) -> None:
        self._step_no = 0
        self.commands = []
        self.state = 0
        self.fail_probe = fail_probe

    def reset(self, task_config=None, soft=False):
        self._step_no = 0
        return {}

    def step(self, action, pause=2.0):
        self._step_no += 1
        self.state += 1
        return {}, 0.0, False, {}

    def _execute_shell(self, command):
        self.commands.append(command)
        if command == "probe" and self.fail_probe:
            return {"returncode": 1, "output": "", "error": "boom"}
        return {"returncode": 0, "output": "state=%d" % self.state}


class EnvironmentHookTests(unittest.TestCase):
    def _hooks(self, **overrides):
        values = dict(
            determinism_commands={"no_updates": "disable"},
            determinism_on_error="fail",
            probe_enabled=True,
            probe_commands={"db": "probe"},
            probe_on_error="warn",
            probe_file="state_probes.jsonl",
            record_output=True,
            output_max_chars=100,
        )
        values.update(overrides)
        return state_probe.EnvironmentHooks(**values)

    def test_environment_config_parses(self) -> None:
        hooks = state_probe.EnvironmentHooks.from_config(
            REPOSITORY / "configs/environments/mypcbench_1280x800.yaml"
        )
        self.assertTrue(hooks.determinism_commands)
        self.assertTrue(hooks.probe_enabled and hooks.probe_commands)

    def test_reset_disables_nondeterminism_and_steps_are_fingerprinted(self) -> None:
        env_cls = type("Env", (_FakeEnv,), {})
        with tempfile.TemporaryDirectory() as tmp:
            state_probe.install(env_cls, self._hooks(), Path(tmp))
            env = env_cls()
            env.reset(task_config={"id": "task-1"})
            self.assertEqual(env.commands[0], "disable")
            traj = Path(tmp) / "task-1" / "traj.jsonl"
            for index in range(2):
                env.step("pyautogui.click(1, 1)")
                with traj.open("a") as handle:  # runner 在 step 返回后才写这一行
                    handle.write("{}\n")
            lines = [json.loads(line) for line in (Path(tmp) / "task-1/state_probes.jsonl").open()]
        self.assertEqual([line["traj_index"] for line in lines], [-1, 0, 1])
        self.assertEqual(lines[0]["determinism"], {"no_updates": {"returncode": 0}})
        self.assertEqual(lines[2]["probe_output"], {"db": "state=2"})
        self.assertNotEqual(
            lines[1]["state_fingerprint"]["sha256"], lines[2]["state_fingerprint"]["sha256"]
        )

    def test_error_policies(self) -> None:
        env_cls = type("Env", (_FakeEnv,), {})
        with tempfile.TemporaryDirectory() as tmp:
            state_probe.install(env_cls, self._hooks(), Path(tmp))
            env = env_cls(fail_probe=True)
            env.reset(task_config={"id": "t"})
            env.step("x")
            last = json.loads((Path(tmp) / "t/state_probes.jsonl").read_text().splitlines()[-1])
            self.assertIn("probe_error", last)

        failing = type("Env", (_FakeEnv,), {"_execute_shell": lambda self, c: {"returncode": 3}})
        state_probe.install(failing, self._hooks(probe_enabled=False), None)
        with self.assertRaises(RuntimeError):
            failing().reset(task_config={"id": "t"})


if __name__ == "__main__":
    unittest.main()
