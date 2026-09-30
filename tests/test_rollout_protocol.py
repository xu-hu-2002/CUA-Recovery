from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import yaml

from derail.rollout import state_probe
from derail.rollout.tasks import load_source_tasks

REPOSITORY = Path(__file__).resolve().parents[1]
RUNTIME = yaml.safe_load((REPOSITORY / "configs/collection/mypcbench_runtime.yaml").read_text())
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
            with self.assertRaises(ValueError):
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
                with traj.open("a") as handle:
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
