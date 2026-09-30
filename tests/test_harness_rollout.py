"""Rollout harness with fakes: cursor discipline, ledger attribution, rollout-trace/1.0 output."""

from __future__ import annotations

import hashlib
import json
import unittest
from pathlib import Path

from derail.derived.schema import validate_schema
from derail.harness.control_client import DerailControlClient
from derail.harness.env_wrapper import DerailEnvWrapper
from derail.harness.trace_builder import (
    TraceBuilder,
    observation_from_trace_record,
    pages_from_trace_records,
    sql_tables,
)

REPOSITORY = Path(__file__).resolve().parents[1]


class FakeControl:
    def __init__(self):
        self.cursor_log = []
        self.cursor = None
        self.rows = []
        self.trace = []
        self.seq = {"hoolicalendar": 40, "workbuzz": 7}

    def transport(self, method, path, payload):
        if path == "/derail/cursor":
            self.cursor = payload["action_index"]
            self.cursor_log.append(self.cursor)
            return {"action_index": self.cursor, "databases": list(self.seq)}
        if path == "/derail/seq":
            return {"seq": dict(self.seq)}
        if path.startswith("/derail/changelog"):
            since = json.loads(__import__("urllib.parse").parse.unquote(path.split("since=")[1]))
            return {"rows": [r for r in self.rows if r["seq"] > since.get(r["db"], 0)]}
        if path.startswith("/derail/observations"):
            index = int(path.split("=")[1])
            return {"records": [t for t in self.trace if t["action_index"] == index]}
        raise AssertionError(path)

    def write(self, db, tbl, rowid, op, new):
        self.seq[db] += 1
        self.rows.append(
            {
                "schema_version": "changelog-row/1.0",
                "db": db,
                "seq": self.seq[db],
                "ts": 1.0,
                "tbl": tbl,
                "rowid": rowid,
                "op": op,
                "old_json": None,
                "new_json": json.dumps(new),
                "action_index": self.cursor,
            }
        )

    def read(self, db, route, sql, rows):
        self.trace.append(
            {
                "ts": 1.0,
                "action_index": self.cursor,
                "route": route,
                "method": "GET",
                "db": db,
                "sql": sql,
                "rows_returned": len(rows),
                "rows": rows,
                "params": [],
            }
        )


class FakeEnv:
    def __init__(self, control):
        self.control = control
        self.instruction = "Move Jim's 1:1"

    def reset(self, task_config=None, soft=False):
        return {"screenshot": b"shot0", "accessibility_tree": None, "instruction": self.instruction}

    def step(self, action, pause=2.0):
        if isinstance(action, str) and "click" in action:
            self.control.read(
                "hoolicalendar",
                "/api/events/530",
                "SELECT * FROM events WHERE id = ?",
                [{"id": 530, "title": "1:1 with Jim Halpert", "start_at": "2026-04-07T11:00:00"}],
            )
        if isinstance(action, str) and "write" in action:
            self.control.write(
                "hoolicalendar", "events", 530, "UPDATE", {"start_at": "2026-04-08T11:00:00"}
            )
        done = isinstance(action, str) and action.strip().upper() == "DONE"
        return (
            {
                "screenshot": b"shot" + str(len(self.control.cursor_log)).encode(),
                "accessibility_tree": "tree",
            },
            0.0,
            done,
            {},
        )

    def _execute_shell(self, command):
        self.control.trace.append(
            {
                "ts": 1.0,
                "action_index": self.control.cursor,
                "source": "cli",
                "db": "/data/workbuzz.sqlite",
                "sql": command,
                "rows_returned": 1,
            }
        )
        return {"returncode": 0, "output": "371\n"}


class HarnessTests(unittest.TestCase):
    def _run(self):
        control = FakeControl()
        client = DerailControlClient("http://fake", transport=control.transport)
        env = FakeEnv(control)
        wrapper = DerailEnvWrapper(
            env,
            client,
            lambda: TraceBuilder(
                rollout_id="r1",
                task_id="mw-001",
                world_id="miniworld",
                agent="fake",
                seed=0,
                step_budget=100,
            ),
        )
        wrapper.reset(task_config={"id": "mw-001"})
        wrapper.note_thought("open the event")
        wrapper.step("pyautogui.click(400, 300)")
        wrapper._execute_shell("sqlite3 /data/workbuzz.sqlite 'select count(*) from messages'")
        wrapper.step("pyautogui.write('2026-04-08')")
        wrapper.step("DONE")
        return control, wrapper, wrapper.finish(step_budget_reached=False)

    def test_cursor_precedes_every_action_and_reset_writes_minus_one(self):
        control, wrapper, trace = self._run()
        self.assertEqual(control.cursor_log, [-1, 0, 1, 2, 3])
        self.assertEqual(trace["changelog_start_seq"], {"hoolicalendar": 40, "workbuzz": 7})
        self.assertEqual(wrapper.actions_taken, 4)

    def test_screenshot_sink_receives_post_action_bytes(self):
        control = FakeControl()
        screenshots = []
        wrapper = DerailEnvWrapper(
            FakeEnv(control),
            DerailControlClient("http://fake", transport=control.transport),
            lambda: TraceBuilder("r", "t", "w", "a", 0, 10),
            screenshot_sink=lambda index, data: screenshots.append((index, data)),
        )
        wrapper.reset(task_config={"id": "t"})
        wrapper.step("pyautogui.click(1, 1)")
        self.assertEqual(screenshots, [(0, b"shot2")])

    def test_trace_validates_and_attributes_ledgers(self):
        control, _, trace = self._run()
        validate_schema(trace, "rollout_trace.schema.json", REPOSITORY)
        steps = trace["steps"]
        self.assertEqual([s["modality"] for s in steps], ["gui", "cli", "gui", "gui"])
        self.assertEqual(steps[0]["thought"], "open the event")
        self.assertEqual(steps[0]["pages"][0]["route"], "/api/events/530")
        self.assertEqual(steps[0]["observations"][0]["facts"][0]["entity"], "events:530")
        self.assertEqual(steps[0]["observations"][0]["facts"][0]["table"], "hoolicalendar.events")
        self.assertEqual(steps[1]["action"]["type"], "shell")
        self.assertEqual(steps[1]["params"][0]["name"], "command")
        self.assertEqual(steps[2]["delta"][0]["rowid"], 530)
        self.assertEqual(steps[2]["delta"][0]["action_index"], 2)
        self.assertEqual(steps[2]["params"], [{"name": "text", "value": "2026-04-08"}])
        self.assertEqual(steps[0]["screenshot_sha256"], hashlib.sha256(b"shot2").hexdigest())
        self.assertEqual(trace["outcome"]["declared_complete"], True)
        self.assertEqual(trace["outcome"]["budget_exhausted"], False)
        self.assertEqual(trace["modality"]["primary"], "gui")

    def test_sql_helpers(self):
        self.assertEqual(
            sql_tables("SELECT m.* FROM messages m JOIN channels c ON c.id = m.channel_id"),
            ["messages", "channels"],
        )
        record = {
            "db": "workbuzz",
            "route": "/api/x",
            "sql": "SELECT id, name FROM channels",
            "rows": {"truncated": True, "sample": [{"id": 1, "name": "general"}]},
        }
        obs = observation_from_trace_record(record, 3)
        self.assertEqual(
            obs["facts"],
            [
                {"table": "workbuzz.channels", "column": "id", "entity": "channels:1", "value": 1},
                {
                    "table": "workbuzz.channels",
                    "column": "name",
                    "entity": "channels:1",
                    "value": "general",
                },
            ],
        )
        self.assertIsNone(
            observation_from_trace_record({"db": "x", "sql": "SELECT 1", "rows": []}, 0)
        )
        self.assertEqual(
            pages_from_trace_records([record])[0]["rendered_fields"][1]["column"], "name"
        )


if __name__ == "__main__":
    unittest.main()


class ReplayRunnerTests(unittest.TestCase):
    def _make(self, flaky_at=None):
        from derail.harness.replay_runner import ReplayRunner

        state = {"control": None, "runs": 0}

        def make_wrapper():
            control = FakeControl()
            state["control"] = control
            env = FakeEnv(control)
            if flaky_at is not None and state["runs"] == flaky_at:
                env._execute_shell = lambda command: {
                    "returncode": 0,
                    "output": "",
                }
                original_step = env.step

                def broken_step(action, pause=2.0):
                    if "write" in str(action):
                        control.write("hoolicalendar", "events", 531, "UPDATE", {"start_at": "x"})
                        return {"screenshot": b"s"}, 0.0, False, {}
                    return original_step(action, pause)

                env.step = broken_step
            state["runs"] += 1
            client = DerailControlClient("http://fake", transport=control.transport)
            return DerailEnvWrapper(
                env,
                client,
                lambda: TraceBuilder(
                    rollout_id="r1-replay",
                    task_id="mw-001",
                    world_id="miniworld",
                    agent="fake",
                    seed=0,
                    step_budget=100,
                ),
            )

        return ReplayRunner(restore=lambda: None, make_wrapper=make_wrapper), state

    def _original(self):
        control = FakeControl()
        client = DerailControlClient("http://fake", transport=control.transport)
        wrapper = DerailEnvWrapper(
            FakeEnv(control),
            client,
            lambda: TraceBuilder(
                rollout_id="r1",
                task_id="mw-001",
                world_id="miniworld",
                agent="fake",
                seed=0,
                step_budget=100,
            ),
        )
        wrapper.reset(task_config={"id": "mw-001"})
        wrapper.step("pyautogui.click(400, 300)")
        wrapper._execute_shell("sqlite3 x 'select 1'")
        wrapper.step("pyautogui.write('2026-04-08')")
        wrapper.step("pyautogui.click(1, 1)")
        wrapper.step("DONE")
        return wrapper.finish(step_budget_reached=False)

    def test_consistent_replays_pass_and_validate(self):
        runner, state = self._make()
        record = runner.verify(
            trace=self._original(),
            task_config={"id": "mw-001"},
            repaired_prefix_action_indices=[0, 1],
            root_cause_action_index=2,
            depths=[0, 1, 5],
            repaired_prefix_ref="pr1",
            snapshot_sha256="0" * 64,
        )
        validate_schema(record, "replay_verification.schema.json", REPOSITORY)
        by_depth = {d["depth"]: d for d in record["depths"]}
        self.assertTrue(by_depth[0]["passed"] and by_depth[1]["passed"])
        self.assertEqual(len(by_depth[1]["attempts"]), 3)
        self.assertEqual(by_depth[5]["code"], "DEPTH_UNAVAILABLE")
        self.assertEqual(state["runs"], 6)

    def test_divergent_replay_is_a_mismatch(self):
        runner, _ = self._make(flaky_at=1)
        record = runner.verify(
            trace=self._original(),
            task_config={"id": "mw-001"},
            repaired_prefix_action_indices=[0, 1],
            root_cause_action_index=2,
            depths=[1],
            repaired_prefix_ref="pr1",
            snapshot_sha256="0" * 64,
        )
        depth = record["depths"][0]
        self.assertFalse(depth["passed"])
        self.assertEqual(depth["code"], "PREFIX_REPLAY_MISMATCH")
        self.assertEqual(depth["attempts"][1]["first_mismatch_action_index"], 2)
