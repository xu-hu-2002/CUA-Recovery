"""Legacy traj.jsonl -> minimal trace -> parameter detector, on a miniworld-derived failure."""

from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path

from derail.derived.schema import validate_schema
from derail.failure_analysis.retrospective import (
    agreement_summary,
    retrospective_row,
    trace_from_traj,
)
from derail.ir.gold_interpreter import GoldInterpreter, InterpreterConfig, WorldCopy
from derail.ir.model import load_task_ir
from derail.world.sqlite_fixture import build_world

REPOSITORY = Path(__file__).resolve().parents[1]
MINIWORLD = REPOSITORY / "tests" / "miniworld"
SEEDS = {
    "hoolicalendar": MINIWORLD / "hoolicalendar.seed.json",
    "workbuzz": MINIWORLD / "workbuzz.seed.json",
}


class RetrospectiveTests(unittest.TestCase):
    def test_legacy_trajectory_is_analysed_by_the_parameter_detector(self):
        rows = [
            {
                "step_num": 1,
                "action": "pyautogui.click(100, 100)",
                "response": "open calendar",
                "screenshot_file": "s1.png",
            },
            {
                "step_num": 2,
                "action": "pyautogui.click(200, 200)",
                "response": "open the event",
                "screenshot_file": "s2.png",
            },
            {
                "step_num": 3,
                "action": "pyautogui.write('Dundies is on 2026-07-03, save the date.')",
                "response": "type it",
                "screenshot_file": "s3.png",
            },
            {
                "step_num": 4,
                "action": "pyautogui.press('enter')",
                "response": "send",
                "screenshot_file": "s4.png",
            },
            {"step_num": 5, "action": "DONE", "response": "done", "screenshot_file": "s5.png"},
        ]
        trace = trace_from_traj(
            rows, task_id="mw-002", agent="legacy_agent", rollout_id="legacy-r1"
        )
        validate_schema(trace, "rollout_trace.schema.json", REPOSITORY)
        self.assertEqual(
            trace["steps"][2]["params"],
            [{"name": "text", "value": "Dundies is on 2026-07-03, save the date."}],
        )
        self.assertTrue(trace["outcome"]["declared_complete"])
        task_ir = load_task_ir(MINIWORLD / "task_ir" / "mw-002.json", REPOSITORY)
        with tempfile.TemporaryDirectory(dir=os.environ.get("DERAIL_TMP_ROOT")) as tmp:
            dbs = build_world(SEEDS, Path(tmp) / "w")
            with WorldCopy.open("miniworld", dbs, Path(tmp) / "g") as world:
                gold = GoldInterpreter(
                    InterpreterConfig.from_yaml(
                        REPOSITORY / "configs/synthesis/gold_interpreter_v1.yaml"
                    )
                ).run(task_ir, world, REPOSITORY)
        row = retrospective_row(trace, gold, human_root=2, human_type="typing_or_parameter_error")
        self.assertEqual(row["auto_root"], 2)
        self.assertTrue(row["exact"] and row["within_one"] and row["covered"])
        summary = agreement_summary(
            [row, dict(row, auto_root=None, covered=False, exact=False, within_one=False)]
        )
        self.assertEqual(summary["coverage"], 0.5)
        self.assertEqual(summary["exact_of_covered"], 1.0)


if __name__ == "__main__":
    unittest.main()
