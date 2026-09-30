"""Prefix repair, depth instantiation and packaging on the miniworld failure traces."""

from __future__ import annotations

import copy
import json
import os
import tempfile
import unittest
from pathlib import Path

import yaml

from derail.cases.instantiate import build_cases, dedup_records, effects_within
from derail.cases.package import funnel, write_bundle
from derail.cases.repair import net_change, neutral_segments, repair_prefix
from derail.derived.schema import validate_schema
from derail.detect.latent_static import static_latent_horizons
from derail.detect.profile import build_profile
from derail.failure_analysis.run import AnalysisConfig, analyze_failure
from derail.ir.gold_interpreter import GoldInterpreter, InterpreterConfig, WorldCopy
from derail.ir.model import load_task_ir
from derail.longhorizon.ontology import Ontology
from derail.world.sqlite_fixture import build_world

REPOSITORY = Path(__file__).resolve().parents[1]
MINIWORLD = REPOSITORY / "tests" / "miniworld"
SEEDS = {
    "hoolicalendar": MINIWORLD / "hoolicalendar.seed.json",
    "workbuzz": MINIWORLD / "workbuzz.seed.json",
}
DEPTHS = list(
    yaml.safe_load((REPOSITORY / "configs/benchmark/derail_v1.yaml").read_text())["depths"]
)
ONTOLOGY = Ontology.from_yaml(REPOSITORY / "configs/synthesis/ontology_v0.2.yaml")


def _trace(name):
    return json.loads((MINIWORLD / "traces" / (name + ".json")).read_text())


def _with_detour(trace):
    """Insert a misclick + back (no delta, same page) before the root cause and shift indices."""

    trace = copy.deepcopy(trace)
    steps = trace["steps"]
    base = steps[0]
    detour_a = {
        **copy.deepcopy(base),
        "action_index": 1,
        "action": {"type": "click", "raw": "pyautogui.click(1, 1)"},
        "params": [],
        "delta": [],
        "observations": [],
        "pages": [{"app": "hoolicalendar", "route": "/settings", "rendered_fields": []}],
    }
    detour_b = {
        **copy.deepcopy(base),
        "action_index": 2,
        "action": {"type": "click", "raw": "pyautogui.click(2, 2)"},
        "params": [],
        "delta": [],
        "observations": [],
        "pages": [
            {"app": "hoolicalendar", "route": "/calendar/week/2026-04-06", "rendered_fields": []}
        ],
    }
    shifted = []
    for step in steps[1:]:
        step = copy.deepcopy(step)
        step["action_index"] += 2
        for row in step["delta"]:
            row["action_index"] += 2
        for obs in step["observations"]:
            obs["action_index"] += 2
        shifted.append(step)
    trace["steps"] = [steps[0], detour_a, detour_b] + shifted
    trace["provenance"]["ground_truth"]["root_cause_action_index"] += 2
    trace["provenance"]["ground_truth"]["earliest_identifiable_action_index"] += 2
    return trace


class RepairTests(unittest.TestCase):
    def test_detour_is_removed_and_original_prefix_unchanged(self):
        trace = _with_detour(_trace("mw001_state_failure"))
        root = trace["provenance"]["ground_truth"]["root_cause_action_index"]
        segments = neutral_segments(trace, root)
        self.assertEqual(
            [(s["start_action_index"], s["end_action_index"]) for s in segments], [(1, 2)]
        )
        record = repair_prefix(trace, root)
        validate_schema(record, "prefix_repair.schema.json", REPOSITORY)
        self.assertEqual(record["status"], "repaired")
        self.assertEqual(record["repaired_prefix_action_indices"], [0, 3, 4])
        plain = repair_prefix(_trace("mw001_state_failure"), 3)
        self.assertEqual(plain["status"], "unchanged")
        self.assertEqual(plain["repaired_prefix_action_indices"], [0, 1, 2])

    def test_non_neutral_pre_root_is_unrepairable(self):
        trace = _trace("mw001_state_failure")
        trace["steps"][1]["delta"].append(
            {
                "schema_version": "changelog-row/1.0",
                "db": "hoolicalendar",
                "seq": 9,
                "ts": 1.0,
                "tbl": "events",
                "rowid": 529,
                "op": "UPDATE",
                "old_json": json.dumps({"location": "Conf A"}),
                "new_json": json.dumps({"location": "Conf Z"}),
                "action_index": 1,
            }
        )
        record = repair_prefix(trace, 3, writes_gold=[])
        self.assertEqual(record["status"], "PREFIX_UNREPAIRABLE")
        self.assertEqual(record["non_neutral_segments"][0]["start_action_index"], 1)
        undo = copy.deepcopy(trace["steps"][1]["delta"][0])
        undo.update(
            {
                "seq": 10,
                "old_json": json.dumps({"location": "Conf Z"}),
                "new_json": json.dumps({"location": "Conf A"}),
                "action_index": 2,
            }
        )
        trace["steps"][2]["delta"].append(undo)
        self.assertEqual(net_change(trace["steps"][:3]), {})


class InstantiateTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        with tempfile.TemporaryDirectory(dir=os.environ.get("DERAIL_TMP_ROOT")) as tmp:
            dbs = build_world(SEEDS, Path(tmp) / "world")
            interpreter = GoldInterpreter(
                InterpreterConfig.from_yaml(
                    REPOSITORY / "configs/synthesis/gold_interpreter_v1.yaml"
                )
            )
            cls.task_ir = load_task_ir(MINIWORLD / "task_ir" / "mw-002.json", REPOSITORY)
            with WorldCopy.open("miniworld", dbs, Path(tmp) / "g") as world:
                cls.gold = interpreter.run(cls.task_ir, world, REPOSITORY)
        cls.analysis = analyze_failure(
            _trace("mw002_param_failure"),
            cls.gold,
            cls.task_ir,
            AnalysisConfig.from_repo(REPOSITORY),
        )
        cls.profile = build_profile(
            cls.task_ir,
            static_latent_horizons(cls.task_ir, cls.gold),
            ["1", "2-3", ">=4", "verifier_only"],
            [">=4"],
        )

    def test_cases_carry_section_10_4_fields_and_validate(self):
        trace = _trace("mw002_param_failure")
        repair = repair_prefix(
            trace, self.analysis["root_cause_action_index"], self.gold["writes_gold"]
        )
        cases, skipped = build_cases(
            trace, self.analysis, self.gold, repair, self.profile, DEPTHS, ONTOLOGY, step_budget=100
        )
        self.assertEqual(
            [c["error_depth"] for c in cases], [0]
        )  # root 5, last 8: only d=0 fits {0,5,...}
        self.assertTrue(all(d in skipped for d in DEPTHS if d > 3))
        case = cases[0]
        validate_schema(case, "derail_case.schema.json", REPOSITORY)
        self.assertEqual(case["schema_version"], "derail-case/1.0")
        self.assertEqual(case["paper_type"], "typing_or_parameter_error")
        self.assertEqual(case["measured_action_horizon"], 2)
        self.assertEqual(
            case["predicted_observability_class"],
            self.profile["provenance"]["node_horizons"]["n3"]["observability_class"],
        )
        self.assertEqual(case["prefix_modality"], "gui")
        # root=5, d=0 covers only step 5 (the typing), before the DM insert at 6: reversible
        self.assertEqual(case["reversibility_stratum"], "reversible")
        effects = effects_within(trace, self.gold, 5, 6)
        self.assertEqual(effects[0]["reversibility_class"], "R3")

    def test_dedup_and_bundle(self):
        trace = _trace("mw002_param_failure")
        repair = repair_prefix(trace, 5, self.gold["writes_gold"])
        cases, _ = build_cases(
            trace, self.analysis, self.gold, repair, self.profile, DEPTHS, ONTOLOGY
        )
        twin = copy.deepcopy(cases[0])
        twin.update(
            {
                "case_id": twin["case_id"] + "_twin",
                "source_rollout_id": "r-twin",
                "root_cause_action_index": 6,
            }
        )
        kept, removed = dedup_records(cases + [twin])
        self.assertEqual(len(kept), 1)
        self.assertEqual(removed[0]["status"], "DUPLICATE_CASE")
        self.assertEqual(
            {kept[0]["case_id"], *kept[0]["merged_from"]}, {cases[0]["case_id"], twin["case_id"]}
        )
        with tempfile.TemporaryDirectory(dir=os.environ.get("DERAIL_TMP_ROOT")) as tmp:
            manifest = write_bundle(
                tmp,
                kept,
                removed,
                [{"failed": True, "analyzed": True, "repaired": True, "cases": 1}],
                DEPTHS,
                {"test": True},
            )
            self.assertEqual(manifest["cases"], 1)
            self.assertEqual(
                manifest["funnel"],
                {"runs": 1, "failures": 1, "auto_analyzed": 1, "repaired": 1, "cases": 1},
            )
            self.assertTrue((Path(tmp) / "cases" / (kept[0]["case_id"] + ".json")).is_file())
        self.assertEqual(funnel([]), {})


if __name__ == "__main__":
    unittest.main()


class GoldActionTests(unittest.TestCase):
    def test_gold_action_names_node_values_writes_and_path(self):
        from derail.cases.gold_action import gold_action

        analysis = {
            "root_cause_node_id": "n4",
            "root_cause_detector": "parameter",
            "provenance": {"root_detail": {"wrong_value": 7, "correct_value": 1}},
        }
        gold = {
            "node_order": ["n1", "n2", "n3", "n4", "n5"],
            "values": [
                {"node_id": "n4", "name": "dm_id", "value": 1},
                {"node_id": "n5", "name": "message_id", "value": 9},
            ],
            "writes_gold": [
                {
                    "node_id": "n5",
                    "table": "workbuzz.dm_messages",
                    "column": "content",
                    "value": "x",
                }
            ],
        }
        task_ir = {
            "nodes": [
                {"node_id": "n4", "app": "workbuzz", "op": "resolve", "semantic_goal": "Find Jim"}
            ]
        }
        action = gold_action(analysis, gold, task_ir)
        self.assertEqual(action["kind"], "redo_node_with_gold_values")
        self.assertEqual(action["parameters"], {"dm_id": 1})
        self.assertEqual(action["continue_with"], ["n5"])
        self.assertEqual(action["gold_writes"], [])
        self.assertEqual((action["wrong_value"], action["correct_value"]), (7, 1))
        self.assertEqual(action["app"], "workbuzz")
        self.assertIsNone(gold_action({"root_cause_node_id": None}, gold, task_ir))
