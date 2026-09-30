"""Automatic failure analysis on the miniworld traces with their planted root causes."""

from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path

from derail.derived.schema import validate_schema
from derail.failure_analysis.detectors import edit_distance
from derail.annotation.labels import Reversibility
from derail.annotation.records import HumanAnnotation
from derail.failure_analysis.run import AnalysisConfig, analyze_failure, annotation_proposal
from derail.ir.gold_interpreter import GoldInterpreter, InterpreterConfig, WorldCopy
from derail.ir.model import load_task_ir
from derail.world.sqlite_fixture import build_world

REPOSITORY = Path(__file__).resolve().parents[1]
MINIWORLD = REPOSITORY / "tests" / "miniworld"
SEEDS = {
    "hoolicalendar": MINIWORLD / "hoolicalendar.seed.json",
    "workbuzz": MINIWORLD / "workbuzz.seed.json",
}


class FailureAnalysisTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = AnalysisConfig.from_repo(REPOSITORY)
        cls.gold, cls.irs = {}, {}
        with tempfile.TemporaryDirectory(dir=os.environ.get("DERAIL_TMP_ROOT")) as tmp:
            dbs = build_world(SEEDS, Path(tmp) / "world")
            interpreter = GoldInterpreter(
                InterpreterConfig.from_yaml(
                    REPOSITORY / "configs/synthesis/gold_interpreter_v1.yaml"
                )
            )
            for task in ("mw-001", "mw-002"):
                cls.irs[task] = load_task_ir(MINIWORLD / "task_ir" / (task + ".json"), REPOSITORY)
                with WorldCopy.open("miniworld", dbs, Path(tmp) / task) as world:
                    cls.gold[task] = interpreter.run(cls.irs[task], world, REPOSITORY)

    def _trace(self, name):
        return json.loads((MINIWORLD / "traces" / (name + ".json")).read_text())

    def test_state_failure_root_cause_type_and_horizon(self):
        trace = self._trace("mw001_state_failure")
        record = analyze_failure(trace, self.gold["mw-001"], self.irs["mw-001"], self.config)
        validate_schema(record, "failure_analysis.schema.json", REPOSITORY)
        truth = trace["provenance"]["ground_truth"]
        self.assertEqual(record["root_cause_action_index"], truth["root_cause_action_index"])
        self.assertEqual(record["root_cause_detector"], "state")
        self.assertEqual(
            record["detectors"]["state"][0]["evidence_pattern"], "wrong_entity_co_displayed"
        )
        self.assertEqual(
            (record["paper_type"], record["paper_category"], record["group"]),
            ("misunderstand_task_objective", "planning", "long_horizon"),
        )
        self.assertEqual(
            record["earliest_identifiable_action_index"],
            truth["earliest_identifiable_action_index"],
        )
        self.assertEqual(record["action_horizon"], 1)
        self.assertEqual(record["semantic_horizon"], 0)
        self.assertFalse(record["horizon_censored"])
        self.assertIsNone(record["residual_code"])

    def test_parameter_failure_is_a_transcription_error(self):
        trace = self._trace("mw002_param_failure")
        record = analyze_failure(trace, self.gold["mw-002"], self.irs["mw-002"], self.config)
        validate_schema(record, "failure_analysis.schema.json", REPOSITORY)
        truth = trace["provenance"]["ground_truth"]
        self.assertEqual(record["root_cause_action_index"], truth["root_cause_action_index"])
        self.assertEqual(record["root_cause_detector"], "parameter")
        self.assertEqual(
            (record["paper_type"], record["group"]), ("typing_or_parameter_error", "local")
        )
        self.assertEqual([c["action_index"] for c in record["detectors"]["state"]], [6])
        self.assertEqual(
            record["earliest_identifiable_action_index"],
            truth["earliest_identifiable_action_index"],
        )
        self.assertEqual(record["earliest_identifiable_node_id"], "n2")
        self.assertEqual(record["action_horizon"], 2)
        self.assertEqual(record["semantic_horizon"], 1)

    def test_success_trace_yields_no_root_cause(self):
        trace = self._trace("mw002_success")
        record = analyze_failure(trace, self.gold["mw-002"], self.irs["mw-002"], self.config)
        self.assertIsNone(record["root_cause_action_index"])
        self.assertEqual(record["detectors"]["state"], [])
        self.assertEqual(record["detectors"]["parameter"], [])

    def test_termination_without_trace_is_typed_from_outcome(self):
        trace = self._trace("mw002_success")
        trace["outcome"].update({"final_verifier": False, "declared_complete": True})
        trace["steps"] = trace["steps"][:4]
        record = analyze_failure(trace, self.gold["mw-002"], self.irs["mw-002"], self.config)
        self.assertEqual(
            (record["paper_type"], record["group"]), ("premature_completion", "termination")
        )
        self.assertEqual(record["root_cause_action_index"], 3)
        trace["outcome"].update({"declared_complete": False, "budget_exhausted": True})
        record = analyze_failure(trace, self.gold["mw-002"], self.irs["mw-002"], self.config)
        self.assertEqual(record["paper_type"], "hit_budget_limit")

    def test_proposal_uses_the_human_annotation_layout_and_routes_by_confidence(self):
        trace = self._trace("mw002_param_failure")
        record = analyze_failure(trace, self.gold["mw-002"], self.irs["mw-002"], self.config)
        threshold = self.config.rules.residual_threshold
        proposal = annotation_proposal(
            record,
            trajectory_id="traj-1",
            source_trajectory_sha256="a" * 64,
            annotator_id="rerail_auto",
            taxonomy_version="frozen",
            confidence_threshold=threshold,
        )
        self.assertEqual(proposal["review_route"], "human_verify")
        self.assertEqual(proposal["annotator_role"], "auto")
        verified = HumanAnnotation.from_dict(
            {**proposal, "reversibility": Reversibility.REVERSIBLE.value}
        )
        self.assertEqual(verified.root_cause_action_index, record["root_cause_action_index"])
        self.assertEqual(verified.error_horizon_actions, record["action_horizon"])
        low = annotation_proposal(
            {**record, "type_confidence": threshold - 0.01},
            trajectory_id="traj-1",
            source_trajectory_sha256="a" * 64,
            annotator_id="rerail_auto",
            taxonomy_version="frozen",
            confidence_threshold=threshold,
        )
        self.assertEqual(low["review_route"], "human_annotate")

    def test_edit_distance(self):
        self.assertEqual(edit_distance("2026-07-02", "2026-07-03"), 1)
        self.assertGreater(edit_distance("abc", "abcdefgh"), 3)


if __name__ == "__main__":
    unittest.main()
