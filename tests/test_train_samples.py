"""Training samples from the miniworld traces: recovery cases (paper format), the token-matched
success control, and the v1.2 step-level samples (base, detection, negative, weights)."""

from __future__ import annotations

import json
import os
import random
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

from derail.derived.layout import DEPTH_GRID
from derail.derived.schema import validate_schema
from derail.failure_analysis.run import AnalysisConfig, analyze_failure
from derail.ir.gold_interpreter import GoldInterpreter, InterpreterConfig, WorldCopy
from derail.ir.model import load_task_ir
from derail.train.build_samples import (
    BuildConfig,
    balance_check_weights,
    base_samples,
    dataset_manifest,
    detection_samples,
    ledger_upto,
    negative_samples,
    recovery_cases,
    success_samples,
    target_loss_weights,
    token_estimate,
    token_matched,
    verification_samples,
)
from derail.train.recovery_gen import RecoveryAttempt, RecoveryResult, recovery_sample
from derail.world.sqlite_fixture import build_world

REPOSITORY = Path(__file__).resolve().parents[1]
MINIWORLD = REPOSITORY / "tests" / "miniworld"
SEEDS = {
    "hoolicalendar": MINIWORLD / "hoolicalendar.seed.json",
    "workbuzz": MINIWORLD / "workbuzz.seed.json",
}
SPLIT = {"mw-001": "test", "mw-002": "train"}
CONFIG = replace(
    BuildConfig.from_yaml(REPOSITORY / "configs/train/sft_v1.yaml"),
    split_function=SPLIT.get,
)
WEIGHTED = replace(CONFIG, objective="weighted", format="steps")


def _trace(name):
    trace = json.loads((MINIWORLD / "traces" / (name + ".json")).read_text())
    trace["instruction"] = "fixture instruction"
    return trace


class TrainSampleTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.task_ir = load_task_ir(MINIWORLD / "task_ir" / "mw-002.json", REPOSITORY)
        with tempfile.TemporaryDirectory(dir=os.environ.get("DERAIL_TMP_ROOT")) as tmp:
            dbs = build_world(SEEDS, Path(tmp) / "w")
            with WorldCopy.open("miniworld", dbs, Path(tmp) / "g") as world:
                cls.gold = GoldInterpreter(
                    InterpreterConfig.from_yaml(
                        REPOSITORY / "configs/synthesis/gold_interpreter_v1.yaml"
                    )
                ).run(cls.task_ir, world, REPOSITORY)
        cls.analysis = analyze_failure(
            _trace("mw002_param_failure"),
            cls.gold,
            cls.task_ir,
            AnalysisConfig.from_repo(REPOSITORY),
        )

    def test_ledger_keeps_values_used_later(self):
        trace = _trace("mw002_success")
        ledger = ledger_upto(trace, 4, CONFIG)
        names = [e["name"] for e in ledger]
        self.assertIn("hoolicalendar.events.start_at@events:545", names)  # typed at step 5
        self.assertNotIn("hoolicalendar.events.location@events:545", names)  # never used

    def test_base_and_negative_samples_validate(self):
        trace = _trace("mw002_success")
        samples = base_samples(trace, CONFIG)
        self.assertEqual(len(samples), 7)  # eight steps minus DONE
        for sample in samples:
            validate_schema(sample, "training_sample.schema.json", REPOSITORY)
        self.assertEqual(samples[5]["target"]["check"], "consistent")
        self.assertTrue(
            any(e["name"].endswith("@events:545") for e in samples[5]["target"]["ledger"])
        )
        negatives = negative_samples(trace, None, CONFIG, 3, random.Random(0))
        self.assertEqual(len(negatives), 3)
        self.assertEqual(negatives[0]["sample_kind"], "negative")

    def test_detection_samples_are_program_verified(self):
        trace = _trace("mw002_param_failure")
        samples = detection_samples(trace, self.analysis, CONFIG)
        self.assertEqual(
            [s["truncation_action_index"] for s in samples], [7]
        )  # evidence at 7, last 8: offsets 0 only
        sample = samples[0]
        validate_schema(sample, "training_sample.schema.json", REPOSITORY)
        refs = sample["evidence_refs"]
        self.assertTrue(refs["verified"])
        self.assertEqual(refs["contradiction_fact"]["entity"], "events:545")
        self.assertEqual(refs["carried_value_ref"]["source_action_index"], 5)
        self.assertEqual(refs["affected_write_ref"]["tbl"], "dm_messages")
        self.assertEqual(sample["target"]["check"], "inconsistent")
        self.assertIsNone(sample["target"]["action"])
        self.assertEqual(len(sample["input"]["history"]), 7)

    def test_verification_only_for_censored_failures(self):
        trace = _trace("mw002_param_failure")
        self.assertEqual(
            verification_samples(trace, self.analysis, self.gold, self.task_ir, None, CONFIG), []
        )
        censored = dict(
            self.analysis, horizon_censored=True, earliest_identifiable_action_index=None
        )
        profile = {
            "provenance": {"node_horizons": {"n2": {"observability_class": "verifier_only"}}}
        }
        samples = verification_samples(trace, censored, self.gold, self.task_ir, profile, CONFIG)
        self.assertEqual(len(samples), 1)
        self.assertEqual(samples[0]["truncation_action_index"], 6)  # before the DM insert (R3)
        self.assertEqual(samples[0]["target"]["action"]["type"], "reread")

    def test_check_weights_and_manifest(self):
        success, failure = _trace("mw002_success"), _trace("mw002_param_failure")
        samples = base_samples(success, WEIGHTED) + detection_samples(
            failure, self.analysis, WEIGHTED
        )
        weights = balance_check_weights(samples, WEIGHTED)
        self.assertGreater(weights["inconsistent"], weights["consistent"])
        self.assertEqual(weights["consistent"], 1.0)
        manifest = dataset_manifest(samples, WEIGHTED, weights)
        self.assertEqual(manifest["by_kind"], {"base": 7, "detection": 1})
        self.assertEqual(manifest["by_split"], {"train": 8})
        # the paper's objective: no class balancing
        self.assertEqual(set(balance_check_weights(samples, CONFIG).values()), {1.0})

    def test_config_defaults_follow_the_paper(self):
        config = BuildConfig.from_yaml(REPOSITORY / "configs/train/sft_v1.yaml")
        self.assertEqual((config.format, config.objective), ("recovery", "nll"))
        self.assertEqual(target_loss_weights(config), {"input": 0.0, "thought": 1.0, "action": 1.0})
        self.assertEqual(target_loss_weights(WEIGHTED)["thought"], 0.5)
        training = config.training
        self.assertEqual(
            (training["epochs"], training["global_batch_size"], training["learning_rate"]),
            (1, 256, 1e-5),
        )
        self.assertEqual(
            (training["max_context_tokens"], training["max_images_per_sample"]), (32768, 20)
        )

    def test_recovery_cases_cut_at_root_plus_depth(self):
        failure = _trace("mw002_param_failure")
        analysis = dict(self.analysis, root_cause_action_index=2)
        cases = recovery_cases(failure, analysis, CONFIG)
        last = max(int(s["action_index"]) for s in failure["steps"])
        self.assertEqual([c["depth"] for c in cases], [d for d in DEPTH_GRID if 2 + d <= last])
        case = cases[-1]
        self.assertEqual(case["cut_action_index"], 2 + case["depth"])
        history = case["input"]["history"]
        self.assertEqual(
            [h["action_index"] for h in history], list(range(case["cut_action_index"] + 1))
        )
        self.assertTrue(all("thought" not in h for h in history))  # reasoning stripped
        self.assertEqual((case["workflow_id"], case["split"]), ("mw-002", "train"))
        self.assertEqual(case["evidence_refs"]["affected_write_ref"]["tbl"], "dm_messages")
        # test-split workflows contribute nothing
        heldout = replace(CONFIG, split_function=lambda w: "test")
        self.assertEqual(recovery_cases(failure, analysis, heldout), [])
        # the (l, H_c, r*) sample keeps H_c and no hint
        accepted = RecoveryAttempt(
            "l1_step", 1, [{"action_index": 5, "thought": "t", "action": {"type": "x"}}], True
        )
        sample = recovery_sample(
            case,
            RecoveryResult(case["sample_id"], accepted, [accepted]),
            target_loss_weights(CONFIG),
        )
        self.assertEqual(sample["input"]["history"], history)
        self.assertEqual(sample["loss_weights"]["input"], 0.0)

    def test_success_control_is_token_matched(self):
        success = _trace("mw002_success")
        success["outcome"] = dict(success.get("outcome") or {}, final_verifier=True)
        controls = []
        for k in range(6):
            trace = dict(success, rollout_id="s%d" % k)
            controls += success_samples(trace, CONFIG)
        self.assertEqual(len(controls), 6)
        self.assertEqual(controls[0]["input"]["history"], [])
        unit = token_estimate(controls[0], CONFIG)
        recovery = [dict(controls[0], workflow_id="mw-002", sample_id="r0")] * 3
        selected, report = token_matched(controls, recovery, CONFIG)
        self.assertEqual(len(selected), 3)
        self.assertTrue(report["within_tolerance"])
        self.assertAlmostEqual(report["matched_tokens"], round(3 * unit, 1))
        other = [dict(r, workflow_id="mw-009") for r in recovery]
        self.assertEqual(token_matched(controls, other, CONFIG)[0], [])  # same workflows only


if __name__ == "__main__":
    unittest.main()
