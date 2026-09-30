"""Recovery generation loop with fakes, and command-line file facts in the trace builder."""

from __future__ import annotations

import unittest
from dataclasses import replace
from pathlib import Path

from derail.harness.trace_builder import observation_from_shell
from derail.train.recovery_gen import (
    HINT_LEVELS,
    RecoveryConfig,
    generate_recovery,
    hint_leaks,
    hint_text,
    recovery_sample,
    sample_key,
)

REPOSITORY = Path(__file__).resolve().parents[1]
CASE = {
    "sample_id": "r1_d0",
    "workflow_id": "mw-002",
    "rollout_id": "r1",
    "agent": "qwen3_5_35b_a3b",
    "split": "train",
    "root_cause_action_index": 5,
    "depth": 0,
    "cut_action_index": 5,
    "input": {
        "instruction": "DM Jim the Dundies date",
        "history": [
            {"action_index": i, "action": {"type": "click"}, "observation_ref": "obs:%d" % i}
            for i in range(6)
        ],
    },
    "evidence_refs": {
        "contradiction_fact": {
            "column": "start_at",
            "value": "2026-07-02T18:00:00",
            "entity": "events:545",
        },
        "carried_value_ref": {"value": "2026-07-03", "source_action_index": 5},
        "affected_write_ref": {
            "action_index": 6,
            "db": "workbuzz",
            "tbl": "dm_messages",
            "rowid": 45,
        },
        "verified": True,
    },
}
WEIGHTS = {"input": 0.0, "thought": 1.0, "action": 1.0}


def _run(config, thought="correcting", teacher_policy=None, fixed_by="compensate"):
    world = {"fixed": False}
    seen = []

    def policy(instruction, history, observation, hint):
        seen.append((hint, list(history)))
        if fixed_by in hint and not world["fixed"]:
            return {"type": "fix", "thought": thought}
        return {"type": "done"}

    def step(action):
        if action["type"] == "fix":
            world["fixed"] = True
        return "obs"

    result = generate_recovery(
        CASE,
        restore=lambda hint, teacher: world.update(fixed=False) or "obs",
        policy=policy,
        step=step,
        verify=lambda: world["fixed"],
        config=config,
        teacher_policy=teacher_policy,
    )
    return result, seen


class RecoveryTests(unittest.TestCase):
    def test_config_defaults_follow_the_paper(self):
        config = RecoveryConfig.from_yaml(REPOSITORY / "configs/train/sft_v1.yaml")
        self.assertEqual(config.levels, HINT_LEVELS)
        self.assertEqual(config.attempts_per_level, 8)
        self.assertEqual((config.teacher_attempts, config.teacher_level), (1, "l3_state_diff"))
        self.assertEqual(config.max_recovery_steps, 100)
        self.assertEqual(config.base_agent, "qwen3_5_35b_a3b")

    def test_hint_levels_escalate(self):
        l1, l2, l3 = (hint_text(level, CASE) for level in HINT_LEVELS)
        self.assertEqual(l1, "Step 5 was wrong.")
        self.assertIn("2026-07-03", l2)
        self.assertIn("compensate", l3)
        self.assertTrue(len(l1) < len(l2) < len(l3))

    def test_rejection_sampling_conditions_on_the_history(self):
        config = RecoveryConfig(attempts_per_level=2, max_recovery_steps=3, leak_policy="off")
        result, seen = _run(config)
        self.assertEqual(result.hint_level, "l3_state_diff")
        self.assertEqual(
            [a.level for a in result.attempts],
            ["l1_step", "l1_step", "l2_evidence", "l2_evidence", "l3_state_diff"],
        )
        # every attempt starts from H_c; the continuation extends it after c
        self.assertEqual(seen[0][1], CASE["input"]["history"])
        self.assertEqual(seen[-1][1][-1]["action_index"], 6)
        sample = recovery_sample(CASE, result, WEIGHTS)
        self.assertEqual(sample["input"]["history"], CASE["input"]["history"])
        self.assertEqual(sample["target"]["steps"][0]["action_index"], 6)
        self.assertEqual((sample["hint_level"], sample["teacher"]), ("l3_state_diff", "self"))
        self.assertNotIn("compensate", str(sample["input"]))  # the hint is not in the input
        self.assertEqual(sample_key(sample), sample_key(dict(sample, sample_id="other")))

    def test_teacher_is_last_resort(self):
        config = RecoveryConfig(attempts_per_level=1, max_recovery_steps=2, leak_policy="off")
        result, _ = _run(config, fixed_by="never")
        self.assertIsNone(result.accepted)
        self.assertIsNone(recovery_sample(CASE, result, WEIGHTS))
        result, _ = _run(
            config, fixed_by="never", teacher_policy=lambda *a: {"type": "fix", "thought": "ok"}
        )
        self.assertEqual(result.accepted.teacher, "external")
        self.assertEqual(result.accepted.level, "l3_state_diff")
        self.assertEqual(len(result.attempts), 4)

    def test_step_budget(self):
        config = RecoveryConfig(attempts_per_level=1, max_recovery_steps=2, leak_policy="off")
        result = generate_recovery(
            CASE,
            restore=lambda hint, teacher: "obs",
            policy=lambda *a: {"type": "wait"},
            step=lambda a: "obs",
            verify=lambda: False,
            config=config,
        )
        self.assertTrue(all(len(a.steps) == 2 for a in result.attempts))

    def test_hint_leak_filter(self):
        config = RecoveryConfig(leak_patterns=(r"\bhint\b",))
        hint = hint_text("l3_state_diff", CASE)
        clean = [{"thought": "The event starts on 2026-07-02, so the DM date is wrong."}]
        self.assertEqual(hint_leaks(hint, clean, CASE, config), [])
        for leaky in (
            "Step 5 was wrong, fix it.",
            "The hint says to compensate that first, then redo the step.",
            "I must fix dm_messages first.",
        ):
            self.assertTrue(hint_leaks(hint, [{"thought": leaky}], CASE, config), leaky)
        leaky = "As noted, step 5 was wrong. I will open the calendar."
        # drop: no sample; resample: the schedule goes on; redact: the sentence goes
        base = RecoveryConfig(attempts_per_level=1, max_recovery_steps=2)
        self.assertIsNone(_run(replace(base, leak_policy="drop"), thought=leaky)[0].accepted)
        resampled, _ = _run(
            replace(base, leak_policy="resample"),
            thought=leaky,
            teacher_policy=lambda *a: {"type": "fix", "thought": "Opening the calendar."},
        )
        self.assertEqual(resampled.accepted.teacher, "external")
        redacted = _run(replace(base, leak_policy="redact"), thought=leaky)[0].accepted
        self.assertEqual(redacted.steps[0]["thought"], "I will open the calendar.")
        self.assertTrue(redacted.leaks)

    def test_shell_file_fact(self):
        event = observation_from_shell(
            "cat /home/user/Documents/Tax_2025/w2_summary.txt", "Employer: Dunder Mifflin", 3
        )
        self.assertEqual(event["facts"][0]["entity"], "file:Documents/Tax_2025/w2_summary.txt")
        self.assertEqual(event["source"], "cli")
        self.assertIsNone(observation_from_shell("ls -la", "x", 3))
        self.assertIsNone(observation_from_shell("cat a.txt", "", 3))


if __name__ == "__main__":
    unittest.main()
