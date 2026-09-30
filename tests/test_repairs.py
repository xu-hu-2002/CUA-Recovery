"""Prefix cleaning preserves source IDs and rejects unsafe or stale decisions."""

import unittest

from derail.canonical.actions import ClickAction
from derail.canonical.trajectory import CanonicalStep
from derail.construction.repair import PrefixRepairError, RepairPatch, apply_repair_patches


def click(x: int) -> ClickAction:
    return ClickAction(kind="click", x_px=x, y_px=100)


class PrefixRepairTests(unittest.TestCase):
    def test_rejects_root_cause_or_suffix_repair(self) -> None:
        with self.assertRaises(PrefixRepairError):
            RepairPatch(
                patch_id="repair-1",
                case_id="case-1",
                step_id=4,
                root_cause_step=4,
                old_action=click(10),
                new_action=click(20),
                reason="missed target",
                annotator_id="ann-1",
            )

    def test_patch_is_applied_without_mutating_source(self) -> None:
        source = (
            CanonicalStep(
                step_id=0,
                action=click(10),
                observation_before_sha256="before-0",
                observation_after_sha256="after-0",
            ),
        )
        patch = RepairPatch(
            patch_id="repair-1",
            case_id="case-1",
            step_id=0,
            root_cause_step=3,
            old_action=click(10),
            new_action=click(20),
            reason="原动作点偏，未触发 Save",
            annotator_id="ann-1",
            evidence=("screenshot:sha256:abc",),
        )
        repaired = apply_repair_patches(source, (patch,))
        self.assertEqual(source[0].action, click(10))
        self.assertEqual(repaired[0].action, click(20))
        self.assertTrue(repaired[0].repaired)
        self.assertEqual(repaired[0].tool_result, "pending_replay")
        self.assertEqual(repaired[0].observation_after_sha256, "")

    def test_rejects_stale_patch(self) -> None:
        source = (
            CanonicalStep(step_id=0, action=click(30), observation_before_sha256="before"),
        )
        stale = RepairPatch(
            patch_id="repair-stale",
            case_id="case-1",
            step_id=0,
            root_cause_step=2,
            old_action=click(10),
            new_action=click(20),
            reason="旧版本 patch",
            annotator_id="ann-1",
        )
        with self.assertRaises(PrefixRepairError):
            apply_repair_patches(source, (stale,))

    def test_drop_removes_action_without_renumbering_later_steps(self) -> None:
        source = tuple(
            CanonicalStep(step_id=index, action=click(10 + index), observation_before_sha256="b")
            for index in range(4)
        )
        patch = RepairPatch(
            patch_id="drop-1",
            case_id="case-1",
            step_id=1,
            root_cause_step=2,
            old_action=source[1].action,
            operation="drop",
            reason="The agent immediately recovered and this click had no persistent effect.",
            annotator_id="ann-1",
            self_recovered=True,
            persistent_state_effect=False,
            causal_to_root_or_task=False,
        )
        cleaned = apply_repair_patches(source, (patch,))
        self.assertEqual([step.action_index_global for step in cleaned], [0, 2, 3])
        self.assertEqual(cleaned[1].tool_result, "pending_replay")

    def test_drop_requires_explicit_noncausal_self_recovery_decision(self) -> None:
        with self.assertRaises(PrefixRepairError):
            RepairPatch(
                patch_id="unsafe-drop",
                case_id="case-1",
                step_id=1,
                root_cause_step=3,
                old_action=click(10),
                operation="drop",
                reason="Looks redundant.",
                annotator_id="ann-1",
                self_recovered=True,
                persistent_state_effect=True,
                causal_to_root_or_task=False,
            )

    def test_later_replacement_does_not_reuse_state_after_an_earlier_patch(self) -> None:
        source = tuple(
            CanonicalStep(
                step_id=index,
                action=click(10 + index),
                observation_before_sha256="before-%d" % index,
                observation_before_uri="image-%d.png" % index,
            )
            for index in range(3)
        )
        patches = (
            RepairPatch(
                patch_id="replace-0",
                case_id="case-1",
                step_id=0,
                root_cause_step=3,
                old_action=source[0].action,
                new_action=click(20),
                reason="Correct the first click.",
                annotator_id="ann-1",
            ),
            RepairPatch(
                patch_id="replace-2",
                case_id="case-1",
                step_id=2,
                root_cause_step=3,
                old_action=source[2].action,
                new_action=click(30),
                reason="Correct the later click.",
                annotator_id="ann-1",
            ),
        )
        repaired = apply_repair_patches(source, patches)
        self.assertEqual(repaired[0].observation_before_sha256, "before-0")
        self.assertEqual(repaired[2].observation_before_sha256, "")
        self.assertEqual(repaired[2].observation_before_uri, "")


if __name__ == "__main__":
    unittest.main()
