"""Clean-prefix consensus, pre-root-only repair, and the shared depth-eligibility rule."""

import importlib.util
import unittest
from pathlib import Path

from derail.annotation.labels import Reversibility
from derail.annotation.records import Adjudication
from derail.canonical.actions import ClickAction, action_to_dict
from derail.canonical.trajectory import CanonicalStep
from derail.construction.cases import (
    SKIP_AFTER_EXPLICIT,
    SKIP_SHORT_SUFFIX,
    SKIP_UNOBSERVED,
    eligible_depths,
)
from derail.construction.repair import PrefixRepairError, RepairPatch


REPOSITORY = Path(__file__).resolve().parents[1]


def _load_script():
    path = REPOSITORY / "scripts" / "benchmark" / "prepare_clean_prefix.py"
    spec = importlib.util.spec_from_file_location("derail_prepare_clean_prefix", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


CLEAN = _load_script()


class CleanPrefixConsensusTests(unittest.TestCase):
    def setUp(self) -> None:
        self.steps = tuple(
            CanonicalStep(
                step_id=index,
                action=ClickAction(kind="click", x_px=10 + index, y_px=20),
                observation_before_sha256="",
            )
            for index in range(19)
        )
        self.adjudication = Adjudication(
            adjudication_id="adj-1",
            trajectory_id="traj-1",
            input_annotation_ids=("ann-1", "ann-2"),
            adjudicator_id="reviewer-3",
            root_cause_action_index=3,
            error_horizon_actions=2,
            identifiable_at_action_index=5,
            error_types=("wrong_subgoal",),
            reversibility=Reversibility.REVERSIBLE,
            resolution_rationale="Reviewers agree.",
            evidence_refs=("step-3",),
            taxonomy_version="draft",
        )

    def _proposal(self, reviewer: str, candidates=(1, 4)):
        return {
            "proposal_id": "proposal-" + reviewer,
            "trajectory_id": "traj-1",
            "source_trajectory_sha256": "a" * 64,
            "reviewer_id": reviewer,
            "root_cause_action_index": 3,
            "audit_end_action_index": 18,
            "audited_action_indices": list(range(19)),
            "drop_candidates": [
                {
                    "action_index_global": index,
                    "recovery_action_index": index + 1,
                    "old_action": action_to_dict(self.steps[index].action),
                    "self_recovered": True,
                    "persistent_state_effect": False,
                    "causal_to_root_or_task": False,
                    "reason": "Self-recovered detour at %d." % index,
                }
                for index in candidates
            ],
            "review_complete": True,
            "schema_version": "0.2.0",
        }

    def test_matching_proposals_create_drop_patches(self) -> None:
        audit, patches, rejected = CLEAN.build_consensus(
            case_id="case-1",
            adjudicator_id="reviewer-3",
            adjudication=self.adjudication,
            steps=self.steps,
            proposals=(
                (Path("proposal-1.json"), self._proposal("reviewer-1")),
                (Path("proposal-2.json"), self._proposal("reviewer-2")),
            ),
        )
        self.assertEqual(audit.audit_end_action_index, 18)
        self.assertEqual(audit.unrelated_error_action_indices, (1,))
        self.assertEqual([patch.operation for patch in patches], ["drop"])
        self.assertEqual([patch.step_id for patch in patches], [1])
        self.assertEqual(rejected, (4,))

    def test_single_reviewer_meets_the_default_and_min_reviewers_is_enforced(self) -> None:
        proposals = ((Path("proposal-1.json"), self._proposal("reviewer-1")),)
        audit, patches, _ = CLEAN.build_consensus(
            case_id="case-1",
            adjudicator_id="reviewer-1",
            adjudication=self.adjudication,
            steps=self.steps,
            proposals=proposals,
        )
        self.assertEqual(audit.reviewer_ids, ("reviewer-1",))
        with self.assertRaisesRegex(RuntimeError, "at least 2"):
            CLEAN.build_consensus(
                case_id="case-1",
                adjudicator_id="reviewer-1",
                adjudication=self.adjudication,
                steps=self.steps,
                proposals=proposals,
                min_reviewers=2,
            )

    def test_drop_at_or_after_root_is_never_a_patch(self) -> None:
        with self.assertRaises(PrefixRepairError):
            RepairPatch(
                patch_id="p",
                case_id="case-1",
                step_id=4,
                root_cause_step=3,
                old_action=self.steps[4].action,
                operation="drop",
                reason="post-root detour",
                annotator_id="reviewer-1",
                self_recovered=True,
                persistent_state_effect=False,
                causal_to_root_or_task=False,
            )

    def test_disagreement_requires_manual_adjudication(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "disagree"):
            CLEAN.build_consensus(
                case_id="case-1",
                adjudicator_id="reviewer-3",
                adjudication=self.adjudication,
                steps=self.steps,
                proposals=(
                    (Path("proposal-1.json"), self._proposal("reviewer-1")),
                    (Path("proposal-2.json"), self._proposal("reviewer-2", ())),
                ),
            )

    def test_root_cause_can_be_the_retained_recovery_boundary(self) -> None:
        proposals = []
        for reviewer in ("reviewer-1", "reviewer-2"):
            proposal = self._proposal(reviewer, (1, 2))
            for candidate in proposal["drop_candidates"]:
                candidate["recovery_action_index"] = 3
            proposals.append((Path("proposal-%s.json" % reviewer), proposal))

        audit, patches, _ = CLEAN.build_consensus(
            case_id="case-1",
            adjudicator_id="reviewer-3",
            adjudication=self.adjudication,
            steps=self.steps,
            proposals=tuple(proposals),
        )

        self.assertEqual(audit.unrelated_error_action_indices, (1, 2))
        self.assertEqual([patch.step_id for patch in patches], [1, 2])


class DepthEligibilityTests(unittest.TestCase):
    def test_skips_short_suffix_and_takeover_after_the_error_is_explicit(self) -> None:
        available, skipped = eligible_depths(3, 20, 13)
        self.assertEqual(available, (0, 5, 10))
        self.assertEqual(skipped[15], SKIP_AFTER_EXPLICIT)
        self.assertEqual(skipped[25], SKIP_SHORT_SUFFIX)

    def test_unobserved_horizon_is_configurable(self) -> None:
        self.assertEqual(eligible_depths(3, 20, None)[0], ())
        self.assertEqual(eligible_depths(3, 20, None)[1][0], SKIP_UNOBSERVED)
        self.assertEqual(eligible_depths(3, 20, None, require_error_explicit=False)[0], (0, 5, 10, 15))


if __name__ == "__main__":
    unittest.main()
