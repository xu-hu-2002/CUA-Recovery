"""Human rubric review: coverage, disagreement bookkeeping, and judge comparison."""

import json
import unittest
from pathlib import Path

from derail.annotation.rubric_review import (
    HumanRubricReview,
    RubricJudgment,
    RubricReviewAdjudication,
    RubricReviewError,
    RubricVerdict,
    disagreement_rubric_ids,
    load_rubric_specs,
    summarize_rubric_agreement,
)
from derail.derived.schema import validate_schema

REPOSITORY = Path(__file__).resolve().parents[1]
TRAJ_SHA = "a" * 64
BUNDLE_SHA = "b" * 64

BUNDLE = {
    "grading_manifest": {
        "rubrics": [
            {"criterion": "opens the app", "type": "llm_judge", "weight": 0.2},
            {"criterion": "clicks check in", "type": "llm_judge", "weight": 0.5},
            {"criterion": "downloads the pass", "type": "llm_judge", "weight": 0.3},
        ]
    }
}


def _judgment(rubric_id: str, verdict: str) -> RubricJudgment:
    return RubricJudgment(
        rubric_id=rubric_id,
        verdict=RubricVerdict(verdict),
        rationale="screenshot %s shows it" % rubric_id,
        evidence_refs=("observation:action-3",),
    )


def _review(review_id: str, reviewer_id: str, verdicts) -> HumanRubricReview:
    return HumanRubricReview(
        review_id=review_id,
        trajectory_id="traj-1",
        source_trajectory_sha256=TRAJ_SHA,
        rubric_bundle_sha256=BUNDLE_SHA,
        reviewer_id=reviewer_id,
        judgments=tuple(_judgment(key, value) for key, value in verdicts.items()),
    )


class RubricReviewTest(unittest.TestCase):
    def test_rubric_ids_follow_the_upstream_positional_convention(self) -> None:
        # MyPCBench bundles carry no `id`; the judge synthesises R1..RN and the
        # human review must attach to exactly the same criteria.
        specs = load_rubric_specs(BUNDLE)
        self.assertEqual([item.rubric_id for item in specs], ["R1", "R2", "R3"])
        self.assertEqual([item.weight for item in specs], [0.2, 0.5, 0.3])
        self.assertEqual(specs[0].criterion, "opens the app")

    def test_missing_weight_defaults_to_one_like_the_judge(self) -> None:
        specs = load_rubric_specs({"grading_manifest": {"rubrics": [{"criterion": "x"}]}})
        self.assertEqual(specs[0].weight, 1.0)

    def test_partial_review_is_rejected(self) -> None:
        specs = load_rubric_specs(BUNDLE)
        review = _review("rev-1", "alice", {"R1": "success", "R2": "failure"})
        with self.assertRaises(RubricReviewError):
            review.validate_against_bundle(specs)

    def test_review_of_a_rubric_outside_the_bundle_is_rejected(self) -> None:
        specs = load_rubric_specs(BUNDLE)
        review = _review(
            "rev-1",
            "alice",
            {"R1": "success", "R2": "failure", "R3": "success", "R9": "success"},
        )
        with self.assertRaises(RubricReviewError):
            review.validate_against_bundle(specs)

    def test_empty_rationale_or_evidence_is_rejected(self) -> None:
        with self.assertRaises(RubricReviewError):
            RubricJudgment("R1", RubricVerdict.SUCCESS, "  ", ("observation:action-1",))
        with self.assertRaises(RubricReviewError):
            RubricJudgment("R1", RubricVerdict.SUCCESS, "why", ())

    def test_adjudication_needs_two_reviewers_and_the_true_disagreement_set(self) -> None:
        reviews = [
            _review("rev-1", "alice", {"R1": "success", "R2": "failure", "R3": "success"}),
            _review("rev-2", "bob", {"R1": "success", "R2": "success", "R3": "success"}),
        ]
        self.assertEqual(disagreement_rubric_ids(reviews), ("R2",))

        def _adjudication(disagreements):
            return RubricReviewAdjudication(
                adjudication_id="radj-1",
                trajectory_id="traj-1",
                source_trajectory_sha256=TRAJ_SHA,
                rubric_bundle_sha256=BUNDLE_SHA,
                input_review_ids=("rev-1", "rev-2"),
                adjudicator_id="carol",
                judgments=tuple(
                    _judgment(key, value)
                    for key, value in {
                        "R1": "success",
                        "R2": "failure",
                        "R3": "success",
                    }.items()
                ),
                resolution_rationale="R2 的下载确认从未出现",
                disagreement_rubric_ids=disagreements,
            )

        # Under-reporting disagreement would make the paper's agreement number
        # unfalsifiable, so it is a hard error rather than a warning.
        with self.assertRaises(RubricReviewError):
            _adjudication(()).validate_inputs(reviews)
        _adjudication(("R2",)).validate_inputs(reviews)

        same_person = [
            _review("rev-1", "alice", {"R1": "success", "R2": "failure", "R3": "success"}),
            _review("rev-2", "alice", {"R1": "success", "R2": "success", "R3": "success"}),
        ]
        with self.assertRaises(RubricReviewError):
            _adjudication(("R2",)).validate_inputs(same_person)

    def test_weighted_score_matches_the_llm_judge_formula(self) -> None:
        specs = load_rubric_specs(BUNDLE)
        adjudication = RubricReviewAdjudication(
            adjudication_id="radj-1",
            trajectory_id="traj-1",
            source_trajectory_sha256=TRAJ_SHA,
            rubric_bundle_sha256=BUNDLE_SHA,
            input_review_ids=("rev-1", "rev-2"),
            adjudicator_id="carol",
            judgments=(
                _judgment("R1", "success"),
                _judgment("R2", "failure"),
                _judgment("R3", "success"),
            ),
            resolution_rationale="见 R2 证据",
            disagreement_rubric_ids=("R2",),
            judge_model="gpt-5.6-terra",
            judge_verdicts={
                "R1": RubricVerdict.SUCCESS,
                "R2": RubricVerdict.SUCCESS,
                "R3": RubricVerdict.SUCCESS,
            },
        )
        score = adjudication.weighted_score(specs)
        self.assertAlmostEqual(score["weighted_fraction"], 0.5)
        self.assertEqual(score["score"], 50)
        self.assertFalse(score["perfect"])
        # The judge passed R2, the humans failed it: a false success.
        self.assertEqual(adjudication.judge_disagreement_rubric_ids(), ("R2",))

        payload = adjudication.to_dict()
        validate_schema(payload, "rubric_review_adjudication.schema.json", REPOSITORY)
        restored = RubricReviewAdjudication.from_dict(json.loads(json.dumps(payload)))
        self.assertEqual(restored.judge_disagreement_rubric_ids(), ("R2",))

    def test_review_round_trips_through_its_schema(self) -> None:
        review = _review(
            "rev-1", "alice", {"R1": "success", "R2": "failure", "R3": "success"}
        )
        payload = review.to_dict()
        validate_schema(payload, "rubric_review.schema.json", REPOSITORY)
        self.assertEqual(HumanRubricReview.from_dict(payload), review)

    def test_agreement_summary_splits_judge_error_direction(self) -> None:
        reviews = [
            _review("rev-1", "alice", {"R1": "success", "R2": "failure", "R3": "success"}),
            _review("rev-2", "bob", {"R1": "success", "R2": "success", "R3": "success"}),
        ]
        adjudication = RubricReviewAdjudication(
            adjudication_id="radj-1",
            trajectory_id="traj-1",
            source_trajectory_sha256=TRAJ_SHA,
            rubric_bundle_sha256=BUNDLE_SHA,
            input_review_ids=("rev-1", "rev-2"),
            adjudicator_id="carol",
            judgments=(
                _judgment("R1", "success"),
                _judgment("R2", "failure"),
                _judgment("R3", "failure"),
            ),
            resolution_rationale="见证据",
            disagreement_rubric_ids=("R2",),
            judge_model="gpt-5.6-terra",
            judge_verdicts={
                "R1": RubricVerdict.SUCCESS,
                "R2": RubricVerdict.SUCCESS,
                "R3": RubricVerdict.FAILURE,
            },
        )
        summary = summarize_rubric_agreement([reviews], [adjudication])
        self.assertEqual(summary["rubric_item_count"], 3)
        self.assertAlmostEqual(summary["inter_reviewer_item_agreement"], 2 / 3)
        self.assertEqual(summary["inter_reviewer_trajectory_agreement"], 0.0)
        self.assertAlmostEqual(summary["judge_human_item_agreement"], 2 / 3)
        self.assertEqual(summary["judge_marked_success_human_failure"], 1)
        self.assertEqual(summary["judge_marked_failure_human_success"], 0)


if __name__ == "__main__":
    unittest.main()
