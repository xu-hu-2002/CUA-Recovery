"""Root-cause error distribution (primary type) and the App. E annotation-protocol statistics."""

import importlib.util
import sys
import unittest
from pathlib import Path

REPOSITORY = Path(__file__).resolve().parents[1]


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module  # dataclasses resolve their module through sys.modules
    spec.loader.exec_module(module)
    return module


TAXONOMY = _load("derail_error_taxonomy_analyze", REPOSITORY / "analysis/derail_error_taxonomy/analyze.py")


class RootCauseDistributionTests(unittest.TestCase):
    def test_one_primary_type_per_failure_follows_the_configured_priority(self) -> None:
        priority = TAXONOMY.load_category_priority(
            REPOSITORY / "configs/synthesis/failure_taxonomy_v0.1.yaml"
        )
        labels = ["premature_completion", "detail_misperception", "scope_error"]
        self.assertEqual(TAXONOMY.primary_label(labels, sorted(labels), priority), "scope_error")
        rows = [
            {"state": "failure", "primary_error_type": "scope_error",
             "error_types": "detail_misperception|premature_completion|scope_error"},
            {"state": "failure", "primary_error_type": "grounding_failure",
             "error_types": "grounding_failure"},
            {"state": "perfect_pass", "primary_error_type": "", "error_types": ""},
        ]
        share = {r["value"]: r["count"] for r in TAXONOMY.distribution_rows(rows, "overall", "root_cause")
                 if r["level"] == "category"}
        self.assertEqual(share, {"planning": 1, "perception": 0, "execution": 1, "termination": 0})
        labels_share = TAXONOMY.distribution_rows(rows, "overall", "label_share")
        self.assertEqual(labels_share[0]["denominator"], 4)


class CleanStartPassTests(unittest.TestCase):
    def test_every_task_is_in_the_denominator_and_n_a_counts_as_failure(self) -> None:
        rows = [
            {"model": "m", "task_id": "t1", "state": "perfect_pass",
             "weighted_rubric_score": "1.0", "rubric_all_pass": "true"},
            {"model": "m", "task_id": "t2", "state": "failure",
             "weighted_rubric_score": "0.5", "rubric_all_pass": "false"},
            {"model": "m", "task_id": "t3", "state": "n/a",
             "weighted_rubric_score": "", "rubric_all_pass": ""},
        ]
        aggregation = {"repeats": 3, "missing_counts_as_failure": True}
        paper = TAXONOMY.clean_start_pass(rows, aggregation)
        self.assertEqual((paper.solved_count, paper.unit_count), (1, 3))
        former = TAXONOMY.clean_start_pass(rows, aggregation, exclude_na=True)
        self.assertEqual((former.solved_count, former.unit_count), (1, 2))


try:
    import pandas  # noqa: F401
except ImportError:  # the analysis package runs in its own environment
    pandas = None


@unittest.skipIf(pandas is None, "derail_annot_stats needs pandas")
class AgreementTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        sys.path.insert(0, str(REPOSITORY / "analysis/derail_annotation_stats/src"))
        from derail_annot_stats import agreement

        cls.agreement = agreement

    def test_binary_agreement_matches_hand_computation(self) -> None:
        stats = self.agreement.binary_agreement([True, True, False, False], [True, False, False, False])
        self.assertAlmostEqual(stats["agreement"], 0.75)
        self.assertAlmostEqual(stats["kappa"], 0.5)
        self.assertAlmostEqual(stats["precision"], 1.0)
        self.assertAlmostEqual(stats["recall"], 0.5)

    def test_plan_is_stratified_seeded_and_sized(self) -> None:
        trajectories = [
            {"trajectory_id": "t%02d" % i, "agent": "a", "task_category": "x" if i < 7 else "y"}
            for i in range(10)
        ] + [{"trajectory_id": "o", "agent": "other", "task_category": "x"}]
        cfg = {"agents": ["a"], "fraction": 0.3, "stratify_by": "task_category", "seed": 1}
        plan = self.agreement.double_annotation_plan(trajectories, cfg)
        self.assertEqual(len(plan), 3)
        self.assertEqual(sorted(p["task_category"] for p in plan), ["x", "x", "y"])
        self.assertEqual(plan, self.agreement.double_annotation_plan(trajectories, cfg))


if __name__ == "__main__":
    unittest.main()
