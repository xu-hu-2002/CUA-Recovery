"""指标分母和 invalid episode 行为必须明确。"""

import unittest

from derail.evaluation.metrics import EpisodeResult, aggregate_metrics, pass_at_k, rubric_verdict


class MetricsTests(unittest.TestCase):
    def test_micro_average_ignores_invalid_but_reports_valid_count(self) -> None:
        results = [
            EpisodeResult("c1", "a1", 0, error_aware=True, recovered=True),
            EpisodeResult("c2", "a1", 0, error_aware=False, recovered=False),
            EpisodeResult("c3", "a1", 0, error_aware=True, recovered=False, valid=False),
        ]
        metrics = aggregate_metrics(results)
        self.assertEqual(metrics.episode_count, 2)
        self.assertEqual(metrics.error_awareness_rate, 0.5)
        self.assertEqual(metrics.post_error_success_rate, 0.5)

    def test_empty_valid_set_is_error(self) -> None:
        with self.assertRaises(ValueError):
            aggregate_metrics([])


class RubricPassAtKTests(unittest.TestCase):
    def test_v_requires_every_criterion_even_a_tiny_weight(self) -> None:
        rho, passed = rubric_verdict([
            {"success": True, "weight": 0.999},
            {"success": False, "weight": 0.001},
        ])
        self.assertAlmostEqual(rho, 0.999)
        self.assertFalse(passed)  # the judge's round(100*rho) >= 100 would call this a pass

    def test_pass_at_3_uses_the_solving_run_else_the_best_score(self) -> None:
        runs = {
            "s1": [(0.4, False), (1.0, True), (0.2, False)],
            "s2": [(0.3, False), (0.6, False), None],
            "s3": [None, None, None],
        }
        result = pass_at_k(runs, ["s1", "s2", "s3", "s4"], repeats=3)
        self.assertEqual((result.unit_count, result.solved_count, result.missing_count), (4, 1, 2))
        self.assertAlmostEqual(result.pass_at_k, 0.25)
        self.assertAlmostEqual(result.rubric_score, (1.0 + 0.6 + 0 + 0) / 4)
        excluded = pass_at_k(runs, ["s1", "s2", "s3", "s4"], repeats=3,
                             missing_counts_as_failure=False)
        self.assertEqual((excluded.unit_count, excluded.missing_count), (2, 2))
        with self.assertRaises(ValueError):
            pass_at_k({"s1": [(1.0, True)] * 4}, ["s1"], repeats=3)


if __name__ == "__main__":
    unittest.main()
