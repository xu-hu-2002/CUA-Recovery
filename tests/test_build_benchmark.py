"""Benchmark construction gates that are independent of filesystem fixtures."""

import importlib.util
import unittest
from pathlib import Path


REPOSITORY = Path(__file__).resolve().parents[1]


def _load_script():
    path = REPOSITORY / "scripts" / "04_build_benchmark.py"
    spec = importlib.util.spec_from_file_location("derail_build_benchmark", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


BUILD = _load_script()


class FailureOnlyBuildTests(unittest.TestCase):
    def test_perfect_trajectory_is_rejected(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "Only failure trajectories"):
            BUILD._require_failed_trajectory({"perfect": True, "score": 1.0})

    def test_nonperfect_trajectory_is_allowed(self) -> None:
        BUILD._require_failed_trajectory({"perfect": False, "score": 0.8})


if __name__ == "__main__":
    unittest.main()
