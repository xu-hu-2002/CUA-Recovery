"""Workflow train/test split (derail.gen.splits)."""

from __future__ import annotations

import unittest

from derail.gen.splits import split_of, split_workflows

WORKFLOWS = {
    "gen-%010d" % i: ["t%d" % (i % 7), "t%d" % ((i * 3) % 11)] for i in range(40)
}


class WorkflowSplitTests(unittest.TestCase):
    def test_deterministic_sizes_and_lookup(self):
        first = split_workflows(WORKFLOWS, seed=1, test_size=10)
        again = split_workflows(dict(reversed(list(WORKFLOWS.items()))), seed=1, test_size=10)
        self.assertEqual(first["splits"], again["splits"])
        self.assertEqual(first["counts"], {"train": 30, "test": 10, "excluded": 0})
        self.assertNotEqual(first["splits"], split_workflows(WORKFLOWS, 2, 10)["splits"])
        self.assertEqual(split_workflows(WORKFLOWS, 1, test_fraction=0.25)["counts"]["test"], 10)
        wid = next(w for w, s in first["splits"].items() if s == "test")
        self.assertEqual(split_of(wid + "#v1", first), "test")
        self.assertEqual(split_of(wid + "/d5", first), "test")
        self.assertIsNone(split_of("aggregation-f001", first))

    def test_source_disjoint_keeps_test_sources_out_of_train(self):
        result = split_workflows(WORKFLOWS, seed=1, test_size=5, source_disjoint=True)
        self.assertEqual(result["source_tasks"]["shared"], [])
        self.assertEqual(result["counts"]["test"], 5)
        test_sources = set(result["source_tasks"]["test"])
        for wid, split in result["splits"].items():
            if split == "train":
                self.assertFalse(set(WORKFLOWS[wid]) & test_sources)


if __name__ == "__main__":
    unittest.main()
