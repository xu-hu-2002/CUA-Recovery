"""Static latent horizon, mutation library and task profile on the miniworld."""

from __future__ import annotations

import unittest
from pathlib import Path

from derail.derived.schema import validate_schema
from derail.detect.latent_static import facts_intersect, static_latent_horizons
from derail.detect.mutations import MutationLibrary, static_mutation_records
from derail.detect.profile import PROFILE_VERSION, bucket_for, build_profile
from derail.ir.model import load_task_ir

REPOSITORY = Path(__file__).resolve().parents[1]
MINIWORLD = REPOSITORY / "tests" / "miniworld"
BUCKETS = ["1", "2-3", ">=4", "verifier_only"]


class LatentStaticTests(unittest.TestCase):
    def test_fact_intersection_with_wildcards(self):
        self.assertTrue(facts_intersect(("a.t", "c", "t:1"), ("a.t", "c", "t:*")))
        self.assertFalse(facts_intersect(("a.t", "c", "t:1"), ("a.t", "d", "t:1")))
        self.assertFalse(facts_intersect(("a.t", "c", "t:1"), ("a.t", "c", "t:2")))

    def test_mw002_horizons(self):
        task_ir = load_task_ir(MINIWORLD / "task_ir" / "mw-002.json", REPOSITORY)
        horizons = static_latent_horizons(task_ir)
        # n1 resolves the event; n2 re-reads its start_at -> the very next step exposes an error.
        self.assertEqual(horizons["n1"].latent_horizon_static, 1)
        self.assertEqual(horizons["n1"].observability_class, "required_next")
        self.assertEqual(horizons["n1"].visible_node_id, "n2")
        # n2's date only reaches the message: nothing downstream re-reads it, the verifier does.
        self.assertIsNone(horizons["n2"].latent_horizon_static)
        self.assertEqual(horizons["n2"].observability_class, "verifier_only")
        # n4's dm id flows into the send node, which carries a sql verifier.
        self.assertEqual(horizons["n4"].observability_class, "verifier_only")

    def test_mw001_confirm_step_reads_the_written_cell(self):
        task_ir = load_task_ir(MINIWORLD / "task_ir" / "mw-001.json", REPOSITORY)
        horizons = static_latent_horizons(task_ir)
        # n3 decides the new time; n4 writes it; n5 confirms by reading start_at back.
        self.assertEqual(horizons["n3"].visible_node_id, "n5")
        self.assertEqual(horizons["n3"].latent_horizon_static, 2)
        self.assertEqual(horizons["n3"].observability_class, "required_later")
        self.assertEqual(horizons["n3"].static_class, "same_app_later")

    def test_profile_and_mutations_validate(self):
        task_ir = load_task_ir(MINIWORLD / "task_ir" / "mw-003.json", REPOSITORY)
        horizons = static_latent_horizons(task_ir)
        profile = build_profile(task_ir, horizons, BUCKETS, ["2-3", ">=4", "verifier_only"])
        validate_schema(profile, "latent_profile.schema.json", REPOSITORY)
        self.assertEqual(profile["node_count"], 7)
        self.assertEqual(profile["structural"]["independent_component_ratio"], 0.0)
        self.assertGreaterEqual(profile["structural"]["dependency_depth"], 4)
        library = MutationLibrary.from_yaml(REPOSITORY / "configs/synthesis/mutations_v1.yaml")
        self.assertEqual(library.family_of("EventRef"), "entity")
        self.assertEqual(library.family_of("DateTime"), "date")
        self.assertIn("date_plus_minus_day", library.mutations_for_type("DateTime"))
        records = static_mutation_records(task_ir, horizons, library, PROFILE_VERSION)
        for record in records:
            validate_schema(record, "mutation.schema.json", REPOSITORY)
        self.assertTrue(
            any(
                r["mutation_type"] == "write_other_entity" and r["node_id"] == "n7" for r in records
            )
        )

    def test_bucket_labels(self):
        self.assertEqual(bucket_for(1, "required_next", BUCKETS), "1")
        self.assertEqual(bucket_for(3, "required_later", BUCKETS), "2-3")
        self.assertEqual(bucket_for(7, "required_later", BUCKETS), ">=4")
        self.assertEqual(bucket_for(None, "silent", BUCKETS), "verifier_only")


if __name__ == "__main__":
    unittest.main()
