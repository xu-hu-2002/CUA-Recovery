"""Schema graph over the miniworld: explicit foreign keys, inferred references, path queries."""

from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path

from derail.derived.schema import validate_schema
from derail.world.schema_graph import SchemaGraph, SchemaGraphConfig, build_schema_graph
from derail.world.sqlite_fixture import build_world

REPOSITORY = Path(__file__).resolve().parents[1]
MINIWORLD = REPOSITORY / "tests" / "miniworld"
SEEDS = {
    "hoolicalendar": MINIWORLD / "hoolicalendar.seed.json",
    "workbuzz": MINIWORLD / "workbuzz.seed.json",
}
CONFIG = SchemaGraphConfig.from_yaml(REPOSITORY / "configs/synthesis/schema_graph_v1.yaml")


class SchemaGraphTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        with tempfile.TemporaryDirectory(dir=os.environ.get("DERAIL_TMP_ROOT")) as tmp:
            dbs = build_world(SEEDS, Path(tmp))
            cls.record = build_schema_graph(dbs, CONFIG, "miniworld")
        validate_schema(cls.record, "schema_graph.schema.json", REPOSITORY)
        cls.graph = SchemaGraph.from_record(cls.record)

    def _relations(self, kind=None, rule=None):
        return [
            (r["from_table"], r["from_column"], r["to_table"], r["to_column"])
            for r in self.record["relations"]
            if (kind is None or r["kind"] == kind) and (rule is None or r["rule"] == rule)
        ]

    def test_explicit_foreign_keys_are_found(self):
        self.assertIn(
            ("hoolicalendar.event_attendees", "event_id", "hoolicalendar.events", "id"),
            self._relations(kind="explicit"),
        )
        self.assertIn(
            ("workbuzz.dm_messages", "dm_id", "workbuzz.dms", "id"),
            self._relations(kind="explicit"),
        )

    def test_inferred_references_carry_their_rule(self):
        self.assertIn(
            ("workbuzz.channel_members", "channel_id", "workbuzz.channels", "id"),
            self._relations(rule="id_suffix_plural"),
        )
        confidences = {r["rule"]: r["confidence"] for r in self.record["relations"]}
        self.assertEqual(confidences["foreign_key"], 1.0)
        self.assertLess(confidences["email_principal"], confidences["id_suffix_plural"])
        self.assertIn("foreign_key", {r["rule"] for r in self.graph.generation_relations()})
        self.assertGreater(self.record["inferred_ratio"], 0)
        self.assertIn(
            ("workbuzz.messages", "sender_email", "workbuzz.members", "email"),
            self._relations(rule="email_principal"),
        )
        self.assertFalse(
            [
                r
                for r in self._relations(rule="email_principal")
                if r[0].startswith("hoolicalendar.")
            ]
        )

    def test_bookkeeping_tables_are_excluded(self):
        self.assertNotIn("workbuzz.sqlite_sequence", self.graph.tables())
        self.assertIn("sqlite_sequence", self.record["excluded_tables"])

    def test_query_helpers(self):
        self.assertTrue(self.graph.has_column("hoolicalendar.events", "start_at"))
        self.assertFalse(self.graph.has_column("hoolicalendar.events", "nope"))
        path = self.graph.foreign_key_path("workbuzz.dm_messages", "workbuzz.members")
        self.assertEqual(len(path), 1)
        two_hops = self.graph.foreign_key_path(
            "hoolicalendar.event_attendees", "hoolicalendar.calendars"
        )
        self.assertEqual([r["from_column"] for r in two_hops], ["event_id", "calendar_id"])
        self.assertIsNone(self.graph.foreign_key_path("hoolicalendar.events", "workbuzz.members"))


if __name__ == "__main__":
    unittest.main()
