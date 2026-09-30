"""World layer: facts, equivalence registry, fixture builder, state digests, trace fixtures."""

from __future__ import annotations

import json
import os
import sqlite3
import tempfile
import unittest
from pathlib import Path

from derail.derived.schema import validate_schema
from derail.world.facts import (
    EquivalenceRegistry,
    Fact,
    FactKey,
    entity_id,
    entity_rowid,
    facts_from_row,
    first_observation_of,
    normalize_value,
    values_match,
)
from derail.world.sqlite_fixture import build_world, load_seed
from derail.world.state import (
    database_and_table_digests,
    database_digest,
    snapshot_row,
    table_digests,
)
from derail.world.volatile import VolatileColumns

REPOSITORY = Path(__file__).resolve().parents[1]
MINIWORLD = REPOSITORY / "tests" / "miniworld"
SEEDS = {
    "hoolicalendar": MINIWORLD / "hoolicalendar.seed.json",
    "workbuzz": MINIWORLD / "workbuzz.seed.json",
}
_TMP_ROOT = os.environ.get("DERAIL_TMP_ROOT")


def _trace(name):
    return json.loads((MINIWORLD / "traces" / (name + ".json")).read_text())


class FactTests(unittest.TestCase):
    def test_normalize_dates_and_text(self):
        self.assertEqual(normalize_value("2026-07-18T10:43:01+00:00"), "2026-07-18T10:43:01")
        self.assertEqual(normalize_value("2026-04-07 11:00"), "2026-04-07T11:00:00")
        self.assertEqual(normalize_value("  Jim   Halpert "), "Jim Halpert")
        self.assertEqual(normalize_value(3.0), 3)
        self.assertIsNone(normalize_value(None))

    def test_match_rules(self):
        self.assertTrue(values_match("2026-07-02T18:00:00", "2026-07-02", "date"))
        self.assertFalse(values_match("2026-07-02T18:00:00", "2026-07-03", "date"))
        self.assertTrue(
            values_match(
                "2026-07-02T18:00:00",
                "THE DUNDIES is confirmed for 2026-07-02 at Chili's",
                "contains",
            )
        )
        self.assertTrue(values_match("Jim Halpert", " Jim Halpert", "exact"))
        with self.assertRaises(ValueError):
            values_match(1, 1, "fuzzy")

    def test_entity_ids_and_row_facts(self):
        self.assertEqual(entity_id("hoolicalendar.events", 530), "events:530")
        self.assertEqual(entity_rowid("events:530"), 530)
        with self.assertRaises(ValueError):
            entity_rowid("derived:n1:event_id")
        facts = facts_from_row(
            "hoolicalendar",
            "events",
            530,
            {"title": "1:1 with Jim Halpert", "start_at": "x"},
            ["title"],
        )
        self.assertEqual(
            facts, [Fact("hoolicalendar.events", "title", "events:530", "1:1 with Jim Halpert")]
        )
        with self.assertRaises(ValueError):
            FactKey("events", "title", "events:530")

    def test_equivalence_registry_is_symmetric(self):
        registry = EquivalenceRegistry.from_records(
            json.loads((MINIWORLD / "equivalences.json").read_text())
        )
        primary = FactKey("hoolicalendar.events", "start_at", "events:545")
        other = FactKey("workbuzz.messages", "content", "messages:9001")
        self.assertEqual([s.key for s in registry.equivalents(primary)], [primary, other])
        self.assertEqual(registry.equivalents(other)[1].key, primary)
        self.assertEqual(registry.match_rule(primary, other), "contains")
        self.assertEqual(
            EquivalenceRegistry.from_records(registry.to_records()).to_records(),
            registry.to_records(),
        )

    def test_first_observation_uses_equivalent_sources(self):
        registry = EquivalenceRegistry.from_records(
            json.loads((MINIWORLD / "equivalences.json").read_text())
        )
        fact = Fact("hoolicalendar.events", "start_at", "events:545", "2026-07-02T18:00:00")
        events = [
            {
                "action_index": 0,
                "facts": [
                    {
                        "table": "workbuzz.messages",
                        "column": "content",
                        "entity": "messages:9001",
                        "value": "THE DUNDIES is confirmed for 2026-07-02 at Chili's",
                    }
                ],
            },
            {"action_index": 1, "facts": [fact.to_dict()]},
        ]
        self.assertEqual(first_observation_of(fact, events, registry)["action_index"], 0)
        self.assertEqual(first_observation_of(fact, events)["action_index"], 1)
        self.assertIsNone(first_observation_of(fact, events[:0]))


class FixtureTests(unittest.TestCase):
    def test_seeds_build_with_real_ddl_and_stable_digest(self):
        seed = load_seed(SEEDS["hoolicalendar"])
        self.assertEqual(seed["database"], "hoolicalendar")
        self.assertTrue(
            any(
                t["name"] == "events" and "start_at TEXT NOT NULL" in t["ddl"]
                for t in seed["tables"]
            )
        )
        digests = []
        for _ in range(2):
            with tempfile.TemporaryDirectory(dir=_TMP_ROOT) as tmp:
                dbs = build_world(SEEDS, Path(tmp))
                conn = sqlite3.connect(str(dbs["hoolicalendar"]))
                digests.append(database_digest(conn, ["sqlite_sequence"]))
                row = snapshot_row(conn, "events", 530)
                self.assertEqual(row["title"], "1:1 with Jim Halpert")
                self.assertIn("events", table_digests(conn, ["sqlite_sequence"]))
                conn.close()
        self.assertEqual(digests[0], digests[1])

    def test_trace_fixtures_validate_and_encode_the_intended_failures(self):
        for name in ("mw002_success", "mw002_param_failure", "mw001_state_failure"):
            validate_schema(_trace(name), "rollout_trace.schema.json", REPOSITORY)
        state = _trace("mw001_state_failure")
        delta_entities = {(r["tbl"], r["rowid"]) for s in state["steps"] for r in s["delta"]}
        self.assertEqual(delta_entities, {("events", 531)})
        self.assertEqual(state["provenance"]["ground_truth"]["root_cause_action_index"], 3)
        param = _trace("mw002_param_failure")
        typed = [p["value"] for s in param["steps"] for p in s["params"] if s["action_index"] == 5][
            0
        ]
        self.assertIn("2026-07-03", typed)
        seen = [
            f["value"]
            for s in param["steps"]
            for o in s["observations"]
            for f in o["facts"]
            if f["entity"] == "events:545" and f["column"] == "start_at" and s["action_index"] < 5
        ]
        self.assertTrue(seen and all(v.startswith("2026-07-02") for v in seen))


if __name__ == "__main__":
    unittest.main()


class VolatileColumnTests(unittest.TestCase):
    def test_rules_and_deployed_copy_agree(self):
        rules = VolatileColumns.from_yaml(REPOSITORY / "configs/synthesis/volatile_columns_v1.yaml")
        deployed = VolatileColumns.from_json(REPOSITORY / "infra" / "volatile_columns.json")
        self.assertEqual(rules.to_mapping(), deployed.to_mapping())
        self.assertTrue(rules.is_volatile("hoolicalendar.events", "updated_at"))
        self.assertTrue(rules.is_volatile("workbuzz.dm_messages", "timestamp"))
        self.assertFalse(rules.is_volatile("hoolicalendar.events", "start_at"))
        self.assertFalse(rules.is_volatile("workbuzz.reminders", "remind_at"))
        self.assertFalse(rules.is_volatile("workbuzz.messages", "content"))

    def test_digest_ignores_volatile_columns(self):
        rules = VolatileColumns.from_yaml(REPOSITORY / "configs/synthesis/volatile_columns_v1.yaml")
        with tempfile.TemporaryDirectory(dir=_TMP_ROOT) as tmp:
            dbs = build_world(SEEDS, Path(tmp))
            conn = sqlite3.connect(str(dbs["hoolicalendar"]))
            skip = rules.excluded_for(
                "hoolicalendar", [("events", ["id", "updated_at", "start_at"])]
            )
            self.assertEqual(skip, {"events": {"updated_at"}})
            before = database_digest(conn, ["sqlite_sequence"], skip)
            conn.execute("UPDATE events SET updated_at = '2030-01-01 00:00:00' WHERE id = 530")
            conn.commit()
            self.assertEqual(database_digest(conn, ["sqlite_sequence"], skip), before)
            self.assertNotEqual(database_digest(conn, ["sqlite_sequence"]), before)
            conn.close()


class DigestEquivalenceTests(unittest.TestCase):
    def test_single_pass_digests_match_the_separate_functions(self):
        with tempfile.TemporaryDirectory(dir=os.environ.get("DERAIL_TMP_ROOT")) as tmp:
            dbs = build_world(
                {"workbuzz": Path(__file__).parent / "miniworld" / "workbuzz.seed.json"},
                Path(tmp),
            )
            conn = sqlite3.connect(str(dbs["workbuzz"]))
            excluded = ("sqlite_sequence",)
            columns = {"messages": {"created_at"}}
            whole, per_table = database_and_table_digests(conn, excluded, columns)
            self.assertEqual(whole, database_digest(conn, excluded, columns))
            self.assertEqual(per_table, table_digests(conn, excluded, columns))
            conn.close()
