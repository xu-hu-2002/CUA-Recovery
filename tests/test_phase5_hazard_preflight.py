"""The preflight must measure without ever writing the golden world, and must refuse inert patches."""

from __future__ import annotations

import hashlib
import sqlite3
import sys
import tempfile
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts" / "phase5"))

from derail.gen.hazards import HazardConfig  # noqa: E402
from preflight_hazard_realization import (  # noqa: E402
    MODELS, calendar_override_survives, classify, classify_distractor,
    realize_option_a, singleton_prefix_titles,
)

COLUMNS = ("id INTEGER PRIMARY KEY AUTOINCREMENT, calendar_id INTEGER, user_email TEXT, "
           "title TEXT, description TEXT, location TEXT, start_at TEXT, end_at TEXT, "
           "is_all_day INTEGER, status TEXT, source_app TEXT, source_ref TEXT, "
           "source_kind TEXT, created_by TEXT, updated_by TEXT, organizer_email TEXT")

CONFIG = """
schema_version: hazards-config/1.0
base_rate_multiplier_max: 2.0
base_rate_queries:
  distractor_candidates:
    - {app: hoolicalendar, unit: same_title_prefix_events, query: "SELECT COUNT(*) FROM \
(SELECT substr(title, 1, 12) p FROM events GROUP BY p HAVING COUNT(*) > 1)"}
patches: {}
"""


def record(summary, start):
    return {"injection_id": "hz-test", "hazard_type": "distractor_candidates", "status": "pending",
            "base_world_id": "world-test", "variant_world_id": "world-test+hz-test",
            "hazard_provenance": {"base_rate_before": 19.0, "base_rate_after": 20.0},
            "persona_diff": [{"op": "add", "path": "/app_overrides/calendar/events/-",
                              "value": {"summary": summary, "start": start, "end": start}}]}


def world(titles):
    root = Path(tempfile.mkdtemp(prefix="phase5-preflight-test-"))
    db = root / "hoolicalendar.sqlite"
    conn = sqlite3.connect(db)
    conn.execute("CREATE TABLE events (%s)" % COLUMNS)
    for title in titles:
        conn.execute("INSERT INTO events (calendar_id, title, start_at) VALUES (1, ?, '2026-06-01')",
                     (title,))
    conn.commit()
    conn.close()
    return db


def config():
    path = Path(tempfile.mkdtemp(prefix="phase5-preflight-cfg-")) / "hazards.yaml"
    path.write_text(CONFIG)
    return HazardConfig.from_yaml(path)


class PreflightTests(unittest.TestCase):
    def setUp(self):
        self.config = config()

    def test_guard_matches_the_guest_seeder(self):
        self.assertFalse(calendar_override_survives({"summary": "X", "start": ""}))
        self.assertFalse(calendar_override_survives({"summary": "", "start": "2026-06-01"}))
        self.assertTrue(calendar_override_survives({"summary": "X", "start": "2026-06-01"}))

    def test_inert_patch_is_refused_without_touching_the_database(self):
        db = world(["Standup meeting A", "Standup meeting B", "Solo meeting"])
        digest = hashlib.sha256(db.read_bytes()).hexdigest()
        result = classify_distractor(record("THE DUNDIES ARE BACK BABY (prep)", ""), self.config,
                                     db.parent)
        self.assertFalse(result["survives_seeder_guards"])
        self.assertFalse(result["admissible"])
        self.assertIsNone(result["realized_rate"])
        self.assertEqual(result["base_rate"], 1)
        self.assertEqual(hashlib.sha256(db.read_bytes()).hexdigest(), digest)

    def test_measured_rate_beats_the_recorded_prediction(self):
        db = world(["Standup meeting A", "Standup meeting B", "Solo meeting"])
        result = classify(record("Solo meeting followup", "2026-07-01"), self.config, db.parent)
        self.assertEqual(
            (result["base_rate"], result["realized_rate"], result["delta"], result["admissible"]),
            (1, 2, 1, True))
        self.assertEqual(result["recorded_base_rate_before"], 19.0)
        self.assertEqual(result["recorded_base_rate_after"], 20.0)

    def test_row_that_does_not_collide_leaves_the_unit_unchanged(self):
        db = world(["Standup meeting A", "Standup meeting B", "Solo meeting"])
        result = classify_distractor(record("Entirely new title", "2026-07-01"), self.config,
                                     db.parent)
        self.assertTrue(result["survives_seeder_guards"])
        self.assertEqual(result["delta"], 0)
        self.assertFalse(result["admissible"])
        self.assertIn("does not change", result["note"])

    def test_option_a_selects_a_base_singleton_and_measures_plus_one(self):
        db = world(["Standup meeting A", "Standup meeting B", "Solo meeting"])
        source = record("Unrelated gold title (prep)", "")
        repaired = realize_option_a(source, db.parent, "2026-09-01T07:24:24+00:00")
        result = classify_distractor(repaired, self.config, db.parent)
        self.assertEqual(singleton_prefix_titles(db), ["Solo meeting"])
        self.assertEqual(repaired["persona_diff"][0]["value"]["summary"],
                         "Solo meeting (prep)")
        self.assertEqual((result["base_rate"], result["realized_rate"], result["delta"]),
                         (1, 2, 1))
        self.assertTrue(result["admissible"])
        self.assertEqual(repaired["provenance"]["source_injection_id"], "hz-test")
        self.assertNotEqual(repaired["injection_id"], "hz-test")

    def test_unmodeled_hazard_type_is_reported_not_guessed(self):
        db = world(["Standup alpha"])
        rec = record("x", "2026-06-01")
        rec["hazard_type"] = "identity_alias"
        rec["status"] = "BASE_RATE_ZERO"
        self.assertNotIn("identity_alias", MODELS)
        result = classify(rec, self.config, db.parent)
        self.assertFalse(result["modeled"])
        self.assertFalse(result["admissible"])
        self.assertEqual(result["status"], "BASE_RATE_ZERO")


if __name__ == "__main__":
    unittest.main()
