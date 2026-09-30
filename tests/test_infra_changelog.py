"""Trigger change log and changelog replay on the miniworld."""

from __future__ import annotations

import json
import os
import shutil
import sqlite3
import tempfile
import unittest
from pathlib import Path

from derail.derived.schema import validate_schema
from derail.infra import load_vm_script
from derail.world.sqlite_fixture import build_world
from derail.world.state import database_digest

REPOSITORY = Path(__file__).resolve().parents[1]
MINIWORLD = REPOSITORY / "tests" / "miniworld"
SEEDS = {"hoolicalendar": MINIWORLD / "hoolicalendar.seed.json"}
triggers = load_vm_script("triggers/install_triggers.py")
replayer = load_vm_script("snapshot/changelog_replay.py")
EXCLUDE = ("sqlite_sequence", "_changelog", "_cursor", "_seed_meta", "_meta")


class ChangelogTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(dir=os.environ.get("DERAIL_TMP_ROOT"))
        root = Path(self.tmp.name)
        self.live = build_world(SEEDS, root / "live")["hoolicalendar"]
        self.baseline = root / "baseline.sqlite"
        shutil.copyfile(self.live, self.baseline)
        self.conn = sqlite3.connect(str(self.live))
        self.report = triggers.install(self.conn)

    def tearDown(self):
        self.conn.close()
        self.tmp.cleanup()

    def _act(self, action_index, sql, params=()):
        triggers.set_cursor(self.conn, action_index)
        self.conn.execute(sql, params)
        self.conn.commit()

    def test_install_is_idempotent_and_covers_every_user_table(self):
        self.assertEqual(self.report["tables"], 4)
        self.assertEqual(self.report["triggers"], 12)
        again = triggers.install(self.conn)
        self.assertEqual(again, self.report)
        self.assertEqual(len(triggers.installed_triggers(self.conn)), 12)

    def test_install_skips_virtual_tables(self):
        self.conn.execute("CREATE VIRTUAL TABLE search_index USING fts5(content)")
        self.conn.execute("INSERT INTO search_index(content) VALUES ('hello')")
        self.conn.commit()
        report = triggers.install(self.conn)
        self.assertNotIn("search_index", triggers.user_tables(self.conn))
        self.assertEqual(report["triggers"], 12)
        self.assertEqual(
            replayer.digest(str(self.live), EXCLUDE), database_digest(self.conn, EXCLUDE)
        )

    def test_changes_are_attributed_to_the_cursor(self):
        self._act(3, "UPDATE events SET start_at = '2026-04-08T14:00:00' WHERE id = 531")
        self._act(
            5,
            "INSERT INTO events (calendar_id, user_email, title, start_at, end_at) "
            "VALUES (1, 'michael.scott@dundermifflin.com', 'X', '2026-04-09T10:00:00', "
            "'2026-04-09T10:30:00')",
        )
        self._act(6, "DELETE FROM event_reminders WHERE event_id = 531")
        rows = triggers.read_changelog(self.conn, db="hoolicalendar")
        for row in rows:
            validate_schema(row, "changelog_row.schema.json", REPOSITORY)
        ops = [(r["op"], r["tbl"], r["action_index"]) for r in rows]
        self.assertEqual(ops[0], ("UPDATE", "events", 3))
        self.assertEqual(ops[1], ("INSERT", "events", 5))
        self.assertTrue(all(op == ("DELETE", "event_reminders", 6) for op in ops[2:]))
        update = rows[0]
        self.assertEqual(json.loads(update["old_json"])["start_at"], "2026-04-07T14:00:00")
        self.assertEqual(json.loads(update["new_json"])["start_at"], "2026-04-08T14:00:00")
        self.assertEqual(update["rowid"], 531)
        self.assertIsNone(rows[1]["old_json"])
        self.assertIsNone(rows[2]["new_json"])

    def test_changes_before_any_cursor_are_marked_unset(self):
        self.conn.execute("UPDATE events SET location = 'Conf C' WHERE id = 530")
        self.conn.commit()
        rows = triggers.read_changelog(self.conn)
        self.assertEqual(rows[-1]["action_index"], triggers.CURSOR_UNSET)

    def test_replay_reproduces_the_direct_state(self):
        self._act(
            1,
            "UPDATE events SET start_at = '2026-04-08T11:00:00', "
            "end_at = '2026-04-08T11:30:00' WHERE id = 530",
        )
        self._act(
            2,
            "INSERT INTO event_attendees (event_id, email, display_name, "
            "response_status, is_organizer) VALUES (530, 'pam.beesly@dundermifflin.com', "
            "'Pam Beesly', 'needsAction', 0)",
        )
        self._act(3, "DELETE FROM event_reminders WHERE event_id = 530")
        self._act(4, "UPDATE calendars SET name = 'Work (renamed)' WHERE id = 1")
        rows = triggers.read_changelog(self.conn, db="hoolicalendar")
        direct_full = database_digest(self.conn, EXCLUDE)
        out = Path(self.tmp.name) / "replayed.sqlite"
        report = replayer.replay(str(self.baseline), rows, str(out))
        self.assertEqual(report["skipped"], 0)
        self.assertEqual(replayer.digest(str(out), EXCLUDE), direct_full)
        partial = Path(self.tmp.name) / "partial.sqlite"
        replayer.replay(str(self.baseline), rows, str(partial), until_action=2)
        direct2 = sqlite3.connect(str(Path(self.tmp.name) / "direct2.sqlite"))
        shutil.copyfile(self.baseline, Path(self.tmp.name) / "direct2.sqlite")
        direct2 = sqlite3.connect(str(Path(self.tmp.name) / "direct2.sqlite"))
        direct2.execute(
            "UPDATE events SET start_at = '2026-04-08T11:00:00', "
            "end_at = '2026-04-08T11:30:00' WHERE id = 530"
        )
        direct2.execute(
            "INSERT INTO event_attendees (event_id, email, display_name, "
            "response_status, is_organizer) VALUES (530, "
            "'pam.beesly@dundermifflin.com', 'Pam Beesly', 'needsAction', 0)"
        )
        direct2.commit()
        self.assertEqual(replayer.digest(str(partial), EXCLUDE), database_digest(direct2, EXCLUDE))
        self.assertNotEqual(replayer.digest(str(partial), EXCLUDE), direct_full)
        direct2.close()

    def test_replay_digest_matches_repository_digest(self):
        self.assertEqual(
            replayer.digest(str(self.live), EXCLUDE), database_digest(self.conn, EXCLUDE)
        )

    def test_digest_normalizes_equivalent_sqlite_real_values(self):
        self.conn.execute("CREATE TABLE balances (value REAL)")
        self.conn.execute("INSERT INTO balances VALUES (?)", (64093.229999999996,))
        self.conn.commit()
        other_path = Path(self.tmp.name) / "other.sqlite"
        shutil.copyfile(self.live, other_path)
        other = sqlite3.connect(other_path)
        try:
            other.execute("UPDATE balances SET value = ?", (64093.23,))
            other.commit()
            self.assertEqual(database_digest(self.conn, EXCLUDE), database_digest(other, EXCLUDE))
            self.assertEqual(
                replayer.digest(str(self.live), EXCLUDE),
                replayer.digest(str(other_path), EXCLUDE),
            )
        finally:
            other.close()

    def test_replay_digest_honours_volatile_rules(self):
        rules = replayer.load_volatile_rules(str(REPOSITORY / "infra" / "volatile_columns.json"))
        from derail.world.volatile import VolatileColumns
        from derail.world.state import table_columns, table_names

        volatile = VolatileColumns.from_json(REPOSITORY / "infra" / "volatile_columns.json")
        skip = volatile.excluded_for(
            "hoolicalendar", [(t, table_columns(self.conn, t)) for t in table_names(self.conn)]
        )
        self._act(1, "UPDATE events SET updated_at = '2031-01-01 00:00:00' WHERE id = 530")
        self.assertEqual(
            replayer.digest(str(self.live), EXCLUDE, rules, "hoolicalendar"),
            database_digest(self.conn, EXCLUDE, skip),
        )
        self.assertEqual(
            replayer.digest(str(self.live), EXCLUDE, rules, "hoolicalendar"),
            replayer.digest(str(self.baseline), EXCLUDE, rules, "hoolicalendar"),
        )
        self.assertNotEqual(
            replayer.digest(str(self.live), EXCLUDE), replayer.digest(str(self.baseline), EXCLUDE)
        )

    def test_digest_tells_different_databases_apart_on_this_sqlite(self):
        paths = []
        for index, title in enumerate(("a", "b")):
            path = Path(self.tmp.name) / ("plain_%d.sqlite" % index)
            conn = sqlite3.connect(str(path))
            conn.execute("CREATE TABLE notes (title TEXT)")
            conn.execute("INSERT INTO notes VALUES (?)", (title,))
            conn.commit()
            conn.close()
            paths.append(str(path))
        self.assertNotEqual(replayer.digest(paths[0]), replayer.digest(paths[1]))
        self.assertNotEqual(replayer.digest(paths[0]), replayer.digest(str(self.baseline)))

    def test_uninstall_removes_triggers_only(self):
        self.assertEqual(triggers.uninstall(self.conn), 12)
        self.assertEqual(triggers.installed_triggers(self.conn), [])
        tables = [
            r[0] for r in self.conn.execute("SELECT name FROM sqlite_master WHERE type='table'")
        ]
        self.assertIn("_changelog", tables)


if __name__ == "__main__":
    unittest.main()
