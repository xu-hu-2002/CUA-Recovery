"""Offline checks of the VM-side control API helpers and the main.py patch."""

from __future__ import annotations

import json
import os
import sqlite3
import tempfile
import unittest
from pathlib import Path

from derail.infra import INFRA_ROOT, load_vm_script
from derail.world.sqlite_fixture import build_world

REPOSITORY = Path(__file__).resolve().parents[1]
MINIWORLD = REPOSITORY / "tests" / "miniworld"
SEEDS = {
    "hoolicalendar": MINIWORLD / "hoolicalendar.seed.json",
    "workbuzz": MINIWORLD / "workbuzz.seed.json",
}
api = load_vm_script("control_api_derail.py")
patch = load_vm_script("control_api_patch.py")
triggers = load_vm_script("triggers/install_triggers.py")
replayer = load_vm_script("snapshot/changelog_replay.py")


class ControlApiHelperTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(dir=os.environ.get("DERAIL_TMP_ROOT"))
        self.db_dir = Path(self.tmp.name) / "data"
        build_world(SEEDS, self.db_dir)
        (self.db_dir / "empty.sqlite").write_bytes(b"")
        os.symlink(self.db_dir / "workbuzz.sqlite", self.db_dir / "hooliwork.sqlite")
        self.trace_dir = Path(self.tmp.name) / "_trace"
        self.cursor_file = Path(self.tmp.name) / "_derail" / "cursor.json"

    def tearDown(self):
        self.tmp.cleanup()

    def test_live_databases_skip_symlinks_and_empty_files(self):
        self.assertEqual(
            sorted(api.live_databases(str(self.db_dir))), ["hoolicalendar", "workbuzz"]
        )

    def test_cursor_changelog_and_digest_round_trip(self):
        report = api.install_all(str(self.db_dir), triggers=triggers)
        self.assertEqual(report["hoolicalendar"]["tables"], 4)
        written = api.write_cursor(7, str(self.db_dir), str(self.cursor_file), triggers=triggers)
        self.assertEqual(written["databases"], ["hoolicalendar", "workbuzz"])
        self.assertEqual(json.loads(self.cursor_file.read_text())["action_index"], 7)
        conn = sqlite3.connect(str(self.db_dir / "hoolicalendar.sqlite"))
        conn.execute("UPDATE events SET location = 'Conf C' WHERE id = 530")
        conn.commit()
        conn.close()
        rows = api.read_changelog({}, str(self.db_dir), triggers=triggers)
        self.assertEqual(
            [(r["db"], r["tbl"], r["action_index"]) for r in rows], [("hoolicalendar", "events", 7)]
        )
        self.assertEqual(
            api.read_changelog({"hoolicalendar": 1}, str(self.db_dir), triggers=triggers), []
        )
        digests = api.digests(str(self.db_dir), replayer=replayer)
        self.assertEqual(sorted(digests), ["hoolicalendar", "workbuzz"])
        state = api.status(
            str(self.db_dir), str(self.trace_dir), str(self.cursor_file), triggers=triggers
        )
        self.assertEqual(state["cursor"]["action_index"], 7)
        self.assertEqual(state["databases"]["workbuzz"]["triggers"], 18)

    def test_observations_filter_by_action_index_and_tag_source(self):
        self.trace_dir.mkdir()
        (self.trace_dir / "hoolicalendar.jsonl").write_text(
            json.dumps(
                {"action_index": 3, "route": "/api/events", "sql": "SELECT 1", "rows_returned": 2}
            )
            + "\n"
            + json.dumps(
                {
                    "action_index": 4,
                    "route": "/api/events/530",
                    "sql": "SELECT 2",
                    "rows_returned": 1,
                }
            )
            + "\n"
        )
        (self.trace_dir / "cli.jsonl").write_text(
            json.dumps(
                {
                    "action_index": 3,
                    "source": "cli",
                    "db": "/data/workbuzz.sqlite",
                    "sql": "select count(*) from messages",
                    "rows_returned": 1,
                }
            )
            + "\n"
        )
        records = api.read_observations(3, str(self.trace_dir))
        self.assertEqual(
            [(r["source"], r["app"]) for r in records], [("cli", "cli"), ("api", "hoolicalendar")]
        )
        self.assertEqual(api.read_observations(9, str(self.trace_dir)), [])


class PatchTests(unittest.TestCase):
    def test_patch_is_idempotent_and_keeps_the_file_parseable(self):
        original = (
            "import os\nfrom flask import Flask\napp = Flask(__name__)\n\n"
            "@app.route('/x')\ndef x():\n    return 'x'\n"
        )
        once = patch.patched_text(original, "/opt/derail")
        self.assertIn("_derail_register(app)", once)
        self.assertEqual(patch.patched_text(once, "/opt/derail"), once)
        compile(once, "main.py", "exec")
        with self.assertRaises(SystemExit):
            patch.patched_text("print('no flask app here')\n", "/opt/derail")

    def test_deploy_files_exist_and_are_executable(self):
        for relative in (
            "triggers/install_triggers.py",
            "snapshot/changelog_replay.py",
            "control_api_patch.py",
            "sqlite3_wrapper.sh",
        ):
            self.assertTrue(os.access(INFRA_ROOT / relative, os.X_OK), relative)
        self.assertTrue((INFRA_ROOT / "tracer" / "trace.js").is_file())
        self.assertIn(
            'Environment="NODE_OPTIONS=--require /opt/derail/trace.js"',
            (INFRA_ROOT / "tracer" / "systemd-dropin.conf").read_text(),
        )


if __name__ == "__main__":
    unittest.main()
