"""Gold interpreter on the miniworld."""

from __future__ import annotations

import copy
import json
import os
import sqlite3
import tempfile
import unittest
from pathlib import Path

from derail.derived.schema import validate_schema
from derail.ir.gold_interpreter import (
    GoldInterpreter,
    GoldInterpreterError,
    InterpreterConfig,
    WorldCopy,
)
from derail.ir.model import TaskIRError, dag_index, load_task_ir, validate_task_ir
from derail.world.sqlite_fixture import build_world

REPOSITORY = Path(__file__).resolve().parents[1]
MINIWORLD = REPOSITORY / "tests" / "miniworld"
CONFIG = InterpreterConfig.from_yaml(REPOSITORY / "configs/synthesis/gold_interpreter_v1.yaml")
SEEDS = {
    "hoolicalendar": MINIWORLD / "hoolicalendar.seed.json",
    "workbuzz": MINIWORLD / "workbuzz.seed.json",
}
TASKS = ("mw-001", "mw-002", "mw-003", "mw-004")
FILES = MINIWORLD / "files"
_TMP_ROOT = os.environ.get("DERAIL_TMP_ROOT")


def _scratch():
    return tempfile.TemporaryDirectory(dir=_TMP_ROOT)


class GoldInterpreterTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.expected = json.loads((MINIWORLD / "expected_gold.json").read_text())
        cls.task_irs = {
            task: load_task_ir(MINIWORLD / "task_ir" / (task + ".json"), REPOSITORY)
            for task in TASKS
        }

    def _run(self, task_id, task_ir=None, config=CONFIG):
        with _scratch() as tmp:
            dbs = build_world(SEEDS, Path(tmp) / "world")
            with WorldCopy.open("miniworld", dbs, Path(tmp) / "copy", files_root=FILES) as world:
                return GoldInterpreter(config).run(
                    task_ir or self.task_irs[task_id], world, REPOSITORY
                )

    def test_task_irs_validate_and_index(self):
        for task_id, task_ir in self.task_irs.items():
            with self.subTest(task=task_id):
                validate_task_ir(task_ir, REPOSITORY)
                dag = dag_index(task_ir)
                self.assertEqual(len(dag.order), len(task_ir["nodes"]))

    def test_gold_values_match_hand_computed_answers(self):
        for task_id in TASKS:
            with self.subTest(task=task_id):
                record = self._run(task_id)
                validate_schema(record, "gold_lineage.schema.json", REPOSITORY)
                got = {"%s.%s" % (v["node_id"], v["name"]): v["value"] for v in record["values"]}
                for key, value in self.expected[task_id]["values"].items():
                    self.assertEqual(got[key], value, key)
                self.assertIs(record["final_verifier_passed"], True)
                self.assertTrue(all(r["passed"] for r in record["verifier_results"]))
                self.assertEqual(record["reads_undeclared"], [], task_id)
                changed = [t["node_id"] for t in record["state_timeline"] if t["changed"]]
                self.assertEqual(changed, self.expected[task_id]["changed_nodes"])

    def test_writes_gold_are_read_back_from_the_database(self):
        for task_id in TASKS:
            with self.subTest(task=task_id):
                record = self._run(task_id)
                got = [
                    [
                        w["node_id"],
                        w["table"],
                        w["column"],
                        w["entity"],
                        w["value"],
                        w["old_value"],
                        w["reversibility_class"],
                    ]
                    for w in record["writes_gold"]
                ]
                self.assertEqual(got, self.expected[task_id]["writes_gold"])

    def test_aggregate_counts_are_from_the_rows(self):
        record = self._run("mw-003")
        counts = [v["value"] for v in record["values"] if v["name"] == "counts"][0]
        top3 = dict(sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))[:3])
        self.assertEqual(top3, self.expected["mw-003"]["counts_top3"])

    def test_digests_are_reproducible(self):
        first = self._run("mw-001")
        second = self._run("mw-001")
        self.assertEqual(first["initial_state_sha256"], second["initial_state_sha256"])
        self.assertEqual(first["state_timeline"], second["state_timeline"])
        self.assertNotEqual(
            first["state_timeline"][-1]["state_sha256"], first["initial_state_sha256"]
        )

    def test_undeclared_write_aborts(self):
        with _scratch() as tmp:
            dbs = build_world(SEEDS, Path(tmp) / "world")
            conn = sqlite3.connect(str(dbs["hoolicalendar"]))
            conn.execute(
                "CREATE TRIGGER t AFTER UPDATE ON events BEGIN "
                "UPDATE calendars SET name = 'touched' WHERE id = 1; END"
            )
            conn.commit()
            conn.close()
            with WorldCopy.open("miniworld", dbs, Path(tmp) / "copy") as world:
                with self.assertRaises(GoldInterpreterError) as ctx:
                    GoldInterpreter(CONFIG).run(self.task_irs["mw-001"], world, REPOSITORY)
        self.assertEqual(ctx.exception.code, "UNDECLARED_WRITE")

    def test_sql_derivations_are_read_only(self):
        task_ir = copy.deepcopy(self.task_irs["mw-001"])
        node = [n for n in task_ir["nodes"] if n["node_id"] == "n2"][0]
        node["produces"][0]["derivation"]["query"] = (
            "UPDATE events SET location = 'x' WHERE id = :event_id RETURNING start_at"
        )
        with self.assertRaises(GoldInterpreterError) as ctx:
            self._run("mw-001", task_ir)
        self.assertEqual(ctx.exception.code, "SQL_DERIVATION_WRITES")

    def test_undeclared_reads_are_reported(self):
        task_ir = copy.deepcopy(self.task_irs["mw-001"])
        node = [n for n in task_ir["nodes"] if n["node_id"] == "n2"][0]
        node["produces"][0]["derivation"]["query"] = (
            "SELECT start_at FROM events WHERE id = :event_id AND location IS NOT NULL"
        )
        record = self._run("mw-001", task_ir)
        self.assertIn(
            {"node_id": "n2", "table": "hoolicalendar.events", "column": "location"},
            record["reads_undeclared"],
        )

    def test_resolved_reads_and_volatile_writes(self):
        record = self._run("mw-003")
        scan = [r for r in record["resolved_reads"] if r["node_id"] == "n1"][0]
        self.assertFalse(scan["coarse"])
        self.assertEqual(len(scan["entity_set"]), 60)
        self.assertEqual(scan["filter_columns"], ["sender_email", "deleted_at"])
        channel = [r for r in record["resolved_reads"] if r["node_id"] == "n4"][0]
        self.assertEqual(channel["entity_set"], [1])
        record = self._run("mw-002")
        volatile = {w["column"]: w["volatile"] for w in record["writes_gold"]}
        self.assertTrue(volatile["timestamp"])
        self.assertFalse(volatile["content"])
        self.assertEqual(
            [w["column"] for w in record["writes_gold"] if w["volatile"]],
            self.expected["mw-002"]["volatile_writes"],
        )

    def test_control_predicate_skips_node(self):
        task_ir = copy.deepcopy(self.task_irs["mw-002"])
        send = [n for n in task_ir["nodes"] if n["node_id"] == "n5"][0]
        send["inputs"].append(
            {
                "port_id": "gate",
                "type": "DateTime",
                "grounding": "derived:n2:start_at",
                "source": "upstream",
            }
        )
        task_ir["edges"].append(
            {
                "edge_id": "ctl",
                "kind": "control_dependency",
                "from": {"node_id": "n2", "port_id": "start_at"},
                "to": {"node_id": "n5", "port_id": "gate"},
                "predicate": 'date_of(V["n2"]["start_at"]) == "1999-01-01"',
            }
        )
        record = self._run("mw-002", task_ir)
        self.assertIn("n5", record["provenance"]["skipped_nodes"])
        self.assertEqual(record["writes_gold"], [])
        self.assertIs(record["final_verifier_passed"], False)

    def test_gate_on_upstream_node_skips_its_dependents(self):
        task_ir = copy.deepcopy(self.task_irs["mw-002"])
        resolve = [n for n in task_ir["nodes"] if n["node_id"] == "n4"][0]
        resolve["inputs"].append(
            {
                "port_id": "gate",
                "type": "DateTime",
                "grounding": "derived:n2:start_at",
                "source": "upstream",
            }
        )
        task_ir["edges"].append(
            {
                "edge_id": "ctl",
                "kind": "control_dependency",
                "from": {"node_id": "n2", "port_id": "start_at"},
                "to": {"node_id": "n4", "port_id": "gate"},
                "predicate": 'date_of(V["n2"]["start_at"]) == "1999-01-01"',
            }
        )
        record = self._run("mw-002", task_ir)
        self.assertEqual(record["provenance"]["skipped_nodes"], ["n4", "n5"])
        self.assertEqual(record["writes_gold"], [])

    def test_r0_node_with_side_effects_is_rejected(self):
        task_ir = copy.deepcopy(self.task_irs["mw-001"])
        task_ir["nodes"][0]["side_effects"] = task_ir["nodes"][3]["side_effects"]
        with self.assertRaises(TaskIRError):
            validate_task_ir(task_ir, REPOSITORY)


if __name__ == "__main__":
    unittest.main()


class FileFactTests(unittest.TestCase):
    def test_file_read_is_audited_and_digested(self):
        from derail.world.files import FILE_TABLE, FileInventory, extract_text

        inventory = FileInventory.build(FILES)
        entry = inventory.get("file:Documents/Dundies_2026_Categories.txt")
        self.assertIsNotNone(entry)
        self.assertEqual(entry.mime, "text/plain")
        self.assertIn("Best Office DJ", entry.content)
        self.assertIsNone(inventory.get("Documents/missing.txt"))
        task_ir = load_task_ir(MINIWORLD / "task_ir" / "mw-004.json", REPOSITORY)
        with _scratch() as tmp:
            dbs = build_world(SEEDS, Path(tmp) / "world")
            with WorldCopy.open("miniworld", dbs, Path(tmp) / "copy", files_root=FILES) as world:
                record = GoldInterpreter(CONFIG).run(task_ir, world, REPOSITORY)
                writing = copy.deepcopy(task_ir)
                note = {
                    "node_id": "n5",
                    "op": "create",
                    "app": "files",
                    "semantic_goal": "Save a note",
                    "route_hint": [],
                    "reads": [],
                    "produces": [
                        {
                            "name": "saved",
                            "type": "Int",
                            "derivation": {"kind": "literal", "value": 1},
                        }
                    ],
                    "writes": [
                        {
                            "table": FILE_TABLE,
                            "column": "content",
                            "entity_ref": "file:Documents/dundies_note.txt",
                            "effect_type": "write_file",
                            "value_ref": "message",
                        }
                    ],
                    "reversibility_class": "R1",
                    "side_effects": [
                        {
                            "effect_id": "fx-n5",
                            "effect_type": "write_file",
                            "target_ref": "file:Documents/dundies_note.txt",
                            "reversibility_class": "R1",
                            "commit_scope": "local_filesystem",
                            "compensation_available": True,
                            "compensating_effect_type": "delete_file",
                            "compensation_verifier_id": None,
                            "checkpoint_required": False,
                            "checkpoint_id": None,
                            "restore_verifier_id": None,
                        }
                    ],
                    "inputs": [
                        {
                            "port_id": "message",
                            "type": "Text",
                            "grounding": "derived:n3:message",
                            "source": "upstream",
                            "cardinality": "one",
                        }
                    ],
                    "outputs": [
                        {
                            "port_id": "saved",
                            "type": "Int",
                            "grounding": "derived:n5:saved",
                            "source": "upstream",
                            "cardinality": "one",
                        }
                    ],
                    "verifier": None,
                    "critical": False,
                }
                writing["nodes"].append(note)
                writing["edges"].append(
                    {
                        "edge_id": "e4",
                        "from": {"node_id": "n3", "port_id": "message"},
                        "to": {"node_id": "n5", "port_id": "message"},
                        "kind": "data_dependency",
                    }
                )
            with WorldCopy.open("miniworld", dbs, Path(tmp) / "copy2", files_root=FILES) as world:
                written = GoldInterpreter(CONFIG).run(writing, world, REPOSITORY)
                self.assertEqual(
                    extract_text(Path(world.files_root) / "Documents/dundies_note.txt"),
                    "You're up for Best Office DJ at the Dundies.",
                )
        self.assertEqual(record["reads_undeclared"], [])
        file_read = [r for r in record["resolved_reads"] if r["table"] == FILE_TABLE][0]
        self.assertEqual(file_read["entity_ref"], "file:Documents/Dundies_2026_Categories.txt")
        file_write = [w for w in written["writes_gold"] if w["table"] == FILE_TABLE][0]
        self.assertEqual(file_write["value"], "You're up for Best Office DJ at the Dundies.")
        self.assertNotEqual(
            written["state_timeline"][-1]["state_sha256"],
            record["state_timeline"][-1]["state_sha256"],
        )

    def test_docx_and_xlsx_text_extraction(self):
        import zipfile
        from derail.world.files import extract_text

        with _scratch() as tmp:
            docx = Path(tmp) / "a.docx"
            with zipfile.ZipFile(docx, "w") as z:
                z.writestr(
                    "word/document.xml",
                    "<w:document><w:body><w:p><w:r><w:t>Hello</w:t></w:r><w:r><w:t>World</w:t></w:r></w:p><w:p><w:r><w:t>Second</w:t></w:r></w:p></w:body></w:document>",
                )
            self.assertEqual(extract_text(docx), "Hello World\nSecond")
            xlsx = Path(tmp) / "b.xlsx"
            with zipfile.ZipFile(xlsx, "w") as z:
                z.writestr(
                    "xl/sharedStrings.xml", "<sst><si><t>Movie</t></si><si><t>Date</t></si></sst>"
                )
                z.writestr(
                    "xl/worksheets/sheet1.xml",
                    "<worksheet><sheetData>"
                    '<row r="1"><c r="A1" t="s"><v>0</v></c><c r="B1" t="s"><v>1</v></c></row>'
                    '<row r="2"><c r="A2" t="inlineStr"><is><t>x</t></is></c>'
                    '<c r="B2"><v>42</v></c></row>'
                    "</sheetData></worksheet>",
                )
            self.assertEqual(extract_text(xlsx).splitlines()[0], "Movie\tDate")
            fake_odt = Path(tmp) / "Morning_Brief.odt"
            fake_odt.write_text("Brief for Monday\nSecond line", encoding="utf-8")
            self.assertEqual(extract_text(fake_odt), "Brief for Monday\nSecond line")
            binary_odt = Path(tmp) / "broken.odt"
            binary_odt.write_bytes(b"PK\x00\x01garbage")
            self.assertIsNone(extract_text(binary_odt))
