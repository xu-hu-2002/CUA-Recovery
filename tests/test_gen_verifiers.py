"""Verifier bundle, mutation values and the mutation test on the miniworld (doc 6.3, 7.5), and
the final-state verdict on real composed workflows of final_v1."""

from __future__ import annotations

import json
import os
import random
import tempfile
import unittest
from pathlib import Path

import yaml

from derail.detect.mutations import MutationLibrary
from derail.gen.pipeline import GoldExecutor
from derail.gen.verifiers import (
    compile_verifiers,
    mutation_test,
    mutation_values,
    rejection_rate,
    verify_final_state,
    world_sources,
)
from derail.ir.gold_interpreter import GoldInterpreter, InterpreterConfig, WorldCopy
from derail.ir.model import load_task_ir
from derail.world.files import FileInventory
from derail.world.sqlite_fixture import build_world

REPOSITORY = Path(__file__).resolve().parents[1]
MINIWORLD = REPOSITORY / "tests" / "miniworld"
SEEDS = {
    "hoolicalendar": MINIWORLD / "hoolicalendar.seed.json",
    "workbuzz": MINIWORLD / "workbuzz.seed.json",
}


class VerifierTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(dir=os.environ.get("DERAIL_TMP_ROOT"))
        self.dbs = build_world(SEEDS, Path(self.tmp.name) / "world")
        self.interpreter = GoldInterpreter(
            InterpreterConfig.from_yaml(REPOSITORY / "configs/synthesis/gold_interpreter_v1.yaml")
        )
        self.library = MutationLibrary.from_yaml(REPOSITORY / "configs/synthesis/mutations_v1.yaml")

    def tearDown(self):
        self.tmp.cleanup()

    def _gold(self, task):
        task_ir = load_task_ir(MINIWORLD / "task_ir" / (task + ".json"), REPOSITORY)
        with WorldCopy.open("miniworld", self.dbs, Path(self.tmp.name) / ("gold_" + task)) as world:
            return task_ir, self.interpreter.run(task_ir, world, REPOSITORY)

    def test_bundle_layers(self):
        task_ir, _ = self._gold("mw-002")
        bundle = compile_verifiers(task_ir)
        self.assertEqual([m["node_id"] for m in bundle["milestone_verifiers"]], ["n5"])
        self.assertEqual(bundle["verifier_kind"], "state")
        self.assertEqual(len(bundle["equivalent_final_states"]), 1)

    def test_overrides_reach_downstream_and_are_recorded(self):
        task_ir, _ = self._gold("mw-002")
        with WorldCopy.open("miniworld", self.dbs, Path(self.tmp.name) / "ov") as world:
            record = self.interpreter.run(
                task_ir, world, REPOSITORY, overrides={("n2", "start_at"): "2026-07-03T18:00:00"}
            )
        message = next(v["value"] for v in record["values"] if v["name"] == "message")
        self.assertIn("2026-07-03", message)
        self.assertIs(record["final_verifier_passed"], False)
        self.assertEqual(record["provenance"]["overrides"][0]["node_id"], "n2")

    def test_mutation_values_come_from_the_world(self):
        task_ir, gold = self._gold("mw-002")
        rng = random.Random(1)
        with WorldCopy.open("miniworld", self.dbs, Path(self.tmp.name) / "mv") as world:
            other = mutation_values(task_ir, gold, "n1", "event_id", "entity_other", world, rng)
            self.assertIsInstance(other, int)
            self.assertNotEqual(other, 545)
            shifted = mutation_values(
                task_ir, gold, "n2", "start_at", "date_plus_minus_day", world, rng
            )
            self.assertIn(shifted, ("2026-07-01T18:00:00", "2026-07-03T18:00:00"))
            self.assertIsNone(
                mutation_values(task_ir, gold, "n2", "start_at", "boolean_flip", world, rng)
            )

    def test_mutation_test_catches_date_and_entity_errors(self):
        task_ir, gold = self._gold("mw-002")
        outcomes = mutation_test(
            task_ir,
            gold,
            self.library,
            self.interpreter,
            self.dbs,
            Path(self.tmp.name) / "mt",
            REPOSITORY,
            seed=3,
        )
        by_key = {(o.node_id, o.mutation): o for o in outcomes}
        date = by_key[("n2", "date_plus_minus_day")]
        self.assertEqual(date.caught_by, "final")  # no downstream node re-reads the date
        self.assertEqual(date.observability_class, "verifier_only")
        entity = by_key[("n1", "entity_other")]
        self.assertIsNotNone(entity.caught_by)  # v-n1 expects 545; downstream verifiers or final
        self.assertGreaterEqual(rejection_rate(outcomes), 0.5)
        self.assertTrue(all(o.to_dict()["mutated_value"] is not None for o in outcomes))


FINAL_V1 = REPOSITORY / "data" / "synthesis" / "generation" / "final_v1"
VM_DB_DIR = os.environ.get("DERAIL_VM_DB_DIR")  # e.g. the offline VM image extract
VM_FILES_DIR = os.environ.get("DERAIL_VM_FILES_DIR")
# (workflow, kind of its final verifier, targeted tamper of the gold end state)
REAL_TASKS = (
    # conditionalized composition: all_of(A.final, B.final) over confirm nodes
    (
        "gen-04648f234f",
        "all_of",
        ("dinoco-airlines", "UPDATE flights SET status = 'scheduled' WHERE rowid = 2"),
    ),
    # the saved report is read back from the file tree
    ("gen-033574ebd2", "derived", ("file", "Documents/Morning_Brief.odt")),
    ("gen-cb32e220e8", "sql", ("mail", "DELETE FROM sent WHERE rowid = 1")),
    # the final predicate reads a decide node that aggregates confirm nodes
    ("gen-9ef0c5efd1", "derived", ("mail", "DELETE FROM drafts WHERE rowid = 1")),
)


@unittest.skipUnless(
    VM_DB_DIR and VM_FILES_DIR and FINAL_V1.is_dir(),
    "needs DERAIL_VM_DB_DIR / DERAIL_VM_FILES_DIR (VM databases and home tree)",
)
class FinalStateVerdictTests(unittest.TestCase):
    """Gold end state -> True; untouched start state and a tampered end state -> False."""

    def test_real_workflows(self):
        extractor = yaml.safe_load(
            (REPOSITORY / "configs/synthesis/task_ir_v1_extractor.yaml").read_text()
        )["v1"]
        interpreter = GoldInterpreter(
            InterpreterConfig.from_yaml(REPOSITORY / "configs/synthesis/gold_interpreter_v1.yaml")
        )
        executor = GoldExecutor(
            interpreter,
            VM_DB_DIR,
            extractor["world_id"],
            "unused",
            REPOSITORY,
            files_root=VM_FILES_DIR,
            app_databases=extractor["app_databases"],
        )
        every_db = world_sources(VM_DB_DIR, extractor["app_databases"])
        for task_id, kind, (app, tamper) in REAL_TASKS:
            with (
                self.subTest(task=task_id),
                tempfile.TemporaryDirectory(dir=os.environ.get("DERAIL_TMP_ROOT")) as tmp,
            ):
                task_ir = json.loads((FINAL_V1 / "task_ir" / (task_id + ".json")).read_text())
                gold = json.loads(
                    (FINAL_V1 / "gold_lineage" / (task_id + ".gold_lineage.json")).read_text()
                )
                self.assertEqual(task_ir["final_verifier"]["kind"], kind)
                with WorldCopy.open(task_id, every_db, Path(tmp) / "start", VM_FILES_DIR) as world:
                    verdict = verify_final_state(task_ir, gold, world, interpreter)
                    self.assertIs(verdict["recovered"], False, verdict)
                with WorldCopy.open(
                    task_id, executor.sources_for(task_ir), Path(tmp) / "gold", VM_FILES_DIR
                ) as world:
                    interpreter.run(task_ir, world, REPOSITORY)
                    verdict = verify_final_state(task_ir, gold, world, interpreter)
                    self.assertIs(verdict["recovered"], True, verdict)
                    if app == "file":
                        (world.files_root / tamper).unlink()
                        world.inventory = FileInventory.build(world.files_root)
                    else:
                        world.connection(app).execute(tamper)
                        world.connection(app).commit()
                    verdict = verify_final_state(task_ir, gold, world, interpreter)
                    self.assertIs(verdict["recovered"], False, verdict)


if __name__ == "__main__":
    unittest.main()
