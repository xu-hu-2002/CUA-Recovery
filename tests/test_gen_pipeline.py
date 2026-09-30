"""Phase 4 driver on the miniworld, from seeds to bundle."""

from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path

import yaml

from derail.derived.schema import validate_schema
from derail.detect.mutations import MutationLibrary
from derail.gen.compat import CompatConfig
from derail.gen.graft import SearchConfig
from derail.gen.hazards import HazardConfig
from derail.gen.pipeline import (
    Candidate,
    GenerationConfig,
    GoldExecutor,
    generation_record,
    hazard_candidates,
    load_seeds,
    mutation_rejection_rate,
    run_generation,
    select_by_targets,
    write_bundle,
)
from derail.gen.realize import RealizationConfig
from derail.gen.verifiers import MutationOutcome
from derail.ir.gold_interpreter import GoldInterpreter, InterpreterConfig
from derail.ir.rubric_check import RubricCheckConfig
from derail.longhorizon.types import ValueTypeRegistry
from derail.world.sqlite_fixture import build_world

REPOSITORY = Path(__file__).resolve().parents[1]
MINIWORLD = REPOSITORY / "tests" / "miniworld"
SEEDS = {
    "hoolicalendar": MINIWORLD / "hoolicalendar.seed.json",
    "workbuzz": MINIWORLD / "workbuzz.seed.json",
}


def _config(**overrides):
    sampling = yaml.safe_load(
        (REPOSITORY / "configs/synthesis/sampling_v1.yaml").read_text(encoding="utf-8")
    )
    sampling["target_count"] = 6
    sampling["secondary_constraints"]["dependency_depth"] = [3, 12]
    hazards_raw = yaml.safe_load(
        (REPOSITORY / "configs/synthesis/hazards_v1.yaml").read_text(encoding="utf-8")
    )
    search = SearchConfig.from_sampling(sampling)
    search = SearchConfig(
        **{
            **search.__dict__,
            "beam_width": 3,
            "max_grafts": 1,
            "compat": CompatConfig(require_cell=False),
        }
    )
    fields = dict(
        sampling=sampling,
        search=search,
        mutation_library=MutationLibrary.from_yaml(
            REPOSITORY / "configs/synthesis/mutations_v1.yaml"
        ),
        hazards=HazardConfig.from_yaml(REPOSITORY / "configs/synthesis/hazards_v1.yaml"),
        hazard_selectors=hazards_raw["edge_selectors"],
        realization=RealizationConfig.from_yaml(
            REPOSITORY / "configs/synthesis/realization_v1.yaml", REPOSITORY
        ),
        registry=ValueTypeRegistry.from_yaml(REPOSITORY / "configs/synthesis/value_types_v0.yaml"),
        schema_graph=None,
        world_id="miniworld",
        mutation_max_per_node=1,
        persona_literals=("Michael Scott", "michael.scott@dundermifflin.com"),
    )
    fields.update(overrides)
    return GenerationConfig(**fields)


class PipelineTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(dir=os.environ.get("DERAIL_TMP_ROOT"))
        self.root = Path(self.tmp.name)
        build_world(SEEDS, self.root / "world")
        self.interpreter = GoldInterpreter(
            InterpreterConfig.from_yaml(REPOSITORY / "configs/synthesis/gold_interpreter_v1.yaml")
        )

    def tearDown(self):
        self.tmp.cleanup()

    def _executor(self):
        return GoldExecutor(
            self.interpreter,
            self.root / "world",
            "miniworld",
            self.root / "gold",
            REPOSITORY,
            files_root=MINIWORLD / "files",
        )

    def test_executor_caches_and_rejects(self):
        executor = self._executor()
        seeds = load_seeds(MINIWORLD / "task_ir", REPOSITORY)
        ir = next(s for s in seeds if s["task_id"] == "mw-002")
        self.assertTrue(executor.executable(ir))
        self.assertTrue(executor.executable(ir))
        self.assertEqual(executor.runs, 1)
        broken = json.loads(json.dumps(ir))
        broken["nodes"][0]["produces"][0]["derivation"] = {
            "kind": "sql",
            "sql": "SELECT * FROM nope",
        }
        self.assertFalse(executor.executable(broken))
        self.assertTrue(executor.errors)

    def test_selection_fills_quotas_and_keeps_seeds_out_of_target_buckets(self):
        sampling = {
            "target_count": 4,
            "latent_horizon_semantic_bucket_targets": {"1": 0.5, "2-3": 0.5},
            "target_buckets": ["2-3"],
            "gates": {"min_mutation_rejection_rate": 0.5},
        }

        def cand(task_id, bucket, origin, score, rate=1.0, rejections=()):
            c = Candidate(
                {"task_id": task_id},
                {"bucket": bucket},
                score,
                list(rejections),
                0,
                origin,
                gold={"values": []},
            )
            c.mutation_rejection_rate = rate
            return c

        pool = [
            cand("s1", "1", "seed", 0.9),
            cand("s2", "2-3", "seed", 0.8),
            cand("c1", "2-3", "composed", 0.7),
            cand("c2", "2-3", "composed", 0.6, rate=0.1),
            cand("c3", "2-3", "composed", 0.5),
            cand("c4", "2-3", "composed", 0.4),
            cand("c5", "1", "composed", 0.3, rejections=["DECORATIVE_CARRY"]),
        ]
        summary = select_by_targets(pool, sampling)
        chosen = {c.task_id for c in pool if c.selected}
        self.assertEqual(chosen, {"s1", "c1", "c3"})
        self.assertEqual(summary["shortfall"], {"1": 1})
        self.assertEqual(pool[1].selection_reason, "seed_only_fills_control_buckets")
        self.assertTrue(pool[3].selection_reason.startswith("mutation_rejection_rate_below"))

    def test_rejection_rate_execution_error_policy(self):
        def outcome(caught_by):
            return MutationOutcome("n1", "x", "m", 1, caught_by, None, "silent")

        outcomes = [outcome("n2"), outcome("final"), outcome(None), outcome("execution_error")]
        self.assertEqual(mutation_rejection_rate(outcomes), round(2 / 3, 4))
        self.assertEqual(mutation_rejection_rate(outcomes, "caught"), 0.75)
        self.assertEqual(mutation_rejection_rate(outcomes, "not_caught"), 0.5)
        with self.assertRaises(ValueError):
            mutation_rejection_rate(outcomes, "bogus")

    def test_hazard_candidates_from_gold_values(self):
        executor = self._executor()
        ir = next(
            s for s in load_seeds(MINIWORLD / "task_ir", REPOSITORY) if s["task_id"] == "mw-002"
        )
        gold = executor.run(ir)
        config = _config()
        from derail.gen.hazards import base_rates

        rates = base_rates(config.hazards, self.root / "world")
        records = hazard_candidates(
            ir, gold, config.hazards, rates, "miniworld", config.hazard_selectors, 2
        )
        self.assertTrue(records)
        for record in records:
            validate_schema(record, "hazard_injection.schema.json", REPOSITORY)
            self.assertEqual(record["task_id"], "mw-002")
        self.assertLessEqual(len(records), 2)

    def test_end_to_end_bundle(self):
        executor = self._executor()
        seeds = load_seeds(MINIWORLD / "task_ir", REPOSITORY)
        config = _config(
            workers=2,
            rubric_check=RubricCheckConfig.from_yaml(
                REPOSITORY / "configs/synthesis/rubric_check_v1.yaml"
            ),
            source_rubrics={
                s["task_id"]: [{"criterion": "Does %s" % s["task_id"], "weight": 1.0}]
                for s in seeds
            },
        )
        messages = []
        result = run_generation(
            seeds,
            config,
            executor,
            self.root / "mut",
            seed_instructions=["Move my 1:1 with Jim to Tuesday.", "Tell Pam about the party."],
            progress=messages.append,
        )
        candidates = result["candidates"]
        self.assertGreater(len(candidates), len(seeds))
        composed = [c for c in candidates if c.origin == "composed"]
        self.assertTrue(composed)
        self.assertTrue(any(c.gold for c in composed))
        self.assertTrue(any(c.mutation_outcomes for c in candidates if c.gold))
        self.assertTrue(any(c.selected for c in candidates))
        for candidate in candidates:
            record = generation_record(candidate, config.generation_version)
            validate_schema(record, "generation_record.schema.json", REPOSITORY)
            if candidate.selected:
                self.assertEqual(candidate.realization["status"], "dry_run")
                self.assertIn("graph", candidate.realization["prompt"].lower())
        manifest = write_bundle(result, self.root / "bundle", config)
        self.assertEqual(manifest["schema_version"], "generation-bundle/1.0")
        self.assertEqual(manifest["selected"], sum(manifest["selected_by_bucket"].values()))
        out = self.root / "bundle"
        for name in (
            "generation_bundle.json",
            "candidates.jsonl",
            "profiles.jsonl",
            "hazards.jsonl",
            "compat_index.jsonl",
            "summary.md",
        ):
            self.assertTrue((out / name).is_file(), name)
        self.assertTrue(list((out / "realization/prompts").glob("*.txt")))
        self.assertTrue(list((out / "records").glob("*.json")))
        rubric = json.loads(next((out / "rubrics").glob("gen-*.json")).read_text("utf-8"))
        self.assertEqual(
            [i["source_task_id"] for i in rubric["rubrics"]], rubric["source_task_ids"]
        )
        self.assertTrue(any("compat index" in m for m in messages))


if __name__ == "__main__":
    unittest.main()
