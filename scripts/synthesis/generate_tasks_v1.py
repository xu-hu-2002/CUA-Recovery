#!/usr/bin/env python3
"""Compose, verify, profile and select generated tasks from the accepted IR set."""

from __future__ import annotations

import argparse
import json
from dataclasses import replace
import os
import sys
import tempfile
from pathlib import Path

import yaml

REPO_ROOT = Path(os.environ.get("RECOVERY_REPO_ROOT", Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(REPO_ROOT / "src"))

from recovery.derived.schema import validate_schema  # noqa: E402
from recovery.detect.mutations import MutationLibrary  # noqa: E402
from recovery.gen.graft import SearchConfig  # noqa: E402
from recovery.gen.hazards import HazardConfig  # noqa: E402
from recovery.gen.pipeline import (  # noqa: E402
    GenerationConfig,
    GoldExecutor,
    generation_record,
    load_seeds,
    run_generation,
    write_bundle,
)
from recovery.gen.realize import RealizationConfig  # noqa: E402
from recovery.ir.extract import V1ExtractorConfig  # noqa: E402
from recovery.ir.gold_interpreter import GoldInterpreter, InterpreterConfig  # noqa: E402
from recovery.ir.rubric_check import RubricCheckConfig  # noqa: E402
from recovery.longhorizon.types import ValueTypeRegistry  # noqa: E402
from recovery.world.schema_graph import SchemaGraph  # noqa: E402

PURPOSE = "realization"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--ir-dir", type=Path, required=True, help="accepted task_ir/ directory")
    parser.add_argument(
        "--tasks",
        type=Path,
        help="seed tasks json (instruction style stats; grading.rubrics -> composed rubrics)",
    )
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--db-dir", type=Path, default=os.environ.get("RECOVERY_VM_DB_DIR"))
    parser.add_argument("--files-root", default=os.environ.get("RECOVERY_VM_FILES_DIR"))
    parser.add_argument("--workdir", default=os.environ.get("RECOVERY_TMP_ROOT"))
    parser.add_argument("--world-id", help="default: extractor config v1.world_id")
    parser.add_argument("--generation-version", default="gen_v1")
    parser.add_argument("--max-seeds", type=int)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--skip-mutation-test", action="store_true")
    parser.add_argument("--realize", action="store_true", help="call the realization model")
    parser.add_argument(
        "--workers",
        type=int,
        default=int(os.environ.get("RECOVERY_GEN_WORKERS", "1")),
        help="worker processes for the verification stage (default RECOVERY_GEN_WORKERS or 1)",
    )
    cfg = REPO_ROOT / "configs/synthesis"
    parser.add_argument("--extractor-config", type=Path, default=cfg / "task_ir_v1_extractor.yaml")
    parser.add_argument("--interpreter-config", type=Path, default=cfg / "gold_interpreter_v1.yaml")
    parser.add_argument("--sampling", type=Path, default=cfg / "sampling_v1.yaml")
    parser.add_argument("--mutations", type=Path, default=cfg / "mutations_v1.yaml")
    parser.add_argument("--hazards", type=Path, default=cfg / "hazards_v1.yaml")
    parser.add_argument("--realization", type=Path, default=cfg / "realization_v1.yaml")
    parser.add_argument("--value-types", type=Path, default=cfg / "value_types_v0.yaml")
    parser.add_argument("--rubric-check", type=Path, default=cfg / "rubric_check_v1.yaml")
    parser.add_argument(
        "--schema-graph", type=Path, help="default: extractor config v1.schema_graph"
    )
    args = parser.parse_args()
    if not args.db_dir:
        raise SystemExit("need --db-dir or RECOVERY_VM_DB_DIR")

    extractor = V1ExtractorConfig.from_yaml(args.extractor_config, REPO_ROOT)
    sampling = yaml.safe_load(args.sampling.read_text(encoding="utf-8"))
    hazards_raw = yaml.safe_load(args.hazards.read_text(encoding="utf-8")) or {}
    graph_path = args.schema_graph or extractor.schema_graph_path
    search = SearchConfig.from_sampling(sampling)
    search = replace(
        search, compat=replace(search.compat, excluded_literals=tuple(extractor.persona_literals))
    )
    config = GenerationConfig(
        sampling=sampling,
        search=search,
        mutation_library=MutationLibrary.from_yaml(args.mutations),
        hazards=HazardConfig.from_yaml(args.hazards),
        hazard_selectors=hazards_raw.get("edge_selectors") or {},
        realization=RealizationConfig.from_yaml(args.realization, REPO_ROOT),
        registry=ValueTypeRegistry.from_yaml(args.value_types),
        schema_graph=SchemaGraph.from_record(
            json.loads(Path(graph_path).read_text(encoding="utf-8"))
        )
        if Path(graph_path).is_file()
        else None,
        world_id=args.world_id or extractor.world_id,
        generation_version=args.generation_version,
        mutation_max_per_node=int((sampling.get("mutation_test") or {}).get("max_per_node", 4)),
        run_mutation_test=not args.skip_mutation_test,
        seed=args.seed,
        max_seeds=args.max_seeds,
        persona_literals=extractor.persona_literals,
        workers=max(1, args.workers),
        rubric_check=RubricCheckConfig.from_yaml(args.rubric_check),
    )
    client = None
    if args.realize:
        from recovery.longhorizon.extraction import OpenAICompatibleClient

        client = OpenAICompatibleClient(config.realization.base, PURPOSE)
    instructions, instruction_of, rubrics = [], {}, {}
    if args.tasks:
        for t in json.loads(args.tasks.read_text(encoding="utf-8")):
            text = str(t.get("instruction") or t.get("task") or "")
            instructions.append(text)
            instruction_of[str(t.get("id"))] = text
            if (t.get("grading") or {}).get("rubrics"):
                rubrics[str(t.get("id"))] = t["grading"]["rubrics"]
    config = replace(config, source_rubrics=rubrics)
    seeds = load_seeds(args.ir_dir, REPO_ROOT)
    interpreter = GoldInterpreter(InterpreterConfig.from_yaml(args.interpreter_config))
    print("seeds %d from %s\nwriting %s" % (len(seeds), args.ir_dir, args.out), file=sys.stderr)
    with tempfile.TemporaryDirectory(dir=args.workdir) as tmp:
        executor = GoldExecutor(
            interpreter,
            args.db_dir,
            config.world_id,
            Path(tmp) / "gold",
            REPO_ROOT,
            files_root=args.files_root,
            app_databases=extractor.app_databases,
        )
        result = run_generation(
            seeds,
            config,
            executor,
            Path(tmp) / "mut",
            seed_instructions=[i for i in instructions if i],
            realization_client=client,
            progress=lambda msg: print(msg, file=sys.stderr),
            instruction_of=instruction_of,
        )
    for candidate in result["candidates"][:1]:
        validate_schema(
            generation_record(candidate, config.generation_version),
            "generation_record.schema.json",
            REPO_ROOT,
        )
    manifest = write_bundle(result, args.out, config)
    print(
        json.dumps(
            {
                k: manifest[k]
                for k in (
                    "seeds",
                    "candidates",
                    "composed",
                    "gold_executed",
                    "selected",
                    "selected_by_bucket",
                )
            }
        ),
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
