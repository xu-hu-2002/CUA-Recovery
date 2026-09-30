#!/usr/bin/env python3
"""One fix round of Task IR v1 extraction."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import yaml

REPO_ROOT = Path(os.environ.get("RECOVERY_REPO_ROOT", Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(REPO_ROOT / "src"))

from recovery.ir.extract import V1ExtractorConfig  # noqa: E402
from recovery.ir.fix_loop import run_fix_round  # noqa: E402
from recovery.longhorizon.extraction import OpenAICompatibleClient  # noqa: E402
from recovery.longhorizon.ontology import Ontology  # noqa: E402
from recovery.longhorizon.types import ValueTypeRegistry  # noqa: E402
from recovery.longhorizon.world import AppAliases, load_sources  # noqa: E402
from recovery.world.schema_graph import SchemaGraph  # noqa: E402

PURPOSE = "task_ir_extraction"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "--config", type=Path, default=REPO_ROOT / "configs/synthesis/task_ir_v1_extractor.yaml"
    )
    parser.add_argument(
        "--tasks",
        type=Path,
        default=REPO_ROOT / "third_party/MyPCBench/tasks/final/all_tasks_with_grading.json",
    )
    parser.add_argument("--previous-run", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--round", type=int, required=True, help="1-based round number")
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args()
    raw = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    loop = raw.get("fix_loop") or {}
    max_rounds = int(loop.get("max_rounds", 3))
    if args.round > max_rounds:
        raise SystemExit("round %d exceeds fix_loop.max_rounds=%d" % (args.round, max_rounds))
    config = V1ExtractorConfig.from_yaml(args.config, REPO_ROOT)
    database_dir = os.environ.get(config.database_dir_env)
    schema_graph = SchemaGraph.from_record(
        json.loads(config.schema_graph_path.read_text(encoding="utf-8"))
    )
    sources = load_sources(config.base.world_sources_config)
    aliases = AppAliases.from_config(sources["app_aliases"])
    ontology = Ontology.from_yaml(config.base.ontology_config)
    registry = ValueTypeRegistry.from_yaml(config.base.value_types_config)
    tasks = json.loads(args.tasks.read_text(encoding="utf-8"))
    client = OpenAICompatibleClient(config.base, PURPOSE)
    started = time.monotonic()

    def progress(done, total, task_id, seconds):
        elapsed = time.monotonic() - started
        print(
            "[%d/%d] %s %.0fs | elapsed %.0fm eta %.0fm"
            % (done, total, task_id, seconds, elapsed / 60, (elapsed / done) * (total - done) / 60),
            file=sys.stderr,
            flush=True,
        )

    print(
        "previous %s\nwriting %s (round %d/%d)"
        % (args.previous_run, args.output_dir, args.round, max_rounds),
        file=sys.stderr,
    )
    manifest = run_fix_round(
        tasks,
        previous_run_dir=args.previous_run,
        output_dir=args.output_dir,
        config=config,
        template_path=REPO_ROOT
        / loop.get("feedback_prompt", "prompts/synthesis/task_ir_v1_fix_user.txt"),
        client=client,
        schema_graph=schema_graph,
        database_dir=Path(database_dir) if database_dir else None,
        ontology=ontology,
        aliases=aliases,
        registry=registry,
        repository=REPO_ROOT,
        workers=args.workers,
        progress=progress,
    )
    print(json.dumps(manifest.get("counts")), file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
