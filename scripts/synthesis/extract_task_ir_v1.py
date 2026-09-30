#!/usr/bin/env python3
"""Task IR v1 extraction for seed tasks."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

REPO_ROOT = Path(os.environ.get("DERAIL_REPO_ROOT", Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(REPO_ROOT / "src"))

from derail.ir.extract import V1ExtractorConfig, run_extraction_v1  # noqa: E402
from derail.longhorizon.extraction import OpenAICompatibleClient  # noqa: E402
from derail.longhorizon.ontology import Ontology  # noqa: E402
from derail.longhorizon.types import ValueTypeRegistry  # noqa: E402
from derail.longhorizon.world import AppAliases, load_sources  # noqa: E402
from derail.world.schema_graph import SchemaGraph  # noqa: E402

PURPOSE = "task_ir_extraction"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "--config", type=Path, default=REPO_ROOT / "configs/synthesis/task_ir_v1_extractor.yaml"
    )
    parser.add_argument("--tasks", type=Path, required=True, help="all_tasks_with_grading.json")
    parser.add_argument("--splits", type=Path, help="source_splits.json (for --dev-sample)")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--task-id", action="append", default=[])
    parser.add_argument("--dev-sample", action="store_true")
    parser.add_argument("--call-model", action="store_true")
    parser.add_argument("--reuse-replies", type=Path)
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args()

    config = V1ExtractorConfig.from_yaml(args.config, REPO_ROOT)
    database_dir = os.environ.get(config.database_dir_env)
    if database_dir and not Path(database_dir).is_dir():
        raise SystemExit("%s=%s is not a directory" % (config.database_dir_env, database_dir))
    schema_graph = SchemaGraph.from_record(
        json.loads(config.schema_graph_path.read_text(encoding="utf-8"))
    )
    sources = load_sources(config.base.world_sources_config)
    aliases = AppAliases.from_config(sources["app_aliases"])
    ontology = Ontology.from_yaml(config.base.ontology_config)
    registry = ValueTypeRegistry.from_yaml(config.base.value_types_config)
    tasks = json.loads(args.tasks.read_text(encoding="utf-8"))
    wanted = set(args.task_id)
    if args.dev_sample:
        if not args.splits:
            raise SystemExit("--dev-sample needs --splits")
        wanted |= set(json.loads(args.splits.read_text(encoding="utf-8"))["dev_sample"]["task_ids"])
    if wanted:
        tasks = [task for task in tasks if task["id"] in wanted]
    if not tasks:
        raise SystemExit("no tasks selected")
    client = OpenAICompatibleClient(config.base, PURPOSE) if args.call_model else None
    print(
        "reading %s (%d tasks), databases=%s\nwriting %s"
        % (args.tasks, len(tasks), database_dir or "(none)", args.output_dir),
        file=sys.stderr,
    )
    started = time.monotonic()

    def progress(done, total, task_id, seconds):
        elapsed = time.monotonic() - started
        eta = (elapsed / done) * (total - done) if done else 0.0
        print(
            "[%d/%d] %s %.0fs | elapsed %.0fm eta %.0fm"
            % (done, total, task_id, seconds, elapsed / 60, eta / 60),
            file=sys.stderr,
            flush=True,
        )

    manifest = run_extraction_v1(
        tasks,
        config=config,
        schema_graph=schema_graph,
        database_dir=Path(database_dir) if database_dir else None,
        ontology=ontology,
        aliases=aliases,
        registry=registry,
        output_dir=args.output_dir,
        repository=REPO_ROOT,
        client=client,
        saved_replies_dir=args.reuse_replies,
        workers=args.workers,
        progress=progress,
    )
    print(json.dumps(manifest["counts"]), file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
