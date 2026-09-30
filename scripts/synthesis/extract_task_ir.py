#!/usr/bin/env python3
"""Render Task-IR extraction prompts; call the model only if approved."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Iterable

from recovery.derived.layout import atomic_write_json, sha256_file
from recovery.longhorizon.extraction import (
    ExtractorConfig,
    OpenAICompatibleClient,
    run_extraction,
)
from recovery.longhorizon.ontology import Ontology
from recovery.longhorizon.types import ValueTypeRegistry
from recovery.longhorizon.world import AppAliases, load_sources

PURPOSE = "task_ir_extraction"


def main(argv: Iterable[str] = ()) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--tasks", type=Path, required=True, help="all_tasks_with_grading.json")
    parser.add_argument("--world", type=Path, required=True, help="world_graph.json")
    parser.add_argument("--splits", type=Path, required=True, help="source_splits.json")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--task-id", action="append", default=[], help="restrict to these ids")
    parser.add_argument("--dev-sample", action="store_true", help="use the frozen dev sample")
    parser.add_argument(
        "--call-model", action="store_true", help="spend budget (needs approval env)"
    )
    parser.add_argument(
        "--reuse-replies", type=Path, help="re-parse <task_id>.txt replies from this directory"
    )
    parser.add_argument("--workers", type=int, default=1, help="concurrent model calls")
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[2])
    args = parser.parse_args(list(argv) if argv else None)
    root = args.repo_root.resolve()

    config = ExtractorConfig.from_yaml(args.config, root)
    sources = load_sources(config.world_sources_config)
    aliases = AppAliases.from_config(sources["app_aliases"])
    ontology = Ontology.from_yaml(config.ontology_config)
    registry = ValueTypeRegistry.from_yaml(config.value_types_config)
    world = json.loads(args.world.read_text(encoding="utf-8"))
    splits_record = json.loads(args.splits.read_text(encoding="utf-8"))
    tasks = json.loads(args.tasks.read_text(encoding="utf-8"))
    wanted = set(args.task_id)
    if args.dev_sample:
        wanted |= set(splits_record["dev_sample"]["task_ids"])
    if wanted:
        tasks = [task for task in tasks if task["id"] in wanted]
    if not tasks:
        raise SystemExit("no tasks selected")
    client = OpenAICompatibleClient(config, PURPOSE) if args.call_model else None
    started = time.monotonic()

    def _progress(done: int, total: int, task_id: str, seconds: float) -> None:
        elapsed = time.monotonic() - started
        eta = (elapsed / done) * (total - done) if done else 0.0
        print(
            "[%d/%d] %s %.0fs | elapsed %.0fm eta %.0fm"
            % (done, total, task_id, seconds, elapsed / 60, eta / 60),
            file=sys.stderr,
            flush=True,
        )

    manifest = run_extraction(
        tasks,
        world=world,
        aliases=aliases,
        ontology=ontology,
        config=config,
        splits=splits_record["assignment"],
        output_dir=args.output_dir,
        source_uri=str(args.tasks.resolve()),
        source_sha256=sha256_file(args.tasks),
        client=client,
        saved_replies_dir=args.reuse_replies,
        registry=registry,
        workers=max(1, args.workers),
        progress=_progress if (args.call_model or args.reuse_replies) else None,
    )
    atomic_write_json(args.output_dir / "stage_manifest.json", manifest)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
