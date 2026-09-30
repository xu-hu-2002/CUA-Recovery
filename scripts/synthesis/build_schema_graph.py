#!/usr/bin/env python3
"""Build the schema-graph/1.0 record over a set of application databases."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

REPO_ROOT = Path(os.environ.get("RECOVERY_REPO_ROOT", Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(REPO_ROOT / "src"))

from recovery.derived.schema import validate_schema  # noqa: E402
from recovery.world.schema_graph import SchemaGraphConfig, build_schema_graph  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--db", nargs="*", default=[], help="app=path.sqlite")
    parser.add_argument("--db-dir", help="directory holding <app>.sqlite files")
    parser.add_argument("--graph-id", required=True)
    parser.add_argument(
        "--config", default=str(REPO_ROOT / "configs/synthesis/schema_graph_v1.yaml")
    )
    parser.add_argument("--out", required=True)
    parser.add_argument(
        "--source-note", default="", help="free-text provenance (image tag, commit)"
    )
    args = parser.parse_args()

    sources = {}
    if args.db_dir:
        for path in sorted(Path(args.db_dir).glob("*.sqlite")):
            if path.is_symlink() or path.stat().st_size == 0:
                continue
            sources[path.stem] = path
    for item in args.db:
        app, _, path = item.partition("=")
        sources[app] = Path(path)
    if not sources:
        raise SystemExit("no databases given")
    config = SchemaGraphConfig.from_yaml(args.config)
    print("reading %d databases\nwriting %s" % (len(sources), args.out), file=sys.stderr)
    started = time.time()

    def progress(done, total, app):
        elapsed = time.time() - started
        print(
            "[%d/%d] %s | elapsed %.1fs eta %.1fs"
            % (done, total, app, elapsed, elapsed / done * (total - done)),
            file=sys.stderr,
        )

    record = build_schema_graph(
        sources,
        config,
        args.graph_id,
        progress=progress,
        provenance={"source_note": args.source_note, "config": str(args.config)},
    )
    validate_schema(record, "schema_graph.schema.json", REPO_ROOT)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(record, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    tables = sum(len(d["tables"]) for d in record["databases"])
    kinds = {}
    for relation in record["relations"]:
        kinds[relation["kind"]] = kinds.get(relation["kind"], 0) + 1
    manifest = {
        "graph_id": args.graph_id,
        "databases": len(record["databases"]),
        "tables": tables,
        "relations": kinds,
        "out_sha256": hashlib.sha256(out.read_bytes()).hexdigest(),
        "seconds": round(time.time() - started, 2),
    }
    out.with_name(out.stem + ".manifest.json").write_text(json.dumps(manifest, indent=1) + "\n")
    print(json.dumps(manifest), file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
