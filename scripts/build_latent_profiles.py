#!/usr/bin/env python3
"""Static detectability profiles for a directory of Task IRs (execution doc v1.2 §6.2, §6.6).

    python scripts/build_latent_profiles.py --ir-dir <run>/task_ir --gold-dir <run>/rubric_check \
        --sampling configs/synthesis/sampling_v1.yaml --out <run>/profiles

Writes ``profiles.jsonl`` (``latent-profile/1.0``), ``mutations.jsonl`` (``mutation/1.0``,
static) and ``summary.md`` (bucket distribution).  Gold lineages, when present, resolve
``<table>:*`` reads to concrete row ids.  Progress on stderr.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import Counter
from pathlib import Path

import yaml

REPO_ROOT = Path(os.environ.get("DERAIL_REPO_ROOT", Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(REPO_ROOT / "src"))

from derail.derived.schema import validate_schema  # noqa: E402
from derail.detect.latent_static import static_latent_horizons  # noqa: E402
from derail.detect.mutations import MutationLibrary, static_mutation_records  # noqa: E402
from derail.detect.profile import PROFILE_VERSION, build_profile  # noqa: E402
from derail.ir.model import load_task_ir  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--ir-dir", type=Path, required=True)
    parser.add_argument("--gold-dir", type=Path, help="directory with <task_id>.gold_lineage.json")
    parser.add_argument(
        "--sampling", type=Path, default=REPO_ROOT / "configs/synthesis/sampling_v1.yaml"
    )
    parser.add_argument(
        "--mutations", type=Path, default=REPO_ROOT / "configs/synthesis/mutations_v1.yaml"
    )
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    sampling = yaml.safe_load(args.sampling.read_text(encoding="utf-8"))
    buckets = list(sampling["latent_horizon_semantic_bucket_targets"])
    targets = list(sampling.get("target_buckets", [b for b in buckets if b != "1"]))
    library = MutationLibrary.from_yaml(args.mutations)
    files = sorted(args.ir_dir.glob("*.json"))
    args.out.mkdir(parents=True, exist_ok=True)
    print("reading %d IRs\nwriting %s" % (len(files), args.out), file=sys.stderr)
    profiles, mutations, errors, started = [], [], [], time.time()
    for index, path in enumerate(files, 1):
        try:
            task_ir = load_task_ir(path, REPO_ROOT)
            gold = None
            if args.gold_dir:
                gold_path = args.gold_dir / ("%s.gold_lineage.json" % path.stem)
                gold = (
                    json.loads(gold_path.read_text(encoding="utf-8"))
                    if gold_path.is_file()
                    else None
                )
            horizons = static_latent_horizons(task_ir, gold)
            profile = build_profile(
                task_ir,
                horizons,
                buckets,
                targets,
                int(sampling.get("secondary_constraints", {}).get("carry_threshold", 3)),
            )
            validate_schema(profile, "latent_profile.schema.json", REPO_ROOT)
            records = static_mutation_records(task_ir, horizons, library, PROFILE_VERSION)
            for record in records[:1]:
                validate_schema(record, "mutation.schema.json", REPO_ROOT)
            profiles.append(profile)
            mutations.extend(records)
        except Exception as exc:  # recorded, not fatal
            errors.append({"task": path.stem, "error": str(exc)[:300]})
        elapsed = time.time() - started
        print(
            "[%d/%d] %s | elapsed %.0fs eta %.0fs"
            % (index, len(files), path.stem, elapsed, elapsed / index * (len(files) - index)),
            file=sys.stderr,
        )
    (args.out / "profiles.jsonl").write_text(
        "".join(json.dumps(p, ensure_ascii=False) + "\n" for p in profiles)
    )
    (args.out / "mutations.jsonl").write_text(
        "".join(json.dumps(m, ensure_ascii=False) + "\n" for m in mutations)
    )
    if errors:
        (args.out / "errors.jsonl").write_text("".join(json.dumps(e) + "\n" for e in errors))
    bucket_counts = Counter(p["bucket"] for p in profiles)
    class_counts = Counter()
    for p in profiles:
        for k, v in p["class_counts"].items():
            class_counts[k] += v
    lines = [
        "# Static detectability summary",
        "",
        "tasks: %d (errors %d)" % (len(profiles), len(errors)),
        "",
        "| task bucket | tasks |",
        "|---|---|",
    ] + ["| %s | %d |" % kv for kv in sorted(bucket_counts.items())]
    lines += ["", "| node class | nodes |", "|---|---|"] + [
        "| %s | %d |" % kv for kv in sorted(class_counts.items())
    ]
    lines += [
        "",
        "tasks with >= min_nodes_in_target_bucket (%d): %d"
        % (
            int(sampling.get("min_nodes_in_target_bucket", 2)),
            sum(
                1
                for p in profiles
                if p["nodes_in_target_bucket"] >= int(sampling.get("min_nodes_in_target_bucket", 2))
            ),
        ),
    ]
    (args.out / "summary.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines[2:]), file=sys.stderr)
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
