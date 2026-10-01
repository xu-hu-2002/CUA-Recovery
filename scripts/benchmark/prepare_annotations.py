#!/usr/bin/env python3
"""Initialize a derived build and normalize one MyPCBench task for annotation."""

from __future__ import annotations

import argparse
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

from recovery.canonical.mypcbench import load_canonical_jsonl, normalize_task_directory
from recovery.derived.layout import (
    DerivedBuild,
    atomic_write_json,
    sha256_file,
    sha256_json,
)
from recovery.derived.schema import validate_schema

FINGERPRINT_SCRIPTS = (
    "benchmark/prepare_annotations.py",
    "takeover/build_native_history.py",
)
DEFAULT_TARGET_AGENTS = (
    "qwen3_5_35b_a3b",
    "evocua_32b",
    "opencua_72b",
    "gpt_5_5",
)


def _git_commit(repository: Path) -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=repository, capture_output=True, text=True, check=True
    )
    return result.stdout.strip()


def _taxonomy_record(path: Path, frozen: bool) -> dict:
    path = path.resolve()
    lines = path.read_text(encoding="utf-8").splitlines()
    version = next(
        (line.split(":", 1)[1].strip() for line in lines if line.startswith("version:")), ""
    )
    labels = sorted(
        line.strip()[2:].strip()
        for line in lines
        if line.strip().startswith("- ")
    )
    if not version or not labels:
        raise RuntimeError("taxonomy must contain version and labels")
    if frozen and version == "draft":
        raise RuntimeError("a draft taxonomy cannot be marked frozen")
    return {
        "version": version,
        "frozen": frozen,
        "mode": "frozen" if frozen else "open_coding",
        "uri": str(path),
        "sha256": sha256_file(path),
        "seed_labels": labels,
        "labels": labels,
    }


def _source_tree_fingerprint(root: Path, suffix: str) -> str:
    return sha256_json(
        [
            (str(path.relative_to(root)), sha256_file(path))
            for path in sorted(root.rglob("*"))
            if path.is_file() and path.name.endswith(suffix)
        ]
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--collection-root", type=Path, required=True)
    parser.add_argument("--collection-id", required=True)
    parser.add_argument("--collection-manifest", type=Path)
    parser.add_argument("--build-root", type=Path, required=True)
    parser.add_argument("--build-id", required=True)
    parser.add_argument("--task-dir", type=Path, required=True)
    parser.add_argument(
        "--task-config-source",
        type=Path,
        help="Explicit immutable task-list JSON; use when a resumed run overwrote _tasks/batch.json.",
    )
    parser.add_argument("--trajectory-id", required=True)
    parser.add_argument("--source-agent", required=True)
    parser.add_argument(
        "--append",
        action="store_true",
        help="Append one new trajectory to an existing immutable build contract.",
    )
    parser.add_argument("--target-agents", nargs="+", default=DEFAULT_TARGET_AGENTS)
    parser.add_argument(
        "--wait-seconds",
        type=float,
        required=True,
        help="Frozen runner sleep_after used for raw WAIT rows; never inferred.",
    )
    parser.add_argument("--frame-width", type=int)
    parser.add_argument("--frame-height", type=int)
    parser.add_argument("--repository", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--taxonomy", type=Path)
    parser.add_argument("--taxonomy-frozen", action="store_true")
    args = parser.parse_args()

    collection_root = args.collection_root.resolve()
    collection_manifest = (
        args.collection_manifest.resolve()
        if args.collection_manifest
        else collection_root / "collection_manifest.json"
    )
    build = DerivedBuild(
        root=args.build_root.resolve(),
        build_id=args.build_id,
        collection_root=collection_root,
        collection_id=args.collection_id,
    )
    repository = args.repository.resolve()
    implementation_fingerprint = sha256_json(
        {
            "src_recovery": _source_tree_fingerprint(repository / "src" / "recovery", ".py"),
            "schemas": _source_tree_fingerprint(repository / "schemas", ".json"),
            "scripts": {
                path.name: path.read_bytes().hex()
                for path in (repository / "scripts" / name for name in FINGERPRINT_SCRIPTS)
            },
        }
    )
    taxonomy = _taxonomy_record(
        args.taxonomy or repository / "prompts" / "annotation" / "taxonomy.yaml",
        args.taxonomy_frozen,
    )
    builder_config = {
        "wait_seconds": args.wait_seconds,
        "frame": (
            [args.frame_width, args.frame_height]
            if args.frame_width is not None and args.frame_height is not None
            else "auto_from_png_ihdr"
        ),
        "depths": [0, 5, 10, 15, 20, 25],
        "implementation_sha256": implementation_fingerprint,
        "target_agents": list(args.target_agents),
    }
    if build.manifest_path.is_file():
        if not args.append:
            raise RuntimeError("build already exists; pass --append to add trajectories")
        manifest = build.read_manifest()
        if manifest["source_collection"]["uri"] != str(collection_root):
            raise RuntimeError("append collection root does not match the build")
        if manifest["source_collection"]["collection_id"] != args.collection_id:
            raise RuntimeError("append collection ID does not match the build")
        if manifest["builder_config"] != builder_config or manifest["taxonomy"] != taxonomy:
            raise RuntimeError("append must not change the builder/taxonomy contract")
        if not build.verify_raw_unchanged():
            raise RuntimeError("raw collection changed before append")
    else:
        if args.append:
            raise RuntimeError("--append only works on an existing build")
        manifest = build.initialize(
            collection_manifest_path=collection_manifest,
            git_commit=_git_commit(repository),
            builder_config=builder_config,
            taxonomy=taxonomy,
            created_at=datetime.now(timezone.utc).isoformat(),
        )
    output_dir = build.artifact_path("canonical", args.trajectory_id, "trajectory.jsonl").parent
    if output_dir.exists() and any(output_dir.iterdir()):
        raise RuntimeError("trajectory_id already exists, refusing to overwrite: %s" % args.trajectory_id)
    report = normalize_task_directory(
        args.task_dir.resolve(),
        output_dir,
        trajectory_id=args.trajectory_id,
        source_agent=args.source_agent,
        wait_seconds=args.wait_seconds,
        frame_width=args.frame_width,
        frame_height=args.frame_height,
        task_config_source=args.task_config_source,
    )
    steps = (
        load_canonical_jsonl(output_dir / "trajectory.jsonl")
        if report.get("annotation_complete", report["normalization_complete"])
        else ()
    )
    for step in steps:
        validate_schema(step.to_dict(), "canonical_trajectory.schema.json", repository)
    annotation_task = {
        "trajectory_id": args.trajectory_id,
        "agent_id": args.source_agent,
        "source_trajectory_sha256": report["source_trajectory_sha256"],
        "instructions": {
            "root_unit": "action_index_global",
            "horizon_unit": "executed_actions_after_root",
            "human_verification_required": True,
            "minimum_independent_annotations": 2,
            "depths": [0, 5, 10, 15, 20, 25],
        },
        "root_action_zero_warning": (
            "raw runner did not save an initial pre-action observation"
            if steps and not report["root_action_zero_eligible"]
            else None
        ),
        "root_action_zero_eligible": report["root_action_zero_eligible"],
        "normalization_gate_passed": report["normalization_complete"],
        "annotation_gate_passed": report.get(
            "annotation_complete", report["normalization_complete"]
        ),
        "actions": [step.to_dict() for step in steps],
    }
    atomic_write_json(output_dir / "annotation_task.json", annotation_task)
    manifest = build.read_manifest()
    counts = dict(manifest["counts"])
    count_key = (
        "canonical_trajectories" if report["normalization_complete"] else "rejected_trajectories"
    )
    counts[count_key] = counts.get(count_key, 0) + 1
    normalization_reports = list(manifest.get("normalization_reports", ()))
    normalization_reports.append(
        {
            "trajectory_id": args.trajectory_id,
            "uri": str((output_dir / "normalization_report.json").resolve()),
            "case_eligible": report["case_eligible"],
        }
    )
    build.update_manifest(
        {
            "status": "canonicalized",
            "counts": counts,
            "normalization_reports": normalization_reports,
        }
    )
    validate_schema(build.read_manifest(), "derived_build_manifest.schema.json", repository)
    if not build.verify_raw_unchanged():
        raise RuntimeError("raw collection changed after derived initialization")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["normalization_complete"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
