#!/usr/bin/env python3
"""Canonicalize every human-labelled trajectory for one takeover source."""

from __future__ import annotations

import argparse
import json
import re
import subprocess
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Mapping, Sequence

from recovery.canonical.mypcbench import load_canonical_jsonl, normalize_task_directory
from recovery.derived.layout import DerivedBuild, sha256_file, sha256_json
from recovery.derived.schema import validate_schema


FINGERPRINT_SCRIPTS = (
    "benchmark/prepare_annotations.py",
    "benchmark/build_benchmark.py",
    "benchmark/replay_instance.py",
    "takeover/build_native_history.py",
)

RUN_TRAJECTORY_PATTERN = re.compile(r"^.+-r\d+-vm\d+-(?P<task_id>.+)$")
SHORT_TRAJECTORY_PATTERN = re.compile(r"^[^-]+-(?P<task_id>.+)$")


def read_object(path: Path) -> Mapping[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def git_commit(repository: Path) -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=repository, capture_output=True, text=True, check=True
    )
    return result.stdout.strip()


def labelled_trajectories(
    labels: Path, annotator: str, trajectory_prefixes: Sequence[str]
) -> Iterable[tuple[str, Mapping[str, object]]]:
    paths = {
        path
        for prefix in trajectory_prefixes
        for path in labels.glob(f"{prefix}*__{annotator}.json")
    }
    for path in sorted(paths):
        annotation = read_object(path)
        if annotation.get("annotator_role") == "human":
            yield str(annotation["trajectory_id"]), annotation


def index_raw_tasks(raw_root: Path) -> dict[str, list[Path]]:
    index: dict[str, list[Path]] = {}
    for path in raw_root.rglob("traj.jsonl"):
        index.setdefault(path.parent.name, []).append(path)
    return index


def find_raw_task(
    raw_tasks: Mapping[str, Sequence[Path]], trajectory_id: str, expected_sha256: str
) -> Path:
    match = RUN_TRAJECTORY_PATTERN.match(trajectory_id)
    if match is None:
        match = SHORT_TRAJECTORY_PATTERN.match(trajectory_id)
    if match is None:
        raise ValueError(f"invalid trajectory id: {trajectory_id}")
    candidates = raw_tasks.get(match.group("task_id"), ())
    matching = [path.parent for path in candidates if sha256_file(path) == expected_sha256]
    if len(matching) != 1:
        raise ValueError(f"{trajectory_id}: expected one hash-matched raw task, found {len(matching)}")
    return matching[0]


def taxonomy_record(repository: Path) -> dict[str, object]:
    path = repository / "prompts" / "annotation" / "taxonomy.yaml"
    lines = path.read_text(encoding="utf-8").splitlines()
    labels = sorted(line.strip()[2:].strip() for line in lines if line.strip().startswith("- "))
    version = next(line.split(":", 1)[1].strip() for line in lines if line.startswith("version:"))
    return {
        "version": version,
        "frozen": False,
        "mode": "open_coding",
        "uri": str(path.resolve()),
        "sha256": sha256_file(path),
        "seed_labels": labels,
        "labels": labels,
    }


def implementation_fingerprint(repository: Path) -> str:
    paths = sorted((repository / "src" / "recovery").rglob("*.py"))
    paths += sorted((repository / "schemas").glob("*.json"))
    paths += [repository / "scripts" / name for name in FINGERPRINT_SCRIPTS]
    return sha256_json([(str(path.relative_to(repository)), sha256_file(path)) for path in paths])


def initialize_build(args: argparse.Namespace, manifest_path: Path) -> DerivedBuild:
    build = DerivedBuild(args.build_root.resolve(), args.build_id, args.raw_root.resolve(), args.collection_id)
    config = {
        "wait_seconds": args.wait_seconds,
        "frame": "auto_from_png_ihdr",
        "depths": [0, 5, 10, 15, 20, 25],
        "implementation_sha256": implementation_fingerprint(args.repository),
        "target_agents": list(args.target_agents),
    }
    build.initialize(
        collection_manifest_path=manifest_path,
        git_commit=git_commit(args.repository),
        builder_config=config,
        taxonomy=taxonomy_record(args.repository),
        created_at=datetime.now(timezone.utc).isoformat(),
    )
    return build


def normalize_one(args: argparse.Namespace, build: DerivedBuild, raw_tasks, labelled) -> dict:
    trajectory_id, annotation = labelled
    try:
        task_dir = find_raw_task(
            raw_tasks, trajectory_id, str(annotation["source_trajectory_sha256"])
        )
    except ValueError as exc:
        return {
            "trajectory_id": trajectory_id,
            "uri": "",
            "case_eligible": False,
            "normalization_complete": False,
            "rejection_reason": str(exc),
        }
    output = build.path / "canonical" / trajectory_id
    task_config_source = task_dir.parent / "_tasks" / "batch.json"
    if args.task_config_root is not None:
        task_config_source = args.task_config_root / trajectory_id / "task_config.json"
    report = normalize_task_directory(
        task_dir,
        output,
        trajectory_id=trajectory_id,
        source_agent=args.source_agent,
        wait_seconds=args.wait_seconds,
        task_config_source=task_config_source,
    )
    if report["normalization_complete"]:
        steps = load_canonical_jsonl(output / "trajectory.jsonl")
        for step in steps:
            validate_schema(step.to_dict(), "canonical_trajectory.schema.json", args.repository)
    return {
        "trajectory_id": trajectory_id,
        "uri": str((output / "normalization_report.json").resolve()),
        "case_eligible": report["case_eligible"],
        "normalization_complete": report["normalization_complete"],
    }


def normalize_labelled_tasks(args: argparse.Namespace, build: DerivedBuild) -> tuple[int, int, list[dict]]:
    raw_tasks = index_raw_tasks(args.raw_root)
    labelled = tuple(labelled_trajectories(args.labels, args.annotator, args.trajectory_prefix))
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        reports = list(pool.map(lambda item: normalize_one(args, build, raw_tasks, item), labelled))
    accepted = sum(bool(report["normalization_complete"]) for report in reports)
    rejected = len(reports) - accepted
    return accepted, rejected, reports


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--annotator", required=True)
    parser.add_argument(
        "--trajectory-prefix",
        action="append",
        required=True,
        help="Annotation trajectory prefix; repeat for sources with multiple naming families.",
    )
    parser.add_argument("--source-agent", required=True)
    parser.add_argument("--collection-id", required=True)
    parser.add_argument(
        "--collection-manifest",
        type=Path,
        help="Explicit source manifest/provenance file when the archive has no collection_manifest.json.",
    )
    parser.add_argument("--build-root", type=Path, required=True)
    parser.add_argument("--build-id", required=True)
    parser.add_argument("--wait-seconds", type=float, default=1.0)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument(
        "--task-config-root",
        type=Path,
        help="Optional canonical/<trajectory-id>/ root containing immutable task_config.json files.",
    )
    parser.add_argument("--target-agents", nargs="+", required=True)
    parser.add_argument("--repository", type=Path, default=Path(__file__).resolve().parents[2])
    args = parser.parse_args()
    if args.workers <= 0:
        raise ValueError("workers must be positive")
    manifests = sorted(args.raw_root.rglob("collection_manifest*.json"))
    if args.collection_manifest is not None:
        manifests = [args.collection_manifest.expanduser().resolve()]
    if not manifests or not manifests[0].is_file():
        raise ValueError("raw root has no collection manifest")
    build = initialize_build(args, manifests[0])
    accepted, rejected, reports = normalize_labelled_tasks(args, build)
    manifest = build.read_manifest()
    manifest.update(status="canonicalized", normalization_reports=reports)
    manifest["counts"].update(canonical_trajectories=accepted, rejected_trajectories=rejected)
    from recovery.derived.layout import atomic_write_json
    atomic_write_json(build.manifest_path, manifest)
    validate_schema(build.read_manifest(), "derived_build_manifest.schema.json", args.repository)
    if not build.verify_raw_unchanged():
        raise RuntimeError("raw collection changed during canonicalization")
    print(json.dumps({"accepted": accepted, "rejected": rejected, "build": str(build.path)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
