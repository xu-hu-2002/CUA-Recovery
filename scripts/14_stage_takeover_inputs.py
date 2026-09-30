#!/usr/bin/env python3
"""Stage portable, hash-audited inputs for Nebula takeover workers."""

from __future__ import annotations

import argparse
import json
import shutil
import urllib.parse
from pathlib import Path
from typing import Mapping

from derail.derived.layout import atomic_write_json, sha256_file
from derail.rollout.state_probe import EnvironmentHooks
from derail.takeover.protocol import DEFAULT_CONFIG_PATH, load_takeover_config


def read_object(path: Path) -> Mapping[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def copy_file(source: Path, destination: Path) -> dict[str, object]:
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)
    return {
        "source": str(source),
        "relative_path": str(destination),
        "sha256": sha256_file(destination),
        "size_bytes": destination.stat().st_size,
    }


def repository_destination(source: Path, repository: Path, output: Path) -> Path:
    try:
        relative = source.resolve().relative_to(repository)
    except ValueError:
        relative = Path("external_sources") / sha256_file(source) / source.name
    return output / relative


def source_record_path(uri: str) -> Path:
    value = uri.rsplit("#line=", 1)[0]
    if value.startswith("file://"):
        value = urllib.parse.unquote(urllib.parse.urlparse(value).path)
    return Path(value).expanduser().resolve()


def selected_files(
    config: Mapping[str, object], repository: Path, takeover: Mapping[str, object]
) -> tuple[set[Path], set[Path]]:
    required: set[Path] = set()
    missing: set[Path] = set()
    prefix = takeover["prefix"]
    # Source-side state probes (the replay gate's expected fingerprints).
    environment = Path(str(takeover["environment_config"]))
    sidecar_name = EnvironmentHooks.from_config(environment).probe_file
    for experiment in config["experiments"]:
        selection_uri = experiment.get("selection_manifest")
        if not selection_uri:
            continue
        selection_path = repository / str(selection_uri)
        selection = read_object(selection_path)
        required.add(selection_path)
        build = repository / str(experiment["build_dir"])
        required.add(build / "build_manifest.json")
        for item in selection["included"]:
            for key in ("canonical_trajectory_uri", "normalization_report_uri", "task_config_uri", "annotation_uri", "rubric_review_uri"):
                required.add(Path(str(item[key])).resolve())
            trajectory = Path(str(item["canonical_trajectory_uri"]))
            if not trajectory.is_file():
                missing.add(trajectory.resolve())
                continue
            trajectories = [trajectory]
            if prefix["source"] == "repaired":
                # Same layout the runner resolves: <build>/<repaired_dir>/<id>/<file>.
                repaired_root = Path(str(prefix["repaired_dir"]))
                if not repaired_root.is_absolute():
                    repaired_root = trajectory.resolve().parents[2] / repaired_root
                repaired = repaired_root / trajectory.parent.name / str(prefix["trajectory_filename"])
                (required if repaired.is_file() else missing).add(repaired.resolve())
                trajectories += [repaired] if repaired.is_file() else []
            for path in trajectories:
                for line in path.read_text(encoding="utf-8").splitlines():
                    row = json.loads(line)
                    source = source_record_path(str(row["source_record_uri"]))
                    (required if source.is_file() else missing).add(source)
                    sidecar = source.parent / sidecar_name
                    if sidecar.is_file():
                        required.add(sidecar)
                    if "claude" in str(row.get("source_agent", "")).lower():
                        messages = source.parent / "messages.json"
                        (required if messages.is_file() else missing).add(messages)
    return required, missing


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--base-manifest",
        type=Path,
        help="Previously uploaded bundle manifest used as an OSS-resident baseline.",
    )
    parser.add_argument("--matrix-manifest", type=Path)
    parser.add_argument("--takeover-config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--repository", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    repository = args.repository.resolve()
    output = args.output.resolve()
    if output.exists():
        raise RuntimeError(f"output already exists: {output}")
    config = read_object(args.config)
    takeover = load_takeover_config(args.takeover_config)
    required, missing = selected_files(config, repository, takeover)
    required.add(args.config.resolve())
    required.add(args.takeover_config.resolve())
    required.add(Path(takeover["environment_config"]))
    if args.matrix_manifest:
        required.add(args.matrix_manifest.resolve())
    base = read_object(args.base_manifest) if args.base_manifest else {}
    records_by_path = {
        str(item["relative_path"]): dict(item)
        for item in base.get("files", ())
        if isinstance(item, Mapping) and item.get("relative_path")
    }
    path_mappings = dict(base.get("path_mappings", {}))
    baseline_sources = set(path_mappings)
    missing = {path for path in missing if str(path) not in baseline_sources}
    for source in sorted(required):
        if not source.is_file():
            if str(source) not in baseline_sources:
                missing.add(source)
            continue
        if str(source) in baseline_sources:
            continue
        destination = repository_destination(source, repository, output)
        record = copy_file(source, destination)
        record["relative_path"] = str(destination.relative_to(output))
        records_by_path[record["relative_path"]] = record
        path_mappings[str(source)] = record["relative_path"]
    records = [records_by_path[path] for path in sorted(records_by_path)]
    payload = {
        "schema_version": "1.0",
        "base_manifest": str(args.base_manifest.resolve()) if args.base_manifest else "",
        "config_sha256": sha256_file(args.config),
        "file_count": len(records),
        "total_size_bytes": sum(int(item["size_bytes"]) for item in records),
        "files": records,
        "path_mappings": path_mappings,
        "missing_source_records": [str(path) for path in sorted(missing)],
    }
    atomic_write_json(output / "bundle_manifest.json", payload)
    print(json.dumps({"output": str(output), "files": len(records), "missing": len(missing)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
