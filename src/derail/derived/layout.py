"""Filesystem contract for a DERAIL derived build."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Tuple

DEPTH_GRID: Tuple[int, ...] = (0, 5, 10, 15, 20, 25)
BUILD_SCHEMA_VERSION = "0.2.0"


class DerivedBuildError(ValueError):
    """A build violates provenance, layout, or immutability requirements."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_json(value: Any) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def tree_fingerprint(root: Path) -> str:
    root = root.resolve()
    if not root.is_dir():
        raise DerivedBuildError("collection root 不存在或不是目录: %s" % root)
    records = []
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            records.append((str(path.relative_to(root)), "symlink", os.readlink(path)))
        elif path.is_file():
            records.append((str(path.relative_to(root)), path.stat().st_size, sha256_file(path)))
    return sha256_json(records)


def _safe_component(value: str, name: str) -> str:
    if not value or value in {".", ".."} or "/" in value or "\\" in value:
        raise DerivedBuildError("%s 不是安全的路径组件: %r" % (name, value))
    return value


def atomic_write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=".%s." % path.name, dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except OSError:
            pass
        raise


def atomic_write_jsonl(path: Path, records: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=".%s." % path.name, dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            for record in records:
                handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except OSError:
            pass
        raise


@dataclass(frozen=True)
class DerivedBuild:
    root: Path
    build_id: str
    collection_root: Path
    collection_id: str

    @property
    def path(self) -> Path:
        return self.root / _safe_component(self.build_id, "build_id")

    @property
    def manifest_path(self) -> Path:
        return self.path / "build_manifest.json"

    def initialize(
        self,
        *,
        collection_manifest_path: Path,
        git_commit: str,
        builder_config: Mapping[str, Any],
        created_at: str,
        taxonomy: Mapping[str, Any],
    ) -> Dict[str, Any]:
        if self.path.exists() and any(self.path.iterdir()):
            raise DerivedBuildError("derived build 已存在且非空: %s" % self.path)
        collection_root = self.collection_root.resolve()
        build_path = self.path.resolve()
        if build_path == collection_root or collection_root in build_path.parents:
            raise DerivedBuildError("derived build 不能位于 raw collection 内")
        manifest_path = collection_manifest_path.resolve()
        if collection_root not in manifest_path.parents:
            raise DerivedBuildError("collection manifest 必须位于 collection root 内")
        if not git_commit.strip():
            raise DerivedBuildError("git_commit 不能为空")

        raw_tree_sha256 = tree_fingerprint(collection_root)
        manifest: Dict[str, Any] = {
            "build_id": self.build_id,
            "schema_version": BUILD_SCHEMA_VERSION,
            "status": "initialized",
            "created_at": created_at,
            "depths": list(DEPTH_GRID),
            "raw_immutable": True,
            "source_collection": {
                "collection_id": self.collection_id,
                "uri": str(collection_root),
                "manifest_uri": str(manifest_path),
                "manifest_sha256": sha256_file(manifest_path),
                "tree_sha256_at_init": raw_tree_sha256,
            },
            "git_commit": git_commit,
            "builder_config": dict(builder_config),
            "builder_config_sha256": sha256_json(builder_config),
            "taxonomy": dict(taxonomy),
            "counts": {
                "canonical_trajectories": 0,
                "rejected_trajectories": 0,
                "candidate_cases": 0,
                "accepted_cases": 0,
                "rejected_cases": 0,
            },
        }
        for directory in (
            "canonical",
            "annotations",
            "repairs",
            "replay",
            "cases",
            "evaluations",
            "reports",
        ):
            (self.path / directory).mkdir(parents=True, exist_ok=True)
        atomic_write_json(self.manifest_path, manifest)
        return manifest

    def read_manifest(self) -> Dict[str, Any]:
        with self.manifest_path.open(encoding="utf-8") as handle:
            value = json.load(handle)
        if tuple(value.get("depths", ())) != DEPTH_GRID:
            raise DerivedBuildError("build depth grid 必须固定为 %s" % (DEPTH_GRID,))
        return value

    def verify_raw_unchanged(self) -> bool:
        manifest = self.read_manifest()
        expected = manifest["source_collection"]["tree_sha256_at_init"]
        return tree_fingerprint(self.collection_root) == expected

    def artifact_path(self, layer: str, artifact_id: str, filename: str) -> Path:
        allowed = {"canonical", "annotations", "repairs", "replay", "cases", "evaluations"}
        if layer not in allowed:
            raise DerivedBuildError("未知 derived layer: %s" % layer)
        return self.path / layer / _safe_component(artifact_id, "artifact_id") / filename

    def update_manifest(self, updates: Mapping[str, Any]) -> Dict[str, Any]:
        manifest = self.read_manifest()
        manifest.update(dict(updates))
        atomic_write_json(self.manifest_path, manifest)
        return manifest
