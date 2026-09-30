"""Resolve an immutable Phase 5 run manifest into executable shard inputs."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


class WorkloadError(ValueError):
    """The frozen workload contract is invalid or has been mutated."""


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def _instruction_id(row: dict[str, Any]) -> str:
    payload = json.dumps(row, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return "instruction-" + hashlib.sha256(payload.encode()).hexdigest()[:16]


def _combination_id(row: dict[str, Any]) -> str:
    fields = {key: row[key] for key in ("task_id", "world_variant_id", "instruction_id")}
    payload = json.dumps(fields, sort_keys=True, separators=(",", ":"))
    return "combo-" + hashlib.sha256(payload.encode()).hexdigest()[:16]


@dataclass(frozen=True)
class FrozenPhase5Manifest:
    repository: Path
    manifest_path: Path
    expected_sha256: str

    def load(self) -> dict[str, Any]:
        actual = sha256(self.manifest_path)
        if actual != self.expected_sha256:
            raise WorkloadError(f"manifest SHA mismatch: expected {self.expected_sha256}, got {actual}")
        manifest = _read_json(self.manifest_path)
        self._validate_manifest(manifest)
        self._validate_inputs(manifest["input_sha256"])
        return manifest

    def resolve(self, model: str, shard_count: int, shard_index: int) -> dict[str, Any]:
        manifest = self.load()
        self._validate_selection(manifest, model, shard_count, shard_index)
        instructions = self._instructions(manifest)
        selected = self._shard(manifest["combinations"], shard_count, shard_index)
        rows = [self._resolve_row(row, instructions) for row in selected]
        return self._workload(manifest, model, shard_count, shard_index, rows)

    def _validate_manifest(self, manifest: dict[str, Any]) -> None:
        combinations = manifest.get("combinations") or []
        if manifest.get("schema") != "phase5-run-manifest/1.0":
            raise WorkloadError("unsupported Phase 5 manifest schema")
        if manifest.get("combination_count") != len(combinations):
            raise WorkloadError("combination_count does not match combinations")
        keys = [self._combination_key(row) for row in combinations]
        if len(keys) != len(set(keys)):
            raise WorkloadError("duplicate task/world/instruction combination")

    def _validate_inputs(self, inputs: dict[str, str]) -> None:
        for relative, expected in inputs.items():
            path = self.repository / relative
            if not path.is_file():
                raise WorkloadError(f"frozen input is missing: {relative}")
            actual = sha256(path)
            if actual != expected:
                raise WorkloadError(f"frozen input SHA mismatch: {relative}")

    @staticmethod
    def _validate_selection(
        manifest: dict[str, Any], model: str, shard_count: int, shard_index: int
    ) -> None:
        if model not in manifest["models"]:
            raise WorkloadError(f"model is not frozen in manifest: {model}")
        if shard_count < 1 or shard_index not in range(shard_count):
            raise WorkloadError("shard_index must be in [0, shard_count)")

    def _instructions(self, manifest: dict[str, Any]) -> dict[str, dict[str, Any]]:
        relative = next(key for key in manifest["input_sha256"] if key.endswith("instructions.jsonl"))
        rows = _read_jsonl(self.repository / relative)
        return {_instruction_id(row): row for row in rows}

    @staticmethod
    def _shard(rows: Iterable[dict[str, Any]], count: int, index: int) -> list[dict[str, Any]]:
        return [row for position, row in enumerate(rows) if position % count == index]

    def _resolve_row(
        self, row: dict[str, Any], instructions: dict[str, dict[str, Any]]
    ) -> dict[str, Any]:
        if row["instruction_id"].startswith("accepted-"):
            return self._resolve_accepted(row)
        instruction = instructions.get(row["instruction_id"])
        if instruction is None:
            raise WorkloadError(f"instruction is missing: {row['instruction_id']}")
        task_path = self.repository / "data/synthesis/generation/final_v1" / row["task_config_ref"]
        verifier_path = self.repository / "data/synthesis/generation/final_v1" / row["verifier_ref"]
        gold_path = self.repository / "data/synthesis/generation/final_v1/gold_lineage" / (
            row["task_id"] + ".gold_lineage.json"
        )
        return {
            **row,
            "combination_id": _combination_id(row),
            "instruction": instruction["instruction"],
            "task_config_path": str(task_path),
            "task_config_sha256": sha256(task_path),
            "verifier_path": str(verifier_path),
            "verifier_sha256": sha256(verifier_path),
            "gold_lineage_path": str(gold_path),
            "gold_lineage_sha256": sha256(gold_path),
        }

    def _resolve_accepted(self, row: dict[str, Any]) -> dict[str, Any]:
        task_path = self.repository / "data/synthesis/task_ir_v1/accepted/task_ir" / (
            row["task_id"] + ".json"
        )
        task = _read_json(task_path)
        gold_path = self.repository / "data/synthesis/task_ir_v1/accepted/gold_lineage" / (
            row["task_id"] + ".gold_lineage.json"
        )
        digest = sha256(task_path)
        return {
            **row,
            "combination_id": _combination_id(row),
            "instruction": task["instruction"],
            "task_config_path": str(task_path),
            "task_config_sha256": digest,
            "verifier_path": str(task_path),
            "verifier_sha256": digest,
            "gold_lineage_path": str(gold_path),
            "gold_lineage_sha256": sha256(gold_path),
        }

    def _workload(
        self, manifest: dict[str, Any], model: str, count: int, index: int, rows: list[dict[str, Any]]
    ) -> dict[str, Any]:
        return {
            "schema": "phase5-rollout-workload/1.0",
            "source_manifest": str(self.manifest_path),
            "source_manifest_sha256": self.expected_sha256,
            "model": model,
            "shard_count": count,
            "shard_index": index,
            "combination_count": len(rows),
            "combinations": rows,
        }

    @staticmethod
    def _combination_key(row: dict[str, Any]) -> tuple[str, str, str]:
        return row["task_id"], row["world_variant_id"], row["instruction_id"]
