#!/usr/bin/env python3
"""Freeze a deterministic Phase 5 run catalog after the hazard gate passes."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from recovery.phase5.catalog import Budget, select_prefix, validate_hazards_before_freeze


def _rows(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _instruction_id(row: dict[str, Any]) -> str:
    payload = json.dumps(row, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return "instruction-" + hashlib.sha256(payload.encode()).hexdigest()[:16]


def _combinations(
    accepted: list[dict[str, Any]], candidates: list[dict[str, Any]],
    instructions: list[dict[str, Any]], hazards: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    by_task: dict[str, list[dict[str, Any]]] = {}
    for row in instructions:
        if row.get("status") == "accepted":
            by_task.setdefault(str(row["task_id"]), []).append(row)
    combinations = [{
        "task_id": str(row["task_id"]),
        "instruction_id": "accepted-" + str(row["task_id"]),
        "world_variant_id": "mypcbench_caf9c754_bake20260719+base",
        "hazard_ref": None,
        "verifier_ref": f"accepted/{row['task_id']}",
    } for row in accepted]
    hazard_by_id = {str(row["injection_id"]): row for row in hazards}
    for row in hazards:
        source_id = row.get("provenance", {}).get("source_injection_id")
        if source_id:
            hazard_by_id[str(source_id)] = row
    for candidate in candidates:
        if candidate.get("status") != "accepted":
            continue
        task_id = str(candidate["candidate_id"])
        for instruction in by_task.get(task_id, []):
            variants = [None] + [str(hazard) for hazard in candidate.get("hazards", [])]
            for hazard_id in variants:
                hazard = hazard_by_id.get(hazard_id) if hazard_id is not None else None
                if hazard_id is not None and (hazard is None or hazard["status"] != "injected"):
                    continue
                combinations.append({
                    "task_id": task_id,
                    "instruction_id": _instruction_id(instruction),
                    "world_variant_id": hazard["variant_world_id"] if hazard else
                    "mypcbench_caf9c754_bake20260719+base",
                    "hazard_ref": hazard["injection_id"] if hazard else None,
                    "verifier_ref": candidate.get("verifier_bundle_ref"),
                    "task_config_ref": f"task_ir/{task_id}.json",
                })
    return combinations


def freeze(args: argparse.Namespace) -> dict[str, Any]:
    hazard_rows = _rows(args.hazards)
    validate_hazards_before_freeze(hazard_rows)
    accepted = _rows(args.accepted)
    candidates = _rows(args.candidates)
    instructions = _rows(args.instructions)
    combinations = _combinations(accepted, candidates, instructions, hazard_rows)
    selected = select_prefix(combinations, budget=Budget())
    sources = [args.accepted, args.hazards, args.candidates, args.instructions]
    return {
        "schema": "phase5-run-manifest/1.0",
        "selection": {"algorithm": "stable_sha256_prefix", "minimum": 10000, "maximum": 10500},
        "models": ["gpt_5_5", "claude_opus_4_8", "kimi_k3", "qwen3_5_35b_a3b",
                   "evocua_32b", "opencua_72b"],
        "input_sha256": {str(path): _sha256(path) for path in sources},
        "combination_count": len(selected),
        "combinations": selected,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--accepted", type=Path, required=True)
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument("--instructions", type=Path, required=True)
    parser.add_argument("--hazards", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    manifest = freeze(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"schema": manifest["schema"], "combination_count": manifest["combination_count"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
