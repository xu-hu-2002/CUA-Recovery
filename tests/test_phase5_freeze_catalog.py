from __future__ import annotations

import json
from argparse import Namespace
from pathlib import Path

import pytest

from scripts.phase5.freeze_catalog import freeze
from scripts.phase5.freeze_catalog import _combinations
from derail.phase5.hazard_records import HazardRecordError


def _write(path: Path, rows: list[dict]) -> Path:
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")
    return path


def test_freeze_refuses_pending_hazards(tmp_path: Path) -> None:
    args = Namespace(
        accepted=_write(tmp_path / "accepted.jsonl", []),
        candidates=_write(tmp_path / "candidates.jsonl", []),
        instructions=_write(tmp_path / "instructions.jsonl", []),
        hazards=_write(tmp_path / "hazards.jsonl", [{"injection_id": "hz", "status": "pending"}]),
        output=tmp_path / "manifest.json",
    )
    with pytest.raises(HazardRecordError, match="non-terminal"):
        freeze(args)


def test_combinations_resolve_option_a_source_hazard_id() -> None:
    candidate = {"candidate_id": "task-1", "status": "accepted", "hazards": ["hz-old"],
                 "verifier_bundle_ref": "verifiers/task-1.json"}
    instruction = {"task_id": "task-1", "status": "accepted", "instruction": "do it"}
    hazard = {"injection_id": "hz-new", "variant_world_id": "world+hz-new",
              "status": "injected", "provenance": {"source_injection_id": "hz-old"}}
    rows = _combinations([], [candidate], [instruction], [hazard])
    variant = next(row for row in rows if row["hazard_ref"] is not None)
    assert variant["hazard_ref"] == "hz-new"
    assert variant["world_variant_id"] == "world+hz-new"
