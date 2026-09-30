from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from derail.phase5.workload import FrozenPhase5Manifest, WorkloadError

REPOSITORY = Path(__file__).resolve().parents[1]
MANIFEST = REPOSITORY / "artifacts/phase5/run-manifest-option-a-20260912.json"
pytestmark = pytest.mark.skipif(
    not MANIFEST.is_file(), reason="frozen Phase 5 manifest is run data, not shipped with the code"
)
MANIFEST_SHA = "228c209ed2d892c34adc5539884ab7f591de0dd8a27a38a417f2c49642887126"


def frozen() -> FrozenPhase5Manifest:
    return FrozenPhase5Manifest(REPOSITORY, MANIFEST, MANIFEST_SHA)


def test_resolves_frozen_gpt55_workload() -> None:
    workload = frozen().resolve("gpt_5_5", 1, 0)
    assert workload["combination_count"] == 814
    assert workload["source_manifest_sha256"] == MANIFEST_SHA
    row = workload["combinations"][0]
    assert row["instruction"]
    assert len(row["task_config_sha256"]) == 64
    assert len(row["verifier_sha256"]) == 64
    assert len(row["gold_lineage_sha256"]) == 64


def test_four_shards_are_complete_and_disjoint() -> None:
    shards = [frozen().resolve("opencua_72b", 4, index) for index in range(4)]
    keys = [
        (row["task_id"], row["world_variant_id"], row["instruction_id"])
        for shard in shards
        for row in shard["combinations"]
    ]
    assert len(keys) == 814
    assert len(set(keys)) == 814


def test_rejects_manifest_mutation(tmp_path: Path) -> None:
    mutated = tmp_path / "manifest.json"
    mutated.write_bytes(MANIFEST.read_bytes() + b"\n")
    with pytest.raises(WorkloadError, match="manifest SHA mismatch"):
        FrozenPhase5Manifest(REPOSITORY, mutated, MANIFEST_SHA).load()


def test_rejects_non_frozen_model() -> None:
    with pytest.raises(WorkloadError, match="model is not frozen"):
        frozen().resolve("replacement_model", 1, 0)


def test_workload_is_deterministic() -> None:
    first = frozen().resolve("gpt_5_5", 4, 2)
    second = frozen().resolve("gpt_5_5", 4, 2)
    encode = lambda value: json.dumps(value, sort_keys=True).encode()
    assert hashlib.sha256(encode(first)).digest() == hashlib.sha256(encode(second)).digest()


def test_smoke_selection_is_frozen_base_world_without_file_writes() -> None:
    config = json.loads((REPOSITORY / "configs/phase5/smoke_v1.json").read_text())
    workload = frozen().resolve("gpt_5_5", 1, 0)
    by_id = {row["combination_id"]: row for row in workload["combinations"]}
    for combination_id in config["consecutive_three"]:
        row = by_id[combination_id]
        task = json.loads(Path(row["task_config_path"]).read_text())
        assert row["hazard_ref"] is None
        assert all(node.get("app") != "files" for node in task["nodes"])
        assert any(node.get("writes") for node in task["nodes"])
