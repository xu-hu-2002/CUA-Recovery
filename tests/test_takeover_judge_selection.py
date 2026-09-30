import json
from pathlib import Path

import pytest

from scripts.judge.takeover_judge_selection import (
    completed_task_dirs, excluded_task_ids, load_exclusions, resolve_annotation,
)


def write_manifest(root: Path, records: list[dict]) -> None:
    (root / "judge_exclusions.json").write_text(json.dumps({
        "schema_version": 1,
        "exclusions": records,
    }))


def episode(cell: Path, task_id: str, response: object) -> Path:
    task = cell / task_id
    task.mkdir(parents=True)
    (task / "result.txt").write_text("1.0\n")
    (task / "traj.jsonl").write_text(json.dumps({"response": response}) + "\n")
    (task / "rubric_bundle.json").write_text("{}\n")
    return task


def test_shared_selection_excludes_only_manifest_task(tmp_path):
    cell = tmp_path / "depth_20" / "unaware"
    kept = episode(cell, "legal-short", "I should stop")
    episode(cell, "dropped", None)
    write_manifest(tmp_path, [{
        "depth": 20, "condition": "unaware", "task_id": "dropped",
        "classification": "PROTOCOL_EXCLUSION", "reason": "no_model_output",
    }])
    excluded = excluded_task_ids(tmp_path, 20, "unaware")
    assert completed_task_dirs(cell, excluded, "traj.jsonl") == [kept]
    assert completed_task_dirs(cell, excluded, "rubric_bundle.json") == [kept]


def test_manifest_rejects_duplicate_and_non_protocol_records(tmp_path):
    record = {"depth": 20, "condition": "unaware", "task_id": "x",
              "classification": "infra_retry", "reason": "no_model_output"}
    write_manifest(tmp_path, [record])
    with pytest.raises(ValueError, match="invalid judge exclusion record"):
        load_exclusions(tmp_path)
    record["classification"] = "PROTOCOL_EXCLUSION"
    write_manifest(tmp_path, [record, record])
    with pytest.raises(ValueError, match="duplicate judge exclusion"):
        load_exclusions(tmp_path)


def test_annotation_fallback_is_hash_bound(tmp_path, monkeypatch):
    label = tmp_path / "label.json"
    label.write_text('{"error_types": ["x"]}\n')
    digest = __import__("hashlib").sha256(label.read_bytes()).hexdigest()
    assert resolve_annotation(str(label), digest) == label.resolve()
    with pytest.raises(ValueError, match="SHA mismatch"):
        resolve_annotation(str(label), "0" * 64)
