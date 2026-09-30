from __future__ import annotations

import pytest

from derail.phase5.hazard_records import HazardRecordError, assert_manifest_safe, terminalize


def _record() -> dict:
    return {"injection_id": "hz-1", "status": "pending", "profile_after": {"x": 1}}


def test_terminalize_injected_requires_verified_invariants() -> None:
    record = _record() | {"gold_unique": True, "seeder_consistent": True}
    result = terminalize(record, accepted=True)
    assert result["status"] == "injected"
    assert result["attempt_lineage"][-1]["status"] == "injected"


def test_terminalize_rejected_requires_reason_and_is_idempotency_guarded() -> None:
    result = terminalize(_record(), accepted=False, reason="PATCH_FAILED")
    assert result["status"] == "rejected"
    with pytest.raises(HazardRecordError):
        terminalize(result, accepted=False, reason="SECOND_ATTEMPT")


def test_manifest_rejects_pending_and_duplicate_records() -> None:
    with pytest.raises(HazardRecordError, match="non-terminal"):
        assert_manifest_safe([_record()])
    with pytest.raises(HazardRecordError, match="duplicate"):
        assert_manifest_safe([{"injection_id": "x", "status": "rejected"}] * 2)
