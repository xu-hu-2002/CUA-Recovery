from __future__ import annotations

import pytest

from derail.phase5.executor import apply_results
from derail.phase5.hazard_records import HazardRecordError


def _pending() -> dict:
    return {"injection_id": "hz-1", "status": "pending", "profile_after": None}


def _result() -> dict:
    return {
        "accepted": True,
        "profile_after": {"x": 1},
        "gold_unique": True,
        "seeder_consistent": True,
        "attempt": {"attempt_id": "a1", "sandbox_id": "sb"},
    }


def test_executor_requires_complete_guest_results() -> None:
    with pytest.raises(HazardRecordError, match="missing guest result"):
        apply_results([_pending()], {})


def test_executor_preserves_terminal_and_records_attempt() -> None:
    first = apply_results([_pending()], {"hz-1": _result()})
    second = apply_results(first, {"hz-1": {"accepted": False, "reason": "ignored"}})
    assert second == first
    assert first[0]["attempt_lineage"][-1]["attempt_id"] == "a1"
