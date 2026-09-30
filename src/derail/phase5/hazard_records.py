"""Validation and terminalization rules for Phase 5 hazard records."""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Mapping


TERMINAL_STATUSES = frozenset(("injected", "rejected"))


class HazardRecordError(ValueError):
    """Raised when a hazard record is unsafe for a frozen manifest."""


def terminalize(
    record: Mapping[str, Any], *, accepted: bool, reason: str | None = None,
    attempt: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Return an immutable terminal record without overwriting prior attempts."""

    if record.get("status") in TERMINAL_STATUSES:
        raise HazardRecordError("terminal hazard records are immutable")
    if not accepted and not reason:
        raise HazardRecordError("rejected hazards require a reason")
    result = deepcopy(dict(record))
    result["schema_version"] = "phase5-hazard/1.0"
    result["status"] = "injected" if accepted else "rejected"
    result["rejection_reason"] = None if accepted else reason
    attempts = list(result.get("attempt_lineage") or [])
    attempt_record = {"status": result["status"], "reason": result["rejection_reason"]}
    if attempt:
        attempt_record.update(deepcopy(dict(attempt)))
    attempts.append(attempt_record)
    result["attempt_lineage"] = attempts
    if result["status"] == "injected" and not result.get("gold_unique"):
        raise HazardRecordError("injected hazards require gold_unique=true")
    if result["status"] == "injected" and not result.get("seeder_consistent"):
        raise HazardRecordError("injected hazards require seeder_consistent=true")
    if result.get("profile_after") is None:
        raise HazardRecordError("terminal hazards require profile_after")
    return result


def assert_manifest_safe(records: list[Mapping[str, Any]]) -> None:
    """Reject duplicate IDs and every non-terminal record before rollout."""

    ids: set[str] = set()
    for record in records:
        injection_id = str(record.get("injection_id", ""))
        if not injection_id or injection_id in ids:
            raise HazardRecordError(f"duplicate or missing injection_id: {injection_id!r}")
        ids.add(injection_id)
        if record.get("status") not in TERMINAL_STATUSES:
            raise HazardRecordError(f"non-terminal hazard: {injection_id}")
