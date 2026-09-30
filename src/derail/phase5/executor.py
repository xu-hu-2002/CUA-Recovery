"""Idempotent application of guest hazard-verification results."""

from __future__ import annotations

from typing import Any, Mapping

from .hazard_records import TERMINAL_STATUSES, HazardRecordError, terminalize


def apply_results(
    records: list[Mapping[str, Any]], results: Mapping[str, Mapping[str, Any]]
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    seen: set[str] = set()
    for record in records:
        injection_id = str(record.get("injection_id", ""))
        if not injection_id or injection_id in seen:
            raise HazardRecordError(f"duplicate or missing injection_id: {injection_id!r}")
        seen.add(injection_id)
        if record.get("status") in TERMINAL_STATUSES:
            output.append(dict(record))
            continue
        result = results.get(injection_id)
        if result is None:
            raise HazardRecordError(f"missing guest result: {injection_id}")
        enriched = dict(record)
        enriched.update({
            "profile_after": result.get("profile_after"),
            "gold_unique": bool(result.get("gold_unique")),
            "seeder_consistent": bool(result.get("seeder_consistent")),
        })
        output.append(terminalize(
            enriched,
            accepted=bool(result.get("accepted")),
            reason=result.get("reason"),
            attempt=result.get("attempt"),
        ))
    unknown = set(results) - seen
    if unknown:
        raise HazardRecordError(f"guest results reference unknown hazards: {sorted(unknown)!r}")
    return output
