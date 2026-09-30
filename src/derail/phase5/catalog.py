"""Deterministic Phase 5 combination selection and budget accounting."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any, Iterable, Mapping

from .hazard_records import assert_manifest_safe


@dataclass(frozen=True)
class Budget:
    rollout_models: int = 6
    takeover_models: int = 4
    failure_rate: float = 0.45
    usable_takeover_depths: float = 3.5

    @property
    def trajectories_per_combination(self) -> float:
        return self.rollout_models + self.failure_rate * self.usable_takeover_depths * self.takeover_models


def stable_key(row: Mapping[str, Any]) -> str:
    payload = json.dumps(row, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def projected_total(count: int, budget: Budget = Budget()) -> float:
    """Project actual selected combinations without an average-variant shortcut."""

    return count * budget.trajectories_per_combination


def select_prefix(
    rows: Iterable[Mapping[str, Any]],
    *,
    minimum: float = 10_000,
    maximum: float = 10_500,
    budget: Budget = Budget(),
) -> list[dict[str, Any]]:
    """Select the first stable-SHA prefix whose projected count is in range."""

    ordered = [dict(row) for row in sorted(rows, key=stable_key)]
    for size in range(1, len(ordered) + 1):
        total = projected_total(size, budget)
        if minimum <= total <= maximum:
            return ordered[:size]
        if total > maximum:
            break
    raise ValueError("input combinations cannot produce a budget in the requested range")


def validate_hazards_before_freeze(records: list[Mapping[str, Any]]) -> None:
    """Require all hazard inputs to be terminal before a run manifest is frozen."""

    assert_manifest_safe(records)
