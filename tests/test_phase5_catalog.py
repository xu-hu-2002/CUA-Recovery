from __future__ import annotations

import pytest

from derail.phase5.catalog import Budget, projected_total, select_prefix, stable_key, validate_hazards_before_freeze
from derail.phase5.hazard_records import HazardRecordError


def test_budget_uses_actual_combination_count() -> None:
    budget = Budget()
    assert budget.trajectories_per_combination == pytest.approx(12.3)
    assert projected_total(100, budget) == pytest.approx(1230)


def test_selection_is_stable_and_reaches_range() -> None:
    rows = [{"task_id": f"task-{i}", "variant": 0} for i in range(900)]
    first = select_prefix(rows)
    second = select_prefix(list(reversed(rows)))
    assert first == second
    assert 10_000 <= projected_total(len(first)) <= 10_500


def test_pending_hazard_blocks_freeze() -> None:
    with pytest.raises(HazardRecordError, match="non-terminal"):
        validate_hazards_before_freeze([{"injection_id": "hz-1", "status": "pending"}])


def test_stable_key_is_canonical() -> None:
    assert stable_key({"b": 2, "a": 1}) == stable_key({"a": 1, "b": 2})
