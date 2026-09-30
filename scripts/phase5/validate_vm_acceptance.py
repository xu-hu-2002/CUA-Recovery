#!/usr/bin/env python3
"""Validate a collected Phase 5 VM acceptance report and fail closed."""

from __future__ import annotations

import argparse
import json
from collections.abc import Iterable
from pathlib import Path
from typing import Optional

REQUIRED_COVERAGE = {"gui", "bash", "cross_app", "reset"}


def trajectory_errors(record: dict, max_overhead: float) -> list[str]:
    errors = []
    trajectory_id = record.get("trajectory_id", "<missing>")
    actions = record.get("actions", [])
    if not actions:
        return [f"{trajectory_id}: no actions"]
    if record.get("reset_cursor") != -1:
        errors.append(f"{trajectory_id}: reset cursor is not -1")
    if record.get("direct_digest") != record.get("replay_digest"):
        errors.append(f"{trajectory_id}: replay digest mismatch")
    for expected, action in enumerate(actions):
        actual = action.get("action_index")
        if actual != expected:
            errors.append(f"{trajectory_id}: action index {actual!r}, expected {expected}")
        if not action.get("cursor_written_before_dispatch"):
            errors.append(f"{trajectory_id}: cursor not written before action {expected}")
        changed = action.get("changed", False)
        if changed and not action.get("changelog_attributed", False):
            errors.append(f"{trajectory_id}: changelog not attributed at action {expected}")
        lag = action.get("observation_lag_actions")
        if lag is not None and not 0 <= int(lag) <= 1:
            errors.append(f"{trajectory_id}: observation lag {lag} at action {expected}")
        if not action.get("screenshot_sha256"):
            errors.append(f"{trajectory_id}: screenshot missing at action {expected}")
        if int(action.get("api_observation_count", 0)) < 1:
            errors.append(f"{trajectory_id}: page observation missing at action {expected}")
        overhead = float(action.get("tracing_overhead_seconds", 0.0))
        if overhead > max_overhead:
            errors.append(f"{trajectory_id}: overhead {overhead:.3f}s at action {expected}")
    return errors


def validate(report: dict) -> list[str]:
    errors = []
    records = report.get("trajectories", [])
    if len(records) != 10:
        errors.append(f"expected 10 trajectories, found {len(records)}")
    coverage = set(report.get("coverage", []))
    missing = sorted(REQUIRED_COVERAGE - coverage)
    if missing:
        errors.append("missing coverage: " + ", ".join(missing))
    for record in records:
        errors.extend(trajectory_errors(record, 1.0))
    ids = [record.get("trajectory_id") for record in records]
    if len(ids) != len(set(ids)):
        errors.append("trajectory ids are not unique")
    return errors


def main(argv: Optional[Iterable[str]] = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("report", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    report = json.loads(args.report.read_text())
    errors = validate(report)
    verdict = {
        "schema": "vm-acceptance-verdict/1.0",
        "accepted": not errors,
        "trajectory_count": len(report.get("trajectories", [])),
        "errors": errors,
    }
    rendered = json.dumps(verdict, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered)
    print(rendered, end="")
    return 0 if not errors else 1


if __name__ == "__main__":
    raise SystemExit(main())
