"""Verify Phase 5 environment writes from the immutable changelog ledger."""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping


def _expected_entities(gold: Mapping[str, Any]) -> dict[tuple[str, str], dict[str, Any]]:
    entities: dict[tuple[str, str], dict[str, Any]] = defaultdict(dict)
    for write in gold.get("writes_gold", ()):
        if write.get("volatile") or write.get("table") == "files.documents":
            continue
        key = (str(write["table"]), str(write["entity_ref"]))
        entities[key][str(write["column"])] = write.get("value")
    return dict(entities)


def _actual_rows(steps: Iterable[Mapping[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    rows: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for step in steps:
        for delta in step.get("delta", ()):
            payload = delta.get("new_json")
            if not payload:
                continue
            row = json.loads(payload) if isinstance(payload, str) else dict(payload)
            app = Path(str(delta.get("db", ""))).stem
            rows[f"{app}.{delta['tbl']}"].append(row)
    return dict(rows)


def _matches(expected: Mapping[str, Any], actual: Mapping[str, Any]) -> bool:
    return all(actual.get(column) == value for column, value in expected.items())


def verify_changelog(gold: Mapping[str, Any], steps: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    expected = _expected_entities(gold)
    actual = _actual_rows(steps)
    checks = []
    for (table, entity_ref), columns in sorted(expected.items()):
        matched = any(_matches(columns, row) for row in actual.get(table, ()))
        checks.append({"table": table, "entity_ref": entity_ref, "matched": matched})
    missing = [check for check in checks if not check["matched"]]
    return {
        "schema": "phase5-verifier-result/1.0",
        "passed": bool(checks) and not missing,
        "kind": "changelog-gold-write-match",
        "checks": checks,
        "missing": missing,
    }
