from __future__ import annotations

import json
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from recovery.world.volatile import VolatileColumns

REPAIR_VERSION = "prefix-repair-v1/1.0"


def _last_page(step: Mapping[str, Any]) -> Optional[str]:
    pages = step.get("pages") or []
    if not pages:
        return None
    last = pages[-1]
    return "%s%s" % (last.get("app", ""), last.get("route", ""))


def _row_state(
    row: Mapping[str, Any], which: str, volatile: Optional[VolatileColumns]
) -> Optional[Dict[str, Any]]:
    payload = row.get(which)
    if not payload:
        return None
    cells = json.loads(payload)
    table = "%s.%s" % (row["db"], row["tbl"])
    if volatile is not None:
        cells = {k: v for k, v in cells.items() if not volatile.is_volatile(table, k)}
    return cells


def net_change(
    steps: Sequence[Mapping[str, Any]], volatile: Optional[VolatileColumns] = None
) -> Dict[Tuple[str, str, int], Tuple[Any, Any]]:
    first: Dict[Tuple[str, str, int], Any] = {}
    last: Dict[Tuple[str, str, int], Any] = {}
    for step in steps:
        for row in sorted(step.get("delta", ()), key=lambda r: (r["db"], int(r["seq"]))):
            key = (row["db"], row["tbl"], int(row["rowid"]))
            before = _row_state(row, "old_json", volatile)
            after = _row_state(row, "new_json", volatile)
            if key not in first:
                first[key] = before if row["op"] != "INSERT" else None
            last[key] = after if row["op"] != "DELETE" else None
    return {key: (first[key], last[key]) for key in first if first[key] != last[key]}


def _typed_values(steps: Sequence[Mapping[str, Any]]) -> List[str]:
    out = []
    for step in steps:
        for param in step.get("params", ()):
            value = param.get("value")
            if isinstance(value, str) and len(value.strip()) >= 3:
                out.append(value.strip().lower())
    return out


def _feeds_later_change(
    window: Sequence[Mapping[str, Any]], later: Sequence[Mapping[str, Any]]
) -> bool:
    typed = _typed_values(window)
    if not typed:
        return False
    for step in later:
        for row in step.get("delta", ()):
            payload = (row.get("new_json") or "").lower()
            if any(value in payload for value in typed):
                return True
    return False


def neutral_segments(
    trace: Mapping[str, Any],
    root_cause_action_index: int,
    volatile: Optional[VolatileColumns] = None,
) -> List[Dict[str, Any]]:
    steps = [s for s in trace["steps"] if int(s["action_index"]) < root_cause_action_index]
    pages_before: Dict[int, Optional[str]] = {}
    previous = None
    for step in steps:
        pages_before[int(step["action_index"])] = previous
        previous = _last_page(step) or previous
    segments: List[Dict[str, Any]] = []
    i = 0
    while i < len(steps):
        best = None
        for j in range(len(steps) - 1, i - 1, -1):
            window = steps[i : j + 1]
            if net_change(window, volatile):
                continue
            later = [
                s for s in trace["steps"] if int(s["action_index"]) > int(steps[j]["action_index"])
            ]
            if _feeds_later_change(window, later):
                continue
            page_before = pages_before[int(steps[i]["action_index"])]
            page_after = _last_page(steps[j]) or page_before
            if page_after == page_before:
                best = (i, j, page_before, page_after)
                break
        if best is None:
            i += 1
            continue
        a, b, page_before, page_after = best
        segments.append(
            {
                "start_action_index": int(steps[a]["action_index"]),
                "end_action_index": int(steps[b]["action_index"]),
                "net_delta_empty": True,
                "page_before": page_before,
                "page_after": page_after,
                "evidence": [
                    {"kind": "changelog", "detail": "no net change over %d actions" % (b - a + 1)}
                ],
            }
        )
        i = b + 1
    return segments


def non_neutral_pre_root(
    trace: Mapping[str, Any],
    root_cause_action_index: int,
    writes_gold: Sequence[Mapping[str, Any]],
    volatile: Optional[VolatileColumns] = None,
) -> List[Dict[str, Any]]:
    steps = [s for s in trace["steps"] if int(s["action_index"]) < root_cause_action_index]
    gold = {("%s" % w["table"], w["entity"]) for w in writes_gold}
    out = []
    for key, (before, after) in net_change(steps, volatile).items():
        db, tbl, rowid = key
        if ("%s.%s" % (db, tbl), "%s:%s" % (tbl, rowid)) in gold:
            continue
        touched = [
            int(s["action_index"])
            for s in steps
            if any((r["db"], r["tbl"], int(r["rowid"])) == key for r in s.get("delta", ()))
        ]
        out.append(
            {
                "start_action_index": min(touched),
                "end_action_index": max(touched),
                "net_delta_empty": False,
                "page_before": None,
                "page_after": None,
                "evidence": [
                    {
                        "kind": "changelog",
                        "detail": "%s.%s:%s changed before the root cause and is not a gold write"
                        % (db, tbl, rowid),
                    }
                ],
            }
        )
    return out


def repair_prefix(
    trace: Mapping[str, Any],
    root_cause_action_index: int,
    writes_gold: Sequence[Mapping[str, Any]] = (),
    volatile: Optional[VolatileColumns] = None,
) -> Dict[str, Any]:
    removed = neutral_segments(trace, root_cause_action_index, volatile)
    non_neutral = non_neutral_pre_root(trace, root_cause_action_index, writes_gold, volatile)
    removed_indices = {
        i for seg in removed for i in range(seg["start_action_index"], seg["end_action_index"] + 1)
    }
    prefix = [
        int(s["action_index"])
        for s in trace["steps"]
        if int(s["action_index"]) < root_cause_action_index
        and int(s["action_index"]) not in removed_indices
    ]
    if non_neutral:
        status = "PREFIX_UNREPAIRABLE"
    elif removed:
        status = "repaired"
    else:
        status = "unchanged"
    return {
        "schema_version": "prefix-repair/1.0",
        "rollout_id": str(trace["rollout_id"]),
        "task_id": str(trace["task_id"]),
        "root_cause_action_index": int(root_cause_action_index),
        "removed_segments": removed,
        "non_neutral_segments": non_neutral,
        "repaired_prefix_action_indices": prefix,
        "status": status,
        "repair_version": REPAIR_VERSION,
        "provenance": {
            "original_prefix_length": sum(
                1 for s in trace["steps"] if int(s["action_index"]) < root_cause_action_index
            )
        },
    }
