from __future__ import annotations

from typing import Any, Dict, Mapping, Optional

GOLD_ACTION_VERSION = "gold-action/1.0"

_KIND_BY_DETECTOR = {
    "parameter": "redo_node_with_gold_values",
    "omission": "execute_skipped_node",
    "state": "compensate_writes_then_redo_node",
}


def gold_action(
    analysis: Mapping[str, Any],
    gold: Mapping[str, Any],
    task_ir: Optional[Mapping[str, Any]] = None,
) -> Optional[Dict[str, Any]]:
    node_id = analysis.get("root_cause_node_id")
    if not node_id:
        return None
    detector = str(analysis.get("root_cause_detector") or "parameter")
    node = next((n for n in (task_ir or {}).get("nodes", ()) if str(n["node_id"]) == node_id), {})
    values = {
        str(v["name"]): v.get("value")
        for v in gold.get("values", ())
        if str(v["node_id"]) == node_id
    }
    writes = [
        {k: w.get(k) for k in ("table", "column", "entity_ref", "value", "reversibility_class")}
        for w in gold.get("writes_gold", ())
        if str(w.get("node_id")) == node_id
    ]
    order = [str(n) for n in gold.get("node_order", ())]
    after = order[order.index(node_id) + 1 :] if node_id in order else []
    detail = (analysis.get("provenance") or {}).get("root_detail") or {}
    return {
        "schema_version": GOLD_ACTION_VERSION,
        "kind": _KIND_BY_DETECTOR.get(detector, "redo_node_with_gold_values"),
        "node_id": node_id,
        "app": node.get("app"),
        "op": node.get("op"),
        "semantic_goal": node.get("semantic_goal"),
        "parameters": values,
        "gold_writes": writes,
        "wrong_value": detail.get("wrong_value", detail.get("actual")),
        "correct_value": detail.get("correct_value", detail.get("expected")),
        "continue_with": after,
        "source": "programme",
    }

