"""Programme-derived correct action at a takeover point (D-046).

Every case gets a ground-truth action even when no verified recovery trajectory exists yet:
the failure analysis names the root-cause node and detector, the gold lineage holds the
correct produced values and the correct written cells, and the node order gives the gold
path to continue on.  The action is expressed at node level (app, op, parameters, cells),
the granularity of the canonical action space; a verified recovery trajectory (base model
under hints, or the teacher) is its realisation in concrete GUI/CLI actions.
"""

from __future__ import annotations

from typing import Any, Dict, List, Mapping, Optional

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
    """``gold-action/1.0`` for the root cause of ``analysis``; None without a root node."""

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


def attach_gold_actions(
    records: List[Dict[str, Any]],
    analysis: Mapping[str, Any],
    gold: Mapping[str, Any],
    task_ir: Optional[Mapping[str, Any]] = None,
) -> None:
    action = gold_action(analysis, gold, task_ir)
    if action is None:
        return
    for record in records:
        record["gold_action"] = action
