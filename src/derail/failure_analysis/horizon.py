"""Earliest identifiable point and horizons (execution doc v1.2 sections 2.3, 9.5).

After the root cause at ``t_root`` the earliest identifiable action is the first ``t > t_root``
whose observations show either (a) a gold-written cell with a value that diverges from the
gold state, or (b) a correct source fact that contradicts the value the agent carried (the
reads of the lineage that produced the mistyped gold value).  ``action_horizon`` is the step
difference; ``semantic_horizon`` the dependency-path length between the root node and the
node whose fact was observed; ``horizon_censored`` when nothing exposes the error before the
trace ends.
"""

from __future__ import annotations

from typing import Any, Dict, List, Mapping, Optional, Tuple

from derail.detect.latent_static import _path_length
from derail.failure_analysis.detectors import Candidate, _same
from derail.ir.model import dag_index


def _lineage_source_facts(
    task_ir: Mapping[str, Any], gold: Mapping[str, Any], node_id: str
) -> List[Tuple[str, str, str, Any]]:
    """``(table, column, entity, gold value)`` of the reads feeding ``node_id`` (itself and
    its ancestors), with concrete entities from ``resolved_reads``."""

    dag = dag_index(task_ir)
    nodes = dag.ancestors(node_id) | {node_id}
    facts = []
    for read in gold.get("resolved_reads", ()):
        if read["node_id"] not in nodes or not read.get("entity_set"):
            continue
        short = read["table"].split(".", 1)[1]
        for entity in read["entity_set"]:
            facts.append((read["table"], read["column"], "%s:%s" % (short, entity), None))
    return facts


def earliest_identifiable(
    trace: Mapping[str, Any], gold: Mapping[str, Any], task_ir: Mapping[str, Any], root: Candidate
) -> Tuple[Optional[int], Optional[str], Optional[str]]:
    """``(action_index, node_id of the exposing fact, source)`` or ``(None, None, None)``."""

    gold_cells = {
        (w["table"], w["entity"], w["column"]): w
        for w in gold.get("writes_gold", ())
        if not w.get("volatile")
    }
    contradiction_sources: List[Tuple[str, str, str, Any]] = []
    gold_value = root.detail.get("gold_value")
    if root.node_id:
        contradiction_sources = _lineage_source_facts(task_ir, gold, root.node_id)
    read_node: Dict[Tuple[str, str, str], str] = {}
    for read in gold.get("resolved_reads", ()):
        short = read["table"].split(".", 1)[1]
        for entity in read.get("entity_set") or []:
            read_node[(read["table"], read["column"], "%s:%s" % (short, entity))] = read["node_id"]
    for step in trace["steps"]:
        t = int(step["action_index"])
        if t <= root.action_index:
            continue
        for event in step.get("observations", ()):
            source = event.get("source", "gui")
            for fact in event.get("facts", ()):
                key = (fact.get("table"), fact.get("entity"), fact.get("column"))
                gold_cell = gold_cells.get(key)
                wrong_value = root.detail.get("wrong_value")
                if (
                    gold_cell is not None
                    and not _same(fact.get("value"), gold_cell["value"])
                    and not (wrong_value is not None and _same(fact.get("value"), wrong_value))
                ):
                    # The agent seeing its own wrong value back is not identification (doc 2.3
                    # example: the submitted order shows the wrong date, the calendar exposes it).
                    return t, gold_cell["node_id"], source
                fact_key = (fact.get("table"), fact.get("column"), fact.get("entity"))
                if (
                    any(fact_key == (s[0], s[1], s[2]) for s in contradiction_sources)
                    and gold_value is not None
                ):
                    if _same(fact.get("value"), gold_value) or (
                        isinstance(gold_value, str)
                        and str(fact.get("value"))
                        and str(fact.get("value"))[:10] in gold_value
                    ):
                        return t, read_node.get(fact_key, root.node_id), source
    return None, None, None


def semantic_horizon(
    task_ir: Mapping[str, Any], root_node: Optional[str], identifiable_node: Optional[str]
) -> Optional[int]:
    if not root_node or not identifiable_node:
        return None
    if root_node == identifiable_node:
        return 0
    dag = dag_index(task_ir)
    forward = _path_length(dag, root_node, identifiable_node)
    backward = _path_length(dag, identifiable_node, root_node)
    lengths = [x for x in (forward, backward) if x is not None]
    return min(lengths) if lengths else None
