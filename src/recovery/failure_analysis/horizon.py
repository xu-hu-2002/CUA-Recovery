from __future__ import annotations

from typing import Any, Dict, List, Mapping, Optional, Tuple

from recovery.detect.latent_static import _path_length
from recovery.failure_analysis.detectors import Candidate, _same
from recovery.ir.model import dag_index


def _lineage_source_facts(
    task_ir: Mapping[str, Any], gold: Mapping[str, Any], node_id: str
) -> List[Tuple[str, str, str, Any]]:
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
