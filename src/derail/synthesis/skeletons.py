"""Canonicalize Task IR and extract empirical skeleton candidates."""

from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Set, Tuple

from derail.synthesis.graph import validate_grounded_module, validate_task_fragment


def _digest(value: Any) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _endpoint(edge: Mapping[str, Any], side: str) -> Tuple[str, str]:
    return str(edge[side]["node_id"]), str(edge[side]["port_id"])


def _port_type(node: Mapping[str, Any], direction: str, port_id: str) -> str:
    return next(str(port["type"]) for port in node.get(direction, ()) if port["port_id"] == port_id)


def canonical_structural_signature(fragment: Mapping[str, Any]) -> Dict[str, Any]:
    """Return a node-ID-, entity-, and app-invariant graph signature.

    Iterative neighborhood hashing preserves operation, typed ports,
    dependency kinds, branch predicates, and effect placement.  The resulting
    representation is deterministic for the small semantic DAGs used here.
    """

    validate_task_fragment(fragment)
    nodes = {str(node["node_id"]): node for node in fragment["nodes"]}
    incoming = defaultdict(list)
    outgoing = defaultdict(list)
    for edge in fragment["edges"]:
        source, source_port = _endpoint(edge, "from")
        target, target_port = _endpoint(edge, "to")
        edge_label = {
            "kind": edge["kind"],
            "source_type": _port_type(nodes[source], "outputs", source_port),
            "target_type": _port_type(nodes[target], "inputs", target_port),
            "has_predicate": bool(edge.get("predicate")),
        }
        outgoing[source].append((target, edge_label))
        incoming[target].append((source, edge_label))

    labels = {}
    for node_id, node in nodes.items():
        labels[node_id] = _digest(
            {
                "op": node["op"],
                "input_types": sorted(str(port["type"]) for port in node.get("inputs", ())),
                "output_types": sorted(str(port["type"]) for port in node.get("outputs", ())),
                "effects": sorted(
                    (
                        str(effect.get("effect_type", "")),
                        bool(effect.get("irreversible")),
                    )
                    for effect in node.get("side_effects", ())
                ),
            }
        )
    for _ in range(len(nodes)):
        refined = {}
        for node_id in nodes:
            refined[node_id] = _digest(
                {
                    "self": labels[node_id],
                    # Sort on a JSON rendering: two neighbours with equal labels would
                    # otherwise make Python compare the edge-label dicts and fail.
                    "incoming": sorted(
                        ((labels[source], edge_label) for source, edge_label in incoming[node_id]),
                        key=lambda item: json.dumps(item, sort_keys=True),
                    ),
                    "outgoing": sorted(
                        ((labels[target], edge_label) for target, edge_label in outgoing[node_id]),
                        key=lambda item: json.dumps(item, sort_keys=True),
                    ),
                }
            )
        labels = refined

    abstract_edges = []
    for edge in fragment["edges"]:
        source, source_port = _endpoint(edge, "from")
        target, target_port = _endpoint(edge, "to")
        abstract_edges.append(
            {
                "from_label": labels[source],
                "to_label": labels[target],
                "kind": edge["kind"],
                "value_type": _port_type(nodes[source], "outputs", source_port),
                "accepted_type": _port_type(nodes[target], "inputs", target_port),
                "has_predicate": bool(edge.get("predicate")),
            }
        )
    representation = {
        "node_labels": sorted(labels.values()),
        "edges": sorted(
            abstract_edges,
            key=lambda value: json.dumps(value, sort_keys=True, separators=(",", ":")),
        ),
    }
    return {"sha256": _digest(representation), "canonical_graph": representation}


def _root_and_terminal_ops(fragment: Mapping[str, Any]) -> Tuple[Sequence[str], Sequence[str]]:
    nodes = {str(node["node_id"]): node for node in fragment["nodes"]}
    incoming = Counter()
    outgoing = Counter()
    for edge in fragment["edges"]:
        source, _ = _endpoint(edge, "from")
        target, _ = _endpoint(edge, "to")
        outgoing[source] += 1
        incoming[target] += 1
    roots = sorted(str(node["op"]) for node_id, node in nodes.items() if not incoming[node_id])
    terminals = sorted(str(node["op"]) for node_id, node in nodes.items() if not outgoing[node_id])
    return roots, terminals


def _rules_for_modules(modules: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    occurrences = Counter()
    task_support: Dict[Tuple[str, str, str], Set[str]] = defaultdict(set)
    for module in modules:
        fragment = module["fragment"]
        nodes = {str(node["node_id"]): node for node in fragment["nodes"]}
        for edge in fragment["edges"]:
            source, source_port = _endpoint(edge, "from")
            target, _ = _endpoint(edge, "to")
            key = (
                str(nodes[source]["op"]),
                str(nodes[target]["op"]),
                _port_type(nodes[source], "outputs", source_port),
            )
            occurrences[key] += 1
            task_support[key].add(str(module["source_task_id"]))
    denominator = sum(occurrences.values())
    rules = []
    for key in sorted(occurrences):
        producer_op, consumer_op, value_type = key
        identity = {
            "producer_op": producer_op,
            "consumer_op": consumer_op,
            "value_type": value_type,
        }
        rules.append(
            {
                "rule_id": "rule_%s" % _digest(identity)[:16],
                **identity,
                "support_task_count": len(task_support[key]),
                "occurrence_count": occurrences[key],
                "approved": False,
                "probability": occurrences[key] / float(denominator),
                "source_task_ids": sorted(task_support[key]),
            }
        )
    return rules


def extract_observed_skeletons(
    modules: Iterable[Mapping[str, Any]],
) -> List[Dict[str, Any]]:
    """Group isomorphic source fragments into reviewable observed skeletons."""

    module_list = [dict(module) for module in modules]
    for module in module_list:
        validate_grounded_module(module)
    groups = defaultdict(list)
    signatures = {}
    for module in module_list:
        signature = canonical_structural_signature(module["fragment"])
        signatures[str(module["module_id"])] = signature
        groups[signature["sha256"]].append(module)
    total = len(module_list)
    skeletons = []
    for signature_hash, members in sorted(groups.items()):
        rules = _rules_for_modules(members)
        if not rules:
            continue
        roots, terminals = _root_and_terminal_ops(members[0]["fragment"])
        source_task_ids = sorted({str(member["source_task_id"]) for member in members})
        skeletons.append(
            {
                "schema_version": "skeleton/0.1",
                "skeleton_id": "skeleton_%s" % signature_hash[:16],
                "kind": "observed",
                "motif_family": "%s=>%s" % ("+".join(roots), "+".join(terminals)),
                "source_task_ids": source_task_ids,
                "source_module_ids": sorted(str(member["module_id"]) for member in members),
                "support_task_count": len(source_task_ids),
                "empirical_probability": len(members) / float(total),
                "complexity_prior": 1.0,
                # A fragment may have several roots with the same op; the schema wants a set.
                "anchor_ops": sorted(set(roots)),
                "composition_rules": rules,
                "canonical_signature": signature_hash,
                "canonical_graph": signatures[str(members[0]["module_id"])]["canonical_graph"],
                "review_status": "needs_review",
            }
        )
    return skeletons
