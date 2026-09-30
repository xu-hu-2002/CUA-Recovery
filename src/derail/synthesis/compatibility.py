"""Typed compatibility graph construction for GroundedTaskModules."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Set, Tuple

from derail.synthesis.graph import validate_grounded_module


def _stable_id(prefix: str, value: Any) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return "%s_%s" % (prefix, hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16])


@dataclass(frozen=True)
class TypeSystem:
    """Small deterministic type hierarchy and executable converter registry."""

    parents: Mapping[str, Sequence[str]] = field(default_factory=dict)
    converters: Mapping[Tuple[str, str], str] = field(default_factory=dict)

    @classmethod
    def from_config(cls, raw: Mapping[str, Any]) -> "TypeSystem":
        parents = {
            str(child): tuple(str(parent) for parent in parent_types)
            for child, parent_types in raw.get("parents", {}).items()
        }
        converters = {}
        for converter in raw.get("converters", ()):
            key = (str(converter["from"]), str(converter["to"]))
            converters[key] = str(converter["converter_id"])
        return cls(parents=parents, converters=converters)

    def is_subtype(self, child: str, parent: str) -> bool:
        if child == parent:
            return True
        frontier = list(self.parents.get(child, ()))
        seen: Set[str] = set()
        while frontier:
            candidate = frontier.pop()
            if candidate == parent:
                return True
            if candidate not in seen:
                seen.add(candidate)
                frontier.extend(self.parents.get(candidate, ()))
        return False

    def unify(self, produced: str, accepted: str) -> Optional[Tuple[str, Optional[str], float]]:
        if produced == accepted:
            return produced, None, 1.0
        if self.is_subtype(produced, accepted):
            return accepted, None, 0.97
        converter = self.converters.get((produced, accepted))
        if converter:
            return accepted, converter, 0.92
        return None


def _constraints_match(
    produced: Mapping[str, Any], accepted: Mapping[str, Any]
) -> Tuple[bool, List[str]]:
    left = produced.get("constraints", {})
    right = accepted.get("constraints", {})
    conflicts = []
    for key in sorted(set(left) & set(right)):
        if left[key] != right[key]:
            conflicts.append("constraint:%s" % key)
    return not conflicts, conflicts


def _pair_conflicts(source: Mapping[str, Any], target: Mapping[str, Any]) -> List[str]:
    source_interface = source["interface"]
    target_interface = target["interface"]
    conflicts: List[str] = []

    source_identity = set(str(value) for value in source_interface.get("identity_scope", ()))
    target_identity = set(str(value) for value in target_interface.get("identity_scope", ()))
    if source_identity and target_identity and not source_identity.intersection(target_identity):
        conflicts.append("identity_scope")

    source_timezone = source_interface.get("time_scope", {}).get("timezone")
    target_timezone = target_interface.get("time_scope", {}).get("timezone")
    if source_timezone and target_timezone and source_timezone != target_timezone:
        conflicts.append("timezone")

    target_reads = set(str(value) for value in target_interface.get("reads", ()))
    source_reads = set(str(value) for value in source_interface.get("reads", ()))
    source_effects = source_interface.get("side_effects", ())
    target_effects = target_interface.get("side_effects", ())
    for effect in source_effects:
        target_id = str(effect.get("target_id", ""))
        if effect.get("effect_type") == "delete" and target_id in target_reads:
            conflicts.append("source_deletes_target_read:%s" % target_id)
    for effect in target_effects:
        target_id = str(effect.get("target_id", ""))
        if effect.get("effect_type") == "delete" and target_id in source_reads:
            conflicts.append("target_deletes_source_read:%s" % target_id)
    for left in source_effects:
        for right in target_effects:
            if left.get("target_id") != right.get("target_id"):
                continue
            if left.get("effect_type") != right.get("effect_type") and {
                left.get("effect_type"),
                right.get("effect_type"),
            }.intersection({"delete", "modify"}):
                conflicts.append("effect_conflict:%s" % left.get("target_id"))
    return sorted(set(conflicts))


def _environment_status(source: Mapping[str, Any], target: Mapping[str, Any]) -> str:
    if (
        source["environment_id"] == target["environment_id"]
        and source["snapshot_id"] == target["snapshot_id"]
    ):
        return "direct"
    source_materializable = set(source.get("materializable_environments", ()))
    target_materializable = set(target.get("materializable_environments", ()))
    if source["environment_id"] in target_materializable:
        return "materializable"
    if target["environment_id"] in source_materializable:
        return "materializable"
    if source.get("world_schema_version") == target.get("world_schema_version"):
        return "schema_only"
    return "incompatible"


def _node_operation(module: Mapping[str, Any], node_id: str) -> str:
    return next(
        str(node["op"])
        for node in module["fragment"]["nodes"]
        if node["node_id"] == node_id
    )


def _binding_records(
    source: Mapping[str, Any],
    target: Mapping[str, Any],
    type_system: TypeSystem,
) -> Tuple[List[Dict[str, Any]], List[str]]:
    bindings: List[Dict[str, Any]] = []
    rejected: List[str] = []
    for output_port in source["interface"].get("produces", ()):
        for input_port in target["interface"].get("accepts", ()):
            unified = type_system.unify(str(output_port["type"]), str(input_port["type"]))
            if unified is None:
                continue
            cardinality_out = output_port.get("cardinality", "one")
            cardinality_in = input_port.get("cardinality", "one")
            if cardinality_out != cardinality_in:
                rejected.append(
                    "cardinality:%s->%s" % (cardinality_out, cardinality_in)
                )
                continue
            matches, constraint_conflicts = _constraints_match(output_port, input_port)
            if not matches:
                rejected.extend(constraint_conflicts)
                continue
            unified_type, converter_id, score = unified
            identity = {
                "from_module": source["module_id"],
                "to_module": target["module_id"],
                "output_node_id": output_port["node_id"],
                "output_port": output_port["port_id"],
                "input_node_id": input_port["node_id"],
                "input_port": input_port["port_id"],
            }
            bindings.append(
                {
                    "binding_id": _stable_id("binding", identity),
                    **identity,
                    "produced_type": output_port["type"],
                    "accepted_type": input_port["type"],
                    "unified_type": unified_type,
                    "output_grounding": output_port["grounding"],
                    "input_grounding": input_port["grounding"],
                    "output_cardinality": cardinality_out,
                    "input_cardinality": cardinality_in,
                    "converter_id": converter_id,
                    "dependency_kind": input_port.get(
                        "dependency_kind", "data_dependency"
                    ),
                    "source_op": _node_operation(source, str(output_port["node_id"])),
                    "target_op": _node_operation(target, str(input_port["node_id"])),
                    "score": score,
                }
            )
    return bindings, rejected


def build_compatibility_edges(
    modules: Iterable[Mapping[str, Any]],
    type_system: TypeSystem,
) -> List[Dict[str, Any]]:
    """Build pairwise compatibility candidates while preserving every binding.

    Only ``direct`` edges are executable without fixture materialization.  The
    other statuses remain useful audit records but are excluded by the
    synthesis search.
    """

    module_list = [dict(module) for module in modules]
    for module in module_list:
        validate_grounded_module(module)
    result: List[Dict[str, Any]] = []
    for source in sorted(module_list, key=lambda value: value["module_id"]):
        for target in sorted(module_list, key=lambda value: value["module_id"]):
            if source["module_id"] == target["module_id"]:
                continue
            bindings, rejected = _binding_records(source, target, type_system)
            if not bindings:
                continue
            environment_status = _environment_status(source, target)
            conflicts = _pair_conflicts(source, target)
            if environment_status == "incompatible":
                conflicts.append("environment")
            identity = {
                "from_module": source["module_id"],
                "to_module": target["module_id"],
                "binding_ids": [binding["binding_id"] for binding in bindings],
            }
            result.append(
                {
                    "schema_version": "compat-edge/0.1",
                    "edge_id": _stable_id("compat", identity),
                    "from_module": source["module_id"],
                    "to_module": target["module_id"],
                    "bindings": bindings,
                    "environment_status": environment_status,
                    "environment_scope": {
                        "source_environment_id": source["environment_id"],
                        "source_snapshot_id": source["snapshot_id"],
                        "target_environment_id": target["environment_id"],
                        "target_snapshot_id": target["snapshot_id"],
                        "world_schema_version": source.get("world_schema_version"),
                    },
                    "conflicts": sorted(set(conflicts)),
                    "rejected_binding_checks": sorted(set(rejected)),
                    "compatibility_score": sum(binding["score"] for binding in bindings)
                    / len(bindings),
                    "check_results": [
                        {"check_id": "type_match", "passed": True},
                        {
                            "check_id": "environment_direct",
                            "passed": environment_status == "direct",
                        },
                        {"check_id": "global_pair_conflicts", "passed": not conflicts},
                    ],
                }
            )
    return result
