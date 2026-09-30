"""Mutation library (execution doc v1.2 section 6.1), static side.

Every produce type maps to the type-preserving mutations that could corrupt it; the static
profile attaches those mutation types to each node's latent horizon (``mutation/1.0`` with
``mode: static``).  Concrete mutated values come from the world and are produced by the
dynamic module (section 6.3, deferred), so ``mutated_value_ref`` here names the *kind* of
substitute, not a value.

The type -> mutation table lives in ``configs/synthesis/mutations_v1.yaml`` so the port-type
vocabulary and the mutation vocabulary can evolve without code changes.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Tuple, Union

import yaml

MUTATIONS_CONFIG_VERSION = "mutations-config/1.0"


@dataclass(frozen=True)
class MutationLibrary:
    by_family: Mapping[str, Tuple[str, ...]]
    family_patterns: Tuple[Tuple[str, str], ...]  # (regex over the type name, family)
    write_mutations: Tuple[str, ...]
    default_family: str

    @classmethod
    def from_yaml(cls, path: Union[str, Path]) -> "MutationLibrary":
        raw = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
        if raw.get("schema_version") != MUTATIONS_CONFIG_VERSION:
            raise ValueError("unsupported mutations config %r" % raw.get("schema_version"))
        return cls(
            by_family={str(k): tuple(v) for k, v in (raw.get("by_family") or {}).items()},
            family_patterns=tuple(
                (str(item["pattern"]), str(item["family"]))
                for item in raw.get("family_patterns", ())
            ),
            write_mutations=tuple(raw.get("write_mutations", ())),
            default_family=str(raw.get("default_family", "text")),
        )

    def family_of(self, type_name: str) -> str:
        for pattern, family in self.family_patterns:
            if re.search(pattern, type_name, re.IGNORECASE):
                return family
        return self.default_family

    def mutations_for_type(self, type_name: str) -> Tuple[str, ...]:
        return self.by_family.get(self.family_of(type_name), ())

    def mutations_for_node(self, node: Mapping[str, Any]) -> List[Tuple[str, str, str]]:
        """``(produce_name, produce_type, mutation_type)`` for every produce, plus write
        mutations once per node that writes."""

        out: List[Tuple[str, str, str]] = []
        for produce in node.get("produces", ()):
            for mutation in self.mutations_for_type(str(produce.get("type", ""))):
                out.append((str(produce["name"]), str(produce.get("type", "")), mutation))
        if node.get("writes"):
            for mutation in self.write_mutations:
                out.append(("<writes>", "WriteEffect", mutation))
        return out


def static_mutation_records(
    task_ir: Mapping[str, Any],
    horizons: Mapping[str, Any],
    library: MutationLibrary,
    profile_version: str,
) -> List[Dict[str, Any]]:
    """One ``mutation/1.0`` record per (node, mutation type), static mode."""

    records: List[Dict[str, Any]] = []
    for node in task_ir["nodes"]:
        node_id = str(node["node_id"])
        horizon = horizons[node_id]
        seen: set = set()
        for produce_name, produce_type, mutation in library.mutations_for_node(node):
            if mutation in seen:
                continue
            seen.add(mutation)
            records.append(
                {
                    "schema_version": "mutation/1.0",
                    "task_id": str(task_ir["task_id"]),
                    "node_id": node_id,
                    "mutation_type": mutation,
                    "mutated_value_ref": "static:%s:%s" % (produce_name, mutation),
                    "original_value_ref": "derived:%s:%s" % (node_id, produce_name)
                    if produce_name != "<writes>"
                    else None,
                    "latent_horizon_static": horizon.latent_horizon_static,
                    "latent_horizon_dynamic": None,
                    "observability_class": horizon.observability_class,
                    "static_class": horizon.static_class,
                    "cross_app": horizon.cross_app,
                    "visible_node_id": horizon.visible_node_id,
                    "visible_pages": [],
                    "contradiction_visible": False,
                    "silent": horizon.observability_class == "silent",
                    "verifier_rejects": None,
                    "mode": "static",
                    "provenance": {
                        "profile_version": profile_version,
                        "produce_type": produce_type,
                    },
                }
            )
    return records
