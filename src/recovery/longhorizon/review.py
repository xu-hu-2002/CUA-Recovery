from __future__ import annotations

import copy
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Tuple

import yaml

from recovery.longhorizon.ontology import Ontology
from recovery.longhorizon.types import ValueTypeRegistry, normalize_module_types

VERDICT_SCHEMA_VERSION = "task-ir-review/0.1"
VERDICTS = ("accept", "fix", "reject")
METHODS = ("per_module", "blanket_signoff")


class ReviewError(ValueError):
    """The verdict file is malformed or names unknown modules."""


@dataclass(frozen=True)
class ReviewVerdicts:
    reviewer: str
    reviewed_at: str
    method: str
    note: str
    verdicts: Mapping[str, Mapping[str, Any]]

    @classmethod
    def from_yaml(cls, path: Path) -> "ReviewVerdicts":
        raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
        if not isinstance(raw, Mapping) or raw.get("schema_version") != VERDICT_SCHEMA_VERSION:
            raise ReviewError(
                "verdict file must declare schema_version %s" % VERDICT_SCHEMA_VERSION
            )
        method = str(raw.get("method", ""))
        if method not in METHODS:
            raise ReviewError("method must be one of %s" % (METHODS,))
        verdicts = raw.get("verdicts")
        if not isinstance(verdicts, Mapping) or not verdicts:
            raise ReviewError("verdicts must be a non-empty mapping of task_id -> verdict")
        for task_id, record in verdicts.items():
            if not isinstance(record, Mapping) or record.get("verdict") not in VERDICTS:
                raise ReviewError("verdict for %s must be one of %s" % (task_id, VERDICTS))
        return cls(
            reviewer=str(raw["reviewer"]),
            reviewed_at=str(raw["reviewed_at"]),
            method=method,
            note=str(raw.get("note", "")),
            verdicts={str(task_id): dict(record) for task_id, record in verdicts.items()},
        )


def project_interface_effects(
    module: Mapping[str, Any], ontology: Ontology
) -> List[Dict[str, Any]]:
    projected = []
    for node in module["fragment"]["nodes"]:
        for effect in node.get("side_effects", ()):
            projected.append(
                {
                    "effect_type": ontology.effect_category(str(effect["effect_type"])),
                    "target_id": str(effect["target_ref"]),
                    "irreversible": str(effect["reversibility_class"])
                    in ontology.irreversible_classes,
                    "concrete_effect_type": str(effect["effect_type"]),
                    "reversibility_class": str(effect["reversibility_class"]),
                    "effect_id": str(effect.get("effect_id", "")),
                    "node_id": str(node["node_id"]),
                }
            )
    return projected


def latest_by_task(module_files: Iterable[Path]) -> Dict[str, Dict[str, Any]]:
    import json

    merged: Dict[str, Dict[str, Any]] = {}
    for path in module_files:
        for line in Path(path).read_text(encoding="utf-8").splitlines():
            if line.strip():
                module = json.loads(line)
                module.setdefault("provenance", {})["module_file"] = str(path)
                merged[str(module["source_task_id"])] = module
    return merged


def apply_review(
    modules: Mapping[str, Mapping[str, Any]],
    verdicts: ReviewVerdicts,
    *,
    registry: ValueTypeRegistry,
    ontology: Ontology,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    unknown = sorted(set(verdicts.verdicts) - set(modules))
    if unknown:
        raise ReviewError("verdicts name modules that were not extracted: %s" % unknown)
    accepted: List[Dict[str, Any]] = []
    skipped: Dict[str, str] = {}
    unregistered: Dict[str, List[str]] = {}
    for task_id in sorted(modules):
        verdict = verdicts.verdicts.get(task_id)
        if verdict is None or verdict["verdict"] != "accept":
            skipped[task_id] = verdict["verdict"] if verdict else "no_verdict"
            continue
        module = copy.deepcopy(dict(modules[task_id]))
        if not module["provenance"].get("static_valid", False):
            skipped[task_id] = "static_invalid"
            continue
        module, type_report = normalize_module_types(module, registry)
        for name, ports in type_report["unregistered_types"].items():
            unregistered.setdefault(name, []).extend("%s:%s" % (task_id, port) for port in ports)
        module["interface"]["side_effects"] = project_interface_effects(module, ontology)
        module["provenance"]["review_status"] = "human_verified"
        module["provenance"]["review"] = {
            "schema_version": VERDICT_SCHEMA_VERSION,
            "reviewer": verdicts.reviewer,
            "reviewed_at": verdicts.reviewed_at,
            "method": verdicts.method,
            "note": verdict.get("note", verdicts.note),
            "entity_precision": verdict.get("entity_precision"),
            "edge_precision": verdict.get("edge_precision"),
        }
        module["provenance"]["type_normalisation"] = {
            "registry_version": registry.schema_version,
            "ports_changed": type_report["ports_changed"],
        }
        accepted.append(module)
    report = {
        "accepted": [module["source_task_id"] for module in accepted],
        "skipped": skipped,
        "unregistered_types": {name: sorted(ports) for name, ports in sorted(unregistered.items())},
    }
    return accepted, report
