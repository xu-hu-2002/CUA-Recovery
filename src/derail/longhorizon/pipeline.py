"""v0.2 post-filter over ``derail.synthesis`` results (manual sections 10.3, 16.1).

The v0.1 pipeline already searches compositions and applies its own targets.  Rather than fork
it, this module re-scores every symbolic candidate with the v0.2 metrics and hard gates and moves
failures into the rejection log with the new codes.  Candidates keep their generation-record
shape; only ``complexity`` and ``validation`` gain fields.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Sequence, Tuple

from derail.longhorizon.complexity import compute_complexity_v02
from derail.longhorizon.effects import EffectError, node_reversibility_class
from derail.longhorizon.gates import GateConfig, failures_to_dicts, run_hard_gates
from derail.longhorizon.ontology import Ontology
from derail.longhorizon.reversibility_sampling import module_reversibility_class
from derail.synthesis.graph import SynthesisValidationError


def annotate_missing_reversibility(
    fragment: Mapping[str, Any], ontology: Ontology
) -> Dict[str, Any]:
    """Return a copy where read-only nodes without ``reversibility_class`` get the R0 class.

    Nodes that carry effects must already declare a class; v0.1 style effects (``irreversible``
    booleans) are not translated because the effect type and commit scope cannot be inferred.
    """

    result = copy.deepcopy(dict(fragment))
    for node in result["nodes"]:
        if "reversibility_class" not in node and not node.get("side_effects"):
            node["reversibility_class"] = ontology.read_only_class
    return result


@dataclass(frozen=True)
class FilteredResult:
    accepted: Tuple[Dict[str, Any], ...]
    rejected: Tuple[Dict[str, Any], ...]


def apply_v02_gates(
    accepted: Sequence[Mapping[str, Any]],
    *,
    gate_config: GateConfig,
    ontology: Ontology,
) -> FilteredResult:
    """Re-evaluate v0.1 accepted candidates and split them by the v0.2 hard gates."""

    kept: List[Dict[str, Any]] = []
    dropped: List[Dict[str, Any]] = []
    for original in accepted:
        candidate = copy.deepcopy(dict(original))
        fragment = annotate_missing_reversibility(candidate["task_ir"], ontology)
        try:
            metrics = compute_complexity_v02(
                fragment,
                ontology,
                anchor_module_id=candidate.get("anchor_module_id"),
                carry_threshold=gate_config.carry_threshold,
            )
        except (EffectError, SynthesisValidationError) as exc:
            dropped.append(_reject(candidate, exc.code, exc.message))
            continue
        failures = run_hard_gates(fragment, metrics, gate_config, ontology)
        candidate["task_ir"] = fragment
        candidate["complexity"] = metrics
        candidate["reversibility_class"] = module_reversibility_class(
            {"fragment": fragment}, ontology
        )
        candidate.setdefault("validation", {})["v02_hard_gates"] = failures_to_dicts(failures)
        if failures:
            first = failures[0]
            dropped.append(_reject(candidate, first.code, first.detail))
        else:
            kept.append(candidate)
    return FilteredResult(tuple(kept), tuple(dropped))


def _reject(candidate: Mapping[str, Any], code: str, detail: str) -> Dict[str, Any]:
    return {
        "schema_version": "generation-record/0.1",
        "candidate_id": "rejected_v02_%s" % candidate["candidate_id"],
        "status": "rejected",
        "rejection_code": code,
        "rejection_detail": detail,
        "attempt": 0,
        "source_skeleton_ids": list(candidate.get("source_skeleton_ids", [])),
        "anchor_module_id": candidate.get("anchor_module_id"),
        "release_eligible": False,
        "provenance": {
            "candidate_fingerprint": candidate["provenance"]["candidate_fingerprint"],
            "llm_calls": [],
            "rejected_by": "derail.longhorizon.pipeline.apply_v02_gates",
            "source_candidate_id": candidate["candidate_id"],
        },
    }


def summarize_reversibility(
    accepted: Sequence[Mapping[str, Any]], ontology: Ontology
) -> Dict[str, int]:
    """Count accepted candidates per module-level reversibility class."""

    counts = {reversibility: 0 for reversibility in ontology.reversibility_classes}
    for candidate in accepted:
        classes = tuple(
            node_reversibility_class(node, ontology) for node in candidate["task_ir"]["nodes"]
        )
        counts[ontology.max_class(classes)] += 1
    return counts
