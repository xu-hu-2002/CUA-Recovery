"""Intrinsic long-horizon metrics, v0.2 (manual section 2 and 2.1).

The v0.1 metrics from ``derail.synthesis.graph.compute_complexity`` are kept verbatim; this
module overrides the two reversibility-based fields with the R-class definitions and adds the
profile, the high-consequence prerequisite depth, lineage depth and decorative-carry findings.
"""

from __future__ import annotations

from typing import Any, Dict, Mapping, Optional

from derail.longhorizon.carry import decorative_carry_violations
from derail.longhorizon.dag import DagIndex
from derail.longhorizon.effects import (
    count_nodes_with_classes,
    high_consequence_prerequisite_depth,
    irreversible_action_depth,
    reversibility_profile,
    validate_fragment_effects,
)
from derail.longhorizon.ontology import Ontology
from derail.synthesis.graph import compute_complexity


def compute_complexity_v02(
    fragment: Mapping[str, Any],
    ontology: Ontology,
    *,
    anchor_module_id: Optional[str] = None,
    carry_threshold: int = 3,
) -> Dict[str, Any]:
    """Return the full v0.2 metric dictionary for a validated fragment."""

    validate_fragment_effects(fragment, ontology)
    metrics = compute_complexity(fragment, anchor_module_id)
    dag = DagIndex.from_fragment(fragment)
    metrics["irreversible_action_depth"] = irreversible_action_depth(dag, ontology)
    metrics["irreversible_action_count"] = count_nodes_with_classes(
        dag, ontology, tuple(ontology.irreversible_classes)
    )
    metrics["high_consequence_prerequisite_depth"] = high_consequence_prerequisite_depth(
        dag, ontology
    )
    metrics["high_consequence_action_count"] = count_nodes_with_classes(
        dag, ontology, tuple(ontology.high_consequence_classes)
    )
    metrics["reversibility_profile"] = reversibility_profile(fragment, ontology)
    metrics["max_lineage_depth"] = max(dag.longest_depth().values())
    metrics["decorative_carry_violations"] = decorative_carry_violations(
        dag, threshold=carry_threshold
    )
    metrics["carry_threshold"] = carry_threshold
    return metrics
