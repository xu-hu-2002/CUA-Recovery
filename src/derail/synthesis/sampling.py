"""Empirical and bounded failure-informed sampling utilities."""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Dict, Iterable, Mapping, Optional


def _clamp(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, value))


def compute_skeleton_weights(
    skeletons: Iterable[Mapping[str, Any]],
    human_failure_stats: Iterable[Mapping[str, Any]] = (),
    *,
    enabled: bool = False,
    alpha: float = 1.0,
    beta: float = 1.0,
    min_exposures: int = 10,
    adjustment_min: float = 0.5,
    adjustment_max: float = 2.0,
    excluded_model_id: Optional[str] = None,
) -> Dict[str, Dict[str, float]]:
    """Return normalized skeleton weights and their auditable components.

    Failure statistics are accepted only when they are explicitly marked as
    human-adjudicated.  ``excluded_model_id`` implements leave-one-model-out
    weighting for evaluations of a model that contributed source failures.
    """

    skeleton_list = [dict(item) for item in skeletons]
    by_structure = defaultdict(lambda: {"exposures": 0, "failures": 0})
    by_family = defaultdict(lambda: {"exposures": 0, "failures": 0})
    for record in human_failure_stats:
        if record.get("review_status") not in {"adjudicated", "human_verified"}:
            raise ValueError("failure-informed sampling requires human-adjudicated statistics")
        if excluded_model_id and record.get("model_id") == excluded_model_id:
            continue
        exposures = int(record.get("exposures", 0))
        failures = int(record.get("attributed_failures", 0))
        if exposures < 0 or failures < 0 or failures > exposures:
            raise ValueError("invalid failure exposure counts")
        structure_id = str(record["structure_id"])
        family = str(record.get("family", structure_id))
        by_structure[structure_id]["exposures"] += exposures
        by_structure[structure_id]["failures"] += failures
        by_family[family]["exposures"] += exposures
        by_family[family]["failures"] += failures

    raw: Dict[str, Dict[str, float]] = {}
    for skeleton in skeleton_list:
        skeleton_id = str(skeleton["skeleton_id"])
        family = str(skeleton.get("motif_family", skeleton_id))
        base = float(skeleton.get("empirical_probability", 0.0))
        if base <= 0:
            raise ValueError("skeleton empirical_probability must be positive")
        adjustment = 1.0
        fragility = 0.0
        family_fragility = 0.0
        if enabled:
            structure_counts = by_structure[skeleton_id]
            family_counts = by_family[family]
            selected = (
                structure_counts
                if structure_counts["exposures"] >= min_exposures
                else family_counts
            )
            fragility = (selected["failures"] + alpha) / (
                selected["exposures"] + alpha + beta
            )
            family_fragility = (family_counts["failures"] + alpha) / (
                family_counts["exposures"] + alpha + beta
            )
            adjustment = _clamp(
                fragility / family_fragility if family_fragility else 1.0,
                adjustment_min,
                adjustment_max,
            )
        complexity_prior = float(skeleton.get("complexity_prior", 1.0))
        raw[skeleton_id] = {
            "empirical_probability": base,
            "complexity_prior": complexity_prior,
            "fragility": fragility,
            "family_fragility": family_fragility,
            "fragility_adjustment": adjustment,
            "raw_weight": base * complexity_prior * adjustment,
        }
    denominator = sum(item["raw_weight"] for item in raw.values())
    if denominator <= 0:
        raise ValueError("skeleton sampling weights sum to zero")
    for item in raw.values():
        item["sampling_probability"] = item["raw_weight"] / denominator
    return raw
