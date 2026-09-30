from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Sequence, Tuple

from derail.longhorizon.effects import node_reversibility_class
from derail.longhorizon.ontology import Ontology

MODE_EMPIRICAL_STRATIFIED = "empirical_stratified"
MODE_NONE = "none"


def module_reversibility_class(module: Mapping[str, Any], ontology: Ontology) -> str:
    classes = tuple(
        node_reversibility_class(node, ontology) for node in module["fragment"]["nodes"]
    )
    return ontology.max_class(classes)


def empirical_class_distribution(
    modules: Sequence[Mapping[str, Any]], ontology: Ontology
) -> Dict[str, int]:
    counts = {reversibility: 0 for reversibility in ontology.reversibility_classes}
    for module in modules:
        counts[module_reversibility_class(module, ontology)] += 1
    return counts


@dataclass(frozen=True)
class ReversibilitySamplingConfig:
    mode: str = MODE_EMPIRICAL_STRATIFIED
    classes: Tuple[str, ...] = ("R0", "R1", "R2", "R3")
    min_pilot_per_available_class: int = 10
    force_uniform_distribution: bool = False
    allowed_irreversible_effect_types: Tuple[str, ...] = ()

    @classmethod
    def from_dict(
        cls, raw: Mapping[str, Any], allowed_irreversible: Sequence[str] = ()
    ) -> "ReversibilitySamplingConfig":
        mode = str(raw.get("mode", MODE_EMPIRICAL_STRATIFIED))
        if mode not in (MODE_EMPIRICAL_STRATIFIED, MODE_NONE):
            raise ValueError("unknown reversibility sampling mode %r" % mode)
        return cls(
            mode=mode,
            classes=tuple(str(item) for item in raw.get("classes", cls.classes)),
            min_pilot_per_available_class=int(raw.get("min_pilot_per_available_class", 10)),
            force_uniform_distribution=bool(raw.get("force_uniform_distribution", False)),
            allowed_irreversible_effect_types=tuple(str(item) for item in allowed_irreversible),
        )


@dataclass(frozen=True)
class Allocation:
    targets: Mapping[str, int]
    coverage_gaps: Tuple[Mapping[str, str], ...]
    source_distribution: Mapping[str, int]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "targets": dict(self.targets),
            "coverage_gaps": [dict(gap) for gap in self.coverage_gaps],
            "source_distribution": dict(self.source_distribution),
        }


def _largest_remainder(total: int, weights: Mapping[str, float]) -> Dict[str, int]:
    if total < 0:
        raise ValueError("total cannot be negative")
    mass = sum(weights.values())
    if mass <= 0:
        return {key: 0 for key in weights}
    raw = {key: total * value / mass for key, value in weights.items()}
    floors = {key: int(value) for key, value in raw.items()}
    remaining = total - sum(floors.values())
    for key in sorted(weights, key=lambda item: (-(raw[item] - floors[item]), item))[:remaining]:
        floors[key] += 1
    return floors


def allocate_targets(
    target_count: int,
    source_distribution: Mapping[str, int],
    config: ReversibilitySamplingConfig,
    ontology: Ontology,
) -> Allocation:
    gaps: List[Dict[str, str]] = []
    available: Dict[str, int] = {}
    for reversibility in config.classes:
        count = int(source_distribution.get(reversibility, 0))
        if count <= 0:
            gaps.append({"class": reversibility, "reason": "absent_in_source"})
            continue
        if (
            reversibility in ontology.irreversible_classes
            and not config.allowed_irreversible_effect_types
        ):
            gaps.append({"class": reversibility, "reason": "no_sandbox_support"})
            continue
        available[reversibility] = count
    if config.mode == MODE_NONE or not available:
        return Allocation(
            targets={}, coverage_gaps=tuple(gaps), source_distribution=dict(source_distribution)
        )

    weights = (
        {key: 1.0 for key in available}
        if config.force_uniform_distribution
        else {key: float(value) for key, value in available.items()}
    )
    targets = _largest_remainder(target_count, weights)
    floor = min(config.min_pilot_per_available_class, target_count // max(1, len(available)))
    for reversibility in sorted(available):
        if targets[reversibility] < floor:
            deficit = floor - targets[reversibility]
            donor = max(targets, key=lambda key: (targets[key], key))
            if donor == reversibility or targets[donor] - deficit < floor:
                continue
            targets[donor] -= deficit
            targets[reversibility] += deficit
    return Allocation(
        targets=targets, coverage_gaps=tuple(gaps), source_distribution=dict(source_distribution)
    )


def stratified_pick(
    candidates_by_class: Mapping[str, Sequence[Any]],
    allocation: Allocation,
    seed: int,
) -> Dict[str, List[Any]]:
    rng = random.Random(seed)
    picked: Dict[str, List[Any]] = {}
    for reversibility in sorted(allocation.targets):
        pool = list(candidates_by_class.get(reversibility, ()))
        wanted = min(allocation.targets[reversibility], len(pool))
        picked[reversibility] = rng.sample(pool, wanted) if wanted else []
    return picked
