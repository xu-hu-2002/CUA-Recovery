from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

from derail.longhorizon.effects import effect_safety_violations
from derail.longhorizon.ontology import Ontology
from derail.synthesis.graph import SynthesisValidationError, validate_task_fragment

REJECTION_CODES = (
    "UNBOUND_PORT",
    "INVALID_GROUNDING",
    "TYPE_MISMATCH",
    "GLOBAL_CONSTRAINT_UNSAT",
    "WORLD_NOT_MATERIALIZABLE",
    "FIXTURE_RESET_FAILED",
    "VERIFIER_INVALID",
    "REALIZATION_DRIFT",
    "INSUFFICIENT_DEPENDENCY",
    "DECORATIVE_CARRY",
    "SHORTCUT_DETECTED",
    "DUPLICATE_TASK",
    "NEEDS_HUMAN_REVIEW",
    "EFFECT_SAFETY_UNSAT",
)


@dataclass(frozen=True)
class GateFailure:
    code: str
    detail: str

    def __post_init__(self) -> None:
        if self.code not in REJECTION_CODES:
            raise ValueError("unknown rejection code %s" % self.code)


@dataclass(frozen=True)
class GateConfig:
    """Pre-registered structural targets."""

    dependency_depth: Tuple[int, int]
    cross_app_dependencies: Tuple[int, int]
    max_independent_component_ratio: float
    required_delayed_reuse: bool
    carry_threshold: int
    max_irreversible_actions: int
    allowed_irreversible_effect_types: Tuple[str, ...]
    min_verifier_coverage: float

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "GateConfig":
        def _range(key: str) -> Tuple[int, int]:
            value = raw[key]
            if not isinstance(value, (list, tuple)) or len(value) != 2 or value[0] > value[1]:
                raise ValueError("%s must be [min, max]" % key)
            return int(value[0]), int(value[1])

        return cls(
            dependency_depth=_range("dependency_depth"),
            cross_app_dependencies=_range("cross_app_dependencies"),
            max_independent_component_ratio=float(raw["max_independent_component_ratio"]),
            required_delayed_reuse=bool(raw.get("required_delayed_reuse", True)),
            carry_threshold=int(raw.get("carry_threshold", 3)),
            max_irreversible_actions=int(raw.get("max_irreversible_actions", 1)),
            allowed_irreversible_effect_types=tuple(
                str(item) for item in raw.get("allowed_irreversible_actions", ())
            ),
            min_verifier_coverage=float(raw.get("min_verifier_coverage", 1.0)),
        )


Gate = Callable[[Mapping[str, Any], Mapping[str, Any], GateConfig, Ontology], Optional[GateFailure]]


def verifier_coverage(fragment: Mapping[str, Any]) -> float:
    total = 0.0
    covered = 0.0
    for node in fragment["nodes"]:
        if not bool(node.get("critical", True)):
            continue
        weight = float(
            node.get("weight", 2.0 if node.get("side_effects") or node["op"] == "decide" else 1.0)
        )
        total += weight
        verifier = node.get("verifier") or {}
        if verifier.get("verifier_id") and verifier.get("observability") != "unobservable":
            covered += weight
    return covered / total if total else 1.0


def gate_structure(fragment, metrics, config, ontology) -> Optional[GateFailure]:
    try:
        validate_task_fragment(fragment)
    except SynthesisValidationError as exc:
        return GateFailure(exc.code, exc.message)
    return None


def gate_dependency_bucket(fragment, metrics, config, ontology) -> Optional[GateFailure]:
    low, high = config.dependency_depth
    if not low <= int(metrics["dependency_depth"]) <= high:
        return GateFailure(
            "INSUFFICIENT_DEPENDENCY", "dependency_depth outside [%d, %d]" % (low, high)
        )
    low, high = config.cross_app_dependencies
    if not low <= int(metrics["cross_app_dependency_count"]) <= high:
        return GateFailure(
            "INSUFFICIENT_DEPENDENCY", "cross_app_dependency_count outside [%d, %d]" % (low, high)
        )
    if float(metrics["independent_component_ratio"]) > config.max_independent_component_ratio:
        return GateFailure("INSUFFICIENT_DEPENDENCY", "independent_component_ratio too high")
    if config.required_delayed_reuse and int(metrics["max_information_carry_distance"]) < 2:
        return GateFailure("INSUFFICIENT_DEPENDENCY", "required delayed reuse is absent")
    return None


def gate_decorative_carry(fragment, metrics, config, ontology) -> Optional[GateFailure]:
    violations = list(metrics.get("decorative_carry_violations", ()))
    if violations:
        return GateFailure(
            "DECORATIVE_CARRY",
            "edges without a second lineage: %s" % ",".join(str(v["edge_id"]) for v in violations),
        )
    return None


def gate_irreversible_budget(fragment, metrics, config, ontology) -> Optional[GateFailure]:
    if int(metrics["irreversible_action_count"]) > config.max_irreversible_actions:
        return GateFailure("GLOBAL_CONSTRAINT_UNSAT", "too many irreversible actions")
    allowed = set(config.allowed_irreversible_effect_types)
    for node in fragment["nodes"]:
        for effect in node.get("side_effects", ()):
            if (
                effect.get("reversibility_class") in ontology.irreversible_classes
                and effect.get("effect_type") not in allowed
            ):
                return GateFailure(
                    "GLOBAL_CONSTRAINT_UNSAT",
                    "irreversible effect %s is not in the allowed list" % effect.get("effect_type"),
                )
    return None


def gate_effect_safety(fragment, metrics, config, ontology) -> Optional[GateFailure]:
    problems = []
    for node in fragment["nodes"]:
        for violation in effect_safety_violations(node, ontology):
            problems.append("%s:%s" % (violation["effect_id"], violation["reason"]))
    if problems:
        return GateFailure("EFFECT_SAFETY_UNSAT", "; ".join(problems))
    return None


def gate_verifier_coverage(fragment, metrics, config, ontology) -> Optional[GateFailure]:
    coverage = verifier_coverage(fragment)
    if coverage < config.min_verifier_coverage:
        return GateFailure(
            "VERIFIER_INVALID", "critical verifier coverage %.2f is insufficient" % coverage
        )
    return None


HARD_GATES: Tuple[Gate, ...] = (
    gate_structure,
    gate_dependency_bucket,
    gate_decorative_carry,
    gate_irreversible_budget,
    gate_effect_safety,
    gate_verifier_coverage,
)


def run_hard_gates(
    fragment: Mapping[str, Any],
    metrics: Mapping[str, Any],
    config: GateConfig,
    ontology: Ontology,
    gates: Sequence[Gate] = HARD_GATES,
) -> List[GateFailure]:
    failures: List[GateFailure] = []
    for gate in gates:
        failure = gate(fragment, metrics, config, ontology)
        if failure is not None:
            failures.append(failure)
    return failures


def failures_to_dicts(failures: Sequence[GateFailure]) -> List[Dict[str, str]]:
    return [{"code": failure.code, "detail": failure.detail} for failure in failures]
