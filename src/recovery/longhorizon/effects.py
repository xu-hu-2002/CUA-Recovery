from __future__ import annotations

from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from recovery.longhorizon.dag import DagIndex
from recovery.longhorizon.ontology import Ontology


class EffectError(ValueError):
    """A side-effect or node reversibility record violates the ontology."""

    def __init__(self, code: str, message: str):
        super().__init__("%s: %s" % (code, message))
        self.code = code
        self.message = message


_REQUIRED_EFFECT_FIELDS = (
    "effect_id",
    "effect_type",
    "target_ref",
    "reversibility_class",
    "commit_scope",
    "compensation_available",
    "checkpoint_required",
)


def validate_side_effect(effect: Mapping[str, Any], ontology: Ontology) -> None:
    for field in _REQUIRED_EFFECT_FIELDS:
        if field not in effect:
            raise EffectError("INVALID_GROUNDING", "side effect lacks %s" % field)
    if not str(effect["effect_id"]).strip() or not str(effect["target_ref"]).strip():
        raise EffectError("INVALID_GROUNDING", "effect_id and target_ref must be non-empty")
    if effect["effect_type"] not in ontology.effect_types:
        raise EffectError("INVALID_GROUNDING", "unknown effect_type %r" % effect["effect_type"])
    if effect["reversibility_class"] not in ontology.side_effect_classes:
        raise EffectError(
            "INVALID_GROUNDING",
            "side effect class must be one of %s" % sorted(ontology.side_effect_classes),
        )
    if effect["commit_scope"] not in ontology.commit_scopes:
        raise EffectError("INVALID_GROUNDING", "unknown commit_scope %r" % effect["commit_scope"])
    for flag in ("compensation_available", "checkpoint_required"):
        if not isinstance(effect[flag], bool):
            raise EffectError("INVALID_GROUNDING", "%s must be a JSON boolean" % flag)
    compensating = effect.get("compensating_effect_type")
    if effect["compensation_available"]:
        if compensating not in ontology.effect_types:
            raise EffectError(
                "INVALID_GROUNDING",
                "compensation_available requires a known compensating_effect_type",
            )
    elif compensating is not None:
        raise EffectError(
            "INVALID_GROUNDING", "compensating_effect_type must be null when unavailable"
        )


def node_reversibility_class(node: Mapping[str, Any], ontology: Ontology) -> str:
    classes = tuple(str(effect["reversibility_class"]) for effect in node.get("side_effects", ()))
    return ontology.max_class(classes)


def validate_node_reversibility(node: Mapping[str, Any], ontology: Ontology) -> None:
    effects = node.get("side_effects", ())
    for effect in effects:
        validate_side_effect(effect, ontology)
    if effects and node.get("op") not in ontology.state_changing_operations:
        raise EffectError(
            "INVALID_GROUNDING",
            "operation %r cannot carry side effects" % node.get("op"),
        )
    declared = node.get("reversibility_class")
    derived = node_reversibility_class(node, ontology)
    if declared is None:
        raise EffectError(
            "INVALID_GROUNDING", "node %s lacks reversibility_class" % node["node_id"]
        )
    if declared != derived:
        raise EffectError(
            "INVALID_GROUNDING",
            "node %s declares %s but its effects imply %s" % (node["node_id"], declared, derived),
        )


def validate_fragment_effects(fragment: Mapping[str, Any], ontology: Ontology) -> None:
    for node in fragment["nodes"]:
        validate_node_reversibility(node, ontology)


def effect_safety_violations(node: Mapping[str, Any], ontology: Ontology) -> List[Dict[str, str]]:
    violations: List[Dict[str, str]] = []
    for effect in node.get("side_effects", ()):
        effect_id = str(effect.get("effect_id", ""))
        reversibility = str(effect.get("reversibility_class", ""))
        if reversibility in ontology.checkpoint_required_classes:
            if not effect.get("checkpoint_required"):
                violations.append(
                    {
                        "effect_id": effect_id,
                        "reason": "checkpoint_required must be true for %s" % reversibility,
                    }
                )
            elif not effect.get("checkpoint_id") or not effect.get("restore_verifier_id"):
                violations.append(
                    {
                        "effect_id": effect_id,
                        "reason": "missing checkpoint_id or restore_verifier_id",
                    }
                )
        if effect.get("compensation_available"):
            if not effect.get("compensation_verifier_id"):
                violations.append(
                    {"effect_id": effect_id, "reason": "missing compensation_verifier_id"}
                )
    return violations


def reversibility_profile(fragment: Mapping[str, Any], ontology: Ontology) -> Dict[str, int]:
    profile = {reversibility: 0 for reversibility in ontology.reversibility_classes}
    for node in fragment["nodes"]:
        profile[node_reversibility_class(node, ontology)] += 1
    return profile


def _nodes_with_classes(
    dag: DagIndex, ontology: Ontology, classes: Sequence[str]
) -> Tuple[str, ...]:
    wanted = set(classes)
    return tuple(
        node_id
        for node_id in dag.order
        if node_reversibility_class(dag.nodes[node_id], ontology) in wanted
    )


def irreversible_action_depth(dag: DagIndex, ontology: Ontology) -> Optional[int]:
    candidates = _nodes_with_classes(dag, ontology, tuple(ontology.irreversible_classes))
    if not candidates:
        return None
    shortest = dag.shortest_depth()
    return min(shortest[node_id] for node_id in candidates)


def high_consequence_prerequisite_depth(dag: DagIndex, ontology: Ontology) -> Optional[int]:
    candidates = _nodes_with_classes(dag, ontology, tuple(ontology.high_consequence_classes))
    if not candidates:
        return None
    shortest = dag.shortest_depth()
    longest = dag.longest_depth()
    earliest = min(candidates, key=lambda node_id: (shortest[node_id], node_id))
    return longest[earliest]


def count_nodes_with_classes(dag: DagIndex, ontology: Ontology, classes: Sequence[str]) -> int:
    return len(_nodes_with_classes(dag, ontology, classes))
