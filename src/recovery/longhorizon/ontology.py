from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, FrozenSet, Mapping, Tuple

import yaml

from recovery.synthesis.graph import EDGE_KINDS, OPERATIONS


class OntologyError(ValueError):
    """The ontology file is malformed or inconsistent with the symbolic core."""


@dataclass(frozen=True)
class Ontology:
    """Immutable view of one ontology version."""

    schema_version: str
    operations: FrozenSet[str]
    state_changing_operations: FrozenSet[str]
    edge_kinds: FrozenSet[str]
    reversibility_classes: Tuple[str, ...]
    side_effect_classes: FrozenSet[str]
    irreversible_classes: FrozenSet[str]
    high_consequence_classes: FrozenSet[str]
    effect_types: Mapping[str, Mapping[str, Any]]
    commit_scopes: FrozenSet[str]
    checkpoint_required_classes: FrozenSet[str]
    milestone_observability: FrozenSet[str]

    @property
    def read_only_class(self) -> str:
        return self.reversibility_classes[0]

    def class_rank(self, reversibility_class: str) -> int:
        try:
            return self.reversibility_classes.index(reversibility_class)
        except ValueError as exc:
            raise OntologyError("unknown reversibility class %r" % reversibility_class) from exc

    def max_class(self, classes: Tuple[str, ...]) -> str:
        if not classes:
            return self.read_only_class
        return max(classes, key=self.class_rank)

    def effect_category(self, effect_type: str) -> str:
        spec = self.effect_types.get(effect_type)
        if spec is None or not spec.get("category"):
            raise OntologyError("effect type %r has no category" % effect_type)
        return str(spec["category"])

    def default_class_for(self, effect_type: str) -> str:
        spec = self.effect_types.get(effect_type)
        if spec is None:
            raise OntologyError("unknown effect type %r" % effect_type)
        return str(spec["default_class"])

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "Ontology":
        def _strings(key: str) -> Tuple[str, ...]:
            value = raw.get(key)
            if (
                not isinstance(value, list)
                or not value
                or not all(isinstance(item, str) and item.strip() for item in value)
            ):
                raise OntologyError("%s must be a non-empty list of strings" % key)
            return tuple(value)

        operations = frozenset(_strings("operations"))
        if operations != OPERATIONS:
            raise OntologyError(
                "ontology operations drift from recovery.synthesis.graph.OPERATIONS: %s"
                % sorted(operations ^ OPERATIONS)
            )
        edge_kinds = frozenset(_strings("dependency_edge_kinds"))
        if edge_kinds != EDGE_KINDS:
            raise OntologyError("ontology edge kinds drift from the symbolic core")

        classes_raw = raw.get("reversibility_classes")
        if not isinstance(classes_raw, Mapping) or not classes_raw:
            raise OntologyError("reversibility_classes must be a non-empty mapping")
        classes = tuple(str(item) for item in classes_raw)
        side_effect_classes = frozenset(_strings("side_effect_reversibility_classes"))
        irreversible = frozenset(_strings("irreversible_classes"))
        high_consequence = frozenset(_strings("high_consequence_classes"))
        checkpoint_classes = frozenset(_strings("checkpoint_required_classes"))
        for name, subset in (
            ("side_effect_reversibility_classes", side_effect_classes),
            ("irreversible_classes", irreversible),
            ("high_consequence_classes", high_consequence),
            ("checkpoint_required_classes", checkpoint_classes),
        ):
            unknown = subset - set(classes)
            if unknown:
                raise OntologyError("%s references unknown classes %s" % (name, sorted(unknown)))
        if classes[0] in side_effect_classes:
            raise OntologyError("the first reversibility class is reserved for read-only nodes")
        if not irreversible <= high_consequence:
            raise OntologyError("irreversible classes must also be high-consequence")

        effect_types_raw = raw.get("effect_types")
        if not isinstance(effect_types_raw, Mapping) or not effect_types_raw:
            raise OntologyError("effect_types must be a non-empty mapping")
        effect_types: Dict[str, Dict[str, Any]] = {}
        for effect_type, spec in effect_types_raw.items():
            if (
                not isinstance(spec, Mapping)
                or spec.get("default_class") not in side_effect_classes
            ):
                raise OntologyError(
                    "effect type %r needs a default_class among %s"
                    % (effect_type, sorted(side_effect_classes))
                )
            effect_types[str(effect_type)] = dict(spec)

        state_changing = frozenset(_strings("state_changing_operations"))
        if not state_changing <= operations:
            raise OntologyError("state_changing_operations must be a subset of operations")

        return cls(
            schema_version=str(raw.get("schema_version", "")),
            operations=operations,
            state_changing_operations=state_changing,
            edge_kinds=edge_kinds,
            reversibility_classes=classes,
            side_effect_classes=side_effect_classes,
            irreversible_classes=irreversible,
            high_consequence_classes=high_consequence,
            effect_types=effect_types,
            commit_scopes=frozenset(_strings("commit_scopes")),
            checkpoint_required_classes=checkpoint_classes,
            milestone_observability=frozenset(_strings("milestone_observability")),
        )

    @classmethod
    def from_yaml(cls, path: Path) -> "Ontology":
        raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
        if not isinstance(raw, Mapping):
            raise OntologyError("ontology file must contain a YAML mapping: %s" % path)
        return cls.from_dict(raw)
