"""Controlled value-type vocabulary and port normalisation (manual v0.2 section 6.2).

Typed compatibility only works over a shared vocabulary.  The registry lives in
``configs/synthesis/value_types_*.yaml``; this module folds free-form extractor names onto it,
parses collection spellings into ``many`` cardinality, and reports what it could not resolve.
Unresolved names are kept verbatim and flagged, never guessed.
"""

from __future__ import annotations

import copy
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Tuple

import yaml

_COLLECTION_PATTERNS = (
    re.compile(r"^(?P<inner>.+)\[\]$"),
    re.compile(r"^(?:List|Set|Array)<(?P<inner>.+)>$"),
    re.compile(r"^(?P<inner>.+?)(?:Collection|List|Set|Array)$"),
)
_CAMEL = re.compile(r"(?<=[a-z0-9])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])")


class TypeRegistryError(ValueError):
    """The value-type configuration is inconsistent."""


def _snake(name: str) -> str:
    return _CAMEL.sub("_", name).replace("-", "_").replace(" ", "_").lower()


@dataclass(frozen=True)
class TypeResolution:
    original: str
    canonical: str
    registered: bool
    collection: bool


@dataclass(frozen=True)
class ValueTypeRegistry:
    schema_version: str
    types: Mapping[str, Mapping[str, Any]]
    aliases: Mapping[str, str]
    converters: Tuple[Mapping[str, str], ...]
    cardinality_aliases: Mapping[str, str]

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "ValueTypeRegistry":
        types = raw.get("types")
        if not isinstance(types, Mapping) or not types:
            raise TypeRegistryError("types must be a non-empty mapping")
        for name, spec in types.items():
            for parent in (spec or {}).get("parents", ()):
                if parent not in types:
                    raise TypeRegistryError("%s names unknown parent %s" % (name, parent))
        aliases = {}
        for alias, target in (raw.get("aliases") or {}).items():
            if target not in types:
                raise TypeRegistryError("alias %s targets unknown type %s" % (alias, target))
            aliases[_snake(str(alias))] = str(target)
        for name in types:
            aliases.setdefault(_snake(name), str(name))
        cardinalities = {
            str(key).lower(): str(value)
            for key, value in (raw.get("cardinality_aliases") or {}).items()
        }
        if set(cardinalities.values()) - {"one", "optional", "many"}:
            raise TypeRegistryError("cardinality aliases must map to one/optional/many")
        return cls(
            schema_version=str(raw.get("schema_version", "")),
            types={str(name): dict(spec or {}) for name, spec in types.items()},
            aliases=aliases,
            converters=tuple(dict(item) for item in raw.get("converters", ())),
            cardinality_aliases=cardinalities,
        )

    @classmethod
    def from_yaml(cls, path: Path) -> "ValueTypeRegistry":
        raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
        if not isinstance(raw, Mapping):
            raise TypeRegistryError("value-types file must be a YAML mapping")
        return cls.from_dict(raw)

    def resolve(self, name: str) -> TypeResolution:
        """Fold one type name; collection spellings resolve their element type."""

        original = str(name or "").strip()
        collection = False
        inner = original
        for pattern in _COLLECTION_PATTERNS:
            match = pattern.match(inner)
            if match and match.group("inner") != inner:
                inner = match.group("inner")
                collection = True
                break
        canonical = self.aliases.get(_snake(inner))
        if canonical is None:
            # Unregistered: keep the element name (collection wrapper stripped) for review.
            return TypeResolution(original, inner, False, collection)
        return TypeResolution(original, canonical, True, collection)

    def resolve_cardinality(self, value: Any, *, collection: bool) -> str:
        if collection:
            return "many"
        text = str(value if value is not None else "one").lower()
        return self.cardinality_aliases.get(
            text, "many" if text.isdigit() and int(text) > 1 else "one"
        )

    def type_system_config(self) -> Dict[str, Any]:
        """The ``type_system`` block ``derail.synthesis.compatibility.TypeSystem`` consumes."""

        parents = {
            name: list(spec.get("parents", ()))
            for name, spec in sorted(self.types.items())
            if spec.get("parents")
        }
        return {"parents": parents, "converters": [dict(item) for item in self.converters]}

    def vocabulary_text(self) -> str:
        """Compact listing for prompts: ``Name (< Parent)``."""

        parts = []
        for name, spec in self.types.items():
            parents = spec.get("parents")
            parts.append("%s (< %s)" % (name, ", ".join(parents)) if parents else name)
        return ", ".join(parts)


def normalize_module_types(
    module: Mapping[str, Any], registry: ValueTypeRegistry
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Return a copy with every fragment and interface port normalised, plus a report.

    Ports keep ``original_type`` and ``original_cardinality`` for audit.  The report lists
    unregistered type names with the ports that use them.
    """

    result = copy.deepcopy(dict(module))
    unregistered: Dict[str, List[str]] = {}
    changed = 0

    def _normalise(port: Dict[str, Any], where: str) -> None:
        nonlocal changed
        resolution = registry.resolve(port.get("type", ""))
        cardinality = registry.resolve_cardinality(
            port.get("cardinality"), collection=resolution.collection
        )
        if resolution.canonical != port.get("type") or cardinality != port.get("cardinality"):
            changed += 1
        port["original_type"] = port.get("type")
        port["original_cardinality"] = port.get("cardinality")
        port["type"] = resolution.canonical
        port["cardinality"] = cardinality
        if not resolution.registered:
            unregistered.setdefault(resolution.canonical, []).append(where)

    for node in result["fragment"]["nodes"]:
        for direction in ("inputs", "outputs"):
            for port in node.get(direction, ()) or ():
                _normalise(port, "%s.%s.%s" % (node["node_id"], direction, port.get("port_id")))
    for direction in ("produces", "accepts"):
        for port in result["interface"].get(direction, ()) or ():
            _normalise(port, "interface.%s.%s" % (direction, port.get("port_id")))
    report = {
        "ports_changed": changed,
        "unregistered_types": {
            name: sorted(set(ports)) for name, ports in sorted(unregistered.items())
        },
    }
    return result, report
