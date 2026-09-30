"""World graph ``W`` from the frozen MyPCBench seed (manual v0.2 sections 4.1, 6.1, 11.1).

The persona file seeds every application in the VM image, and ``variables.json`` holds the
ground-truth reference values the rubrics cite.  This module turns both into a ``world-graph/0.1``
record: entities with stable ids, the applications where each fact is observable, whether the
fact is mutable, and provenance back to the seed path.  What to extract is configuration
(``configs/synthesis/world_sources_*.yaml``); this module only walks the seed.

Ids are ``seed_derived`` until a per-app check confirms the in-VM store exposes the same key.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import yaml

WORLD_SCHEMA_VERSION = "world-graph/0.1"
_SCALARS = (str, int, float, bool)


class WorldError(ValueError):
    """The world-sources configuration or the seed data is malformed."""


def slugify(value: Any) -> str:
    """Lower-case, non-alphanumerics folded to ``_``; used for stable id segments.

    Dots are folded too: ``entity_id.attribute`` groundings split on the first dot, so an id
    must never contain one.
    """

    if isinstance(value, (list, tuple)):
        value = "_".join(str(item) for item in value)
    text = re.sub(r"[^a-z0-9-]+", "_", str(value).lower()).strip("_")
    return text or "empty"


@dataclass(frozen=True)
class AppAliases:
    canonical: Mapping[str, str]  # any spelling (lower-cased) -> canonical id

    @classmethod
    def from_config(cls, raw: Mapping[str, Sequence[str]]) -> "AppAliases":
        table: Dict[str, str] = {}
        for app_id, spellings in raw.items():
            table[str(app_id).lower()] = str(app_id)
            for spelling in spellings:
                table[str(spelling).lower()] = str(app_id)
        return cls(canonical=table)

    def resolve(self, spelling: str) -> str:
        try:
            return self.canonical[str(spelling).lower()]
        except KeyError as exc:
            raise WorldError("unknown application spelling %r" % spelling) from exc

    def resolve_all(self, spellings: Iterable[str]) -> List[str]:
        return sorted({self.resolve(item) for item in spellings})

    @property
    def app_ids(self) -> List[str]:
        return sorted(set(self.canonical.values()))


def _walk(root: Mapping[str, Any], path: str) -> Any:
    node: Any = root
    for part in path.split("."):
        if not isinstance(node, Mapping) or part not in node:
            return None
        node = node[part]
    return node


def _scalar_attributes(item: Mapping[str, Any], nested: Sequence[str] = ()) -> Dict[str, Any]:
    """Keep scalars and lists of scalars; nested objects are kept only when listed in ``nested``.

    Nested structures that are neither children nor listed are dropped, so a section must opt
    in explicitly to carry, say, a trip's ``generates`` block into the entity.
    """

    attributes: Dict[str, Any] = {}
    for key, value in item.items():
        if isinstance(value, _SCALARS) or value is None:
            attributes[str(key)] = value
        elif isinstance(value, list) and all(isinstance(v, _SCALARS) for v in value):
            attributes[str(key)] = list(value)
        elif str(key) in nested:
            attributes[str(key)] = json.loads(json.dumps(value))
    return attributes


def _collect_path(value: Any, parts: Sequence[str]) -> List[Any]:
    """Walk ``parts`` through dicts and lists, collecting every leaf value."""

    if not parts:
        # A leaf list contributes its elements, not the list object.
        return list(value) if isinstance(value, list) else [value]
    if isinstance(value, Mapping):
        return _collect_path(value.get(parts[0]), parts[1:]) if parts[0] in value else []
    if isinstance(value, list):
        found: List[Any] = []
        for item in value:
            found.extend(_collect_path(item, parts))
        return found
    return []


def _entity_id(
    entity_type: str, item: Mapping[str, Any], id_fields: Sequence[str], fallback: str
) -> str:
    present = [item[name] for name in id_fields if name in item and item[name] not in (None, "")]
    segment = slugify(present) if present else slugify(fallback)
    return "%s:%s" % (slugify(entity_type), segment)


@dataclass
class _Builder:
    aliases: AppAliases
    entities: Dict[str, Dict[str, Any]]
    relations: List[Dict[str, Any]]

    def add_entity(
        self,
        *,
        entity_type: str,
        item: Mapping[str, Any],
        id_fields: Sequence[str],
        fallback: str,
        surfaces: Sequence[str],
        mutable: bool,
        source_path: str,
        nested: Sequence[str] = (),
    ) -> str:
        entity_id = _entity_id(entity_type, item, id_fields, fallback)
        if entity_id in self.entities:
            # Two seed rows collapsing onto one id is a real ambiguity; keep both by suffixing.
            entity_id = "%s.%s" % (entity_id, slugify(source_path))
        self.entities[entity_id] = {
            "entity_id": entity_id,
            "entity_type": entity_type,
            "attributes": _scalar_attributes(item, nested),
            "observation_surfaces": sorted(set(surfaces)),
            "mutable": bool(mutable),
            "source_path": source_path,
            "id_confidence": "seed_derived",
        }
        return entity_id

    def add_relation(self, relation_type: str, source: str, target: str, source_path: str) -> None:
        digest = hashlib.sha256(("%s|%s|%s" % (relation_type, source, target)).encode()).hexdigest()
        self.relations.append(
            {
                "relation_id": "rel_%s" % digest[:12],
                "type": relation_type,
                "from": source,
                "to": target,
                "source_path": source_path,
            }
        )


def _surfaces_for(
    section: Mapping[str, Any], item: Mapping[str, Any], aliases: AppAliases
) -> List[str]:
    """Union of static surfaces, a (possibly nested) spelling field and key-derived surfaces.

    ``surfaces_field`` may be a dotted path such as ``affected.app``; lists along the path are
    traversed.  ``surfaces_from_keys`` maps the keys of a nested block (for example a trip's
    ``generates``) to applications through ``key_apps``.
    """

    found: List[str] = list(_static_surfaces(section.get("surfaces", ()), aliases))
    field = section.get("surfaces_field")
    if field:
        spellings = _collect_path(item, str(field).split("."))
        found.extend(aliases.resolve_all(str(value) for value in spellings))
    from_keys = section.get("surfaces_from_keys")
    if from_keys:
        block = item.get(str(from_keys["path"]))
        if isinstance(block, Mapping):
            key_apps = from_keys.get("key_apps", {})
            found.extend(aliases.resolve(str(key_apps[key])) for key in block if key in key_apps)
    return sorted(set(found))


def _static_surfaces(raw: Any, aliases: AppAliases) -> List[str]:
    """``surfaces: all`` means every canonical application; otherwise resolve the list."""

    if raw == "all":
        return list(aliases.app_ids)
    return aliases.resolve_all(str(value) for value in raw)


def _emit_section(
    builder: _Builder,
    section: Mapping[str, Any],
    parent: Mapping[str, Any],
    parent_path: str,
    parent_entity: Optional[str],
    root_entity: str,
) -> None:
    path = str(section["path"])
    value = _walk(parent, path)
    if value is None:
        return
    full_path = "%s.%s" % (parent_path, path) if parent_path else path
    if bool(section.get("singleton")) or isinstance(value, Mapping):
        items: List[Tuple[Mapping[str, Any], str]] = [(value, full_path)]
    elif isinstance(value, list):
        items = [
            (item, "%s[%d]" % (full_path, index))
            for index, item in enumerate(value)
            if isinstance(item, Mapping)
        ]
    else:
        raise WorldError("section %s is neither an object nor a list of objects" % full_path)
    entity_type = str(section["entity_type"])
    for item, item_path in items:
        entity_id = builder.add_entity(
            entity_type=entity_type,
            item=item,
            id_fields=[str(name) for name in section.get("id_fields", ())],
            fallback=item_path,
            surfaces=_surfaces_for(section, item, builder.aliases),
            mutable=bool(section.get("mutable", False)),
            source_path=item_path,
            nested=[str(name) for name in section.get("nested_attributes", ())],
        )
        relation = section.get("relation")
        if relation and relation.get("to_root"):
            builder.add_relation(str(relation["type"]), entity_id, root_entity, item_path)
        if section.get("relation_type") and parent_entity:
            builder.add_relation(str(section["relation_type"]), entity_id, parent_entity, item_path)
        for child in section.get("children", ()):
            _emit_section(builder, child, item, item_path, entity_id, root_entity)


def build_world_graph(
    persona: Mapping[str, Any],
    variables: Mapping[str, Any],
    sources: Mapping[str, Any],
    *,
    provenance: Mapping[str, str],
) -> Dict[str, Any]:
    """Build the world-graph record from the persona seed, reference variables and config."""

    aliases = AppAliases.from_config(sources["app_aliases"])
    builder = _Builder(aliases=aliases, entities={}, relations=[])
    persona_cfg = sources["persona"]
    root_cfg = persona_cfg["root_entity"]
    root_item = _walk(persona, str(root_cfg["path"]))
    if not isinstance(root_item, Mapping):
        raise WorldError("root entity path %s missing in persona" % root_cfg["path"])
    root_entity = builder.add_entity(
        entity_type=str(root_cfg["entity_type"]),
        item=root_item,
        id_fields=[str(name) for name in root_cfg.get("id_fields", ())],
        fallback=str(root_cfg["path"]),
        surfaces=_static_surfaces(root_cfg.get("surfaces", ()), aliases),
        mutable=bool(root_cfg.get("mutable", False)),
        source_path=str(root_cfg["path"]),
    )
    for section in persona_cfg.get("sections", ()):
        _emit_section(builder, section, persona, "", None, root_entity)

    variables_cfg = sources.get("variables", {})
    prefixes = variables_cfg.get("prefix_surfaces", {})
    for key in sorted(variables):
        surfaces = list(variables_cfg.get("default_surfaces", ()))
        for prefix, apps in prefixes.items():
            if str(key).startswith(str(prefix)):
                surfaces = list(apps)
                break
        value = variables[key]
        builder.add_entity(
            entity_type=str(variables_cfg.get("entity_type", "ReferenceValue")),
            item={"key": key, "value": value if isinstance(value, _SCALARS) else json.dumps(value)},
            id_fields=["key"],
            fallback=str(key),
            surfaces=aliases.resolve_all(surfaces),
            mutable=False,
            source_path="variables.%s" % key,
        )
    return {
        "schema_version": WORLD_SCHEMA_VERSION,
        "environment_id": str(sources["environment_id"]),
        "snapshot_id": str(sources["snapshot_id"]),
        "world_schema_version": str(sources["world_schema_version"]),
        "source": dict(provenance),
        "app_ids": aliases.app_ids,
        "entities": [builder.entities[key] for key in sorted(builder.entities)],
        "relations": sorted(builder.relations, key=lambda item: item["relation_id"]),
    }


def subgraph_for_apps(world: Mapping[str, Any], app_ids: Iterable[str]) -> Dict[str, Any]:
    """Entities observable in any of ``app_ids`` plus the relations among them (§6.1 W_sub)."""

    wanted = set(app_ids)
    entities = [
        entity for entity in world["entities"] if wanted & set(entity["observation_surfaces"])
    ]
    ids = {entity["entity_id"] for entity in entities}
    relations = [rel for rel in world["relations"] if rel["from"] in ids and rel["to"] in ids]
    return {"app_ids": sorted(wanted), "entities": entities, "relations": relations}


def load_sources(path: Path) -> Dict[str, Any]:
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(raw, Mapping) or "app_aliases" not in raw or "persona" not in raw:
        raise WorldError("world sources config must define app_aliases and persona")
    return dict(raw)
