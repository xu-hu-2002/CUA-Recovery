from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence, Tuple

import yaml


class TaxonomyError(ValueError):
    """The taxonomy file is malformed or a label is outside the frozen vocabulary."""


@dataclass(frozen=True)
class FailureTaxonomy:
    schema_version: str
    taxonomy_version: str
    categories: Mapping[str, Tuple[str, ...]]
    category_to_group: Mapping[str, str]
    paper_type_to_subgroup: Mapping[str, str]
    retired_labels: Tuple[str, ...]
    renamed_labels: Mapping[str, str]
    category_priority: Tuple[str, ...]
    cross_category_labels: Tuple[str, ...]

    @property
    def paper_types(self) -> Tuple[str, ...]:
        return tuple(label for labels in self.categories.values() for label in labels)

    def category_of(self, paper_type: str) -> str:
        for category, labels in self.categories.items():
            if paper_type in labels:
                return category
        raise TaxonomyError("label %r is not a frozen paper type" % paper_type)

    def group_of(self, paper_type: str) -> str:
        return self.category_to_group[self.category_of(paper_type)]

    def normalize(self, raw_labels: Iterable[str]) -> Tuple[Tuple[str, ...], Tuple[str, ...]]:
        kept = []
        dropped = []
        known = set(self.paper_types)
        for label in raw_labels:
            label = self.renamed_labels.get(label, label)
            if label in known:
                if label not in kept:
                    kept.append(label)
            elif label in self.retired_labels:
                dropped.append(label)
            else:
                raise TaxonomyError("unknown error label %r" % label)
        return tuple(kept), tuple(dropped)

    def primary_paper_type(self, labels: Sequence[str]) -> Optional[str]:
        for category in self.category_priority:
            for label in labels:
                if self.category_of(label) == category:
                    return label
        return None

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "FailureTaxonomy":
        categories_raw = raw.get("paper_categories")
        if not isinstance(categories_raw, Mapping) or not categories_raw:
            raise TaxonomyError("paper_categories must be a non-empty mapping")
        categories: Dict[str, Tuple[str, ...]] = {}
        seen: Dict[str, str] = {}
        for category, labels in categories_raw.items():
            if not isinstance(labels, list) or not labels:
                raise TaxonomyError("category %r has no labels" % category)
            for label in labels:
                if label in seen:
                    raise TaxonomyError("label %r appears in two categories" % label)
                seen[str(label)] = str(category)
            categories[str(category)] = tuple(str(label) for label in labels)
        groups = raw.get("category_to_group")
        if not isinstance(groups, Mapping) or set(groups) != set(categories):
            raise TaxonomyError("category_to_group must cover every category exactly")
        priority = raw.get("primary_selection", {}).get("category_priority")
        if not isinstance(priority, list) or sorted(priority) != sorted(categories):
            raise TaxonomyError("primary_selection.category_priority must permute the categories")
        subgroups = raw.get("paper_type_to_subgroup", {})
        if not isinstance(subgroups, Mapping) or set(subgroups) != set(seen):
            raise TaxonomyError("paper_type_to_subgroup must cover every paper type exactly")
        renamed = raw.get("renamed_labels", {})
        if not isinstance(renamed, Mapping) or any(
            target not in seen for target in renamed.values()
        ):
            raise TaxonomyError("renamed_labels must map onto frozen paper types")
        return cls(
            schema_version=str(raw.get("schema_version", "")),
            taxonomy_version=str(raw.get("taxonomy_version", "")),
            categories=categories,
            category_to_group={str(key): str(value) for key, value in groups.items()},
            paper_type_to_subgroup={str(key): str(value) for key, value in subgroups.items()},
            retired_labels=tuple(str(item) for item in raw.get("retired_labels", [])),
            renamed_labels={str(key): str(value) for key, value in renamed.items()},
            category_priority=tuple(str(item) for item in priority),
            cross_category_labels=tuple(str(item) for item in raw.get("cross_category_labels", [])),
        )

    @classmethod
    def from_yaml(cls, path: Path) -> "FailureTaxonomy":
        raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
        if not isinstance(raw, Mapping):
            raise TaxonomyError("taxonomy file must contain a YAML mapping: %s" % path)
        return cls.from_dict(raw)
