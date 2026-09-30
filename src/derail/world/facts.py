from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, FrozenSet, Iterable, List, Mapping, Optional, Sequence, Tuple

DERIVED_PREFIX = "derived:"
_ISO_DATETIME = re.compile(
    r"^(\d{4}-\d{2}-\d{2})[T ](\d{2}:\d{2})(?::(\d{2}))?(?:\.\d+)?(Z|[+-]\d{2}:?\d{2})?$"
)
_ISO_DATE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
_WS = re.compile(r"\s+")


def qualified_table(app: str, table: str) -> str:
    return table if "." in table else "%s.%s" % (app, table)


def split_table(table: str) -> Tuple[str, str]:
    app, _, name = table.partition(".")
    if not name:
        raise ValueError("table %r is not app-qualified" % table)
    return app, name


def entity_id(table: str, rowid: Any) -> str:
    _, name = split_table(table) if "." in table else ("", table)
    return "%s:%s" % (name, rowid)


def entity_rowid(entity: str) -> int:
    if entity.startswith(DERIVED_PREFIX):
        raise ValueError("entity %r is derived, not a row" % entity)
    _, _, rowid = entity.rpartition(":")
    return int(rowid)


def normalize_value(value: Any) -> Any:
    if value is None or isinstance(value, bool):
        return value
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value) if value.is_integer() else value
    if isinstance(value, (bytes, bytearray, memoryview)):
        return bytes(value)
    if isinstance(value, str):
        text = _WS.sub(" ", value).strip()
        match = _ISO_DATETIME.match(text)
        if match:
            seconds = match.group(3) or "00"
            return "%sT%s:%s" % (match.group(1), match.group(2), seconds)
        return text
    if isinstance(value, (list, tuple)):
        return [normalize_value(item) for item in value]
    if isinstance(value, Mapping):
        return {str(key): normalize_value(item) for key, item in value.items()}
    return value


def date_part(value: Any) -> Optional[str]:
    normalized = normalize_value(value)
    if isinstance(normalized, str):
        if _ISO_DATE.match(normalized):
            return normalized
        if _ISO_DATETIME.match(normalized):
            return normalized[:10]
    return None


@dataclass(frozen=True)
class FactKey:
    """Location of a fact without its value."""

    table: str
    column: str
    entity: str

    def __post_init__(self) -> None:
        split_table(self.table)

    def to_dict(self) -> Dict[str, str]:
        return {"table": self.table, "column": self.column, "entity": self.entity}

    @classmethod
    def from_dict(cls, record: Mapping[str, Any]) -> "FactKey":
        return cls(str(record["table"]), str(record["column"]), str(record["entity"]))


@dataclass(frozen=True)
class Fact:
    table: str
    column: str
    entity: str
    value: Any

    @property
    def key(self) -> FactKey:
        return FactKey(self.table, self.column, self.entity)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "table": self.table,
            "column": self.column,
            "entity": self.entity,
            "value": self.value,
        }

    @classmethod
    def from_dict(cls, record: Mapping[str, Any]) -> "Fact":
        return cls(
            str(record["table"]), str(record["column"]), str(record["entity"]), record.get("value")
        )


def facts_from_row(
    app: str,
    table: str,
    rowid: Any,
    row: Mapping[str, Any],
    columns: Optional[Sequence[str]] = None,
) -> List[Fact]:
    qualified = qualified_table(app, table)
    entity = entity_id(qualified, rowid)
    names = list(columns) if columns is not None else [str(key) for key in row]
    return [Fact(qualified, name, entity, row[name]) for name in names if name in row]


MATCH_RULES = ("exact", "date", "contains")


def values_match(primary: Any, other: Any, rule: str = "exact") -> bool:
    if rule == "exact":
        return normalize_value(primary) == normalize_value(other)
    if rule == "date":
        first, second = date_part(primary), date_part(other)
        return first is not None and first == second
    if rule == "contains":
        needle = date_part(primary) or normalize_value(primary)
        haystack = normalize_value(other)
        if not isinstance(haystack, str) or needle is None:
            return False
        return str(needle) in haystack
    raise ValueError("unknown match rule %r" % rule)


@dataclass(frozen=True)
class EquivalentSource:
    key: FactKey
    match: str = "exact"

    def __post_init__(self) -> None:
        if self.match not in MATCH_RULES:
            raise ValueError("unknown match rule %r" % self.match)


@dataclass
class EquivalenceRegistry:
    _links: Dict[FactKey, Dict[FactKey, str]] = field(default_factory=dict)

    @classmethod
    def from_records(cls, records: Iterable[Mapping[str, Any]]) -> "EquivalenceRegistry":
        registry = cls()
        for record in records:
            primary = FactKey.from_dict(record["fact"])
            for equivalent in record.get("equivalents", ()):
                registry.add(
                    primary, FactKey.from_dict(equivalent), str(equivalent.get("match", "exact"))
                )
        return registry

    def add(self, primary: FactKey, other: FactKey, match: str = "exact") -> None:
        if match not in MATCH_RULES:
            raise ValueError("unknown match rule %r" % match)
        self._links.setdefault(primary, {})[other] = match
        self._links.setdefault(other, {})[primary] = match

    def equivalents(self, key: FactKey) -> List[EquivalentSource]:
        out = [EquivalentSource(key, "exact")]
        for other, match in sorted(
            self._links.get(key, {}).items(),
            key=lambda item: (item[0].table, item[0].column, item[0].entity),
        ):
            out.append(EquivalentSource(other, match))
        return out

    def keys(self) -> FrozenSet[FactKey]:
        return frozenset(self._links)

    def match_rule(self, primary: FactKey, other: FactKey) -> Optional[str]:
        if primary == other:
            return "exact"
        return self._links.get(primary, {}).get(other)

    def to_records(self) -> List[Dict[str, Any]]:
        records = []
        for primary in sorted(self._links, key=lambda k: (k.table, k.column, k.entity)):
            records.append(
                {
                    "fact": primary.to_dict(),
                    "equivalents": [
                        dict(other.to_dict(), match=match)
                        for other, match in sorted(
                            self._links[primary].items(),
                            key=lambda item: (item[0].table, item[0].column, item[0].entity),
                        )
                    ],
                }
            )
        return records


def first_observation_of(
    fact: Fact,
    events: Sequence[Mapping[str, Any]],
    registry: Optional[EquivalenceRegistry] = None,
) -> Optional[Mapping[str, Any]]:
    sources = registry.equivalents(fact.key) if registry else [EquivalentSource(fact.key)]
    wanted = {
        (source.key.table, source.key.column, source.key.entity): source.match for source in sources
    }
    for event in events:
        for observed in event.get("facts", ()):
            key = (
                str(observed.get("table")),
                str(observed.get("column")),
                str(observed.get("entity")),
            )
            rule = wanted.get(key)
            if rule is not None and values_match(fact.value, observed.get("value"), rule):
                return event
    return None
