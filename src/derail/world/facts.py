"""Canonical world facts and their equivalent sources (execution doc v1.2 section 2.2).

A *fact* is ``(table, column, entity, value)``: one cell of one database row (or one attribute
of a file).  ``table`` is app-qualified (``"hoolicalendar.events"``) because table names repeat
across the seventeen application databases; ``entity`` is the stable row id ``"events:530"``.

The same fact often has several *equivalent sources* -- the seeder writes a trip's date into
the flight, the hotel booking, the confirmation e-mail and the calendar event.  ``reads(j)`` of
a Task IR node names the primary source; the ``EquivalenceRegistry`` answers "which other
cells carry the same fact", which is what the omission detector and the latent-horizon
predictor need (a fact read from any equivalent source counts as obtained).

Entry points
------------
``Fact`` / ``FactKey``                       canonical records
``normalize_value(value)``                  comparison form (dates to ISO, text stripped ...)
``facts_from_row(app, table, rowid, row)``  every cell of a row as facts
``EquivalenceRegistry.from_records(...)``   symmetric registry of equivalent sources
``registry.equivalents(key)``               all keys carrying the same fact (including key)
``registry.values_match(a, b)``             value comparison honouring the match rule
"""

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
    """``"hoolicalendar.events"``; a name that is already qualified is returned unchanged."""

    return table if "." in table else "%s.%s" % (app, table)


def split_table(table: str) -> Tuple[str, str]:
    app, _, name = table.partition(".")
    if not name:
        raise ValueError("table %r is not app-qualified" % table)
    return app, name


def entity_id(table: str, rowid: Any) -> str:
    """Stable entity id ``"<table>:<rowid>"`` (table without app prefix)."""

    _, name = split_table(table) if "." in table else ("", table)
    return "%s:%s" % (name, rowid)


def entity_rowid(entity: str) -> int:
    """Inverse of :func:`entity_id`; raises on derived or non-numeric ids."""

    if entity.startswith(DERIVED_PREFIX):
        raise ValueError("entity %r is derived, not a row" % entity)
    _, _, rowid = entity.rpartition(":")
    return int(rowid)


def normalize_value(value: Any) -> Any:
    """Comparison form of a cell value.

    - ``None`` and booleans pass through; numbers become ``float`` when they are not integral.
    - ISO date-times are reduced to ``YYYY-MM-DDTHH:MM:SS`` in their own clock (the offset is
      dropped, not applied: application databases mix naive local stamps and ``+00:00`` stamps
      for the same wall time).
    - Text is whitespace-collapsed and stripped; case is preserved because identifiers
      (channel names, e-mail addresses) are case-sensitive in the applications.
    """

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
    """``YYYY-MM-DD`` of a date or date-time value, else ``None``."""

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
    """Every cell of ``row`` (or only ``columns``) as facts of entity ``table:rowid``."""

    qualified = qualified_table(app, table)
    entity = entity_id(qualified, rowid)
    names = list(columns) if columns is not None else [str(key) for key in row]
    return [Fact(qualified, name, entity, row[name]) for name in names if name in row]


MATCH_RULES = ("exact", "date", "contains")


def values_match(primary: Any, other: Any, rule: str = "exact") -> bool:
    """Whether ``other`` carries the same information as ``primary`` under ``rule``.

    ``exact``    normalized values are equal.
    ``date``     the calendar dates agree (time-of-day and clock offset ignored).
    ``contains`` ``other`` is free text that contains the primary value's date or text.
    """

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
    """Symmetric registry: ``primary fact key -> equivalent sources``.

    Records look like ``{"fact": {table, column, entity}, "equivalents": [{table, column,
    entity, match}]}``; they come from the seeder's cross-application links (execution doc
    section 2.1) and from successful rollouts (section 8.4).  Registration is symmetric, so
    querying any member returns the whole equivalence set.
    """

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
        """All keys carrying the same fact, ``key`` itself first with rule ``exact``."""

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
    """Earliest ``observation-event/1.0`` in which ``fact`` (or an equivalent) was observed.

    Events are scanned in the given order.  This is the primitive behind the omission
    detector ("was this required fact ever observed") and the earliest-identifiable rule.
    """

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
