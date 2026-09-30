from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Sequence, Set, Tuple, Union

import yaml

VOLATILE_SCHEMA_VERSION = "volatile-columns/1.0"


@dataclass(frozen=True)
class VolatileColumns:
    patterns: Tuple[str, ...]
    keep: Mapping[str, Tuple[str, ...]]

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any]) -> "VolatileColumns":
        if raw.get("schema_version") != VOLATILE_SCHEMA_VERSION:
            raise ValueError("unsupported volatile columns version %r" % raw.get("schema_version"))
        return cls(
            patterns=tuple(str(p) for p in raw.get("patterns", ())),
            keep={
                str(t): tuple(str(c) for c in cols) for t, cols in (raw.get("keep") or {}).items()
            },
        )

    @classmethod
    def from_yaml(cls, path: Union[str, Path]) -> "VolatileColumns":
        return cls.from_mapping(yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {})

    @classmethod
    def from_json(cls, path: Union[str, Path]) -> "VolatileColumns":
        return cls.from_mapping(json.loads(Path(path).read_text(encoding="utf-8")))

    def is_volatile(self, table: str, column: str) -> bool:
        if column in self.keep.get(table, ()):
            return False
        return any(re.search(pattern, column) for pattern in self.patterns)

    def excluded_for(
        self, app: str, tables: Iterable[Tuple[str, Sequence[str]]]
    ) -> Dict[str, Set[str]]:
        out: Dict[str, Set[str]] = {}
        for table, columns in tables:
            qualified = "%s.%s" % (app, table)
            excluded = {c for c in columns if self.is_volatile(qualified, c)}
            if excluded:
                out[table] = excluded
        return out

    def to_mapping(self) -> Dict[str, Any]:
        return {
            "schema_version": VOLATILE_SCHEMA_VERSION,
            "patterns": list(self.patterns),
            "keep": {t: list(c) for t, c in self.keep.items()},
        }
