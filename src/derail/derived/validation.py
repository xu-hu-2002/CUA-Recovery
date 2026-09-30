"""Strict parsing helpers for evidence-bearing JSON records.

Python's ``bool("false")`` is true.  Evidence gates must therefore reject JSON values with the
wrong type instead of coercing them.
"""

from __future__ import annotations

import re
from typing import Any

SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


def strict_bool(value: Any, field: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError("%s 必须是 JSON boolean" % field)
    return value


def require_sha256(value: Any, field: str, *, allow_empty: bool = False) -> str:
    text = str(value)
    if allow_empty and not text:
        return text
    if not SHA256_RE.fullmatch(text):
        raise ValueError("%s 必须是 64 位小写 SHA-256" % field)
    return text
