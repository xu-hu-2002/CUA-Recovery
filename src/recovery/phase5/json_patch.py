"""Small RFC 6902 subset used by the Phase 5 persona injector."""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Iterable


class PatchError(ValueError):
    """Raised when a frozen persona patch cannot be applied."""


def _tokens(path: str) -> list[str]:
    if not path.startswith("/"):
        raise PatchError(f"invalid JSON pointer: {path!r}")
    return [part.replace("~1", "/").replace("~0", "~") for part in path[1:].split("/")]


def _parent(document: Any, path: str) -> tuple[Any, str]:
    parts = _tokens(path)
    if not parts:
        raise PatchError("root replacement is not supported")
    current = document
    for token in parts[:-1]:
        try:
            current = current[int(token)] if isinstance(current, list) else current[token]
        except (KeyError, IndexError, TypeError, ValueError) as exc:
            raise PatchError(f"missing parent for {path!r}") from exc
    return current, parts[-1]


def _add(parent: Any, token: str, value: Any) -> None:
    if isinstance(parent, list):
        if token == "-":
            parent.append(value)
            return
        if not token.isdigit() or int(token) > len(parent):
            raise PatchError(f"invalid array index: {token!r}")
        parent.insert(int(token), value)
        return
    if isinstance(parent, dict):
        parent[token] = value
        return
    raise PatchError("add target parent is not an object or array")


def _remove(parent: Any, token: str) -> None:
    try:
        if isinstance(parent, list):
            del parent[int(token)]
        elif isinstance(parent, dict):
            del parent[token]
        else:
            raise TypeError
    except (KeyError, IndexError, TypeError, ValueError) as exc:
        raise PatchError(f"cannot remove {token!r}") from exc


def apply_patch(document: Any, operations: Iterable[dict[str, Any]]) -> Any:
    result = deepcopy(document)
    for operation in operations:
        op = operation.get("op")
        parent, token = _parent(result, operation.get("path", ""))
        if op == "add":
            _add(parent, token, deepcopy(operation.get("value")))
        elif op == "remove":
            _remove(parent, token)
        elif op == "replace":
            _remove(parent, token)
            _add(parent, token, deepcopy(operation.get("value")))
        else:
            raise PatchError(f"unsupported operation: {op!r}")
    return result
