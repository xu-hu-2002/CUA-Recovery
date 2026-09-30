"""Runtime JSON-Schema validation for every evidence-bearing CLI boundary."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def validate_schema(value: Any, schema_name: str, repository: Path) -> None:
    try:
        import jsonschema
    except ImportError as exc:
        raise RuntimeError(
            "JSON Schema validation is mandatory; install the project runtime dependencies"
        ) from exc

    schema_dir = repository.resolve() / "schemas"
    schemas = {}
    for path in schema_dir.glob("*.schema.json"):
        schema = json.loads(path.read_text(encoding="utf-8"))
        schemas[str(schema.get("$id", path.as_uri()))] = schema
        schemas[path.name] = schema
    target_path = schema_dir / schema_name
    if not target_path.is_file():
        raise RuntimeError("unknown schema: %s" % schema_name)
    target = json.loads(target_path.read_text(encoding="utf-8"))
    resolver = jsonschema.RefResolver.from_schema(target, store=schemas)
    try:
        jsonschema.Draft202012Validator(target, resolver=resolver).validate(value)
    except jsonschema.ValidationError as exc:
        branch = _branch_for(target, value) if exc.validator == "oneOf" else None
        if branch is None:
            raise
        branch_schema = dict(branch[1])
        branch_schema.setdefault("$id", target.get("$id", ""))
        try:
            jsonschema.Draft202012Validator(branch_schema, resolver=resolver).validate(value)
        except jsonschema.ValidationError as branch_exc:
            branch_exc.message = "[%s branch %s] %s" % (schema_name, branch[0], branch_exc.message)
            raise branch_exc from None
        raise


def _branch_for(schema: dict, value: Any):
    version = value.get("schema_version") if isinstance(value, dict) else None
    for name, sub in (schema.get("$defs") or {}).items():
        const = ((sub.get("properties") or {}).get("schema_version") or {}).get("const")
        if version is not None and const == version:
            return name, sub
    if version is None:
        for name, sub in (schema.get("$defs") or {}).items():
            if "schema_version" not in (sub.get("required") or []):
                return name, sub
    return None
