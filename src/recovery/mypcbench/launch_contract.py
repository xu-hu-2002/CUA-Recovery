"""Collection launch contracts that must be checked before a VM is started."""

from __future__ import annotations

import json
import re
import urllib.request
from typing import Any, Mapping, Optional


class LaunchContractError(ValueError):
    """A frozen serving or collection launch contract was not satisfied."""


def positive_int(name: str, raw_value: Any) -> int:
    """Parse an explicitly supplied positive integer without accepting booleans."""

    if isinstance(raw_value, bool):
        raise LaunchContractError(f"{name} must be a positive integer")
    try:
        value = int(raw_value)
    except (TypeError, ValueError) as exc:
        raise LaunchContractError(f"{name} must be a positive integer") from exc
    if value <= 0 or str(raw_value) != str(value):
        raise LaunchContractError(f"{name} must be a positive integer")
    return value


def model_context_from_payload(payload: Mapping[str, Any], expected_model: str) -> int:
    """Return ``max_model_len`` for one exact model ID from a vLLM /models payload."""

    data = payload.get("data")
    if not isinstance(data, list):
        raise LaunchContractError("endpoint /models response has no data list")
    matches = [item for item in data if isinstance(item, Mapping) and item.get("id") == expected_model]
    if len(matches) != 1:
        served = [item.get("id") for item in data if isinstance(item, Mapping)]
        raise LaunchContractError(
            f"expected exactly one served model {expected_model!r}; endpoint reported {served!r}"
        )
    return positive_int("endpoint max_model_len", matches[0].get("max_model_len"))


def validate_generation_budget(requested_max_tokens: Any, max_model_len: Any) -> tuple[int, int]:
    """Reject a completion budget that leaves no room for the request prompt."""

    requested = positive_int("MYPCBENCH_QWEN_MAX_TOKENS", requested_max_tokens)
    context = positive_int("endpoint max_model_len", max_model_len)
    if requested >= context:
        raise LaunchContractError(
            "MYPCBENCH_QWEN_MAX_TOKENS must be smaller than endpoint max_model_len "
            f"(requested={requested}, max_model_len={context})"
        )
    return requested, context


def fetch_vllm_contract(
    base_url: str,
    expected_model: str,
    requested_max_tokens: Any,
    requested_history_n: Any,
    *,
    api_key: Optional[str] = None,
    timeout: float = 10.0,
) -> dict[str, Any]:
    """Query a vLLM-compatible endpoint and validate the frozen request contract."""

    models_url = base_url.rstrip("/") + "/models"
    headers = {"Accept": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    request = urllib.request.Request(models_url, headers=headers)
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            payload = json.load(response)
    except Exception as exc:
        raise LaunchContractError(f"cannot query {models_url}: {exc}") from exc
    if not isinstance(payload, Mapping):
        raise LaunchContractError("endpoint /models response must be a JSON object")
    context = model_context_from_payload(payload, expected_model)
    requested, context = validate_generation_budget(requested_max_tokens, context)
    history_n = positive_int("MYPCBENCH_QWEN_HISTORY_N", requested_history_n)
    return {
        "base_url": base_url,
        "model": expected_model,
        "max_model_len": context,
        "requested_max_tokens": requested,
        "requested_history_n": history_n,
    }


def formal_collection_authorized(lock_text: str) -> bool:
    """Read the single top-level authorization bit without a YAML dependency."""

    matches = re.findall(
        r"^formal_collection_authorized\s*:\s*(true|false)\s*(?:#.*)?$",
        lock_text,
        flags=re.MULTILINE | re.IGNORECASE,
    )
    if len(matches) != 1:
        raise LaunchContractError(
            "models.lock.yaml must contain exactly one top-level "
            "formal_collection_authorized: true|false"
        )
    return matches[0].lower() == "true"
