from __future__ import annotations

import hashlib
import json
import os
import re
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Protocol, Sequence, Tuple

import yaml

from derail.longhorizon.effects import EffectError, validate_fragment_effects
from derail.longhorizon.ontology import Ontology
from derail.longhorizon.types import ValueTypeRegistry
from derail.longhorizon.world import AppAliases, subgraph_for_apps
from derail.synthesis.graph import SynthesisValidationError, validate_grounded_module

MODULE_SCHEMA_VERSION = "grounded-task-module/0.1"
FRAGMENT_SCHEMA_VERSION = "task-fragment/0.1"
_TRUE_VALUES = {"1", "true", "yes"}


class ApprovalRequired(RuntimeError):
    """The explicit per-run API approval variables are not set."""


class ExtractionParseError(ValueError):
    """The model reply does not contain one parseable JSON object."""


@dataclass(frozen=True)
class ExtractorConfig:
    extractor_id: str
    provider: str
    model: str
    api_key_env: str
    base_url_env: str
    base_url_default: str
    send_temperature: bool
    temperature: float
    reasoning_effort: Optional[str]
    max_completion_tokens: int
    timeout_s: int
    prompt_system: Path
    prompt_user: Path
    world_sources_config: Path
    ontology_config: Path
    value_types_config: Path
    approval_env: str
    purpose_env: str
    allowed_purposes: Tuple[str, ...]

    @classmethod
    def from_yaml(cls, path: Path, repo_root: Path) -> "ExtractorConfig":
        raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
        if not isinstance(raw, Mapping):
            raise ValueError("extractor config must be a YAML mapping")
        policy = raw.get("api_invocation_policy") or {}
        if not policy.get("requires_explicit_user_approval_per_run", False):
            raise ValueError("extractor config must require explicit per-run approval")
        return cls(
            extractor_id=str(raw["extractor_id"]),
            provider=str(raw["provider"]),
            model=str(raw["model"]),
            api_key_env=str(raw["api_key_env"]),
            base_url_env=str(raw["base_url_env"]),
            base_url_default=str(raw.get("base_url_default", "")),
            send_temperature=bool(raw.get("send_temperature", True)),
            temperature=float(raw.get("temperature", 0.0)),
            reasoning_effort=(
                str(raw["reasoning_effort"]) if raw.get("reasoning_effort") else None
            ),
            max_completion_tokens=int(raw.get("max_completion_tokens", 4096)),
            timeout_s=int(raw.get("timeout_s", 300)),
            prompt_system=repo_root / raw["prompt_system"],
            prompt_user=repo_root / raw["prompt_user"],
            world_sources_config=repo_root / raw["world_sources_config"],
            ontology_config=repo_root / raw["ontology_config"],
            value_types_config=repo_root / raw["value_types_config"],
            approval_env=str(policy["approval_env"]),
            purpose_env=str(policy["purpose_env"]),
            allowed_purposes=tuple(str(item) for item in policy.get("allowed_purposes", ())),
        )


@dataclass(frozen=True)
class LLMResponse:
    text: str
    model: str
    usage: Mapping[str, Any]
    request_sha256: str


class LLMClient(Protocol):
    def complete(self, system: str, user: str) -> LLMResponse: ...


class OpenAICompatibleClient:
    """Minimal chat-completions client over the standard library."""

    def __init__(self, config: ExtractorConfig, purpose: str):
        approved = os.environ.get(config.approval_env, "").strip().lower() in _TRUE_VALUES
        declared = os.environ.get(config.purpose_env, "").strip()
        if not approved or declared != purpose or purpose not in config.allowed_purposes:
            raise ApprovalRequired(
                "set %s=1 and %s=%s to allow this run"
                % (config.approval_env, config.purpose_env, purpose)
            )
        api_key = os.environ.get(config.api_key_env, "")
        base_url = os.environ.get(config.base_url_env, "") or config.base_url_default
        if base_url and not base_url.lower().startswith(("http://", "https://")):
            raise ApprovalRequired(
                "%s is not an http(s) URL (length %d); fix the environment / .env"
                % (config.base_url_env, len(base_url))
            )
        if not api_key or not base_url:
            raise ApprovalRequired(
                "%s must be set and %s or base_url_default must name the endpoint"
                % (config.api_key_env, config.base_url_env)
            )
        self._config = config
        self._base_url = base_url.rstrip("/")
        self._api_key = api_key

    def complete(self, system: str, user: str) -> LLMResponse:
        payload: Dict[str, Any] = {
            "model": self._config.model,
            "max_completion_tokens": self._config.max_completion_tokens,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        }
        if self._config.send_temperature:
            payload["temperature"] = self._config.temperature
        if self._config.reasoning_effort:
            payload["reasoning_effort"] = self._config.reasoning_effort
        body = json.dumps(payload).encode("utf-8")
        request = urllib.request.Request(
            "%s/chat/completions" % self._base_url,
            data=body,
            headers={
                "Content-Type": "application/json",
                "Authorization": "Bearer %s" % self._api_key,
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self._config.timeout_s) as response:
                reply = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            raise RuntimeError(
                "extraction request failed: HTTP %s %s"
                % (exc.code, exc.read()[:200].decode("utf-8", "replace"))
            ) from exc
        text = reply["choices"][0]["message"]["content"]
        return LLMResponse(
            text=text,
            model=str(reply.get("model", self._config.model)),
            usage=dict(reply.get("usage", {})),
            request_sha256=hashlib.sha256(body).hexdigest(),
        )


_PLACEHOLDER = re.compile(r"\{\{([a-z_]+)\}\}")


def render_prompt(template: str, fields: Mapping[str, str]) -> str:
    missing = sorted(set(_PLACEHOLDER.findall(template)) - set(fields))
    if missing:
        raise KeyError("prompt template lacks values for %s" % missing)
    return _PLACEHOLDER.sub(lambda match: str(fields[match.group(1)]), template)


_ENTITY_LINE_CHARS = 700


def _entity_line(entity: Mapping[str, Any]) -> str:
    attributes = {
        key: value
        for key, value in entity["attributes"].items()
        if not str(key).startswith("_") and value is not None
    }
    compact = json.dumps(attributes, ensure_ascii=False)
    if len(compact) > _ENTITY_LINE_CHARS:
        compact = compact[: _ENTITY_LINE_CHARS - 3] + "..."
    return "- %s | %s | apps=%s | %s" % (
        entity["entity_id"],
        entity["entity_type"],
        ",".join(entity["observation_surfaces"]),
        compact,
    )


def build_prompt_fields(
    task: Mapping[str, Any],
    world_subgraph: Mapping[str, Any],
    ontology: Ontology,
    app_ids: Sequence[str],
    registry: Optional[ValueTypeRegistry] = None,
) -> Dict[str, str]:
    rubrics = task.get("grading", {}).get("rubrics", [])
    return {
        "task_id": str(task["id"]),
        "category": str(task.get("category", "")),
        "apps": ", ".join(app_ids),
        "instruction": str(task["instruction"]),
        "rubrics": "\n".join(
            "%d. (weight %s) %s" % (index, item.get("weight"), item.get("criterion"))
            for index, item in enumerate(rubrics, start=1)
        )
        or "(none)",
        "required_subtasks": "\n".join("- %s" % item for item in task.get("required_subtasks", []))
        or "(not provided)",
        "world_entities": "\n".join(_entity_line(entity) for entity in world_subgraph["entities"])
        or "(no seeded entities for these applications)",
        "operations": ", ".join(sorted(ontology.operations)),
        "reversibility_classes": ", ".join(ontology.reversibility_classes),
        "effect_types": ", ".join(sorted(ontology.effect_types)),
        "commit_scopes": ", ".join(sorted(ontology.commit_scopes)),
        "value_types": registry.vocabulary_text() if registry else "(no registry supplied)",
    }


def parse_extraction_response(text: str) -> Dict[str, Any]:
    fenced = re.search(r"```(?:json)?\s*(\{.*\})\s*```", text, re.DOTALL)
    candidate = fenced.group(1) if fenced else text[text.find("{") : text.rfind("}") + 1]
    if not candidate.strip():
        raise ExtractionParseError("reply contains no JSON object")
    try:
        parsed = json.loads(candidate)
    except json.JSONDecodeError as exc:
        raise ExtractionParseError("reply is not valid JSON: %s" % exc) from exc
    if not isinstance(parsed, dict) or "fragment" not in parsed or "interface" not in parsed:
        raise ExtractionParseError("reply must contain fragment and interface")
    return parsed


_NODE_SYNONYMS = {"operation": "op", "application": "app", "goal": "semantic_goal"}
_PORT_SYNONYMS = {"name": "port_id", "port": "port_id"}
APP_UNSPECIFIED = "unspecified"


_UNBOUND_FIELDS = ("name", "app", "kind", "needed_by", "why")


def normalize_unbound(entries: Any) -> List[Dict[str, Any]]:
    result: List[Dict[str, Any]] = []
    for entry in entries or ():
        if isinstance(entry, Mapping):
            record = {field: entry.get(field) for field in _UNBOUND_FIELDS}
            if not record["name"]:
                record["name"] = json.dumps(dict(entry), ensure_ascii=False)
        else:
            record = {field: None for field in _UNBOUND_FIELDS}
            record["name"] = str(entry)
        result.append(record)
    return result


def normalize_fragment(fragment: Mapping[str, Any]) -> Dict[str, Any]:
    result = json.loads(json.dumps(fragment))
    for node in result.get("nodes", ()):
        for old, new in _NODE_SYNONYMS.items():
            if old in node and new not in node:
                node[new] = node.pop(old)
        if not node.get("app"):
            node["app"] = APP_UNSPECIFIED
        for direction in ("inputs", "outputs"):
            for port in node.get(direction, ()) or ():
                for old, new in _PORT_SYNONYMS.items():
                    if old in port and new not in port:
                        port[new] = port.pop(old)
    return result


def _grounding_issues(fragment: Mapping[str, Any], entity_ids: Iterable[str]) -> List[str]:
    known = set(entity_ids)
    issues = []
    for node in fragment.get("nodes", ()):
        if node.get("app") == APP_UNSPECIFIED:
            issues.append("APP_UNSPECIFIED: node %s has no application" % node.get("node_id"))
        for port in node.get("inputs", ()):
            if port.get("source") == "world":
                entity = str(port.get("grounding", "")).split(".", 1)[0]
                if entity not in known:
                    issues.append(
                        "INVALID_GROUNDING: %s.%s references unknown entity %s"
                        % (node.get("node_id"), port.get("port_id"), entity)
                    )
    return issues


@dataclass(frozen=True)
class ExtractedModule:
    module: Dict[str, Any]
    issues: Tuple[str, ...] = field(default_factory=tuple)

    @property
    def static_valid(self) -> bool:
        return not self.issues


def to_grounded_module(
    parsed: Mapping[str, Any],
    *,
    task: Mapping[str, Any],
    source_split: str,
    world: Mapping[str, Any],
    world_subgraph: Mapping[str, Any],
    ontology: Ontology,
    source_uri: str,
    source_sha256: str,
    llm_call: Mapping[str, Any],
) -> ExtractedModule:
    task_id = str(task["id"])
    fragment = normalize_fragment(parsed["fragment"])
    fragment["schema_version"] = FRAGMENT_SCHEMA_VERSION
    fragment.setdefault("fragment_id", "tf_%s" % task_id)
    module: Dict[str, Any] = {
        "schema_version": MODULE_SCHEMA_VERSION,
        "module_id": "gm_%s" % task_id,
        "source_task_id": task_id,
        "source_split": source_split,
        "environment_id": str(world["environment_id"]),
        "snapshot_id": str(world["snapshot_id"]),
        "world_schema_version": str(world["world_schema_version"]),
        "world_subgraph_id": "wg_%s" % task_id,
        "instruction": str(task["instruction"]),
        "fragment": fragment,
        "interface": dict(parsed["interface"]),
        "provenance": {
            "source_uri": source_uri,
            "source_sha256": source_sha256,
            "review_status": "needs_review",
            "llm_calls": [dict(llm_call)],
            "unbound_entities": normalize_unbound(parsed.get("unbound_entities")),
            "extractor_notes": str(parsed.get("notes", "")),
            "world_subgraph_entity_ids": [e["entity_id"] for e in world_subgraph["entities"]],
        },
    }
    issues: List[str] = []
    try:
        validate_grounded_module(module)
    except SynthesisValidationError as exc:
        issues.append("%s: %s" % (exc.code, exc.message))
    try:
        validate_fragment_effects(fragment, ontology)
    except EffectError as exc:
        issues.append("%s: %s" % (exc.code, exc.message))
    issues.extend(_grounding_issues(fragment, module["provenance"]["world_subgraph_entity_ids"]))
    module["provenance"]["static_issues"] = list(issues)
    module["provenance"]["static_valid"] = not issues
    return ExtractedModule(module=module, issues=tuple(issues))


def run_extraction(
    tasks: Sequence[Mapping[str, Any]],
    *,
    world: Mapping[str, Any],
    aliases: AppAliases,
    ontology: Ontology,
    config: ExtractorConfig,
    splits: Mapping[str, str],
    output_dir: Path,
    source_uri: str,
    source_sha256: str,
    client: Optional[LLMClient] = None,
    saved_replies_dir: Optional[Path] = None,
    registry: Optional[ValueTypeRegistry] = None,
    workers: int = 1,
    progress: Optional[Callable[[int, int, str, float], None]] = None,
) -> Dict[str, Any]:
    system = config.prompt_system.read_text(encoding="utf-8")
    user_template = config.prompt_user.read_text(encoding="utf-8")
    prompts_dir = output_dir / "prompts"
    replies_dir = output_dir / "replies"
    prompts_dir.mkdir(parents=True, exist_ok=True)
    modules: List[Dict[str, Any]] = []
    failures: List[Dict[str, str]] = []
    prepared = []
    for task in tasks:
        task_id = str(task["id"])
        app_ids = aliases.resolve_all(task.get("apps_involved", []))
        subgraph = subgraph_for_apps(world, app_ids)
        prompt = render_prompt(
            user_template, build_prompt_fields(task, subgraph, ontology, app_ids, registry)
        )
        (prompts_dir / ("%s.md" % task_id)).write_text(prompt, encoding="utf-8")
        prepared.append((task, task_id, subgraph, prompt))

    def _fetch(item) -> Tuple[Optional[LLMResponse], bool]:
        _, task_id, _, prompt = item
        saved = saved_replies_dir / ("%s.txt" % task_id) if saved_replies_dir else None
        if saved is not None and saved.is_file():
            return (
                LLMResponse(
                    text=saved.read_text(encoding="utf-8"),
                    model=config.model,
                    usage={},
                    request_sha256="",
                ),
                True,
            )
        if client is None:
            return None, False
        return client.complete(system, prompt), False

    import time

    def _timed(index: int):
        started = time.monotonic()
        result = _fetch(prepared[index])
        return index, result, time.monotonic() - started

    fetched: List[Tuple[Optional[LLMResponse], bool]] = [(None, False)] * len(prepared)
    done = 0
    if workers > 1 and client is not None:
        from concurrent.futures import ThreadPoolExecutor, as_completed

        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = [pool.submit(_timed, index) for index in range(len(prepared))]
            for future in as_completed(futures):
                index, result, seconds = future.result()
                fetched[index] = result
                done += 1
                if progress:
                    progress(done, len(prepared), prepared[index][1], seconds)
    else:
        for index in range(len(prepared)):
            index, result, seconds = _timed(index)
            fetched[index] = result
            done += 1
            if progress:
                progress(done, len(prepared), prepared[index][1], seconds)

    for (task, task_id, subgraph, prompt), (response, reused) in zip(prepared, fetched):
        if response is None:
            continue
        replies_dir.mkdir(parents=True, exist_ok=True)
        (replies_dir / ("%s.txt" % task_id)).write_text(response.text, encoding="utf-8")
        llm_call = {
            "extractor_id": config.extractor_id,
            "model": response.model,
            "temperature": config.temperature,
            "prompt_sha256": hashlib.sha256((system + prompt).encode("utf-8")).hexdigest(),
            "request_sha256": response.request_sha256,
            "usage": dict(response.usage),
            "called_at": datetime.now(timezone.utc).isoformat(),
            "reused_saved_reply": reused,
            "reply_source": (str(saved_replies_dir / ("%s.txt" % task_id)) if reused else None),
        }
        try:
            parsed = parse_extraction_response(response.text)
        except ExtractionParseError as exc:
            failures.append({"task_id": task_id, "error": str(exc)})
            continue
        extracted = to_grounded_module(
            parsed,
            task=task,
            source_split=splits.get(task_id, "source-development"),
            world=world,
            world_subgraph=subgraph,
            ontology=ontology,
            source_uri=source_uri,
            source_sha256=source_sha256,
            llm_call=llm_call,
        )
        modules.append(extracted.module)
    if modules:
        with (output_dir / "modules.jsonl").open("w", encoding="utf-8") as handle:
            for module in modules:
                handle.write(json.dumps(module, ensure_ascii=False, sort_keys=True) + "\n")
    return {
        "schema_version": "stage-manifest/0.1",
        "stage": "phase1_task_ir_extraction",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "mode": (
            "reparse_saved_replies"
            if saved_replies_dir is not None
            else ("dry_run_prompts_only" if client is None else "model_called")
        ),
        "extractor_id": config.extractor_id,
        "model": config.model if (client is not None or saved_replies_dir) else None,
        "workers": workers,
        "counts": {
            "tasks": len(tasks),
            "prompts_written": len(tasks),
            "modules": len(modules),
            "static_valid_modules": sum(1 for m in modules if m["provenance"]["static_valid"]),
            "parse_failures": len(failures),
        },
        "parse_failures": failures,
        "human_review_status": "pending",
    }
