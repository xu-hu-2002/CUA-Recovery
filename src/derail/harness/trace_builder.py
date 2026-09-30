from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Sequence

from derail.canonical.actions import action_to_dict
from derail.canonical.mypcbench import NormalizationError, PyAutoGUINormalizer

TRACE_SCHEMA_VERSION = "rollout-trace/1.0"
_FROM = re.compile(r"\bFROM\s+[\"`']?([A-Za-z_][A-Za-z0-9_]*)[\"`']?", re.IGNORECASE)
_JOIN = re.compile(r"\bJOIN\s+[\"`']?([A-Za-z_][A-Za-z0-9_]*)[\"`']?", re.IGNORECASE)


def sha256_bytes(data: Optional[bytes]) -> Optional[str]:
    return hashlib.sha256(data).hexdigest() if data else None


def sql_tables(sql: str) -> List[str]:
    out: List[str] = []
    for match in list(_FROM.finditer(sql)) + list(_JOIN.finditer(sql)):
        name = match.group(1)
        if name.lower() not in ("select",) and name not in out:
            out.append(name)
    return out


def _flatten_params(action: Any) -> List[Dict[str, Any]]:
    params: List[Dict[str, Any]] = []
    kind = action.get("kind") if isinstance(action, Mapping) else None
    if kind == "sequence":
        for item in action.get("actions", ()):
            params.extend(_flatten_params(item))
        return params
    if not isinstance(action, Mapping):
        return params
    if "text" in action:
        params.append({"name": "text", "value": action["text"]})
    if "keys" in action:
        params.append({"name": "keys", "value": list(action["keys"])})
    if "key" in action:
        params.append({"name": "key", "value": action["key"]})
    if "x_px" in action and "y_px" in action:
        params.append({"name": "point", "value": [action["x_px"], action["y_px"]]})
    if "commands" in action:
        for command in action["commands"]:
            params.append({"name": "command", "value": command})
    return params


_CAT = re.compile(
    r"^\s*(?:cat|head|tail|less|more)\s+(?:-[a-zA-Z0-9]+\s+)*['\"]?(~?/?[^\s'\"|;&]+)"
)


def observation_from_shell(
    command: str,
    output: Optional[str],
    action_index: int,
    rollout_id: Optional[str] = None,
    home_prefix: str = "/home/user",
) -> Optional[Dict[str, Any]]:
    if not output:
        return None
    match = _CAT.match(command or "")
    if not match:
        return None
    path = match.group(1)
    if path.startswith("~/"):
        path = path[2:]
    elif path.startswith(home_prefix + "/"):
        path = path[len(home_prefix) + 1 :]
    return {
        "schema_version": "observation-event/1.0",
        "rollout_id": rollout_id,
        "action_index": int(action_index),
        "source": "cli",
        "app": "files",
        "route": None,
        "facts": [
            {
                "table": "files.documents",
                "column": "content",
                "entity": "file:%s" % path.lstrip("/"),
                "value": output,
            }
        ],
        "raw_ref": "shell_output:%d" % action_index,
    }


@dataclass
class ActionRecord:
    """What the wrapper knows about one executed action."""

    raw: Any
    modality: str
    thought: Optional[str] = None
    shell_output: Optional[str] = None


def normalize_action(record: ActionRecord, normalizer: PyAutoGUINormalizer) -> Dict[str, Any]:
    raw = record.raw
    if record.modality == "cli":
        command = raw if isinstance(raw, str) else str(raw.get("command", raw))
        return {"type": "shell", "raw": command, "params": [{"name": "command", "value": command}]}
    if isinstance(raw, str) and raw.strip().upper() in ("DONE", "FAIL", "WAIT"):
        return {"type": raw.strip().lower(), "raw": raw, "params": []}
    if isinstance(raw, str):
        try:
            canonical = action_to_dict(normalizer.normalize(raw, wait_seconds=None))
        except (NormalizationError, ValueError) as exc:
            return {
                "type": "pyautogui",
                "raw": raw,
                "params": [],
                "normalization_error": str(exc)[:200],
            }
        return {
            "type": str(canonical.get("kind", canonical.get("type", "pyautogui"))),
            "raw": raw,
            "canonical": canonical,
            "params": _flatten_params(canonical),
        }
    if isinstance(raw, Mapping):
        return {
            "type": str(raw.get("action_type", raw.get("type", "dict"))),
            "raw": dict(raw),
            "params": [
                {"name": k, "value": v}
                for k, v in raw.items()
                if k in ("text", "key", "keys", "x", "y", "coordinate")
            ],
        }
    return {"type": "unknown", "raw": str(raw), "params": []}


def _entity_for(table: str, row: Mapping[str, Any]) -> str:
    rowid = row.get("id", row.get("rowid"))
    return (
        "%s:%s" % (table, rowid)
        if isinstance(rowid, int) and not isinstance(rowid, bool)
        else "%s:*" % table
    )


def observation_from_trace_record(
    record: Mapping[str, Any], action_index: int, rollout_id: Optional[str] = None
) -> Optional[Dict[str, Any]]:
    rows = record.get("rows")
    if not rows:
        return None
    if isinstance(rows, Mapping) and rows.get("truncated"):
        rows = rows.get("sample") or []
    app = str(record.get("db") or record.get("app") or "")
    tables = sql_tables(str(record.get("sql", "")))
    if not tables or not app:
        return None
    table = tables[0]
    facts = []
    for row in rows:
        if not isinstance(row, Mapping):
            continue
        entity = _entity_for(table, row)
        for column, value in row.items():
            if value is None or isinstance(value, (bytes, bytearray)):
                continue
            facts.append(
                {
                    "table": "%s.%s" % (app, table),
                    "column": str(column),
                    "entity": entity,
                    "value": value,
                }
            )
    if not facts:
        return None
    source = "cli" if record.get("source") == "cli" else "api"
    return {
        "schema_version": "observation-event/1.0",
        "rollout_id": rollout_id,
        "action_index": int(action_index),
        "source": source,
        "ts": record.get("ts"),
        "app": app,
        "route": record.get("route"),
        "facts": facts,
        "raw_ref": None,
    }


def pages_from_trace_records(records: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    pages: Dict[tuple, Dict[str, Any]] = {}
    for record in records:
        if record.get("source") == "cli" or not record.get("route"):
            continue
        app = str(record.get("db") or record.get("app") or "")
        key = (app, str(record["route"]))
        page = pages.setdefault(
            key, {"app": app, "route": str(record["route"]), "rendered_fields": []}
        )
        seen = {(f["table"], f["column"], f.get("entity")) for f in page["rendered_fields"]}
        rows = record.get("rows")
        if isinstance(rows, Mapping):
            rows = rows.get("sample") or []
        tables = sql_tables(str(record.get("sql", "")))
        if not tables:
            continue
        table = tables[0]
        for row in rows or []:
            if not isinstance(row, Mapping):
                continue
            entity = _entity_for(table, row)
            for column, value in row.items():
                if (table, column, entity) in seen:
                    continue
                seen.add((table, column, entity))
                page["rendered_fields"].append(
                    {"table": table, "column": str(column), "entity": entity, "value": value}
                )
    return list(pages.values())


@dataclass
class TraceBuilder:
    rollout_id: str
    task_id: str
    world_id: str
    agent: str
    seed: int
    step_budget: int
    sampling_round: int = 1
    image_digest: Optional[str] = None
    instruction: Optional[str] = None
    frame: tuple = (1280, 800)
    primary_modality_threshold: float = 0.7
    steps: List[Dict[str, Any]] = field(default_factory=list)
    changelog_start_seq: Dict[str, int] = field(default_factory=dict)
    _normalizer: PyAutoGUINormalizer = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._normalizer = PyAutoGUINormalizer(*self.frame)

    def start(self, changelog_start_seq: Mapping[str, int]) -> None:
        self.changelog_start_seq = {k: int(v) for k, v in changelog_start_seq.items()}

    def record_step(
        self,
        action_index: int,
        action: ActionRecord,
        *,
        delta: Sequence[Mapping[str, Any]],
        trace_records: Sequence[Mapping[str, Any]],
        screenshot: Optional[bytes],
        a11y_text: Optional[str],
        a11y_text_ref: Optional[str] = None,
        filesystem_delta: Sequence[Mapping[str, Any]] = (),
    ) -> Dict[str, Any]:
        normalized = normalize_action(action, self._normalizer)
        observations = [
            o
            for o in (
                observation_from_trace_record(r, action_index, self.rollout_id)
                for r in trace_records
            )
            if o
        ]
        if action.shell_output:
            file_event = observation_from_shell(
                str(action.raw), action.shell_output, action_index, self.rollout_id
            )
            observations.append(
                file_event
                or {
                    "schema_version": "observation-event/1.0",
                    "rollout_id": self.rollout_id,
                    "action_index": int(action_index),
                    "source": "cli",
                    "app": "shell",
                    "route": None,
                    "facts": [],
                    "raw_ref": "shell_output:%d" % action_index,
                }
            )
        step = {
            "action_index": int(action_index),
            "action": {"type": normalized["type"], "raw": normalized["raw"]},
            "params": normalized["params"],
            "modality": action.modality if action.modality in ("gui", "cli") else "gui",
            "thought": action.thought,
            "screenshot_sha256": sha256_bytes(screenshot),
            "a11y_sha256": hashlib.sha256(a11y_text.encode("utf-8")).hexdigest()
            if a11y_text
            else None,
            "a11y_text_ref": a11y_text_ref,
            "delta": [dict(row) for row in delta],
            "observations": observations,
            "pages": pages_from_trace_records(trace_records),
            "filesystem_delta": list(filesystem_delta),
        }
        if "canonical" in normalized:
            step["action"]["canonical"] = normalized["canonical"]
        if "normalization_error" in normalized:
            step["action"]["normalization_error"] = normalized["normalization_error"]
        self.steps.append(step)
        return step

    def finish(
        self,
        *,
        declared_complete: bool,
        declared_infeasible: bool,
        budget_exhausted: bool,
        final_verifier: Optional[bool] = None,
        verifier_kind: Optional[str] = None,
        extra_side_effects: Optional[bool] = None,
        provenance: Optional[Mapping[str, Any]] = None,
    ) -> Dict[str, Any]:
        counts = {
            "gui": sum(s["modality"] == "gui" for s in self.steps),
            "cli": sum(s["modality"] == "cli" for s in self.steps),
            "editor": 0,
        }
        if counts["gui"] and counts["cli"]:
            major = max(counts["gui"], counts["cli"]) / (counts["gui"] + counts["cli"])
            primary = (
                "mixed"
                if major < self.primary_modality_threshold
                else ("gui" if counts["gui"] >= counts["cli"] else "cli")
            )
        else:
            primary = "cli" if counts["cli"] else "gui"
        return {
            "schema_version": TRACE_SCHEMA_VERSION,
            "rollout_id": self.rollout_id,
            "task_id": self.task_id,
            "world_id": self.world_id,
            "agent": self.agent,
            "seed": int(self.seed),
            "sampling_round": int(self.sampling_round),
            "step_budget": int(self.step_budget),
            "image_digest": self.image_digest,
            "instruction": self.instruction or "",
            "steps": list(self.steps),
            "outcome": {
                "final_verifier": final_verifier,
                "verifier_kind": verifier_kind,
                "declared_complete": declared_complete,
                "declared_infeasible": declared_infeasible,
                "budget_exhausted": budget_exhausted,
                "extra_side_effects": extra_side_effects,
                "steps_taken": len(self.steps),
            },
            "modality": {"primary": primary, "counts": counts},
            "changelog_start_seq": dict(self.changelog_start_seq),
            "provenance": dict(provenance or {}),
        }
