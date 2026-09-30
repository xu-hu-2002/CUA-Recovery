from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple, Union

import yaml

HINT_LEVELS = ("l1_step", "l2_evidence", "l3_state_diff")
LEAK_POLICIES = ("drop", "resample", "redact", "off")
RECOVERY_SAMPLE_VERSION = "recovery-sample/1.0"


def hint_text(level: str, case: Mapping[str, Any]) -> str:
    refs = case.get("evidence_refs") or {}
    root = case.get("root_cause_action_index")
    carried = refs.get("carried_value_ref") or {}
    contradiction = refs.get("contradiction_fact") or {}
    affected = refs.get("affected_write_ref") or {}
    if level == "l1_step":
        return "Step %s was wrong." % root
    if level == "l2_evidence":
        return "Step %s was wrong: the screen shows %s = %r, but you used %r (from step %s)." % (
            root,
            contradiction.get("column"),
            contradiction.get("value"),
            carried.get("value"),
            carried.get("source_action_index"),
        )
    if level == "l3_state_diff":
        text = hint_text("l2_evidence", case)
        if affected:
            text += (
                " Step %s already changed %s.%s (row %s): compensate that first, then redo the step correctly."
                % (
                    affected.get("action_index"),
                    affected.get("db"),
                    affected.get("tbl"),
                    affected.get("rowid"),
                )
            )
        else:
            text += " Nothing has been written yet: redo the step correctly."
        return text
    raise ValueError("unknown hint level %r" % level)


@dataclass(frozen=True)
class RecoveryConfig:
    base_agent: str = "qwen3_5_35b_a3b"
    levels: Tuple[str, ...] = HINT_LEVELS
    attempts_per_level: int = 8
    teacher_agent: Optional[str] = None
    teacher_attempts: int = 1
    teacher_level: str = "l3_state_diff"
    max_recovery_steps: int = 100
    observation_ops: Tuple[str, ...] = ("confirm", "verify")
    leak_policy: str = "drop"
    leak_ngram: int = 5
    leak_patterns: Tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.leak_policy not in LEAK_POLICIES:
            raise ValueError("leak policy must be one of %s" % (LEAK_POLICIES,))
        for level in tuple(self.levels) + (self.teacher_level,):
            if level not in HINT_LEVELS:
                raise ValueError("unknown hint level %r" % level)

    @classmethod
    def from_yaml(cls, path: Union[str, Path]) -> "RecoveryConfig":
        raw = (yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {})["recovery"]
        teacher = raw.get("teacher") or {}
        leak = raw.get("leak_filter") or {}
        return cls(
            base_agent=str(raw["base_agent"]),
            levels=tuple(raw["hint_levels"]),
            attempts_per_level=int(raw["attempts_per_level"]),
            teacher_agent=teacher.get("agent") or None,
            teacher_attempts=int(teacher.get("attempts", 1)),
            teacher_level=str(teacher.get("hint_level", "l3_state_diff")),
            max_recovery_steps=int(raw["max_recovery_steps"]),
            observation_ops=tuple(raw.get("observation_ops") or ("confirm", "verify")),
            leak_policy=str(leak.get("policy", "drop")),
            leak_ngram=int(leak.get("ngram", 5)),
            leak_patterns=tuple(leak.get("patterns") or ()),
        )


@dataclass
class RecoveryAttempt:
    level: str
    attempt: int
    steps: List[Dict[str, Any]]
    verified: bool
    teacher: str = "self"
    leaks: List[str] = field(default_factory=list)


@dataclass
class RecoveryResult:
    sample_id: str
    accepted: Optional[RecoveryAttempt]
    attempts: List[RecoveryAttempt] = field(default_factory=list)

    @property
    def hint_level(self) -> Optional[str]:
        return self.accepted.level if self.accepted else None


Policy = Callable[
    [str, Sequence[Mapping[str, Any]], Any, str], Any
]


def _words(text: str) -> List[str]:
    return re.findall(r"\w+", text.lower())


def _context_text(case: Mapping[str, Any]) -> str:
    history = case["input"]["history"]
    return " ".join(
        [str(case["input"]["instruction"])]
        + [json.dumps(h.get("action"), ensure_ascii=False) for h in history]
    ).lower()


def hint_leaks(
    hint: str, steps: Sequence[Mapping[str, Any]], case: Mapping[str, Any], config: RecoveryConfig
) -> List[str]:
    thoughts = "\n".join(str(s.get("thought") or "") for s in steps).lower()
    if not thoughts.strip():
        return []
    context = _context_text(case)
    out: List[str] = []
    root = case.get("root_cause_action_index")
    if root is not None and re.search(r"\bstep\s*#?\s*%d\b" % int(root), thoughts):
        out.append("step:%s" % root)
    affected = (case.get("evidence_refs") or {}).get("affected_write_ref") or {}
    if affected.get("tbl") and str(affected["tbl"]).lower() in hint.lower():
        table = str(affected["tbl"]).lower()
        if re.search(r"\b%s\b" % re.escape(table), thoughts) and table not in context:
            out.append("table:%s" % affected["tbl"])
    words, seen = _words(hint), set()
    thought_words = " %s " % " ".join(_words(thoughts))
    context_words = " %s " % " ".join(_words(context))
    for i in range(len(words) - config.leak_ngram + 1):
        gram = " ".join(words[i : i + config.leak_ngram])
        if gram in seen:
            continue
        seen.add(gram)
        if " %s " % gram in thought_words and " %s " % gram not in context_words:
            out.append("ngram:%s" % gram)
    for pattern in config.leak_patterns:
        if re.search(pattern, thoughts, re.IGNORECASE):
            out.append("pattern:%s" % pattern)
    return out


def redact_leaks(
    hint: str, steps: Sequence[Mapping[str, Any]], case: Mapping[str, Any], config: RecoveryConfig
) -> List[Dict[str, Any]]:
    out = []
    for step in steps:
        sentences = re.split(r"(?<=[.!?\n])\s+", str(step.get("thought") or ""))
        kept = [s for s in sentences if not hint_leaks(hint, [{"thought": s}], case, config)]
        out.append(dict(step, thought=" ".join(kept).strip()))
    return out


def generate_recovery(
    case: Mapping[str, Any],
    *,
    restore: Callable[[str, str], Any],
    policy: Policy,
    step: Callable[[Any], Any],
    verify: Callable[[], Optional[bool]],
    config: RecoveryConfig,
    teacher_policy: Optional[Policy] = None,
) -> RecoveryResult:
    result = RecoveryResult(sample_id=str(case["sample_id"]), accepted=None)
    instruction = case["input"]["instruction"]
    base_history = list(case["input"]["history"])
    first_index = int(case["cut_action_index"]) + 1
    schedule = [(level, policy, "self", config.attempts_per_level) for level in config.levels]
    if teacher_policy is not None and config.teacher_attempts > 0:
        schedule.append((config.teacher_level, teacher_policy, "external", config.teacher_attempts))
    for level, current_policy, teacher, attempts in schedule:
        hint = hint_text(level, case)
        for attempt in range(1, attempts + 1):
            observation = restore(hint, teacher)
            history = list(base_history)
            steps: List[Dict[str, Any]] = []
            for _ in range(config.max_recovery_steps):
                action = current_policy(instruction, history, observation, hint)
                if action is None:
                    break
                index = first_index + len(steps)
                steps.append(
                    {
                        "action_index": index,
                        "thought": getattr(action, "thought", None)
                        if not isinstance(action, Mapping)
                        else action.get("thought"),
                        "action": action,
                    }
                )
                observation = step(action)
                history.append(
                    {"action_index": index, "action": action, "observation_ref": "obs:%d" % index}
                )
                if isinstance(action, Mapping) and action.get("type") == "done":
                    break
            record = RecoveryAttempt(
                level=level,
                attempt=attempt,
                steps=steps,
                verified=verify() is True,
                teacher=teacher,
            )
            result.attempts.append(record)
            if not record.verified:
                continue
            if config.leak_policy != "off":
                record.leaks = hint_leaks(hint, steps, case, config)
            if record.leaks and config.leak_policy == "drop":
                return result
            if record.leaks and config.leak_policy == "resample":
                continue
            if record.leaks:
                record = replace(record, steps=redact_leaks(hint, steps, case, config))
            result.accepted = record
            return result
    return result


def recovery_sample(
    case: Mapping[str, Any], result: RecoveryResult, loss_weights: Mapping[str, float]
) -> Optional[Dict[str, Any]]:
    if result.accepted is None:
        return None
    return {
        "schema_version": RECOVERY_SAMPLE_VERSION,
        "sample_id": str(case["sample_id"]),
        "sample_kind": "recovery",
        "workflow_id": str(case["workflow_id"]),
        "rollout_id": str(case["rollout_id"]),
        "agent": str(case["agent"]),
        "split": str(case["split"]),
        "root_cause_action_index": int(case["root_cause_action_index"]),
        "depth": int(case["depth"]),
        "cut_action_index": int(case["cut_action_index"]),
        "input": {
            "instruction": str(case["input"]["instruction"]),
            "history": list(case["input"]["history"]),
        },
        "target": {
            "steps": [
                {k: s[k] for k in ("action_index", "thought", "action")}
                for s in result.accepted.steps
            ]
        },
        "loss_weights": dict(loss_weights),
        "hint_level": result.accepted.level,
        "teacher": result.accepted.teacher,
        "provenance": {
            "attempts": len(result.attempts),
            "redacted_leaks": list(result.accepted.leaks),
        },
    }


def sample_key(sample: Mapping[str, Any]) -> str:
    payload = {
        "l": sample["input"]["instruction"],
        "h": [h.get("action") for h in sample["input"]["history"]],
        "r": [s.get("action") for s in sample["target"]["steps"]],
    }
    encoded = json.dumps(payload, sort_keys=True, ensure_ascii=False, default=str)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()
