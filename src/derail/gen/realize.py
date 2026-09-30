"""Instruction realization and round-trip fidelity (doc v1.2 section 7.6; v0.2 section 12).

- ``graph_summary(task_ir)``: the graph as prose facts (ops, apps, literals, side effects),
  without gold values.
- ``hidden_values(task_ir, gold)``: every gold value string the instruction must not leak.
- ``build_realization_fields(...)`` / ``parse_realization_reply(...)``.
- ``round_trip_compare(source_ir, reextracted_ir, gold, instruction, config)``: node-op
  recall, edge recall, unauthorized nodes, literal consistency, gold leak -> verdict.

Model calls go through the same approval-gated client as extraction (``derail.longhorizon
.extraction.OpenAICompatibleClient``); everything here is testable without it.
"""

from __future__ import annotations

import json
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Union

import yaml

from derail.longhorizon.extraction import ExtractionParseError, ExtractorConfig, render_prompt
from derail.world.facts import date_part, normalize_value


@dataclass(frozen=True)
class RealizationConfig:
    base: ExtractorConfig
    min_node_op_recall: float
    min_edge_recall: float
    max_unauthorized_nodes: int
    forbid_gold_leak: bool
    default_length_words: tuple
    realizations_per_task: int = 1

    @classmethod
    def from_yaml(cls, path: Union[str, Path], repo_root: Union[str, Path]) -> "RealizationConfig":
        raw = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
        rt = raw.get("round_trip") or {}
        return cls(
            base=ExtractorConfig.from_yaml(Path(path), Path(repo_root)),
            min_node_op_recall=float(rt.get("min_node_op_recall", 0.8)),
            min_edge_recall=float(rt.get("min_edge_recall", 0.7)),
            max_unauthorized_nodes=int(rt.get("max_unauthorized_nodes", 1)),
            forbid_gold_leak=bool(rt.get("forbid_gold_leak", True)),
            default_length_words=tuple(
                (raw.get("style") or {}).get("default_length_words", (25, 90))
            ),
            realizations_per_task=int(raw.get("realizations_per_task", 1)),
        )


def literals_of(task_ir: Mapping[str, Any]) -> List[Any]:
    out: List[Any] = []
    for node in task_ir["nodes"]:
        for port in node.get("inputs", ()):
            if "literal" in port and port["literal"] not in out:
                out.append(port["literal"])
    return out


def graph_summary(task_ir: Mapping[str, Any]) -> str:
    lines = []
    for node in task_ir["nodes"]:
        literals = ", ".join(
            "%s=%r" % (p["port_id"], p["literal"]) for p in node.get("inputs", ()) if "literal" in p
        )
        effects = ", ".join(
            e["effect_type"] + "(" + e["reversibility_class"] + ")"
            for e in node.get("side_effects", ())
        )
        lines.append(
            "- %s [%s on %s]: %s%s%s"
            % (
                node["node_id"],
                node["op"],
                node["app"],
                node.get("semantic_goal", ""),
                (" | user-known constants: " + literals) if literals else "",
                (" | effects: " + effects) if effects else "",
            )
        )
    for edge in task_ir["edges"]:
        if edge["kind"] == "control_dependency":
            lines.append(
                "- %s happens only if %s"
                % (edge["to"]["node_id"], edge.get("predicate", "a condition holds"))
            )
    return "\n".join(lines)


_SHORT_WORD_CHARS = 8
_WORDS = re.compile(r"[a-z0-9][a-z0-9'-]*")
_YEAR = re.compile(r"(19|20)\d{2}")


def hidden_values(
    task_ir: Mapping[str, Any],
    gold: Optional[Mapping[str, Any]],
    public_texts: Sequence[str] = (),
) -> List[str]:
    """Gold values that are not user-known: not an input literal, not stated (in any word
    order) in the task's own or its parent seeds' instructions (``public_texts``), not a bare
    year, and not a single short common word (a project called "sales" is not a leak).
    These must be looked up, not stated."""

    known = {str(normalize_value(v)).lower() for v in literals_of(task_ir)}
    public = " ".join(
        [str(task_ir.get("instruction") or "")] + [str(t) for t in public_texts] + sorted(known)
    ).lower()
    public_words = set(_WORDS.findall(public))
    hidden: List[str] = []
    for entry in (gold or {}).get("values", ()):
        value = entry.get("value")
        if not isinstance(value, (str, int, float)) or isinstance(value, bool):
            continue
        text = str(normalize_value(value))
        lowered = text.lower()
        if not text or len(text) < 3 or text in hidden or lowered in known:
            continue
        if lowered in public or _YEAR.fullmatch(text):
            continue
        words = _WORDS.findall(lowered)
        if words and set(words) <= public_words:
            continue  # the same name in another word order ("Q2 team morale initiative")
        if (
            isinstance(value, str)
            and len(words) <= 1
            and text.isalpha()
            and len(text) < _SHORT_WORD_CHARS
        ):
            continue
        hidden.append(text)
    return hidden


def build_realization_fields(
    task_ir: Mapping[str, Any],
    gold: Optional[Mapping[str, Any]],
    style_stats: Mapping[str, Any],
    public_context: Sequence[str] = (),
) -> Dict[str, str]:
    return {
        "graph_summary": graph_summary(task_ir),
        "public_context": "\n".join("- %s" % c for c in public_context)
        or "(none beyond the graph)",
        "hidden_values": "\n".join("- %s" % v for v in hidden_values(task_ir, gold, public_context))
        or "(none)",
        "style_stats": json.dumps(dict(style_stats), ensure_ascii=False),
    }


def parse_realization_reply(text: str) -> Dict[str, Any]:
    fenced = re.search(r"```(?:json)?\s*(\{.*\})\s*```", text, re.DOTALL)
    candidate = fenced.group(1) if fenced else text[text.find("{") : text.rfind("}") + 1]
    try:
        parsed = json.loads(candidate)
    except (json.JSONDecodeError, ValueError) as exc:
        raise ExtractionParseError("realization reply is not JSON: %s" % exc) from exc
    if not isinstance(parsed, dict) or not parsed.get("instruction"):
        raise ExtractionParseError("realization reply lacks instruction")
    return parsed


def style_stats_from_instructions(instructions: Sequence[str]) -> Dict[str, Any]:
    lengths = sorted(len(i.split()) for i in instructions) or [0]
    apps = Counter()
    for text in instructions:
        for name in re.findall(
            r"\b(Hooli[A-Za-z]+|Gringotts|LockedIn|SprintBoard|Cheskepdia|BatBucks|OddsMarket|SpeedTax|TableFind|eTaxi|HangryDash|Kwik-E-Mart|Dinoco)\b",
            text,
        ):
            apps[name] += 1
    return {
        "length_words_p25_p75": [lengths[len(lengths) // 4], lengths[3 * len(lengths) // 4]],
        "median_words": lengths[len(lengths) // 2],
        "first_person_share": round(
            sum(1 for i in instructions if re.search(r"\b(I|my|me)\b", i))
            / max(len(instructions), 1),
            2,
        ),
        "app_mentions_per_instruction": round(sum(apps.values()) / max(len(instructions), 1), 2),
    }


def _leaks(instruction: str, hidden: Sequence[str]) -> List[str]:
    lowered = instruction.lower()
    out = []
    for value in hidden:
        day = date_part(value)
        needle = day or value.lower()
        if len(needle) >= 4 and needle in lowered:
            out.append(value)
    return out


def round_trip_compare(
    source_ir: Mapping[str, Any],
    reextracted: Optional[Mapping[str, Any]],
    gold: Optional[Mapping[str, Any]],
    instruction: str,
    config: RealizationConfig,
    persona_literals: Sequence[Any] = (),
    public_texts: Sequence[str] = (),
) -> Dict[str, Any]:
    """Fidelity verdict of a realized instruction (v0.2 section 12.3).  Persona identity
    literals are known to the assistant and need not appear in the instruction;
    ``public_texts`` (parent seed instructions) are not leaks."""

    src_ops = Counter(n["op"] for n in source_ir["nodes"])
    persona = {str(normalize_value(v)).lower() for v in persona_literals}
    literal_texts = [
        str(normalize_value(v)).lower()
        for v in literals_of(source_ir)
        if str(normalize_value(v)).lower() not in persona
    ]
    missing_literals = [t for t in literal_texts if t and t not in instruction.lower()]
    leaks = (
        _leaks(instruction, hidden_values(source_ir, gold, public_texts))
        if config.forbid_gold_leak
        else []
    )
    result: Dict[str, Any] = {
        "instruction": instruction,
        "missing_literals": missing_literals,
        "gold_leaks": leaks,
        "node_op_recall": None,
        "edge_recall": None,
        "unauthorized_nodes": None,
    }
    if reextracted is not None:
        re_ops = Counter(n["op"] for n in reextracted["nodes"])
        recalled = sum(min(src_ops[op], re_ops.get(op, 0)) for op in src_ops)
        result["node_op_recall"] = round(recalled / max(sum(src_ops.values()), 1), 3)
        result["edge_recall"] = round(
            min(len(reextracted["edges"]), len(source_ir["edges"]))
            / max(len(source_ir["edges"]), 1),
            3,
        )
        result["unauthorized_nodes"] = max(0, sum(re_ops.values()) - sum(src_ops.values()))
    ok = not missing_literals and not leaks
    if reextracted is not None:
        ok = (
            ok
            and result["node_op_recall"] >= config.min_node_op_recall
            and result["edge_recall"] >= config.min_edge_recall
            and result["unauthorized_nodes"] <= config.max_unauthorized_nodes
        )
    result["verdict"] = "accepted" if ok else "re_realize"
    return result


def render_realization_prompt(config: RealizationConfig, fields: Mapping[str, str]) -> str:
    return render_prompt(config.base.prompt_user.read_text(encoding="utf-8"), fields)
