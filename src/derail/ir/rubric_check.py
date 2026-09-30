from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, Union

import yaml

from derail.world.facts import date_part, normalize_value

CHECK_VERSION = "rubric-check/1.0"


@dataclass(frozen=True)
class RubricCheckConfig:
    date_formats: Tuple[str, ...]
    default_year: int
    amount_pattern: str
    email_pattern: str
    quoted_pattern: str
    template_marker_pattern: str
    use_instruction: bool
    variable_number_tolerance: float
    match_threshold: float
    min_expectations: int
    min_rubric_weight: float
    lexical_enabled: bool = True
    keyword_threshold: float = 0.5
    criteria_threshold: float = 0.7
    min_word_length: int = 4
    stopwords: Tuple[str, ...] = ()

    @classmethod
    def from_yaml(cls, path: Union[str, Path]) -> "RubricCheckConfig":
        raw = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
        if raw.get("schema_version") != "rubric-check-config/1.0":
            raise ValueError("unsupported rubric check config %r" % raw.get("schema_version"))
        exp, verdict = raw.get("expectations", {}), raw.get("verdict", {})
        return cls(
            date_formats=tuple(exp.get("date_formats", ("%Y-%m-%d",))),
            default_year=int(exp.get("default_year", 2026)),
            amount_pattern=str(exp["amount_pattern"]),
            email_pattern=str(exp["email_pattern"]),
            quoted_pattern=str(exp["quoted_pattern"]),
            template_marker_pattern=str(exp.get("template_marker_pattern", "<[^<>]{1,40}>")),
            use_instruction=bool(exp.get("use_instruction", False)),
            variable_number_tolerance=float(exp.get("variable_number_tolerance", 0.005)),
            match_threshold=float(verdict.get("match_threshold", 1.0)),
            min_expectations=int(verdict.get("min_expectations", 1)),
            min_rubric_weight=float(verdict.get("min_rubric_weight", 0.0)),
            lexical_enabled=bool((verdict.get("lexical") or {}).get("enabled", True)),
            keyword_threshold=float((verdict.get("lexical") or {}).get("keyword_threshold", 0.5)),
            criteria_threshold=float((verdict.get("lexical") or {}).get("criteria_threshold", 0.7)),
            min_word_length=int((verdict.get("lexical") or {}).get("min_word_length", 4)),
            stopwords=tuple(
                str(w).lower() for w in (verdict.get("lexical") or {}).get("stopwords", ())
            ),
        )


_MONTH_DATE = re.compile(
    r"\b((?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Sept|Oct|Nov|Dec)[a-z]*\.?\s+\d{1,2}(?:st|nd|rd|th)?"
    r"(?:,?\s+\d{4})?|\d{4}-\d{2}-\d{2}|\d{1,2}/\d{1,2}(?:/\d{4})?)\b"
)


def _parse_date(text: str, config: RubricCheckConfig) -> Optional[str]:
    cleaned = re.sub(r"(\d)(st|nd|rd|th)", r"\1", text.replace("Sept", "Sep").replace(".", ""))
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    for fmt in config.date_formats:
        try:
            parsed = datetime.strptime(cleaned, fmt)
        except ValueError:
            continue
        if "%Y" not in fmt:
            parsed = parsed.replace(year=config.default_year)
        return parsed.strftime("%Y-%m-%d")
    return None


def _amount(text: str) -> float:
    return float(text.replace(",", ""))


_TODAY = re.compile(r"\$\{TODAY([+-]\d+)?\}")


def expand_variables(variables: Mapping[str, Any], reference_time: str) -> Dict[str, Any]:
    today = datetime.strptime(str(reference_time)[:10], "%Y-%m-%d")

    def _sub(match: "re.Match[str]") -> str:
        return (today + timedelta(days=int(match.group(1) or 0))).strftime("%Y-%m-%d")

    return {
        name: (
            _TODAY.sub(_sub, value) if isinstance(value, str) and _TODAY.search(value) else value
        )
        for name, value in variables.items()
    }


def extract_expectations(
    task: Mapping[str, Any], variables: Mapping[str, Any], config: RubricCheckConfig
) -> List[Dict[str, Any]]:
    sources: List[Tuple[str, Optional[int], float, str]] = []
    if config.use_instruction:
        sources.append(("instruction", None, 1.0, str(task.get("instruction", ""))))
    for index, rubric in enumerate(task.get("grading", {}).get("rubrics", [])):
        sources.append(
            ("rubric", index, float(rubric.get("weight", 0.0)), str(rubric.get("criterion", "")))
        )
    found: Dict[Tuple[str, Any], Dict[str, Any]] = {}

    def add(
        kind: str, value: Any, source: str, index: Optional[int], weight: float, raw: str
    ) -> None:
        key = (kind, value)
        if key in found and found[key]["weight"] >= weight:
            return
        found[key] = {
            "kind": kind,
            "value": value,
            "source": source,
            "rubric_index": index,
            "weight": weight,
            "raw": raw,
        }

    amount_re, email_re, quoted_re = (
        re.compile(config.amount_pattern),
        re.compile(config.email_pattern),
        re.compile(config.quoted_pattern),
    )
    template_re = re.compile(config.template_marker_pattern)
    variable_items = [
        (name, value)
        for name, value in variables.items()
        if isinstance(value, (str, int, float)) and not isinstance(value, bool)
    ]
    for source, index, weight, text in sources:
        for match in _MONTH_DATE.finditer(text):
            iso = _parse_date(match.group(1), config)
            if iso:
                add("date", iso, source, index, weight, match.group(1))
        for match in amount_re.finditer(text):
            add("amount", _amount(match.group(1)), source, index, weight, match.group(0))
        for match in email_re.finditer(text):
            add("email", match.group(0).lower(), source, index, weight, match.group(0))
        for match in quoted_re.finditer(text):
            if template_re.search(match.group(1)):
                continue
            add("text", match.group(1).strip(), source, index, weight, match.group(0))
        lowered = text.lower()
        for name, value in variable_items:
            if isinstance(value, str):
                looks_like_value = any(ch.isdigit() or ch.isupper() or ch in "@ " for ch in value)
                if looks_like_value and len(value) >= 4 and value.lower() in lowered:
                    add("variable", value, source, index, weight, name)
            else:
                for match in amount_re.finditer(text):
                    number = _amount(match.group(1))
                    if (
                        abs(number - float(value))
                        <= abs(float(value)) * config.variable_number_tolerance
                    ):
                        add("variable", float(value), source, index, weight, name)
    return sorted(
        found.values(),
        key=lambda e: (e["source"] != "rubric", e["rubric_index"] or 0, e["kind"], str(e["value"])),
    )


def _flatten(value: Any) -> Iterable[Any]:
    if isinstance(value, Mapping):
        for item in value.values():
            yield from _flatten(item)
    elif isinstance(value, (list, tuple)):
        for item in value:
            yield from _flatten(item)
    else:
        yield value


def gold_values(gold_lineage: Mapping[str, Any]) -> List[Any]:
    out: List[Any] = []
    for entry in gold_lineage.get("values", ()):
        out.extend(_flatten(entry["value"]))
    for write in gold_lineage.get("writes_gold", ()):
        out.extend(_flatten(write.get("value")))
    return [v for v in out if v is not None]


def context_values(
    task_ir: Mapping[str, Any], gold_lineage: Mapping[str, Any], connections: Mapping[str, Any]
) -> List[Any]:
    derived: Dict[Tuple[str, str], Any] = {
        (v["node_id"], v["name"]): v["value"] for v in gold_lineage.get("values", ())
    }
    seen: set = set()
    out: List[Any] = []
    for node in task_ir.get("nodes", ()):
        own = "derived:%s:" % node["node_id"]
        refs = list(node.get("writes", ())) + [
            r for r in node.get("reads", ()) if str(r["entity_ref"]).startswith(own)
        ]
        for ref in refs:
            table = str(ref["table"])
            app, name = table.split(".", 1)
            entity = str(ref["entity_ref"])
            rowid: Optional[int] = None
            if entity.startswith("derived:"):
                _, node_id, value_name = entity.split(":", 2)
                value = derived.get((node_id, value_name))
                if isinstance(value, Mapping):
                    value = value.get("id")
                if isinstance(value, int) and not isinstance(value, bool):
                    rowid = value
            elif ":" in entity and not entity.endswith(":*"):
                try:
                    rowid = int(entity.rsplit(":", 1)[1])
                except ValueError:
                    rowid = None
            conn = connections.get(app)
            if rowid is None or conn is None or (table, rowid) in seen:
                continue
            seen.add((table, rowid))
            row = conn.execute('SELECT * FROM "%s" WHERE rowid = ?' % name, (rowid,)).fetchone()
            if row is not None:
                out.extend(
                    v
                    for v in (
                        list(row) if not hasattr(row, "keys") else [row[k] for k in row.keys()]
                    )
                    if v is not None and not isinstance(v, (bytes, bytearray))
                )
    return out


def expectation_matches(
    expectation: Mapping[str, Any], values: Sequence[Any], config: RubricCheckConfig
) -> bool:
    kind, expected = expectation["kind"], expectation["value"]
    for value in values:
        if kind == "date":
            if date_part(value) == expected:
                return True
            if isinstance(value, str) and expected in value:
                return True
        elif kind in ("amount",) or (kind == "variable" and isinstance(expected, float)):
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                if abs(float(value) - float(expected)) <= max(
                    abs(float(expected)) * config.variable_number_tolerance, 0.005
                ):
                    return True
            elif isinstance(value, str):
                for match in re.finditer(config.amount_pattern, value):
                    if abs(_amount(match.group(1)) - float(expected)) <= 0.005:
                        return True
        else:
            text = str(normalize_value(value)) if value is not None else ""
            if str(expected).lower() == text.lower() or (
                len(str(expected)) >= 4 and str(expected).lower() in text.lower()
            ):
                return True
    return False


_WORD = re.compile(r"[A-Za-z][A-Za-z'\-]{2,}")


def rubric_keywords(criterion: str, config: RubricCheckConfig) -> List[str]:
    words = []
    for word in _WORD.findall(criterion):
        lowered = word.lower().strip("'-")
        if (
            len(lowered) >= config.min_word_length
            and lowered not in config.stopwords
            and lowered not in words
        ):
            words.append(lowered)
    return words


def lexical_check(
    task: Mapping[str, Any],
    task_ir: Optional[Mapping[str, Any]],
    values: Sequence[Any],
    config: RubricCheckConfig,
) -> Dict[str, Any]:
    corpus_parts = [str(normalize_value(v)) for v in values if v is not None]
    if task_ir:
        for node in task_ir.get("nodes", ()):
            corpus_parts.append(str(node.get("semantic_goal", "")))
            corpus_parts.extend(
                str(p.get("name", "")).replace("_", " ") for p in node.get("produces", ())
            )
        corpus_parts.append(str(task_ir.get("provenance", {}).get("notes", "")))
    corpus = " ".join(corpus_parts).lower()
    criteria = []
    for index, rubric in enumerate(task.get("grading", {}).get("rubrics", [])):
        weight = float(rubric.get("weight", 0.0))
        keywords = rubric_keywords(str(rubric.get("criterion", "")), config)
        hits = [k for k in keywords if k in corpus or (k.endswith("s") and k[:-1] in corpus)]
        coverage = len(hits) / len(keywords) if keywords else 1.0
        criteria.append(
            {
                "rubric_index": index,
                "weight": weight,
                "keywords": keywords,
                "hits": hits,
                "coverage": round(coverage, 3),
                "satisfied": coverage >= config.keyword_threshold,
            }
        )
    total = sum(c["weight"] for c in criteria) or float(len(criteria) or 1)
    satisfied = (
        sum(c["weight"] for c in criteria if c["satisfied"])
        if any(c["weight"] for c in criteria)
        else float(sum(1 for c in criteria if c["satisfied"]))
    )
    return {"criteria": criteria, "satisfied_share": round(satisfied / total, 4) if total else 0.0}


def check(
    task_id: str,
    gold_lineage: Mapping[str, Any],
    expectations: Sequence[Mapping[str, Any]],
    config: RubricCheckConfig,
    extra_values: Sequence[Any] = (),
    task: Optional[Mapping[str, Any]] = None,
    task_ir: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    values = gold_values(gold_lineage) + list(extra_values)
    counted = [
        e
        for e in expectations
        if e["source"] != "rubric" or e["weight"] >= config.min_rubric_weight
    ]
    matched = [e for e in counted if expectation_matches(e, values, config)]
    unmatched = [e for e in counted if e not in matched]
    ratio = (len(matched) / len(counted)) if counted else 0.0
    lexical: Optional[Dict[str, Any]] = None
    if len(counted) < config.min_expectations:
        verdict, code = "needs_review", "NO_COMPARABLE_EXPECTATIONS"
        if (
            config.lexical_enabled
            and task is not None
            and gold_lineage.get("final_verifier_passed") is not False
        ):
            lexical = lexical_check(task, task_ir, values, config)
            if lexical["satisfied_share"] >= config.criteria_threshold:
                verdict, code = "auto_validated", "LEXICAL_MATCH"
    elif ratio >= config.match_threshold and gold_lineage.get("final_verifier_passed") is not False:
        verdict, code = "auto_validated", None
    elif gold_lineage.get("final_verifier_passed") is False and ratio >= config.match_threshold:
        verdict, code = "needs_review", "FINAL_VERIFIER_FAILED"
    else:
        verdict, code = "needs_review", "IR_RUBRIC_MISMATCH"
    return {
        "schema_version": CHECK_VERSION,
        "task_id": task_id,
        "verdict": verdict,
        "code": code,
        "matched": matched,
        "unmatched": unmatched,
        "match_ratio": round(ratio, 4),
        "gold_value_count": len(values),
        "lexical": lexical,
        "final_verifier_passed": gold_lineage.get("final_verifier_passed"),
    }
