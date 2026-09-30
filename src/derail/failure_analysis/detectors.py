"""Root-cause detectors over the three ledgers (execution doc v1.2 sections 2.3, 9.2).

Each detector returns candidates ``{action_index, node_id, evidence_pattern, evidence[],
confidence, detail}``; ``pick_root_cause`` takes the earliest high-confidence candidate with
state > parameter > omission on ties.  All comparisons are on facts and values, never on
action sequences (section 6.4).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Mapping, Optional, Set, Tuple

from derail.world.facts import date_part, normalize_value
from derail.world.volatile import VolatileColumns

Fact = Tuple[str, str, str]


@dataclass
class Candidate:
    detector: str
    action_index: int
    evidence_pattern: str
    confidence: float
    node_id: Optional[str] = None
    evidence: List[Dict[str, Any]] = field(default_factory=list)
    detail: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "action_index": self.action_index,
            "node_id": self.node_id,
            "evidence_pattern": self.evidence_pattern,
            "evidence": self.evidence,
            "confidence": self.confidence,
        }


# ------------------------------------------------------------------------------ helpers
def _same(a: Any, b: Any) -> bool:
    na, nb = normalize_value(a), normalize_value(b)
    if na == nb:
        return True
    da, db = date_part(a), date_part(b)
    return (
        da is not None
        and da == db
        and (isinstance(na, str) and isinstance(nb, str) and len(na) <= 10 or len(str(nb)) <= 10)
    )


def edit_distance(a: str, b: str, limit: int = 3) -> int:
    if abs(len(a) - len(b)) > limit:
        return limit + 1
    previous = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        current = [i]
        for j, cb in enumerate(b, 1):
            current.append(min(previous[j] + 1, current[j - 1] + 1, previous[j - 1] + (ca != cb)))
        previous = current
    return previous[-1]


def gold_write_index(
    gold: Mapping[str, Any], volatile: Optional[VolatileColumns]
) -> Dict[Tuple[str, str], List[Dict[str, Any]]]:
    """``(table, entity) -> [gold write cells]`` with volatile cells dropped."""

    out: Dict[Tuple[str, str], List[Dict[str, Any]]] = {}
    for write in gold.get("writes_gold", ()):
        if write.get("volatile") or (
            volatile and volatile.is_volatile(write["table"], write["column"])
        ):
            continue
        out.setdefault((write["table"], write["entity"]), []).append(write)
    return out


def gold_scalar_values(gold: Mapping[str, Any]) -> List[Tuple[str, str, Any]]:
    """``(node_id, name, value)`` for scalar gold values (strings, numbers) worth matching."""

    out = []
    for entry in gold.get("values", ()):
        value = entry.get("value")
        if isinstance(value, (str, int, float)) and not isinstance(value, bool):
            if isinstance(value, str) and len(value) < 3:
                continue
            out.append((entry["node_id"], entry["name"], value))
    for write in gold.get("writes_gold", ()):
        value = write.get("value")
        if (
            isinstance(value, (str, int, float))
            and not isinstance(value, bool)
            and not write.get("volatile")
        ):
            out.append((write["node_id"], write["column"], value))
    return out


def observed_facts(
    trace: Mapping[str, Any], upto: Optional[int] = None, after: Optional[int] = None
) -> List[Tuple[int, Dict[str, Any]]]:
    out = []
    for step in trace["steps"]:
        index = int(step["action_index"])
        if upto is not None and index > upto:
            continue
        if after is not None and index <= after:
            continue
        for event in step.get("observations", ()):
            for fact in event.get("facts", ()):
                out.append((index, fact))
        for page in step.get("pages", ()):
            for cell in page.get("rendered_fields", ()):
                out.append(
                    (
                        index,
                        {
                            "table": "%s.%s" % (page["app"], cell["table"])
                            if "." not in cell["table"]
                            else cell["table"],
                            "column": cell["column"],
                            "entity": cell.get("entity"),
                            "value": cell.get("value"),
                        },
                    )
                )
    return out


def _row_cells(row: Mapping[str, Any]) -> Dict[str, Any]:
    try:
        return json.loads(row["new_json"]) if row.get("new_json") else {}
    except ValueError:
        return {}


# ------------------------------------------------------------------------ state detector
def state_candidates(
    trace: Mapping[str, Any], gold: Mapping[str, Any], volatile: Optional[VolatileColumns] = None
) -> List[Candidate]:
    """Writes outside ``Writes_gold``: wrong entity, wrong value, extra write, duplicate."""

    index = gold_write_index(gold, volatile)
    by_table_col_value: Dict[Tuple[str, str, str], List[Tuple[str, Dict[str, Any]]]] = {}
    for (table, entity), cells in index.items():
        for cell in cells:
            by_table_col_value.setdefault(
                (table, cell["column"], json.dumps(normalize_value(cell["value"]), sort_keys=True)),
                [],
            ).append((entity, cell))
    node_of_entity = {key: cells[0]["node_id"] for key, cells in index.items()}
    displayed = observed_facts(trace)
    matched_gold: Set[Tuple[str, str, str]] = set()
    out: List[Candidate] = []
    for step in trace["steps"]:
        t = int(step["action_index"])
        for row in step.get("delta", ()):
            table = "%s.%s" % (row["db"], row["tbl"])
            entity = "%s:%s" % (row["tbl"], row["rowid"])
            cells = _row_cells(row)
            if row["op"] == "DELETE":
                cells = {"*": None}
            gold_cells = index.get((table, entity))
            if gold_cells is None:
                # Not a gold entity: a created row matching a gold row template counts as gold.
                template_hit = None
                for (g_table, g_entity), g_cells in index.items():
                    if g_table != table or not all(c.get("old_value") is None for c in g_cells):
                        continue
                    if all(
                        _same(cells.get(c["column"]), c["value"])
                        for c in g_cells
                        if c["column"] in cells
                    ):
                        template_hit = (g_entity, g_cells)
                        break
                if template_hit is not None:
                    key = (table, template_hit[0], "row")
                    if key in matched_gold:
                        out.append(
                            Candidate(
                                "state",
                                t,
                                "duplicate_gold_write",
                                0.8,
                                node_of_entity.get((table, template_hit[0])),
                                [
                                    {
                                        "kind": "changelog",
                                        "ref": row["seq"],
                                        "detail": "second write of gold row %s" % template_hit[0],
                                    }
                                ],
                            )
                        )
                    matched_gold.add(key)
                    continue
                for column, value in cells.items():
                    if volatile and volatile.is_volatile(table, column):
                        continue
                    hits = by_table_col_value.get(
                        (table, column, json.dumps(normalize_value(value), sort_keys=True))
                    )
                    if not hits and row["op"] == "UPDATE":
                        # same table and column as a gold write on another row, other value
                        hits = [
                            (g_entity, c)
                            for (g_table, g_entity), g_cells in index.items()
                            if g_table == table
                            for c in g_cells
                            if c["column"] == column and c.get("old_value") is not None
                        ]
                    if hits:
                        g_entity, g_cell = hits[0]
                        co_displayed = any(
                            f.get("entity") == g_entity and f.get("table") == table
                            for _, f in displayed
                            if _ < t
                        ) and any(
                            f.get("entity") == entity and f.get("table") == table
                            for _, f in displayed
                            if _ < t
                        )
                        out.append(
                            Candidate(
                                "state",
                                t,
                                "wrong_entity_co_displayed" if co_displayed else "wrong_entity",
                                0.9,
                                g_cell["node_id"],
                                [
                                    {
                                        "kind": "changelog",
                                        "ref": row["seq"],
                                        "detail": "%s.%s written on %s, gold entity %s"
                                        % (table, column, entity, g_entity),
                                    },
                                    {
                                        "kind": "gold",
                                        "ref": g_entity,
                                        "detail": g_cell.get("value"),
                                    },
                                ],
                                {"gold_entity": g_entity, "wrong_entity": entity, "column": column},
                            )
                        )
                        break
                else:
                    if cells and any(
                        not (volatile and volatile.is_volatile(table, c)) for c in cells
                    ):
                        out.append(
                            Candidate(
                                "state",
                                t,
                                "extra_write",
                                0.6,
                                None,
                                [
                                    {
                                        "kind": "changelog",
                                        "ref": row["seq"],
                                        "detail": "%s %s %s not in Writes_gold"
                                        % (row["op"], table, entity),
                                    }
                                ],
                                {"table": table, "entity": entity},
                            )
                        )
                continue
            for cell in gold_cells:
                column = cell["column"]
                if column not in cells:
                    continue
                key = (table, entity, column)
                if _same(cells[column], cell["value"]):
                    if key in matched_gold:
                        out.append(
                            Candidate(
                                "state",
                                t,
                                "duplicate_gold_write",
                                0.8,
                                cell["node_id"],
                                [
                                    {
                                        "kind": "changelog",
                                        "ref": row["seq"],
                                        "detail": "second gold-equivalent write of %s.%s"
                                        % (entity, column),
                                    }
                                ],
                            )
                        )
                    matched_gold.add(key)
                else:
                    out.append(
                        Candidate(
                            "state",
                            t,
                            "wrong_value_written",
                            0.85,
                            cell["node_id"],
                            [
                                {
                                    "kind": "changelog",
                                    "ref": row["seq"],
                                    "detail": "%s.%s = %r, gold %r"
                                    % (entity, column, cells[column], cell["value"]),
                                }
                            ],
                            {
                                "gold_value": cell["value"],
                                "wrong_value": cells[column],
                                "table": table,
                                "entity": entity,
                                "column": column,
                            },
                        )
                    )
    return out


# -------------------------------------------------------------------- parameter detector
def _param_texts(step: Mapping[str, Any]) -> List[str]:
    out = []
    for param in step.get("params", ()):
        value = param.get("value")
        if (
            isinstance(value, str)
            and len(value.strip()) >= 3
            and param.get("name") in ("text", "command", "value")
        ):
            out.append(value.strip())
    return out


def parameter_candidates(trace: Mapping[str, Any], gold: Mapping[str, Any]) -> List[Candidate]:
    """Typed values that disagree with a gold slot: transcription (edit distance <= 2), a
    seen-but-wrong value (stale / wrong pick), or a value never observed (hallucinated)."""

    golds = gold_scalar_values(gold)
    gold_strings = [(n, name, str(v)) for n, name, v in golds]
    out: List[Candidate] = []
    for step in trace["steps"]:
        t = int(step["action_index"])
        seen_before = observed_facts(trace, upto=t - 1)
        for text in _param_texts(step):
            if any(_same(text, v) for _, _, v in golds):
                continue
            # date / number tokens inside free text: compare token-wise against gold dates
            best = None
            for node_id, name, gold_text in gold_strings:
                if len(gold_text) < 4:
                    continue
                distance = edit_distance(text.lower(), gold_text.lower(), limit=2)
                if distance <= 2 and (best is None or distance < best[0]):
                    best = (distance, node_id, name, gold_text)
            if best is not None:
                out.append(
                    Candidate(
                        "parameter",
                        t,
                        "transcription_error",
                        0.9,
                        best[1],
                        [
                            {"kind": "param", "ref": t, "detail": text},
                            {
                                "kind": "gold",
                                "ref": "%s.%s" % (best[1], best[2]),
                                "detail": best[3],
                            },
                        ],
                        {
                            "gold_value": best[3],
                            "wrong_value": text,
                            "gold_node": best[1],
                            "gold_name": best[2],
                        },
                    )
                )
                continue
            # tokens that look like dates: the gold has a date slot with another date
            typed_date = None
            for token in text.replace(",", " ").split():
                if date_part(token):
                    typed_date = date_part(token)
                    break
            if typed_date:
                gold_dates = [(n, name, date_part(v)) for n, name, v in golds if date_part(v)]
                if gold_dates and typed_date not in {d for _, _, d in gold_dates}:
                    seen = any(date_part(f.get("value")) == typed_date for _, f in seen_before)
                    pattern = "stale_value_from_earlier_page" if seen else "hallucinated_value"
                    out.append(
                        Candidate(
                            "parameter",
                            t,
                            pattern,
                            0.75,
                            gold_dates[0][0],
                            [
                                {"kind": "param", "ref": t, "detail": text},
                                {
                                    "kind": "gold",
                                    "ref": "%s.%s" % gold_dates[0][:2],
                                    "detail": gold_dates[0][2],
                                },
                            ],
                            {
                                "gold_value": gold_dates[0][2],
                                "wrong_value": typed_date,
                                "gold_node": gold_dates[0][0],
                                "gold_name": gold_dates[0][1],
                            },
                        )
                    )
    return out


# --------------------------------------------------------------------- omission detector
def omission_candidates(
    trace: Mapping[str, Any], gold: Mapping[str, Any], task_ir: Mapping[str, Any]
) -> List[Candidate]:
    """A required read (resolved to concrete rows) never observed before a downstream write
    of its node's lineage happened."""

    from derail.ir.model import dag_index

    dag = dag_index(task_ir)
    required: Dict[str, List[Tuple[str, str, str]]] = {}
    for read in gold.get("resolved_reads", ()):
        if not read.get("entity_set") or read["column"] in (read.get("filter_columns") or []):
            continue  # a filter column is consumed by the lookup, it is not a value to be seen
        table = read["table"]
        short = table.split(".", 1)[1]
        for entity in read["entity_set"]:
            required.setdefault(read["node_id"], []).append(
                (table, read["column"], "%s:%s" % (short, entity))
            )
    writes_by_node: Dict[str, List[Dict[str, Any]]] = {}
    for write in gold.get("writes_gold", ()):
        writes_by_node.setdefault(write["node_id"], []).append(write)
    out: List[Candidate] = []
    for step in trace["steps"]:
        t = int(step["action_index"])
        touched = {
            ("%s.%s" % (row["db"], row["tbl"]), "%s:%s" % (row["tbl"], row["rowid"]))
            for row in step.get("delta", ())
        }
        if not touched:
            continue
        writing_nodes = [
            n
            for n, ws in writes_by_node.items()
            if any((w["table"], w["entity"]) in touched for w in ws)
        ]
        if not writing_nodes:
            continue
        seen = observed_facts(trace, upto=t)
        seen_keys = {(f.get("table"), f.get("column"), f.get("entity")) for _, f in seen}
        seen_wild = {
            (f.get("table"), f.get("column"))
            for _, f in seen
            if str(f.get("entity", "")).endswith(":*")
        }
        for writer in writing_nodes:
            for upstream in dag.ancestors(writer) | {writer}:
                for table, column, entity in required.get(upstream, ()):
                    if (table, column, entity) in seen_keys or (table, column) in seen_wild:
                        continue
                    out.append(
                        Candidate(
                            "omission",
                            t,
                            "required_read_never_observed",
                            0.5,
                            upstream,
                            [
                                {
                                    "kind": "gold",
                                    "ref": "%s.%s@%s" % (table, column, entity),
                                    "detail": "required by %s, not observed before write at %d"
                                    % (upstream, t),
                                }
                            ],
                            {"table": table, "column": column, "entity": entity},
                        )
                    )
                    break
    return out


_PRIORITY = {"state": 0, "parameter": 1, "omission": 2}


def pick_root_cause(
    candidates: Iterable[Candidate], residual_threshold: float
) -> Optional[Candidate]:
    strong = [c for c in candidates if c.confidence >= residual_threshold]
    if not strong:
        return None
    return sorted(strong, key=lambda c: (c.action_index, _PRIORITY[c.detector], -c.confidence))[0]
