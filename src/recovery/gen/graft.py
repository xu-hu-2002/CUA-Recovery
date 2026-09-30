from __future__ import annotations

import copy
import hashlib
import re
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Set, Tuple

from recovery.detect.latent_static import static_latent_horizons
from recovery.detect.profile import build_profile
from recovery.gen.compat import (
    CompatConfig,
    literal_input_ports,
    output_ports,
    port_compatibility,
)
from recovery.ir.model import dag_index
from recovery.ir.rubric_check import RubricCheckConfig, expectation_matches, extract_expectations
from recovery.longhorizon.types import ValueTypeRegistry
from recovery.world.facts import date_part
from recovery.world.schema_graph import SchemaGraph

FAN_IN_OPS = {"compare", "aggregate", "decide"}
LOOKUP_OPS = {"retrieve", "resolve", "filter"}
STATE_OPS = {"create", "modify", "delete", "communicate"}
FINAL_VERIFIER_MODES = ("all_of", "receiver")
_V_REF = re.compile(r'V\[\s*["\']([A-Za-z0-9_-]+)["\']\s*\]')


def rename_ir(task_ir: Mapping[str, Any], prefix: str) -> Dict[str, Any]:
    ir = copy.deepcopy(dict(task_ir))
    ids = {str(n["node_id"]) for n in ir["nodes"]}

    def new(node_id: str) -> str:
        return prefix + node_id

    def fix_ref(text: Any) -> Any:
        if isinstance(text, str) and text.startswith("derived:"):
            _, node_id, rest = text.split(":", 2)
            return "derived:%s:%s" % (new(node_id), rest) if node_id in ids else text
        return text

    def fix_param(name: str) -> str:
        node_id, _, rest = name.partition("__")
        return new(node_id) + "__" + rest if rest and node_id in ids else name

    def fix_expr(text: Any) -> Any:
        if not isinstance(text, str):
            return text
        text = _PARAM.sub(lambda m: ":" + fix_param(m.group(1)), text)
        return _V_REF.sub(
            lambda m: 'V["%s"]' % (new(m.group(1)) if m.group(1) in ids else m.group(1)), text
        )

    def fix_final(verifier: Any) -> None:
        if not verifier:
            return
        if verifier.get("kind") == "all_of":
            for part in verifier.get("parts", ()):
                fix_final(part)
            return
        verifier["predicate"] = fix_expr(verifier.get("predicate"))
        for alt in verifier.get("equivalent_final_states", ()):
            alt["predicate"] = fix_expr(alt.get("predicate"))

    for node in ir["nodes"]:
        node["node_id"] = new(str(node["node_id"]))
        for port in list(node.get("inputs", ())) + list(node.get("outputs", ())):
            port["grounding"] = fix_ref(port.get("grounding"))
        for entry in list(node.get("reads", ())) + list(node.get("writes", ())):
            entry["entity_ref"] = fix_ref(entry["entity_ref"])
            if entry.get("value_ref"):
                entry["value_ref"] = fix_ref(entry["value_ref"])
        for produce in node.get("produces", ()):
            derivation = produce["derivation"]
            if "expr" in derivation:
                derivation["expr"] = fix_expr(derivation["expr"])
        for effect in node.get("side_effects", ()):
            effect["target_ref"] = fix_ref(effect.get("target_ref"))
        if node.get("verifier"):
            node["verifier"]["predicate"] = fix_expr(node["verifier"].get("predicate"))
            node["verifier"]["verifier_id"] = prefix + str(node["verifier"].get("verifier_id", "v"))
            if node["verifier"].get("expected_ref"):
                node["verifier"]["expected_ref"] = fix_param(str(node["verifier"]["expected_ref"]))
    for edge in ir["edges"]:
        edge["edge_id"] = prefix + str(edge["edge_id"])
        edge["from"]["node_id"] = new(str(edge["from"]["node_id"]))
        edge["to"]["node_id"] = new(str(edge["to"]["node_id"]))
        if edge.get("predicate"):
            edge["predicate"] = fix_expr(edge["predicate"])
    fix_final(ir.get("final_verifier"))
    return ir


def operator_label(a: Mapping[str, Any], b: Mapping[str, Any], edge: Mapping[str, Any]) -> str:
    src = next(n for n in a["nodes"] if n["node_id"] == edge["from"]["node_id"])
    dst = next(n for n in b["nodes"] if n["node_id"] == edge["to"]["node_id"])
    if dst["op"] in FAN_IN_OPS:
        return "fan_in_extension"
    if src["op"] in LOOKUP_OPS and dst["op"] in LOOKUP_OPS:
        return "motif_repetition"
    dag = dag_index(b)
    depth = dag.longest_depth().get(dst["node_id"], 1)
    if depth >= 3:
        return "delayed_reuse"
    return "typed_grafting"


_QUOTED = re.compile(r"'((?:[^']|'')*)'")
_MIN_LITERAL_CHARS = 3


def _param_name(node_id: str, name: str) -> str:
    return "%s__%s" % (node_id, name)


def rebind_predicate(predicate: str, bindings: Sequence[Tuple[str, Any]]) -> Tuple[str, List[str]]:
    used: List[str] = []

    def quoted(match: "re.Match[str]") -> str:
        inner = match.group(1)
        core = inner.strip("%")
        like = inner.startswith("%") and inner.endswith("%") and core
        texts = [(p, str(v)) for p, v in bindings if len(str(v)) >= _MIN_LITERAL_CHARS]
        for param, text in texts:
            if inner == text:
                used.append(param)
                return ":" + param
            if like and core == text:
                used.append(param)
                return "'%' || :" + param + " || '%'"
        for param, text in texts:
            if like and len(core) == 10 and "T" in text and text.startswith(core):
                used.append(param)
                return "'%' || substr(:" + param + ", 1, 10) || '%'"
        return match.group(0)

    out = _QUOTED.sub(quoted, predicate)
    for param, value in bindings:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            continue
        name = param.split("__", 1)[1]
        pattern = re.compile(r"\b(%s)\s*=\s*%s\b" % (re.escape(name), re.escape(str(value))))
        out, count = pattern.subn(lambda m: "%s = :%s" % (m.group(1), param), out)
        if count:
            used.append(param)
    return out, used


def _changed_nodes(b: Mapping[str, Any], dst_node_raw: str, prefix: str) -> Set[str]:
    dag = dag_index(b)
    return {prefix + n for n in {dst_node_raw} | set(dag.descendants(dst_node_raw))}


def _downstream_bindings(
    gold_b: Mapping[str, Any], prefix: str, changed: Set[str]
) -> List[Tuple[str, Any]]:
    bindings: List[Tuple[str, Any]] = []
    for entry in gold_b.get("values", ()):
        node_id = prefix + str(entry["node_id"])
        if node_id in changed and isinstance(entry.get("value"), (str, int, float)):
            bindings.append((_param_name(node_id, str(entry["name"])), entry["value"]))
    return bindings


def rebind_verifier(
    verifier: Mapping[str, Any], bindings: Sequence[Tuple[str, Any]]
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    out = dict(verifier)
    predicate, used = rebind_predicate(str(verifier.get("predicate", "")), bindings)
    out["predicate"] = predicate
    kept, dropped = [], []
    for alternative in verifier.get("equivalent_final_states", ()):
        alt_predicate, alt_used = rebind_predicate(str(alternative.get("predicate", "")), bindings)
        if used and not alt_used:
            dropped.append(alternative)
            continue
        kept.append({**alternative, "predicate": alt_predicate})
    if "equivalent_final_states" in verifier or kept:
        out["equivalent_final_states"] = kept
    return out, {"params": used, "dropped_alternatives": len(dropped)}


def compose_final_verifier(
    ra: Mapping[str, Any],
    rb: Mapping[str, Any],
    bindings: Sequence[Tuple[str, Any]],
    task_id: str,
    mode: str = "all_of",
) -> Tuple[Optional[Dict[str, Any]], Dict[str, Any]]:
    if mode not in FINAL_VERIFIER_MODES:
        raise ValueError("unknown final verifier mode %r" % mode)
    parts: List[Dict[str, Any]] = []
    note: Dict[str, Any] = {}
    if ra.get("final_verifier") and mode == "all_of":
        parts.append(dict(ra["final_verifier"]))
    if rb.get("final_verifier"):
        rebound, note = rebind_verifier(rb["final_verifier"], bindings)
        parts.append(rebound)
    if not parts:
        return None, note
    if len(parts) == 1:
        return parts[0], note
    return (
        {
            "verifier_id": "final-%s" % task_id,
            "kind": "all_of",
            "observability": "environment_state",
            "parts": parts,
        },
        note,
    )


_IDENT = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
_PARAM = re.compile(r":([A-Za-z_][A-Za-z0-9_]*__[A-Za-z_][A-Za-z0-9_]*)\b")


def _mark_stale_node_verifiers(
    renamed_b: Mapping[str, Any], changed: Set[str], bindings: Sequence[Tuple[str, Any]]
) -> Dict[str, List[str]]:
    marked: List[str] = []
    dropped: List[str] = []
    for node in renamed_b["nodes"]:
        verifier = node.get("verifier")
        if not verifier or node["node_id"] not in changed:
            continue
        if "expected" in verifier:
            produces = {str(p["name"]) for p in node.get("produces", ())}
            names = [t for t in _IDENT.findall(str(verifier.get("predicate", ""))) if t in produces]
            if names:
                node["verifier"] = {
                    **verifier,
                    "expected_ref": _param_name(str(node["node_id"]), names[0]),
                }
                marked.append(str(verifier["verifier_id"]))
            else:
                dropped.append(str(verifier["verifier_id"]))
                node["verifier"] = None
        elif verifier.get("kind") == "sql":
            predicate, _ = rebind_predicate(str(verifier.get("predicate", "")), bindings)
            node["verifier"] = {**verifier, "predicate": predicate}
    return {"refrozen_node_verifiers": marked, "dropped_node_verifiers": dropped}


def _sql_literal(value: Any) -> str:
    if isinstance(value, bool):
        return "1" if value else "0"
    if isinstance(value, (int, float)):
        return repr(value)
    return "'" + str(value).replace("'", "''") + "'"


_BARE_PARAM = re.compile(r":([A-Za-z_][A-Za-z0-9_]*)\b")
_V_INDEX = re.compile(r"""V\[\s*(['"])([A-Za-z0-9_-]+)\1\s*\]\[\s*(['"])([A-Za-z0-9_-]+)\3\s*\]""")


def _sink_nodes(task_ir: Mapping[str, Any]) -> Set[str]:
    dag = dag_index(task_ir)
    fed = {str(e["to"]["node_id"]) for e in dag.value_edges()}
    return {n for n in dag.order if n not in dag.outgoing or not dag.outgoing[n]} | (
        set(dag.order) - fed - {n for n in dag.order if dag.outgoing[n]}
    )


def freeze_verifiers(task_ir: Mapping[str, Any], gold: Mapping[str, Any]) -> Dict[str, Any]:
    values: Dict[str, Any] = {}
    by_name: Dict[str, Any] = {}
    for v in gold.get("values", ()):
        values[_param_name(str(v["node_id"]), str(v["name"]))] = v.get("value")
        by_name.setdefault(str(v["name"]), v.get("value"))
    sinks = _sink_nodes(task_ir)
    missing: List[str] = []

    produces_of = {
        str(n["node_id"]): {str(p["name"]) for p in n.get("produces", ())} for n in task_ir["nodes"]
    }

    def sql_sub(keep: Set[str]):
        live = {name for node_id in keep for name in produces_of.get(node_id, ())}

        def sub(match: "re.Match[str]") -> str:
            param = match.group(1)
            if "__" in param:
                node_id, name = param.split("__", 1)
                if node_id in keep:
                    return match.group(0)
                value = values.get(param)
            else:
                if param in live:
                    return match.group(0)
                value = by_name.get(param)
            if value is None or isinstance(value, (list, dict)):
                missing.append(param)
                return match.group(0)
            return _sql_literal(value)

        return sub

    def expr_sub(keep: Set[str]):
        def sub(match: "re.Match[str]") -> str:
            node_id, name = match.group(2), match.group(4)
            if node_id in keep:
                return match.group(0)
            key = _param_name(node_id, name)
            if key not in values:
                missing.append(key)
                return match.group(0)
            return repr(values[key])

        return sub

    def freeze(verifier: Optional[Mapping[str, Any]], keep: Set[str]) -> Optional[Dict[str, Any]]:
        if not verifier:
            return None if verifier is None else dict(verifier)
        out = dict(verifier)
        if out.get("kind") == "all_of":
            out["parts"] = [freeze(p, keep) for p in out.get("parts", ())]
            return out
        substitute = (
            (lambda text: _BARE_PARAM.sub(sql_sub(keep), text))
            if out.get("kind") == "sql"
            else (lambda text: _V_INDEX.sub(expr_sub(keep), text))
        )
        if "predicate" in out:
            out["predicate"] = substitute(str(out["predicate"]))
        if out.get("equivalent_final_states"):
            out["equivalent_final_states"] = [
                {**alt, "predicate": substitute(str(alt.get("predicate", "")))}
                for alt in out["equivalent_final_states"]
            ]
        ref = out.pop("expected_ref", None)
        if ref is not None:
            if ref in values:
                out["expected"] = values[ref]
            else:
                missing.append(ref)
        return out

    frozen = copy.deepcopy(dict(task_ir))
    frozen["final_verifier"] = freeze(task_ir.get("final_verifier"), sinks)
    for node in frozen["nodes"]:
        node["verifier"] = freeze(node.get("verifier"), {str(node["node_id"])})
    provenance = dict(frozen.get("provenance") or {})
    note = {
        "frozen": True,
        "frozen_from_gold": gold.get("initial_state_sha256"),
        "sink_nodes": sorted(sinks),
        "unresolved_params": sorted(set(missing)),
    }
    if provenance.get("composition"):
        composition = dict(provenance["composition"])
        composition["verifier_rebinding"] = {
            **(composition.get("verifier_rebinding") or {}),
            **note,
        }
        provenance["composition"] = composition
    else:
        provenance["verifier_freeze"] = note
    frozen["provenance"] = provenance
    return frozen


RUBRIC_VERSION = "composed-rubric/1.0"


def _scalar(value: Any) -> bool:
    return isinstance(value, (str, int, float)) and not isinstance(value, bool)


def _rubric_substitutions(
    source_ir: Optional[Mapping[str, Any]],
    source_gold: Optional[Mapping[str, Any]],
    task_ir: Mapping[str, Any],
    values: Mapping[Tuple[str, str], Any],
    prefix: str,
) -> List[Tuple[Any, Any]]:
    subs: List[Tuple[Any, Any]] = []
    for entry in (source_gold or {}).get("values", ()):
        old = entry.get("value")
        new = values.get((prefix + str(entry["node_id"]), str(entry["name"])))
        if _scalar(old) and _scalar(new) and new != old:
            subs.append((old, new))
    composed_nodes = {str(n["node_id"]): n for n in task_ir["nodes"]}
    for node in (source_ir or {}).get("nodes", ()):
        target = composed_nodes.get(prefix + str(node["node_id"]), {})
        ports = {str(p["port_id"]): p for p in target.get("inputs", ())}
        for port in node.get("inputs", ()):
            grounding = str(ports.get(str(port["port_id"]), {}).get("grounding") or "")
            if "literal" not in port or not grounding.startswith("derived:"):
                continue
            _, src_node, src_name = grounding.split(":", 2)
            new = values.get((src_node, src_name))
            if _scalar(port["literal"]) and _scalar(new) and new != port["literal"]:
                subs.append((port["literal"], new))
    return subs


def _render_expected(kind: str, raw: str, value: Any) -> str:
    if kind == "amount" and isinstance(value, (int, float)):
        number = "{:,.2f}".format(value) if isinstance(value, float) else "{:,}".format(value)
        return ("$" if raw.lstrip().startswith("$") else "") + number
    if kind == "date":
        return date_part(value) or str(value)
    return str(value)


def _compose_item(
    source_task_id: str,
    index: int,
    item: Mapping[str, Any],
    subs: Sequence[Tuple[Any, Any]],
    config: RubricCheckConfig,
    min_literal_chars: int,
) -> Dict[str, Any]:
    criterion = str(item.get("criterion", ""))
    text = criterion
    expected: List[Dict[str, Any]] = []
    for exp in extract_expectations({"grading": {"rubrics": [item]}}, {}, config):
        hit = next((new for old, new in subs if expectation_matches(exp, [old], config)), None)
        if hit is None:
            expected.append({"kind": exp["kind"], "value": exp["value"], "raw": exp["raw"]})
            continue
        old_text = str(exp["value"]) if exp["kind"] == "text" else exp["raw"]
        text = text.replace(old_text, _render_expected(exp["kind"], exp["raw"], hit))
        expected.append(
            {"kind": exp["kind"], "value": hit, "source_value": exp["value"], "raw": exp["raw"]}
        )
    for old, new in subs:
        if isinstance(old, str) and len(old) >= min_literal_chars and old in text:
            text = text.replace(old, str(new))
            expected.append({"kind": "text", "value": new, "source_value": old, "raw": old})
    out = {
        **dict(item),
        "criterion": text,
        "source_task_id": source_task_id,
        "source_rubric_index": index,
        "expected": expected,
        "updated": text != criterion,
    }
    if text != criterion:
        out["source_criterion"] = criterion
    return out


def compose_rubric(
    task_ir: Mapping[str, Any],
    gold: Mapping[str, Any],
    sources: Mapping[str, Mapping[str, Any]],
    config: RubricCheckConfig,
    min_literal_chars: int,
) -> Dict[str, Any]:
    values = {(str(v["node_id"]), str(v["name"])): v.get("value") for v in gold.get("values", ())}
    items: List[Dict[str, Any]] = []
    for entry in source_prefixes(task_ir):
        source = sources.get(entry["task_id"]) or {}
        subs = _rubric_substitutions(
            source.get("task_ir"), source.get("gold"), task_ir, values, entry["node_prefix"]
        )
        for index, item in enumerate(source.get("rubrics") or ()):
            items.append(
                _compose_item(entry["task_id"], index, item, subs, config, min_literal_chars)
            )
    return {
        "schema_version": RUBRIC_VERSION,
        "task_id": str(task_ir["task_id"]),
        "source_task_ids": lineage(task_ir)[0],
        "type": "llm_judge",
        "rubrics": items,
        "missing_sources": sorted(
            {e["task_id"] for e in source_prefixes(task_ir) if not sources.get(e["task_id"])}
        ),
    }


def lineage(task_ir: Mapping[str, Any]) -> Tuple[List[str], List[str]]:
    composition = (task_ir.get("provenance") or {}).get("composition") or {}
    sources = list(composition.get("source_task_ids") or [str(task_ir["task_id"])])
    return sources, list(composition.get("operators") or [])


def source_prefixes(task_ir: Mapping[str, Any]) -> List[Dict[str, str]]:
    composition = (task_ir.get("provenance") or {}).get("composition") or {}
    if composition.get("source_prefixes"):
        return [dict(p) for p in composition["source_prefixes"]]
    return [{"task_id": str(task_ir["task_id"]), "node_prefix": ""}]


def _prefixed_sources(
    a: Mapping[str, Any], b: Mapping[str, Any], pa: str, pb: str
) -> List[Dict[str, str]]:
    return [
        {"task_id": p["task_id"], "node_prefix": prefix + p["node_prefix"]}
        for ir, prefix in ((a, pa), (b, pb))
        for p in source_prefixes(ir)
    ]


def graft(
    a: Mapping[str, Any],
    b: Mapping[str, Any],
    edge: Mapping[str, Any],
    task_id: Optional[str] = None,
    gold_a: Optional[Mapping[str, Any]] = None,
    gold_b: Optional[Mapping[str, Any]] = None,
    final_verifier: str = "all_of",
) -> Dict[str, Any]:
    pa, pb = "a_", "b_"
    ra, rb = rename_ir(a, pa), rename_ir(b, pb)
    src_node, src_port = pa + edge["from"]["node_id"], edge["from"]["port_id"]
    dst_node, dst_port = pb + edge["to"]["node_id"], edge["to"]["port_id"]
    for node in rb["nodes"]:
        if node["node_id"] == dst_node:
            for port in node["inputs"]:
                if port["port_id"] == dst_port:
                    port.pop("literal", None)
                    port["source"] = "upstream"
                    port["grounding"] = "derived:%s:%s" % (src_node, src_port)
    label = operator_label(a, b, edge)
    changed = _changed_nodes(b, str(edge["to"]["node_id"]), pb) if gold_b else set()
    bindings = _downstream_bindings(gold_b, pb, changed) if gold_b else []
    marks = (
        _mark_stale_node_verifiers(rb, changed, bindings)
        if gold_b
        else {"refrozen_node_verifiers": [], "dropped_node_verifiers": []}
    )
    composed_id = (
        task_id
        or "gen-%s"
        % hashlib.sha256(
            (
                "%s|%s|%s.%s->%s.%s"
                % (a["task_id"], b["task_id"], src_node, src_port, dst_node, dst_port)
            ).encode()
        ).hexdigest()[:10]
    )
    final, rebinding = compose_final_verifier(ra, rb, bindings, composed_id, final_verifier)
    composed = {
        "schema_version": "task-ir/1.0",
        "task_id": composed_id,
        "source_task_id": None,
        "world_id": str(b.get("world_id") or a.get("world_id")),
        "instruction": "",
        "reference_time": a.get("reference_time") or b.get("reference_time"),
        "nodes": ra["nodes"] + rb["nodes"],
        "edges": ra["edges"]
        + rb["edges"]
        + [
            {
                "edge_id": "graft_%s_%s" % (src_node, dst_node),
                "from": {"node_id": src_node, "port_id": src_port},
                "to": {"node_id": dst_node, "port_id": dst_port},
                "kind": "data_dependency",
            }
        ],
        "final_verifier": final,
        "provenance": {
            "composition": {
                "operator": label,
                "parents": [a["task_id"], b["task_id"]],
                "source_task_ids": lineage(a)[0] + lineage(b)[0],
                "source_prefixes": _prefixed_sources(a, b, pa, pb),
                "operators": lineage(a)[1] + lineage(b)[1] + [label],
                "binding": {
                    "from": {"node_id": src_node, "port_id": src_port},
                    "to": {"node_id": dst_node, "port_id": dst_port},
                },
                "compat": {k: edge[k] for k in ("reasons", "score", "cross_app") if k in edge},
                "verifier_rebinding": {**rebinding, **marks},
            },
            "review_status": "needs_review",
        },
    }
    return composed


def conditionalize(
    a: Mapping[str, Any],
    b: Mapping[str, Any],
    decide_node: str,
    produce: str,
    task_id: Optional[str] = None,
    final_verifier: str = "all_of",
) -> Optional[Dict[str, Any]]:
    first_write = next((n for n in b["nodes"] if n["op"] in STATE_OPS), None)
    if first_write is None:
        return None
    pa, pb = "a_", "b_"
    ra, rb = rename_ir(a, pa), rename_ir(b, pb)
    gate_node = pb + first_write["node_id"]
    if any(str(p.get("port_id")) == "gate" for p in first_write.get("inputs", ())):
        return None
    for node in rb["nodes"]:
        if node["node_id"] == gate_node:
            node["inputs"].append(
                {
                    "port_id": "gate",
                    "type": "Boolean",
                    "grounding": "derived:%s:%s" % (pa + decide_node, produce),
                    "source": "upstream",
                    "cardinality": "one",
                }
            )
    composed_id = (
        task_id
        or "gen-%s"
        % hashlib.sha256(
            ("cond|%s|%s|%s" % (a["task_id"], b["task_id"], decide_node)).encode()
        ).hexdigest()[:10]
    )
    final, _ = compose_final_verifier(ra, rb, (), composed_id, final_verifier)
    return {
        "schema_version": "task-ir/1.0",
        "task_id": composed_id,
        "source_task_id": None,
        "world_id": str(b.get("world_id") or a.get("world_id")),
        "instruction": "",
        "reference_time": a.get("reference_time") or b.get("reference_time"),
        "nodes": ra["nodes"] + rb["nodes"],
        "edges": ra["edges"]
        + rb["edges"]
        + [
            {
                "edge_id": "cond_%s" % gate_node,
                "from": {"node_id": pa + decide_node, "port_id": produce},
                "to": {"node_id": gate_node, "port_id": "gate"},
                "kind": "control_dependency",
                "predicate": 'bool(V["%s"]["%s"])' % (pa + decide_node, produce),
            }
        ],
        "final_verifier": final,
        "provenance": {
            "composition": {
                "operator": "conditionalization",
                "parents": [a["task_id"], b["task_id"]],
                "source_task_ids": lineage(a)[0] + lineage(b)[0],
                "source_prefixes": _prefixed_sources(a, b, pa, pb),
                "operators": lineage(a)[1] + lineage(b)[1] + ["conditionalization"],
                "gate": {"node_id": pa + decide_node, "produce": produce, "gated": gate_node},
            },
            "review_status": "needs_review",
        },
    }


@dataclass(frozen=True)
class SearchConfig:
    beam_width: int = 16
    max_grafts: int = 3
    max_gate_checks_per_round: int = 48
    buckets: Tuple[str, ...] = ("1", "2-3", ">=4", "verifier_only")
    target_buckets: Tuple[str, ...] = ("2-3", ">=4", "verifier_only")
    bucket_targets: Mapping[str, float] = field(
        default_factory=lambda: {"1": 0.2, "2-3": 0.35, ">=4": 0.3, "verifier_only": 0.15}
    )
    min_nodes_in_target_bucket: int = 2
    depth_range: Tuple[int, int] = (5, 9)
    max_independent_component_ratio: float = 0.25
    max_nodes: int = 24
    min_nodes_composed: int = 10
    carry_threshold: int = 3
    compat: CompatConfig = CompatConfig()
    final_verifier: str = "all_of"

    @classmethod
    def from_sampling(cls, sampling: Mapping[str, Any]) -> "SearchConfig":
        sec = sampling.get("secondary_constraints", {})
        beam = sampling.get("beam", {})
        depth = sec.get("dependency_depth", [5, 9])
        compat = sampling.get("compat", {})
        return cls(
            compat=CompatConfig(
                max_fk_hops=int(compat.get("max_fk_hops", 2)),
                allow_cross_app=bool(compat.get("allow_cross_app", True)),
                min_score=float(compat.get("min_score", 0.5)),
                generic_types=tuple(compat.get("generic_types", ())),
                generic_same_type_score=float(compat.get("generic_same_type_score", 0.4)),
                require_cell=bool(compat.get("require_cell", True)),
            ),
            beam_width=int(beam.get("width", 16)),
            max_grafts=int(beam.get("max_grafts", 3)),
            max_gate_checks_per_round=int(beam.get("max_gate_checks_per_round", 48)),
            buckets=tuple(sampling["latent_horizon_semantic_bucket_targets"]),
            target_buckets=tuple(sampling.get("target_buckets", ("2-3", ">=4", "verifier_only"))),
            bucket_targets={
                str(k): float(v)
                for k, v in sampling["latent_horizon_semantic_bucket_targets"].items()
            },
            min_nodes_in_target_bucket=int(sampling.get("min_nodes_in_target_bucket", 2)),
            depth_range=(int(depth[0]), int(depth[1])),
            max_independent_component_ratio=float(sec.get("max_independent_component_ratio", 0.25)),
            max_nodes=int(sec.get("max_nodes", 24)),
            min_nodes_composed=int(sec.get("min_nodes_composed", 10)),
            carry_threshold=int(sec.get("carry_threshold", 3)),
            final_verifier=str((sampling.get("composition") or {}).get("final_verifier", "all_of")),
        )


@dataclass
class Scored:
    task_ir: Dict[str, Any]
    profile: Dict[str, Any]
    score: float
    rejections: List[str]
    grafts: int


def score_profile(profile: Mapping[str, Any], config: SearchConfig) -> Tuple[float, List[str]]:
    rejections: List[str] = []
    structural = profile["structural"]
    if structural["independent_component_ratio"] > config.max_independent_component_ratio:
        rejections.append("INDEPENDENT_COMPONENTS")
    if structural["decorative_carry_violations"]:
        rejections.append("DECORATIVE_CARRY")
    if profile["nodes_in_target_bucket"] < config.min_nodes_in_target_bucket:
        rejections.append("NO_NODE_IN_TARGET_BUCKET")
    if profile.get("node_count", 0) > config.max_nodes:
        rejections.append("TOO_MANY_NODES")
    node_buckets = profile.get("node_buckets", {})
    total = max(len(node_buckets), 1)
    score = 0.0
    for bucket, weight in config.bucket_targets.items():
        share = sum(1 for b in node_buckets.values() if b == bucket) / total
        score += weight * share * (2.0 if bucket in config.target_buckets else 0.5)
    depth = structural["dependency_depth"]
    low, high = config.depth_range
    if depth < low:
        score -= 0.1 * (low - depth)
    elif depth > high:
        score -= 0.1 * (depth - high)
    score += 0.05 * profile["class_counts"].get("cross_app", 0)
    return round(score, 4), rejections


def evaluate(task_ir: Mapping[str, Any], config: SearchConfig, grafts: int) -> Scored:
    horizons = static_latent_horizons(task_ir)
    profile = build_profile(
        task_ir, horizons, list(config.buckets), list(config.target_buckets), config.carry_threshold
    )
    score, rejections = score_profile(profile, config)
    if grafts > 0 and profile.get("node_count", 0) < config.min_nodes_composed:
        rejections.append("TOO_FEW_NODES")
    return Scored(dict(task_ir), profile, score, rejections, grafts)


def edges_touching(
    state: Mapping[str, Any],
    pool: Sequence[Mapping[str, Any]],
    registry: ValueTypeRegistry,
    graph: Optional[SchemaGraph],
    compat: CompatConfig,
) -> List[Dict[str, Any]]:
    others = [p for p in pool if p["task_id"] != state["task_id"]]
    state_out = output_ports(state)
    state_in = literal_input_ports(state, compat.excluded_literals)
    edges: List[Dict[str, Any]] = []
    for other in others:
        for source in state_out:
            for target in literal_input_ports(other, compat.excluded_literals):
                edge = port_compatibility(source, target, registry, graph, compat)
                if edge:
                    edges.append(edge)
        for source in output_ports(other):
            for target in state_in:
                edge = port_compatibility(source, target, registry, graph, compat)
                if edge:
                    edges.append(edge)
    return edges


def beam_search(
    seeds: Sequence[Mapping[str, Any]],
    registry: ValueTypeRegistry,
    config: SearchConfig,
    graph: Optional[SchemaGraph] = None,
    executable: Optional[Callable[[Mapping[str, Any]], bool]] = None,
    progress: Optional[Callable[[int, int, str], None]] = None,
    pool: Optional[Sequence[Mapping[str, Any]]] = None,
    gold_lookup: Optional[Callable[[Mapping[str, Any]], Optional[Mapping[str, Any]]]] = None,
) -> List[Scored]:
    pool = list(pool if pool is not None else seeds)
    beam: List[Scored] = [evaluate(ir, config, 0) for ir in seeds]
    accepted: Dict[str, Scored] = {}
    for round_index in range(config.max_grafts):
        candidates: List[Scored] = []
        for state in beam:
            edges = edges_touching(state.task_ir, pool, registry, graph, config.compat)
            for edge in edges:
                a = (
                    state.task_ir
                    if edge["from"]["task_id"] == state.task_ir["task_id"]
                    else next(p for p in pool if p["task_id"] == edge["from"]["task_id"])
                )
                b = (
                    state.task_ir
                    if edge["to"]["task_id"] == state.task_ir["task_id"]
                    else next(p for p in pool if p["task_id"] == edge["to"]["task_id"])
                )
                if a is b:
                    continue
                try:
                    composed = graft(
                        a,
                        b,
                        edge,
                        gold_a=gold_lookup(a) if gold_lookup else None,
                        gold_b=gold_lookup(b) if gold_lookup else None,
                        final_verifier=config.final_verifier,
                    )
                    scored = evaluate(composed, config, state.grafts + 1)
                except Exception as exc:
                    candidates.append(
                        Scored(
                            dict(a),
                            {},
                            -1.0,
                            ["COMPOSITION_ERROR: %s" % str(exc)[:80]],
                            state.grafts + 1,
                        )
                    )
                    continue
                candidates.append(scored)
            for node in state.task_ir["nodes"]:
                if node["op"] != "decide":
                    continue
                for produce in node.get("produces", ()):
                    if produce.get("type", "").lower() not in ("boolean", "bool"):
                        continue
                    for other in pool:
                        if other["task_id"] == state.task_ir["task_id"]:
                            continue
                        try:
                            composed = conditionalize(
                                state.task_ir,
                                other,
                                str(node["node_id"]),
                                str(produce["name"]),
                                final_verifier=config.final_verifier,
                            )
                            if composed is not None:
                                candidates.append(evaluate(composed, config, state.grafts + 1))
                        except Exception as exc:
                            candidates.append(
                                Scored(
                                    dict(state.task_ir),
                                    {},
                                    -1.0,
                                    ["COMPOSITION_ERROR: %s" % str(exc)[:80]],
                                    state.grafts + 1,
                                )
                            )
        candidates.sort(key=lambda c: c.score, reverse=True)
        beam = []
        gate_checks = 0
        for candidate in candidates:
            if candidate.rejections or not candidate.profile:
                continue
            if executable is not None:
                if gate_checks >= config.max_gate_checks_per_round:
                    candidate.rejections.append("GATE_BUDGET_EXHAUSTED")
                    continue
                gate_checks += 1
                if not executable(candidate.task_ir):
                    candidate.rejections.append("GOLD_EXECUTION_FAILED")
                    continue
            accepted[candidate.task_ir["task_id"]] = candidate
            beam.append(candidate)
            if len(beam) >= config.beam_width:
                break
        if progress:
            progress(
                round_index + 1,
                config.max_grafts,
                "%d candidates, %d kept" % (len(candidates), len(beam)),
            )
        if not beam:
            break
    return sorted(accepted.values(), key=lambda c: c.score, reverse=True)
