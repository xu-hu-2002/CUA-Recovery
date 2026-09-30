"""State verifiers, mutation values, mutation test and dynamic latent horizon."""

from __future__ import annotations

import copy
import json
import random
import re
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Set, Tuple, Union

from derail.detect.mutations import MutationLibrary
from derail.ir.expr import date_add
from derail.ir.gold_interpreter import GoldInterpreter, GoldInterpreterError, WorldCopy
from derail.ir.model import dag_index
from derail.world.facts import date_part

VERIFIER_BUNDLE_VERSION = "verifier-bundle/1.0"


def compile_verifiers(task_ir: Mapping[str, Any]) -> Dict[str, Any]:
    nodes = list(task_ir["nodes"])
    node_verifiers = [
        {"node_id": n["node_id"], "verifier": n["verifier"]} for n in nodes if n.get("verifier")
    ]
    milestones = [
        {"node_id": n["node_id"], "verifier": n["verifier"]}
        for n in nodes
        if n.get("verifier") and n.get("critical")
    ]
    final = task_ir.get("final_verifier")
    return {
        "schema_version": VERIFIER_BUNDLE_VERSION,
        "task_id": str(task_ir["task_id"]),
        "node_verifiers": node_verifiers,
        "milestone_verifiers": milestones,
        "final_verifier": final,
        "verifier_kind": "state"
        if final and final.get("kind") in ("sql", "derived", "all_of")
        else "rubric",
        "equivalent_final_states": list((final or {}).get("equivalent_final_states", ())),
    }


def _other_row_id(
    conn: sqlite3.Connection,
    table: str,
    current: Any,
    rng: random.Random,
    same_name_column: Optional[str] = None,
) -> Optional[int]:
    rows = [
        r[0]
        for r in conn.execute(
            'SELECT rowid FROM "%s" WHERE rowid != ? ORDER BY rowid' % table, (current,)
        )
    ]
    if not rows:
        return None
    if same_name_column:
        try:
            name = conn.execute(
                'SELECT "%s" FROM "%s" WHERE rowid = ?' % (same_name_column, table), (current,)
            ).fetchone()
            if name and name[0] is not None:
                twins = [
                    r[0]
                    for r in conn.execute(
                        'SELECT rowid FROM "%s" WHERE "%s" = ? AND rowid != ?'
                        % (table, same_name_column),
                        (name[0], current),
                    )
                ]
                if twins:
                    return twins[0]
        except sqlite3.Error:
            pass
    return rng.choice(rows)


def _table_for(task_ir: Mapping[str, Any], node_id: str, name: str) -> Optional[Tuple[str, str]]:
    ref = "derived:%s:%s" % (node_id, name)
    for node in task_ir["nodes"]:
        for entry in list(node.get("reads", ())) + list(node.get("writes", ())):
            if entry.get("entity_ref") == ref:
                app, table = str(entry["table"]).split(".", 1)
                return app, table
    return None


def mutation_values(
    task_ir: Mapping[str, Any],
    gold: Mapping[str, Any],
    node_id: str,
    name: str,
    mutation: str,
    world: WorldCopy,
    rng: random.Random,
) -> Optional[Any]:
    value = next(
        (v["value"] for v in gold["values"] if v["node_id"] == node_id and v["name"] == name), None
    )
    if value is None:
        return None
    if mutation in ("entity_other", "entity_same_name", "write_other_entity"):
        located = _table_for(task_ir, node_id, name)
        if located and isinstance(value, int):
            conn = world.connection(located[0])
            columns = [r[1] for r in conn.execute('PRAGMA table_info("%s")' % located[1])]
            name_col = next(
                (c for c in ("name", "title", "display_name", "full_name") if c in columns), None
            )
            return _other_row_id(
                conn, located[1], value, rng, name_col if mutation == "entity_same_name" else None
            )
        return None
    if mutation == "date_plus_minus_day" and date_part(value):
        return date_add(value, days=rng.choice((-1, 1)))
    if mutation == "date_plus_minus_week" and date_part(value):
        return date_add(value, days=rng.choice((-7, 7)))
    if mutation == "time_zone_shift_hour" and date_part(value) and "T" in str(value):
        return date_add(value, hours=rng.choice((-1, 1)))
    if mutation == "amount_pct" and isinstance(value, (int, float)) and not isinstance(value, bool):
        return round(value * rng.choice((0.95, 1.05)), 2)
    if (
        mutation == "amount_unit"
        and isinstance(value, (int, float))
        and not isinstance(value, bool)
    ):
        return value + rng.choice((-1, 1))
    if mutation == "list_omit_element" and isinstance(value, list) and len(value) > 1:
        out = list(value)
        out.pop(rng.randrange(len(out)))
        return out
    if mutation == "list_extra_element" and isinstance(value, list) and value:
        return list(value) + [copy.deepcopy(value[0])]
    if mutation == "list_reorder" and isinstance(value, list) and len(value) > 1:
        out = list(value)
        out.reverse()
        return out
    if mutation == "boolean_flip" and isinstance(value, bool):
        return not value
    if mutation == "text_similar_entity" and isinstance(value, str) and value:
        return value[:-1] if len(value) > 3 else value + "x"
    if mutation == "text_stale_version" and isinstance(value, str):
        return value + " (old)"
    if mutation == "write_mutated_value":
        if isinstance(value, str) and value:
            return value + "x"
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            return value + 1
    return None


@dataclass
class MutationOutcome:
    node_id: str
    produce: str
    mutation: str
    mutated_value: Any
    caught_by: Optional[str]
    dynamic_horizon: Optional[int]
    observability_class: str
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return self.__dict__.copy()


_V_REF = re.compile(r"""V\[\s*(['"])([A-Za-z0-9_-]+)\1\s*\]\[\s*(['"])([A-Za-z0-9_-]+)\3\s*\]""")


def consumed_produces(task_ir: Mapping[str, Any]) -> Set[Tuple[str, str]]:
    out: Set[Tuple[str, str]] = set()
    for edge in task_ir.get("edges", ()):
        if edge.get("kind") != "control_dependency":
            out.add((str(edge["from"]["node_id"]), str(edge["from"]["port_id"])))
    texts: List[str] = []
    for node in task_ir["nodes"]:
        for produce in node.get("produces", ()):
            derivation = produce.get("derivation") or {}
            texts.append(
                str(
                    derivation.get("expr") or derivation.get("query") or derivation.get("sql") or ""
                )
            )
        verifier = node.get("verifier") or {}
        texts.append(str(verifier.get("predicate", "")))
    final = task_ir.get("final_verifier") or {}
    for part in final.get("parts", ()) if final.get("kind") == "all_of" else [final]:
        texts.append(str((part or {}).get("predicate", "")))
        for alt in (part or {}).get("equivalent_final_states", ()):
            texts.append(str(alt.get("predicate", "")))
    for text in texts:
        for match in _V_REF.finditer(text):
            out.add((match.group(2), match.group(4)))
    for node in task_ir["nodes"]:
        produces = list(node.get("produces", ()))
        for index, produce in enumerate(produces):
            derivation = produce.get("derivation") or {}
            text = str(
                derivation.get("expr") or derivation.get("query") or derivation.get("sql") or ""
            )
            for earlier in produces[:index]:
                if re.search(r"\b%s\b" % re.escape(str(earlier["name"])), text):
                    out.add((str(node["node_id"]), str(earlier["name"])))
    return out


def _sink_nodes(task_ir: Mapping[str, Any]) -> Set[str]:
    dag = dag_index(task_ir)
    return {n for n in dag.order if not dag.outgoing[n]}


def _canonical(value: Any) -> Any:
    if isinstance(value, float):
        return round(value, 6)
    if isinstance(value, list):
        return [_canonical(v) for v in value]
    if isinstance(value, dict):
        return {str(k): _canonical(v) for k, v in value.items()}
    return value


def gold_state_divergence(
    task_ir: Mapping[str, Any], gold: Mapping[str, Any], record: Mapping[str, Any]
) -> Optional[str]:
    sinks = _sink_nodes(task_ir)
    expected = {
        (str(v["node_id"]), str(v["name"])): _canonical(v.get("value"))
        for v in gold.get("values", ())
        if str(v["node_id"]) in sinks
    }
    actual = {
        (str(v["node_id"]), str(v["name"])): _canonical(v.get("value"))
        for v in record.get("values", ())
    }
    for key in sorted(expected):
        if actual.get(key) != expected[key]:
            return "sink:%s" % key[0]

    def cells(writes: Sequence[Mapping[str, Any]]) -> Set[Tuple[str, str, str]]:
        return {
            (
                str(w["table"]),
                str(w["column"]),
                json.dumps(_canonical(w.get("value")), sort_keys=True),
            )
            for w in writes
            if not w.get("volatile")
        }

    if cells(gold.get("writes_gold", ())) != cells(record.get("writes_gold", ())):
        return "writes"
    return None


def mutation_test(
    task_ir: Mapping[str, Any],
    gold: Mapping[str, Any],
    library: MutationLibrary,
    interpreter: GoldInterpreter,
    world_sources: Mapping[str, Union[str, Path]],
    workdir: Union[str, Path],
    repository: Union[str, Path],
    seed: int = 0,
    max_per_node: int = 4,
    files_root: Optional[Union[str, Path]] = None,
) -> List[MutationOutcome]:
    rng = random.Random(seed)
    dag = dag_index(task_ir)
    order = list(dag.order)
    consumed = consumed_produces(task_ir)
    outcomes: List[MutationOutcome] = []
    run_index = 0
    for node in task_ir["nodes"]:
        node_id = str(node["node_id"])
        tried = 0
        for produce_name, _type, mutation in library.mutations_for_node(node):
            if produce_name == "<writes>" or tried >= max_per_node:
                continue
            if (node_id, produce_name) not in consumed:
                continue
            run_index += 1
            with WorldCopy.open(
                "mutation",
                world_sources,
                Path(workdir) / ("m%d" % run_index),
                files_root=files_root,
            ) as world:
                value = mutation_values(task_ir, gold, node_id, produce_name, mutation, world, rng)
                if value is None:
                    continue
                tried += 1
                try:
                    record = interpreter.run(
                        task_ir, world, repository, overrides={(node_id, produce_name): value}
                    )
                except GoldInterpreterError as exc:
                    outcomes.append(
                        MutationOutcome(
                            node_id,
                            produce_name,
                            mutation,
                            value,
                            "execution_error",
                            None,
                            "verifier_only",
                            str(exc)[:200],
                        )
                    )
                    continue
            failed = [
                r["node_id"]
                for r in record["verifier_results"]
                if not r["passed"] and r["node_id"] != node_id
            ]
            downstream = [n for n in order[order.index(node_id) + 1 :] if n in failed]
            if downstream:
                first = downstream[0]
                distance = len(
                    [n for n in order[order.index(node_id) + 1 : order.index(first) + 1]]
                )
                klass = "required_next" if distance == 1 else "required_later"
                outcomes.append(
                    MutationOutcome(node_id, produce_name, mutation, value, first, distance, klass)
                )
            elif record.get("final_verifier_passed") is False:
                outcomes.append(
                    MutationOutcome(
                        node_id, produce_name, mutation, value, "final", None, "verifier_only"
                    )
                )
            elif gold_state_divergence(task_ir, gold, record) is not None:
                outcomes.append(
                    MutationOutcome(
                        node_id,
                        produce_name,
                        mutation,
                        value,
                        gold_state_divergence(task_ir, gold, record),
                        None,
                        "verifier_only",
                    )
                )
            else:
                outcomes.append(
                    MutationOutcome(node_id, produce_name, mutation, value, None, None, "silent")
                )
    return outcomes


OBSERVATION_OPS = ("confirm", "verify")
PURE_DERIVATIONS = ("expr", "upstream", "literal")
ABSENT_STATE_ERRORS = ("FILE_MISSING",)


def world_sources(
    database_dir: Union[str, Path], app_databases: Mapping[str, Optional[str]]
) -> Dict[str, Path]:
    out = {p.stem: p for p in sorted(Path(database_dir).glob("*.sqlite")) if p.is_file()}
    for app, stem in app_databases.items():
        if stem and stem in out:
            out.setdefault(str(app), out[stem])
    return out


def _created_in(world: WorldCopy, node: Mapping[str, Any], name: str, value: Any) -> bool:
    if isinstance(value, str) and value.startswith("file:"):
        return world.inventory is not None and world.inventory.get(value) is not None
    ref = "derived:%s:%s" % (node["node_id"], name)
    tables = {str(w["table"]) for w in node.get("writes", ()) if w.get("entity_ref") == ref}
    if len(tables) != 1 or value is None:
        return value is not None
    app, table = tables.pop().split(".", 1)
    row = world.connection(app).execute('SELECT 1 FROM "%s" WHERE rowid = ?' % table, (value,))
    return row.fetchone() is not None


def final_state_scope(
    task_ir: Mapping[str, Any],
    gold: Mapping[str, Any],
    world: WorldCopy,
    interpreter: GoldInterpreter,
    observation_ops: Sequence[str] = OBSERVATION_OPS,
) -> Dict[str, Any]:
    env: Dict[str, Dict[str, Any]] = {str(n["node_id"]): {} for n in task_ir["nodes"]}
    for value in gold.get("values", ()):
        env.setdefault(str(value["node_id"]), {})[str(value["name"])] = value.get("value")
    skipped = set((gold.get("provenance") or {}).get("skipped_nodes", ()))
    base = {"reference_time": task_ir.get("reference_time")}
    binding = interpreter._edge_bindings(task_ir)
    dag = dag_index(task_ir)
    absent: Set[Tuple[str, str]] = set()
    rederived: Set[str] = set()
    for node_id in dag.order:
        node = dag.nodes[node_id]
        if node_id in skipped:
            continue
        sources = [binding.get((node_id, str(p["port_id"]))) for p in node.get("inputs", ())]
        pure = not node.get("writes") and all(
            p["derivation"].get("kind") in PURE_DERIVATIONS for p in node.get("produces", ())
        )
        texts = " ".join(json.dumps(p["derivation"]) for p in node.get("produces", ()))
        fed = {s[0] for s in sources if s} | {m.group(2) for m in _V_REF.finditer(texts)}
        observe = str(node.get("op")) in observation_ops or bool(pure and fed & rederived)
        blind = observe and any(s in absent for s in sources)
        scope = dict(base)
        if observe:
            rederived.add(node_id)
        if observe and not blind:
            scope.update(interpreter._bind_inputs(node, binding, env))
        scope["V"] = env
        for produce in node.get("produces", ()):
            name = str(produce["name"])
            if produce["derivation"].get("kind") == "create":
                if not _created_in(world, node, name, env[node_id].get(name)):
                    env[node_id][name] = None
                    absent.add((node_id, name))
            elif blind:
                env[node_id][name] = None
                absent.add((node_id, name))
            elif observe:
                env[node_id][name] = interpreter._derive(produce, node, scope, world, env, set())
            scope[name] = env[node_id].get(name)
    return dict(base, V=env)


def verify_final_state(
    task_ir: Mapping[str, Any],
    gold: Mapping[str, Any],
    world: WorldCopy,
    interpreter: Optional[GoldInterpreter] = None,
    observation_ops: Sequence[str] = OBSERVATION_OPS,
    absent_state_errors: Sequence[str] = ABSENT_STATE_ERRORS,
) -> Dict[str, Any]:
    final = task_ir.get("final_verifier")
    if not final or final.get("kind") not in ("sql", "derived", "all_of"):
        return {"recovered": None, "detail": "no state verifier"}
    interpreter = interpreter or GoldInterpreter()
    try:
        scope = final_state_scope(task_ir, gold, world, interpreter, observation_ops)
        passed = interpreter._evaluate_final(final, scope, task_ir, world)
    except GoldInterpreterError as exc:
        verdict = False if exc.code in absent_state_errors else None
        return {"recovered": verdict, "detail": str(exc)[:200]}
    except sqlite3.Error as exc:
        return {"recovered": None, "detail": "SQL_ERROR: %s" % str(exc)[:200]}
    return {"recovered": passed, "detail": "final_verifier"}


def rejection_rate(outcomes: Sequence[MutationOutcome]) -> float:
    valid = [o for o in outcomes if o.caught_by != "execution_error"]
    if not valid:
        return 0.0
    return round(sum(1 for o in valid if o.caught_by is not None) / len(valid), 4)
