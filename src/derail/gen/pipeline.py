from __future__ import annotations

import hashlib
import json
import os
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple, Union

from derail.detect.mutations import MutationLibrary
from derail.gen.compat import build_compat_index
from derail.gen.graft import (
    Scored,
    SearchConfig,
    beam_search,
    compose_rubric,
    evaluate,
    freeze_verifiers,
    lineage,
)
from derail.gen.hazards import HazardConfig, base_rates, injection_record
from derail.gen.realize import (
    RealizationConfig,
    build_realization_fields,
    parse_realization_reply,
    render_realization_prompt,
    round_trip_compare,
    style_stats_from_instructions,
)
from derail.gen.verifiers import compile_verifiers, mutation_test, rejection_rate
from derail.ir.gold_interpreter import GoldInterpreter, WorldCopy
from derail.ir.model import dag_index, load_task_ir
from derail.ir.rubric_check import RubricCheckConfig
from derail.longhorizon.types import ValueTypeRegistry
from derail.world.schema_graph import SchemaGraph

GENERATION_RECORD_VERSION = "generation-record/1.0"
BUNDLE_VERSION = "generation-bundle/1.0"
Progress = Callable[[str], None]


def fingerprint(task_ir: Mapping[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(task_ir, sort_keys=True, ensure_ascii=False).encode("utf-8")
    ).hexdigest()


class GoldExecutor:
    def __init__(
        self,
        interpreter: GoldInterpreter,
        database_dir: Union[str, Path],
        world_id: str,
        workdir: Union[str, Path],
        repository: Union[str, Path],
        files_root: Optional[Union[str, Path]] = None,
        app_databases: Optional[Mapping[str, Optional[str]]] = None,
    ):
        self.interpreter = interpreter
        self.database_dir = Path(database_dir)
        self.world_id = world_id
        self.workdir = Path(workdir)
        self.repository = Path(repository)
        self.files_root = files_root
        self.app_databases = dict(app_databases or {})
        self.golds: Dict[str, Dict[str, Any]] = {}
        self.errors: Dict[str, str] = {}
        self.runs = 0

    def sources_for(self, task_ir: Mapping[str, Any]) -> Dict[str, Path]:
        sources: Dict[str, Path] = {}
        for app in sorted({str(n["app"]) for n in task_ir["nodes"]}):
            stem = self.app_databases.get(app, app)
            if stem is None:
                continue
            path = self.database_dir / ("%s.sqlite" % stem)
            if path.is_file():
                sources[app] = path
        return sources

    def run(self, task_ir: Mapping[str, Any]) -> Optional[Dict[str, Any]]:
        key = fingerprint(task_ir)
        if key in self.golds:
            return self.golds[key]
        if key in self.errors:
            return None
        self.runs += 1
        try:
            with WorldCopy.open(
                self.world_id,
                self.sources_for(task_ir),
                self.workdir / ("g%d" % self.runs),
                files_root=self.files_root,
            ) as world:
                gold = self.interpreter.run(task_ir, world, self.repository)
        except Exception as exc:
            self.errors[key] = "%s: %s" % (type(exc).__name__, str(exc)[:300])
            return None
        if gold.get("final_verifier_passed") is False:
            self.errors[key] = "FINAL_VERIFIER_FAILED"
            return None
        self.golds[key] = gold
        return gold

    def executable(self, task_ir: Mapping[str, Any]) -> bool:
        return self.run(task_ir) is not None


@dataclass
class Candidate:
    task_ir: Dict[str, Any]
    profile: Dict[str, Any]
    score: float
    rejections: List[str]
    grafts: int
    origin: str
    gold: Optional[Dict[str, Any]] = None
    gold_error: Optional[str] = None
    verifier_bundle: Optional[Dict[str, Any]] = None
    mutation_outcomes: Optional[List[Dict[str, Any]]] = None
    mutation_rejection_rate: Optional[float] = None
    rubric: Optional[Dict[str, Any]] = None
    hazards: List[Dict[str, Any]] = field(default_factory=list)
    realization: Optional[Dict[str, Any]] = None
    selected: bool = False
    selection_reason: Optional[str] = None

    @property
    def task_id(self) -> str:
        return str(self.task_ir["task_id"])

    @property
    def bucket(self) -> str:
        return str(self.profile.get("bucket", "none")) if self.profile else "none"

    @classmethod
    def from_scored(cls, scored: Scored, origin: str) -> "Candidate":
        return cls(
            dict(scored.task_ir),
            dict(scored.profile),
            scored.score,
            list(scored.rejections),
            scored.grafts,
            origin,
        )


EXECUTION_ERROR_POLICIES = ("exclude", "caught", "not_caught")


def mutation_rejection_rate(outcomes: Sequence[Any], execution_errors: str = "exclude") -> float:
    if execution_errors not in EXECUTION_ERROR_POLICIES:
        raise ValueError("unknown execution_errors policy %r" % execution_errors)
    if execution_errors == "exclude":
        return rejection_rate(outcomes)
    if not outcomes:
        return 0.0
    caught = sum(
        1
        for o in outcomes
        if o.caught_by is not None
        and (o.caught_by != "execution_error" or execution_errors == "caught")
    )
    return round(caught / len(outcomes), 4)


def bucket_quotas(sampling: Mapping[str, Any]) -> Dict[str, int]:
    total = int(sampling.get("target_count", 0))
    shares = sampling["latent_horizon_semantic_bucket_targets"]
    return {str(b): int(round(total * float(share))) for b, share in shares.items()}


def select_by_targets(
    candidates: Sequence[Candidate], sampling: Mapping[str, Any]
) -> Dict[str, Any]:
    quotas = bucket_quotas(sampling)
    target_buckets = set(sampling.get("target_buckets", ()))
    min_rate = sampling.get("gates", {}).get("min_mutation_rejection_rate")
    filled = {b: 0 for b in quotas}
    for candidate in sorted(candidates, key=lambda c: (-c.score, c.task_id)):
        candidate.selected = False
        if candidate.rejections:
            candidate.selection_reason = "rejected:" + ",".join(candidate.rejections)
            continue
        if candidate.gold is None:
            candidate.selection_reason = "gold_execution_failed"
            continue
        if (
            min_rate is not None
            and candidate.mutation_rejection_rate is not None
            and candidate.mutation_rejection_rate < float(min_rate)
        ):
            candidate.selection_reason = "mutation_rejection_rate_below_%s" % min_rate
            continue
        bucket = candidate.bucket
        if bucket not in quotas:
            candidate.selection_reason = "bucket_not_targeted:%s" % bucket
            continue
        if candidate.origin == "seed" and bucket in target_buckets:
            candidate.selection_reason = "seed_only_fills_control_buckets"
            continue
        if filled[bucket] >= quotas[bucket]:
            candidate.selection_reason = "quota_full:%s" % bucket
            continue
        filled[bucket] += 1
        candidate.selected = True
        candidate.selection_reason = "quota:%s" % bucket
    return {
        "quotas": quotas,
        "filled": filled,
        "shortfall": {b: quotas[b] - filled[b] for b in quotas if filled[b] < quotas[b]},
        "selected": sum(filled.values()),
    }


_DATE_NAME = re.compile(r"date|time|start|end|when|day", re.IGNORECASE)


def _slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", ".", str(text).lower()).strip(".") or "alias"


def hazard_candidates(
    task_ir: Mapping[str, Any],
    gold: Mapping[str, Any],
    config: HazardConfig,
    rates: Mapping[str, Mapping[str, Any]],
    world_id: str,
    selectors: Mapping[str, Mapping[str, Any]],
    max_per_task: int,
) -> List[Dict[str, Any]]:
    dag = dag_index(task_ir)
    gold_values: Dict[str, List[Dict[str, Any]]] = {}
    for entry in gold.get("values", ()):
        gold_values.setdefault(str(entry["node_id"]), []).append(entry)
    records: List[Dict[str, Any]] = []
    for node_id in dag.order:
        node = dag.nodes[node_id]
        sources = [
            (str(p["port_id"]), str(p.get("type") or ""), p["literal"])
            for p in node.get("inputs", ())
            if "literal" in p
        ] + [
            (str(v["name"]), str(v.get("type") or ""), v.get("value"))
            for v in gold_values.get(node_id, ())
        ]
        outgoing = next(
            (e for e in dag.value_edges() if str(e["from"]["node_id"]) == node_id), None
        )
        target_edge = {
            "from_node_id": node_id,
            "to_node_id": str(outgoing["to"]["node_id"]) if outgoing else node_id,
            "edge_id": outgoing.get("edge_id") if outgoing else None,
        }
        dates = [
            v["value"]
            for v in gold_values.get(node_id, ())
            if _DATE_NAME.search(str(v["name"])) and isinstance(v["value"], str)
        ]
        written = next(
            (w for w in node.get("writes", ()) if isinstance(w, Mapping) and w.get("entity_ref")),
            None,
        )
        read = next(
            (r for r in node.get("reads", ()) if isinstance(r, Mapping) and r.get("table")), None
        )
        target_cell = (
            {
                "table": read["table"],
                "column": read.get("column"),
                "entity_ref": read.get("entity_ref"),
            }
            if read
            else None
        )
        used = set()
        for hazard, selector in selectors.items():
            if hazard not in config.patches or hazard in used:
                continue
            pattern = re.compile(str(selector.get("pattern", "")), re.IGNORECASE)
            kind = selector.get("value_kind", "any")
            for name, type_name, value in sources:
                if not isinstance(value, (str, int, float)) or isinstance(value, bool):
                    continue
                if kind == "text" and not isinstance(value, str):
                    continue
                if not (pattern.search(type_name) or pattern.search(name)):
                    continue
                used.add(hazard)
                fill = {
                    "name": value,
                    "alias_email": "%s@alias.example" % _slug(str(value)),
                    "app": node["app"],
                    "path": value,
                    "title": value,
                    "start": dates[0] if dates else "",
                    "end": dates[1] if len(dates) > 1 else "",
                    "status_path": "/app_overrides/%s/%s/status"
                    % (node["app"], str(written["entity_ref"]).replace(":", "/"))
                    if written
                    else "/app_overrides/%s/status" % node["app"],
                }
                records.append(
                    injection_record(
                        config,
                        hazard,
                        str(task_ir["task_id"]),
                        world_id,
                        target_edge,
                        fill,
                        rates,
                        target_cell,
                    )
                )
                break
    records.sort(key=lambda r: (r["status"] != "pending", r["hazard_type"]))
    return records[:max_per_task]


def realization_step(
    candidate: Candidate,
    config: RealizationConfig,
    style_stats: Mapping[str, Any],
    persona_literals: Sequence[Any],
    client: Any = None,
    public_texts: Sequence[str] = (),
) -> Dict[str, Any]:
    fields = build_realization_fields(
        candidate.task_ir, candidate.gold, style_stats, public_context=list(public_texts)
    )
    prompt = render_realization_prompt(config, fields)
    result: Dict[str, Any] = {
        "status": "dry_run",
        "prompt_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
        "prompt": prompt,
        "instruction": None,
        "instructions": [],
        "llm_calls": [],
    }
    if client is None:
        return result
    system = config.base.prompt_system.read_text(encoding="utf-8")
    seen = set()
    for _ in range(max(1, config.realizations_per_task)):
        response = client.complete(system, prompt)
        result["llm_calls"].append(
            {
                "model": response.model,
                "usage": dict(response.usage),
                "request_sha256": response.request_sha256,
            }
        )
        try:
            parsed = parse_realization_reply(response.text)
        except Exception as exc:
            result["instructions"].append({"status": "parse_error", "error": str(exc)[:200]})
            continue
        instruction = str(parsed["instruction"]).strip()
        if instruction.lower() in seen:
            continue
        seen.add(instruction.lower())
        verdict = round_trip_compare(
            candidate.task_ir,
            None,
            candidate.gold,
            instruction,
            config,
            persona_literals,
            public_texts,
        )
        result["instructions"].append(
            {
                "instruction": instruction,
                "status": verdict["verdict"],
                "round_trip": {k: v for k, v in verdict.items() if k != "instruction"},
            }
        )
    accepted = [i for i in result["instructions"] if i.get("status") == "accepted"]
    if accepted:
        result["instruction"] = accepted[0]["instruction"]
        result["status"] = "accepted"
    elif any("instruction" in i for i in result["instructions"]):
        result["instruction"] = next(
            i["instruction"] for i in result["instructions"] if "instruction" in i
        )
        result["status"] = "re_realize"
    else:
        result["status"] = "parse_error"
    return result


def generation_record(
    candidate: Candidate, generation_version: str, min_variants: int = 2
) -> Dict[str, Any]:
    variants = [h for h in candidate.hazards if h["status"] == "pending"]
    return {
        "schema_version": GENERATION_RECORD_VERSION,
        "candidate_id": candidate.task_id,
        "status": "accepted" if not candidate.rejections and candidate.gold else "rejected",
        "origin": candidate.origin,
        "source_task_ids": lineage(candidate.task_ir)[0],
        "operators": lineage(candidate.task_ir)[1],
        "grafts": candidate.grafts,
        "score": candidate.score,
        "rejections": list(candidate.rejections),
        "bucket": candidate.bucket,
        "profile_ref": "profiles.jsonl#%s" % candidate.task_id if candidate.profile else None,
        "gold_execution": {
            "passed": candidate.gold is not None,
            "error": candidate.gold_error,
            "final_verifier_passed": (candidate.gold or {}).get("final_verifier_passed"),
        },
        "verifier_bundle_ref": "verifiers/%s.json" % candidate.task_id
        if candidate.verifier_bundle
        else None,
        "mutation_test": {
            "outcomes": len(candidate.mutation_outcomes or []),
            "rejection_rate": candidate.mutation_rejection_rate,
            "ref": "mutations/%s.jsonl" % candidate.task_id
            if candidate.mutation_outcomes
            else None,
        },
        "hazards": [h["injection_id"] for h in candidate.hazards],
        "variants": {
            "base_world": 1,
            "injectable": len(variants),
            "total": 1 + len(variants),
            "status": "ok" if 1 + len(variants) >= min_variants else "short",
        },
        "realization": {
            k: v for k, v in (candidate.realization or {}).items() if k not in ("prompt", "reply")
        },
        "selected": candidate.selected,
        "selection_reason": candidate.selection_reason,
        "release_eligible": False,
        "provenance": {
            "candidate_fingerprint": fingerprint(candidate.task_ir),
            "generation_version": generation_version,
            "llm_calls": list((candidate.realization or {}).get("llm_calls", ())),
        },
    }


@dataclass
class GenerationConfig:
    sampling: Mapping[str, Any]
    search: SearchConfig
    mutation_library: MutationLibrary
    hazards: HazardConfig
    hazard_selectors: Mapping[str, Mapping[str, Any]]
    realization: RealizationConfig
    registry: ValueTypeRegistry
    schema_graph: Optional[SchemaGraph]
    world_id: str
    generation_version: str = "gen_v1"
    mutation_max_per_node: int = 4
    run_mutation_test: bool = True
    seed: int = 0
    max_seeds: Optional[int] = None
    persona_literals: Sequence[Any] = ()
    workers: int = 1
    source_rubrics: Mapping[str, Sequence[Mapping[str, Any]]] = field(default_factory=dict)
    rubric_check: Optional[RubricCheckConfig] = None


def load_seeds(ir_dir: Union[str, Path], repository: Union[str, Path]) -> List[Dict[str, Any]]:
    return [load_task_ir(p, repository) for p in sorted(Path(ir_dir).glob("*.json"))]


def verify_candidate(
    task_ir: Mapping[str, Any],
    executor: GoldExecutor,
    config: GenerationConfig,
    rates: Mapping[str, Mapping[str, Any]],
    workdir: Union[str, Path],
    seeds: Optional[Mapping[str, Mapping[str, Any]]] = None,
) -> Dict[str, Any]:
    out: Dict[str, Any] = {
        "task_ir": dict(task_ir),
        "gold": None,
        "gold_error": None,
        "verifier_bundle": None,
        "mutation_outcomes": None,
        "mutation_rejection_rate": None,
        "rubric": None,
        "hazards": [],
    }
    gold = executor.run(task_ir)
    if gold is None:
        out["gold_error"] = executor.errors.get(fingerprint(task_ir))
        return out
    frozen = freeze_verifiers(task_ir, gold)
    gold = executor.run(frozen)
    if gold is None:
        out["gold_error"] = "FROZEN_VERIFIER_FAILED: %s" % executor.errors.get(fingerprint(frozen))
        return out
    out["task_ir"], out["gold"] = frozen, gold
    out["verifier_bundle"] = compile_verifiers(frozen)
    if config.rubric_check is not None and config.source_rubrics:
        sources = {}
        for task_id in lineage(frozen)[0]:
            seed = (seeds or {}).get(task_id)
            rubrics = config.source_rubrics.get(
                str((seed or {}).get("source_task_id") or task_id)
            )
            if rubrics is not None:
                sources[task_id] = {
                    "task_ir": seed,
                    "gold": executor.run(seed) if seed else None,
                    "rubrics": rubrics,
                }
        composition = config.sampling.get("composition") or {}
        out["rubric"] = compose_rubric(
            frozen,
            gold,
            sources,
            config.rubric_check,
            int((composition.get("rubric") or {}).get("min_literal_chars", 4)),
        )
    if config.run_mutation_test:
        outcomes = mutation_test(
            frozen,
            gold,
            config.mutation_library,
            executor.interpreter,
            executor.sources_for(frozen),
            Path(workdir) / ("mut_%s" % frozen["task_id"]),
            executor.repository,
            seed=config.seed,
            max_per_node=config.mutation_max_per_node,
            files_root=executor.files_root,
        )
        out["mutation_outcomes"] = [o.to_dict() for o in outcomes]
        out["mutation_rejection_rate"] = (
            mutation_rejection_rate(
                outcomes, str(config.sampling.get("gates", {}).get("execution_errors", "exclude"))
            )
            if outcomes
            else None
        )
    out["hazards"] = hazard_candidates(
        frozen,
        gold,
        config.hazards,
        rates,
        config.world_id,
        config.hazard_selectors,
        config.hazards.max_hazards_per_task,
    )
    return out


_WORKER: Dict[str, Any] = {}


def _worker_executor() -> GoldExecutor:
    state = _WORKER
    if "executor" not in state:
        parent: GoldExecutor = state["parent_executor"]
        state["executor"] = GoldExecutor(
            parent.interpreter,
            parent.database_dir,
            parent.world_id,
            parent.workdir / ("w%d" % os.getpid()),
            parent.repository,
            files_root=parent.files_root,
            app_databases=parent.app_databases,
        )
    return state["executor"]


def _worker_search(
    seed: Mapping[str, Any],
) -> List[Tuple[Dict[str, Any], Dict[str, Any], float, List[str], int]]:
    executor = _worker_executor()
    config: GenerationConfig = _WORKER["config"]
    results = beam_search(
        [seed],
        config.registry,
        config.search,
        config.schema_graph,
        executable=executor.executable,
        pool=_WORKER["pool"],
        gold_lookup=executor.run,
    )
    return [
        (dict(r.task_ir), dict(r.profile), r.score, list(r.rejections), r.grafts) for r in results
    ]


def _worker_verify(task_ir: Mapping[str, Any]) -> Dict[str, Any]:
    state = _WORKER
    _worker_executor()
    return verify_candidate(
        task_ir,
        state["executor"],
        state["config"],
        state["rates"],
        Path(state["workdir"]) / ("w%d" % os.getpid()),
        state["seeds"],
    )


def run_generation(
    seeds: Sequence[Mapping[str, Any]],
    config: GenerationConfig,
    executor: GoldExecutor,
    workdir: Union[str, Path],
    seed_instructions: Sequence[str] = (),
    realization_client: Any = None,
    progress: Optional[Progress] = None,
    instruction_of: Optional[Mapping[str, str]] = None,
) -> Dict[str, Any]:
    log = progress or (lambda _msg: None)
    seeds = list(seeds)[: config.max_seeds] if config.max_seeds else list(seeds)
    started = time.time()
    compat_index = build_compat_index(
        seeds, config.registry, config.schema_graph, config.search.compat
    )
    log("compat index: %d edges over %d seeds" % (len(compat_index), len(seeds)))

    candidates: Dict[str, Candidate] = {}
    for ir in seeds:
        scored = evaluate(ir, config.search, 0)
        candidate = Candidate.from_scored(scored, "seed")
        candidates[candidate.task_id] = candidate

    def record_search(
        index: int, ir: Mapping[str, Any], results: Sequence[Any], tick: float
    ) -> None:
        for item in results:
            scored = (
                item
                if isinstance(item, Scored)
                else Scored(item[0], item[1], item[2], item[3], item[4])
            )
            if scored.task_ir["task_id"] not in candidates:
                candidates[scored.task_ir["task_id"]] = Candidate.from_scored(scored, "composed")
        elapsed = time.time() - started
        log(
            "[%d/%d] %s %d composed %.1fs | elapsed %dm eta %dm"
            % (
                index,
                len(seeds),
                ir["task_id"],
                len(results),
                time.time() - tick,
                elapsed // 60,
                (elapsed / index * (len(seeds) - index)) // 60,
            )
        )

    if config.workers > 1 and len(seeds) > 1:
        import multiprocessing

        _WORKER.update({"parent_executor": executor, "config": config, "pool": seeds})
        context = multiprocessing.get_context("fork")
        with context.Pool(processes=min(config.workers, len(seeds))) as pool:
            tick = time.time()
            for index, (ir, results) in enumerate(
                zip(seeds, pool.imap(_worker_search, seeds, chunksize=1)), 1
            ):
                record_search(index, ir, results, tick)
                tick = time.time()
    else:
        for index, ir in enumerate(seeds, 1):
            tick = time.time()
            results = beam_search(
                [ir],
                config.registry,
                config.search,
                config.schema_graph,
                executable=executor.executable,
                pool=seeds,
                gold_lookup=executor.run,
            )
            record_search(index, ir, results, tick)

    rates = base_rates(config.hazards, executor.database_dir)
    style_stats = (
        style_stats_from_instructions(list(seed_instructions))
        if seed_instructions
        else {"default_length_words": list(config.realization.default_length_words)}
    )
    ordered = sorted(candidates.values(), key=lambda c: (-c.score, c.task_id))
    pending = [c for c in ordered if not c.rejections]
    seeds_by_id = {str(ir["task_id"]): ir for ir in seeds}

    def apply(candidate: Candidate, result: Mapping[str, Any]) -> None:
        candidate.task_ir = dict(result["task_ir"])
        for key in (
            "gold",
            "gold_error",
            "verifier_bundle",
            "mutation_outcomes",
            "mutation_rejection_rate",
            "rubric",
            "hazards",
        ):
            setattr(candidate, key, result[key])

    verified = 0
    if config.workers > 1 and len(pending) > 1:
        import multiprocessing

        _WORKER.update(
            {
                "parent_executor": executor,
                "config": config,
                "rates": rates,
                "workdir": workdir,
                "seeds": seeds_by_id,
            }
        )
        context = multiprocessing.get_context("fork")
        with context.Pool(processes=min(config.workers, len(pending))) as pool:
            for candidate, result in zip(
                pending, pool.imap(_worker_verify, [c.task_ir for c in pending], chunksize=1)
            ):
                apply(candidate, result)
                verified += 1
                if verified % 10 == 0 or verified == len(pending):
                    log("verified %d/%d candidates" % (verified, len(pending)))
    else:
        for candidate in pending:
            apply(
                candidate,
                verify_candidate(candidate.task_ir, executor, config, rates, workdir, seeds_by_id),
            )
            verified += 1
            if verified % 10 == 0 or verified == len(pending):
                log("verified %d/%d candidates" % (verified, len(pending)))
    selection = select_by_targets(ordered, config.sampling)
    for candidate in ordered:
        if candidate.selected:
            parents = lineage(candidate.task_ir)[0]
            candidate.realization = realization_step(
                candidate,
                config.realization,
                style_stats,
                config.persona_literals,
                realization_client,
                public_texts=[(instruction_of or {}).get(p, "") for p in parents],
            )
    log(
        "selected %d (%s); shortfall %s"
        % (selection["selected"], selection["filled"], selection["shortfall"])
    )
    return {
        "candidates": ordered,
        "compat_index": compat_index,
        "selection": selection,
        "base_rates": rates,
        "style_stats": style_stats,
        "seed_count": len(seeds),
        "gold_runs": executor.runs,
    }


def _jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8"
    )


def write_bundle(
    result: Mapping[str, Any], out: Union[str, Path], config: GenerationConfig
) -> Dict[str, Any]:
    out = Path(out)
    for sub in (
        "task_ir",
        "gold_lineage",
        "verifiers",
        "rubrics",
        "mutations",
        "records",
        "realization/prompts",
    ):
        (out / sub).mkdir(parents=True, exist_ok=True)
    candidates: Sequence[Candidate] = result["candidates"]
    _jsonl(out / "compat_index.jsonl", result["compat_index"])
    profiles, hazards, records, realized = [], [], [], []
    min_variants = int(config.sampling.get("min_variants_per_task", 2))
    for candidate in candidates:
        record = generation_record(candidate, config.generation_version, min_variants)
        records.append(record)
        (out / "records" / ("%s.json" % candidate.task_id)).write_text(
            json.dumps(record, indent=1, ensure_ascii=False) + "\n", encoding="utf-8"
        )
        if candidate.rejections:
            continue
        (out / "task_ir" / ("%s.json" % candidate.task_id)).write_text(
            json.dumps(candidate.task_ir, indent=1, ensure_ascii=False) + "\n", encoding="utf-8"
        )
        if candidate.profile:
            profiles.append(candidate.profile)
        if candidate.gold:
            (out / "gold_lineage" / ("%s.gold_lineage.json" % candidate.task_id)).write_text(
                json.dumps(candidate.gold, indent=1, ensure_ascii=False) + "\n", encoding="utf-8"
            )
        if candidate.verifier_bundle:
            (out / "verifiers" / ("%s.json" % candidate.task_id)).write_text(
                json.dumps(candidate.verifier_bundle, indent=1, ensure_ascii=False) + "\n",
                encoding="utf-8",
            )
        if candidate.rubric:
            (out / "rubrics" / ("%s.json" % candidate.task_id)).write_text(
                json.dumps(candidate.rubric, indent=1, ensure_ascii=False) + "\n", encoding="utf-8"
            )
        if candidate.mutation_outcomes:
            _jsonl(
                out / "mutations" / ("%s.jsonl" % candidate.task_id), candidate.mutation_outcomes
            )
        hazards.extend(candidate.hazards)
        if candidate.realization:
            (out / "realization/prompts" / ("%s.txt" % candidate.task_id)).write_text(
                candidate.realization["prompt"], encoding="utf-8"
            )
            for index, item in enumerate(candidate.realization.get("instructions", ())):
                if "instruction" in item:
                    realized.append(
                        {
                            "task_id": candidate.task_id,
                            "variant": index,
                            "instruction": item["instruction"],
                            "status": item["status"],
                        }
                    )
    _jsonl(out / "profiles.jsonl", profiles)
    _jsonl(out / "hazards.jsonl", hazards)
    _jsonl(out / "candidates.jsonl", records)
    if realized:
        _jsonl(out / "realization" / "instructions.jsonl", realized)
    selected = [c for c in candidates if c.selected]
    by_bucket: Dict[str, int] = {}
    for c in selected:
        by_bucket[c.bucket] = by_bucket.get(c.bucket, 0) + 1
    variants_short = [
        c.task_id
        for c in selected
        if 1 + sum(1 for h in c.hazards if h["status"] == "pending") < min_variants
    ]
    manifest = {
        "schema_version": BUNDLE_VERSION,
        "generation_version": config.generation_version,
        "world_id": config.world_id,
        "seeds": result["seed_count"],
        "candidates": len(candidates),
        "composed": sum(1 for c in candidates if c.origin == "composed"),
        "gold_executed": sum(1 for c in candidates if c.gold),
        "gold_runs": result["gold_runs"],
        "selected": len(selected),
        "selected_by_bucket": by_bucket,
        "selection": result["selection"],
        "hazard_base_rates": {k: v["count"] for k, v in result["base_rates"].items()},
        "hazard_records": len(hazards),
        "variants_short": variants_short,
        "min_variants_per_task": min_variants,
        "variant_counting": "base world + injectable hazards (D-037)",
        "realized": len(realized),
        "style_stats": result["style_stats"],
        "vm_side_pending": [
            "hazard injection seeding + consistency check (hazards.jsonl, status pending)",
            "round-trip re-extraction of realized instructions",
            "rollouts / replay verification",
        ],
    }
    (out / "generation_bundle.json").write_text(
        json.dumps(manifest, indent=1, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    lines = [
        "# Generation bundle",
        "",
        "seeds %d, candidates %d (composed %d), gold-executed %d, selected %d"
        % (
            manifest["seeds"],
            manifest["candidates"],
            manifest["composed"],
            manifest["gold_executed"],
            manifest["selected"],
        ),
        "",
        "| bucket | quota | filled |",
        "|---|---|---|",
    ]
    sel = result["selection"]
    lines += ["| %s | %d | %d |" % (b, sel["quotas"][b], sel["filled"][b]) for b in sel["quotas"]]
    lines += [
        "",
        "hazard base rates: %s" % manifest["hazard_base_rates"],
        "",
        "variants short (base world + injectable hazards < %d): %d"
        % (min_variants, len(variants_short)),
    ]
    (out / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return manifest
