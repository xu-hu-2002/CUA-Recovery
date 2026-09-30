from __future__ import annotations

import functools
import importlib
import json
import random
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple, Union

import yaml

from recovery.cases.repair import neutral_segments
from recovery.derived.layout import DEPTH_GRID
from recovery.world.facts import date_part, normalize_value

SAMPLE_VERSION = "training-sample/1.0"
CASE_VERSION = "recovery-case/1.0"
SUCCESS_SAMPLE_VERSION = "success-sample/1.0"
FORMATS = ("recovery", "success_control", "steps")
OBJECTIVES = ("nll", "weighted")


def _resolve(dotted: str) -> Any:
    module, _, attr = str(dotted).partition(":")
    return getattr(importlib.import_module(module), attr)


@dataclass(frozen=True)
class BuildConfig:
    format: str
    objective: str
    weighted: Mapping[str, float]
    training: Mapping[str, Any]
    success_control: Mapping[str, Any]
    ratios: Mapping[str, float]
    truncation_offsets: Tuple[int, ...]
    min_value_chars: int
    split_function: Callable[[str], Optional[str]] = field(repr=False)

    @classmethod
    def from_yaml(
        cls, path: Union[str, Path], repository: Optional[Union[str, Path]] = None
    ) -> "BuildConfig":
        raw = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
        if raw.get("schema_version") != "sft-build-config/2.0":
            raise ValueError("unsupported sft build config %r" % raw.get("schema_version"))
        if raw["format"] not in FORMATS or raw["loss"]["objective"] not in OBJECTIVES:
            raise ValueError("format must be in %s, loss.objective in %s" % (FORMATS, OBJECTIVES))
        steps = raw.get("steps") or {}
        return cls(
            format=str(raw["format"]),
            objective=str(raw["loss"]["objective"]),
            weighted={k: float(v) for k, v in raw["loss"]["weighted"].items()},
            training=dict(raw.get("training") or {}),
            success_control=dict(raw.get("success_control") or {}),
            ratios={k: float(v) for k, v in (steps.get("ratios") or {}).items()},
            truncation_offsets=tuple(int(x) for x in steps.get("offsets_after_evidence", ())),
            min_value_chars=int((steps.get("ledger") or {}).get("min_value_chars", 3)),
            split_function=split_function(raw["split"], repository or Path(path).parents[2]),
        )

    @property
    def loss_weights(self) -> Dict[str, float]:
        if self.objective == "weighted":
            return dict(self.weighted)
        return {"input": 0.0, "action": 1.0, "thought": 1.0, "ledger": 1.0, "check_min": 1.0}

    def split_of(self, workflow_id: str) -> Optional[str]:
        return self.split_function(str(workflow_id))

    def cluster_of(self, task_id: str) -> str:
        return str(task_id).split("#", 1)[0]


def split_function(section: Mapping[str, Any], repository: Union[str, Path]) -> Callable:
    function = _resolve(section["function"])
    if not section.get("file"):
        return function
    path = Path(section["file"])
    path = path if path.is_absolute() else Path(repository) / path
    loader = _resolve(section["loader"]) if section.get("loader") else None

    @functools.lru_cache(maxsize=None)
    def table() -> Any:
        return loader(path) if loader else json.loads(path.read_text(encoding="utf-8"))

    return lambda workflow_id: function(workflow_id, table())


def _param_values(step: Mapping[str, Any]) -> List[str]:
    out = []
    for param in step.get("params", ()):
        value = param.get("value")
        if isinstance(value, str):
            out.append(value)
        elif isinstance(value, list):
            out.extend(str(v) for v in value)
    return out


def _value_used(value: Any, later_params: Sequence[str]) -> bool:
    text = str(normalize_value(value))
    day = date_part(value)
    for param in later_params:
        lowered = param.lower()
        if text.lower() in lowered or (day and day in lowered):
            return True
    return False


def ledger_upto(
    trace: Mapping[str, Any],
    upto: int,
    config: BuildConfig,
    gold_facts: Sequence[Tuple[str, str, str]] = (),
) -> List[Dict[str, Any]]:
    steps = [s for s in trace["steps"] if int(s["action_index"]) <= upto]
    later: Dict[int, List[str]] = {}
    for step in trace["steps"]:
        index = int(step["action_index"])
        later[index] = [
            v for s in trace["steps"] if int(s["action_index"]) > index for v in _param_values(s)
        ]
    entries: List[Dict[str, Any]] = []
    seen = set()
    gold_keys = set(gold_facts)
    for step in steps:
        index = int(step["action_index"])
        for event in step.get("observations", ()):
            for fact in event.get("facts", ()):
                value = fact.get("value")
                if value is None or len(str(value)) < config.min_value_chars:
                    continue
                key = (fact["table"], fact["column"], fact["entity"])
                if key in seen:
                    continue
                if key in gold_keys or _value_used(value, later.get(index, [])):
                    seen.add(key)
                    entries.append(
                        {
                            "name": "%s.%s@%s" % key,
                            "value": value,
                            "source_action_index": index,
                            "source_ref": "obs:%d:%s" % (index, event.get("source", "gui")),
                        }
                    )
    return entries


def _history(trace: Mapping[str, Any], upto: int) -> List[Dict[str, Any]]:
    return [
        {
            "action_index": int(s["action_index"]),
            "thought": s.get("thought"),
            "action": s["action"],
            "observation_ref": "obs:%d" % int(s["action_index"]),
        }
        for s in trace["steps"]
        if int(s["action_index"]) < upto
    ]


def _sample(
    kind: str,
    trace: Mapping[str, Any],
    index: int,
    ledger: List[Dict[str, Any]],
    check: str,
    thought: str,
    action: Any,
    config: BuildConfig,
    *,
    hint_level: str = "none",
    teacher: str = "none",
    source: str = "real_rollout",
    evidence: Optional[Mapping[str, Any]] = None,
    check_weight: float = 1.0,
) -> Dict[str, Any]:
    return {
        "schema_version": SAMPLE_VERSION,
        "sample_id": "%s_%s_%d" % (trace["rollout_id"], kind, index),
        "sample_kind": kind,
        "source": source,
        "rollout_id": str(trace["rollout_id"]),
        "task_id": str(trace["task_id"]),
        "agent": str(trace["agent"]),
        "truncation_action_index": int(index),
        "input": {
            "instruction": _instruction(trace),
            "history": _history(trace, index),
            "observation_ref": "obs:%d" % index,
        },
        "target": {"ledger": ledger, "check": check, "thought": thought, "action": action},
        "evidence_refs": dict(evidence or {"verified": kind != "detection"}),
        "loss_weights": {
            "input": config.loss_weights["input"],
            "action": config.loss_weights["action"],
            "thought": config.loss_weights["thought"],
            "ledger": config.loss_weights["ledger"],
            "check": check_weight,
        },
        "hint_level": hint_level,
        "teacher": teacher,
        "modality": str(trace.get("modality", {}).get("primary", "gui")),
        "split": config.split_of(str(trace["task_id"])),
        "cluster_id": config.cluster_of(str(trace["task_id"])),
        "provenance": {},
    }


def base_samples(
    trace: Mapping[str, Any], config: BuildConfig, gold_facts: Sequence[Tuple[str, str, str]] = ()
) -> List[Dict[str, Any]]:
    last = max(int(s["action_index"]) for s in trace["steps"])
    removed = {
        i
        for seg in neutral_segments(trace, last + 1)
        for i in range(seg["start_action_index"], seg["end_action_index"] + 1)
    }
    out = []
    for step in trace["steps"]:
        index = int(step["action_index"])
        if index in removed or step["action"]["type"] in ("done", "fail"):
            continue
        out.append(
            _sample(
                "base",
                trace,
                index,
                ledger_upto(trace, index - 1, config, gold_facts),
                "consistent",
                str(step.get("thought") or ""),
                step["action"],
                config,
            )
        )
    return out


def _wrong_value_entries(
    ledger: Sequence[Mapping[str, Any]], wrong_value: Any
) -> List[Mapping[str, Any]]:
    return [
        e
        for e in ledger
        if wrong_value is not None
        and str(normalize_value(wrong_value)) == str(normalize_value(e["value"]))
    ]


def evidence_refs(
    trace: Mapping[str, Any],
    analysis: Mapping[str, Any],
    cut: int,
    ledger: Sequence[Mapping[str, Any]],
) -> Dict[str, Any]:
    evidence_step = analysis.get("earliest_identifiable_action_index")
    root = analysis.get("root_cause_action_index")
    detail = (analysis.get("provenance") or {}).get("root_detail") or {}
    exposing = next((s for s in trace["steps"] if int(s["action_index"]) == evidence_step), None)
    contradiction = None
    for event in (exposing or {}).get("observations", ()):
        for fact in event.get("facts", ()):
            if fact.get("entity") in (
                detail.get("gold_entity"),
                detail.get("wrong_entity"),
            ) or (
                detail.get("gold_value") is not None
                and str(fact.get("value", ""))[:10] in str(detail["gold_value"])
            ):
                contradiction = dict(fact, action_index=evidence_step)
                break
        if contradiction:
            break
    carried = None
    for entry in ledger:
        if detail.get("wrong_value") is not None and str(
            normalize_value(detail["wrong_value"])
        ) == str(normalize_value(entry["value"])):
            carried = dict(entry)
    if carried is None and detail.get("wrong_value") is not None:
        for s in trace["steps"]:
            if int(s["action_index"]) == root:
                for param in s.get("params", ()):
                    if str(normalize_value(param.get("value"))) == str(
                        normalize_value(detail["wrong_value"])
                    ):
                        carried = {
                            "name": "param:%s" % param["name"],
                            "value": param["value"],
                            "source_action_index": root,
                            "source_ref": "param:%d" % root,
                        }
    affected = None
    for s in trace["steps"]:
        if root is not None and root <= int(s["action_index"]) <= cut:
            for row in s.get("delta", ()):
                affected = {
                    "action_index": int(s["action_index"]),
                    "db": row["db"],
                    "tbl": row["tbl"],
                    "rowid": row["rowid"],
                    "op": row["op"],
                    "seq": row["seq"],
                }
                break
            if affected:
                break
    return {
        "contradiction_fact": contradiction,
        "carried_value_ref": carried,
        "affected_write_ref": affected,
        "verified": contradiction is not None and carried is not None,
    }


def detection_samples(
    trace: Mapping[str, Any],
    analysis: Mapping[str, Any],
    config: BuildConfig,
    gold_facts: Sequence[Tuple[str, str, str]] = (),
    gold_action: Optional[Mapping[str, Any]] = None,
) -> List[Dict[str, Any]]:
    evidence_step = analysis.get("earliest_identifiable_action_index")
    root = analysis.get("root_cause_action_index")
    if evidence_step is None or root is None:
        return []
    last = max(int(s["action_index"]) for s in trace["steps"])
    out = []
    for offset in config.truncation_offsets:
        cut = evidence_step + offset
        if cut > last:
            break
        ledger = ledger_upto(trace, cut - 1, config, gold_facts)
        refs = evidence_refs(trace, analysis, cut, ledger)
        contradiction = refs["contradiction_fact"]
        carried = refs["carried_value_ref"]
        affected = refs["affected_write_ref"]
        thought = "The current observation shows %s = %r, but I carried %r (from step %s)%s." % (
            (contradiction or {}).get("column", "?"),
            (contradiction or {}).get("value"),
            (carried or {}).get("value"),
            (carried or {}).get("source_action_index"),
            (
                "; step %d already wrote %s.%s:%s"
                % (affected["action_index"], affected["db"], affected["tbl"], affected["rowid"])
            )
            if affected
            else "",
        )
        out.append(
            _sample(
                "detection",
                trace,
                cut,
                ledger,
                "inconsistent",
                thought,
                None,
                config,
                evidence=refs,
            )
        )
        if gold_action is not None:
            out[-1]["target"]["action"] = dict(gold_action)
        out[-1]["provenance"] = {
            "action_source": "programme" if gold_action is not None else None,
            "root_cause_action_index": root,
            "evidence_step": evidence_step,
        }
    return [s for s in out if s["evidence_refs"]["verified"]]


def verification_samples(
    trace: Mapping[str, Any],
    analysis: Mapping[str, Any],
    gold: Mapping[str, Any],
    task_ir: Mapping[str, Any],
    profile: Optional[Mapping[str, Any]],
    config: BuildConfig,
) -> List[Dict[str, Any]]:
    if not analysis.get("horizon_censored") or analysis.get("root_cause_action_index") is None:
        return []
    root = int(analysis["root_cause_action_index"])
    horizons = ((profile or {}).get("provenance") or {}).get("node_horizons") or {}
    root_node = analysis.get("root_cause_node_id")
    eligible = (
        any(
            h.get("observability_class") == "verifier_only" or h.get("cross_app")
            for h in horizons.values()
        )
        if horizons
        else False
    )
    if not eligible:
        return []
    r3_tables = {
        w["table"] for w in gold.get("writes_gold", ()) if w.get("reversibility_class") == "R3"
    }
    for step in trace["steps"]:
        index = int(step["action_index"])
        if index <= root:
            continue
        if any("%s.%s" % (r["db"], r["tbl"]) in r3_tables for r in step.get("delta", ())):
            reads = (
                [r for r in gold.get("resolved_reads", ()) if r["node_id"] == root_node]
                if root_node
                else []
            )
            source = reads[0] if reads else None
            ledger = ledger_upto(trace, index - 1, config)
            thought = (
                "About to perform an irreversible action; the value it uses came from step %d "
                "and has not been re-checked since." % root
            )
            action = {"type": "reread", "source": source} if source else {"type": "reread"}
            sample = _sample(
                "verification", trace, index, ledger, "consistent", thought, action, config
            )
            sample["provenance"] = {
                "inserted_before_action_index": index,
                "irreversible_tables": sorted(r3_tables),
            }
            return [sample]
    return []


def negative_samples(
    trace: Mapping[str, Any],
    analysis: Optional[Mapping[str, Any]],
    config: BuildConfig,
    count: int,
    rng: random.Random,
) -> List[Dict[str, Any]]:
    limit = (
        int(analysis["root_cause_action_index"])
        if analysis and analysis.get("root_cause_action_index") is not None
        else None
    )
    candidates = [
        s
        for s in trace["steps"]
        if (limit is None or int(s["action_index"]) < limit)
        and s["action"]["type"] not in ("done", "fail")
    ]
    rng.shuffle(candidates)
    out = []
    for step in candidates[:count]:
        index = int(step["action_index"])
        out.append(
            _sample(
                "negative",
                trace,
                index,
                ledger_upto(trace, index - 1, config),
                "consistent",
                "Consistent with what I have read so far.",
                step["action"],
                config,
            )
        )
    return out


def _instruction(trace: Mapping[str, Any]) -> str:
    return str(
        trace.get("instruction")
        or trace.get("provenance", {}).get("instruction")
        or trace["task_id"]
    )


def stripped_history(trace: Mapping[str, Any], upto: int) -> List[Dict[str, Any]]:
    return [
        {
            "action_index": int(s["action_index"]),
            "action": s["action"],
            "observation_ref": "obs:%d" % int(s["action_index"]),
        }
        for s in trace["steps"]
        if int(s["action_index"]) <= upto
    ]


def target_loss_weights(config: BuildConfig) -> Dict[str, float]:
    weights = config.loss_weights
    return {k: weights[k] for k in ("input", "thought", "action")}


def recovery_cases(
    trace: Mapping[str, Any],
    analysis: Mapping[str, Any],
    config: BuildConfig,
    gold_facts: Sequence[Tuple[str, str, str]] = (),
) -> List[Dict[str, Any]]:
    root = analysis.get("root_cause_action_index")
    workflow = str(trace["task_id"])
    split = config.split_of(workflow)
    if root is None or split != "train":
        return []
    root = int(root)
    last = max(int(s["action_index"]) for s in trace["steps"])
    out = []
    for depth in DEPTH_GRID:
        cut = root + depth
        if cut > last:
            break
        out.append(
            {
                "schema_version": CASE_VERSION,
                "sample_id": "%s_d%d" % (trace["rollout_id"], depth),
                "workflow_id": workflow,
                "rollout_id": str(trace["rollout_id"]),
                "agent": str(trace["agent"]),
                "split": split,
                "root_cause_action_index": root,
                "depth": depth,
                "cut_action_index": cut,
                "evidence_step": analysis.get("earliest_identifiable_action_index"),
                "input": {
                    "instruction": _instruction(trace),
                    "history": stripped_history(trace, cut),
                },
                "evidence_refs": evidence_refs(
                    trace, analysis, cut, ledger_upto(trace, cut, config, gold_facts)
                ),
                "provenance": dict(trace.get("provenance") or {}),
            }
        )
    return out


def success_samples(trace: Mapping[str, Any], config: BuildConfig) -> List[Dict[str, Any]]:
    workflow = str(trace["task_id"])
    split = config.split_of(workflow)
    if split != "train" or (trace.get("outcome") or {}).get("final_verifier") is not True:
        return []
    return [
        {
            "schema_version": SUCCESS_SAMPLE_VERSION,
            "sample_id": "%s_success" % trace["rollout_id"],
            "sample_kind": "success",
            "workflow_id": workflow,
            "rollout_id": str(trace["rollout_id"]),
            "agent": str(trace["agent"]),
            "split": split,
            "input": {"instruction": _instruction(trace), "history": []},
            "target": {
                "steps": [
                    {
                        "action_index": int(s["action_index"]),
                        "thought": s.get("thought"),
                        "action": s["action"],
                    }
                    for s in trace["steps"]
                ]
            },
            "loss_weights": target_loss_weights(config),
        }
    ]


def token_estimate(sample: Mapping[str, Any], config: BuildConfig) -> float:
    control = config.success_control
    steps = list(sample["target"]["steps"])
    parts = [str(s.get("thought") or "") + json.dumps(s.get("action"), default=str) for s in steps]
    images = len(steps)
    if control.get("counted", "input+target") == "input+target":
        history = list(sample["input"]["history"])
        parts.append(str(sample["input"]["instruction"]))
        parts += [json.dumps(h.get("action"), default=str) for h in history]
        images += len(history)
    images = min(images, int(config.training.get("max_images_per_sample", images)))
    chars = sum(len(p) for p in parts)
    return chars / float(control.get("chars_per_token", 4.0)) + images * float(
        control.get("tokens_per_image", 0)
    )


def token_matched(
    control: Sequence[Mapping[str, Any]],
    recovery: Sequence[Mapping[str, Any]],
    config: BuildConfig,
) -> Tuple[List[Mapping[str, Any]], Dict[str, Any]]:
    settings = config.success_control
    tolerance = float(settings.get("tolerance", 0.02))
    per_workflow = settings.get("match", "global") == "per_workflow"
    rng = random.Random(int(settings.get("seed", 0)))
    workflows = {str(s["workflow_id"]) for s in recovery}
    targets: Dict[str, float] = defaultdict(float)
    pools: Dict[str, List[Mapping[str, Any]]] = defaultdict(list)
    for sample in recovery:
        targets[str(sample["workflow_id"]) if per_workflow else "*"] += token_estimate(
            sample, config
        )
    for sample in sorted(control, key=lambda s: str(s["sample_id"])):
        if str(sample["workflow_id"]) in workflows:
            pools[str(sample["workflow_id"]) if per_workflow else "*"].append(sample)
    selected: List[Mapping[str, Any]] = []
    groups = {}
    for key in sorted(targets):
        pool = list(pools.get(key, ()))
        rng.shuffle(pool)
        target, total = targets[key], 0.0
        for sample in pool:
            if total >= target * (1 - tolerance):
                break
            tokens = token_estimate(sample, config)
            if total + tokens <= target * (1 + tolerance):
                selected.append(sample)
                total += tokens
        groups[key] = {
            "target_tokens": round(target, 1),
            "matched_tokens": round(total, 1),
            "within_tolerance": abs(total - target) <= tolerance * target,
        }
    return selected, {
        "match": "per_workflow" if per_workflow else "global",
        "tolerance": tolerance,
        "target_tokens": round(sum(targets.values()), 1),
        "matched_tokens": round(sum(g["matched_tokens"] for g in groups.values()), 1),
        "within_tolerance": all(g["within_tolerance"] for g in groups.values()),
        "groups": groups if per_workflow else None,
    }


def balance_check_weights(samples: List[Dict[str, Any]], config: BuildConfig) -> Dict[str, float]:
    counts = Counter(s["target"]["check"] for s in samples)
    total = sum(counts.values()) or 1
    weights = {}
    for klass, count in counts.items():
        weights[klass] = (
            max(config.loss_weights["check_min"], round(total / (len(counts) * count), 4))
            if config.objective == "weighted"
            else 1.0
        )
    for sample in samples:
        sample["loss_weights"]["check"] = weights[sample["target"]["check"]]
    return weights


def dataset_manifest(
    samples: Sequence[Mapping[str, Any]],
    config: BuildConfig,
    check_weights: Optional[Mapping[str, float]] = None,
    **extra: Any,
) -> Dict[str, Any]:
    manifest = {
        "schema_version": "sft-dataset-manifest/2.0",
        "format": config.format,
        "samples": len(samples),
        "by_kind": dict(Counter(s["sample_kind"] for s in samples)),
        "by_split": dict(Counter(s["split"] for s in samples)),
        "workflows": len(
            {config.cluster_of(s.get("workflow_id") or s["task_id"]) for s in samples}
        ),
        "loss": {"objective": config.objective, "weights": config.loss_weights},
        "training": dict(config.training),
    }
    if config.format == "steps":
        manifest.update(
            by_modality=dict(Counter(s["modality"] for s in samples)),
            check_class_frequencies=dict(Counter(s["target"]["check"] for s in samples)),
            check_weights=dict(check_weights or {}),
            ratios_target=dict(config.ratios),
        )
    manifest.update(extra)
    return manifest
