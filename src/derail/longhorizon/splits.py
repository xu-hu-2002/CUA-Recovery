"""Frozen source splits and the Phase 1 development sample (manual v0.2 sections 17, 18.3).

Source tasks are split ``source-development / source-pilot / source-holdout`` by cluster, so a
story or fixture family never straddles a split.  Until Task IR exists the cluster key is a
configurable tuple of task fields (default: the sorted application set); the choice is recorded
in the output so it can be revisited once semantic clusters are available.
"""

from __future__ import annotations

import hashlib
import json
import random
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

SPLIT_NAMES = ("source-development", "source-pilot", "source-holdout")
SPLITS_SCHEMA_VERSION = "source-splits/0.1"


@dataclass(frozen=True)
class SplitConfig:
    seed: int
    ratios: Tuple[float, float, float] = (0.6, 0.2, 0.2)
    cluster_fields: Tuple[str, ...] = ("apps_involved",)
    dev_sample_size: int = 16
    dev_strata_fields: Tuple[str, ...] = ("category", "multi_app", "has_failure_annotation")

    def __post_init__(self) -> None:
        if len(self.ratios) != 3 or abs(sum(self.ratios) - 1.0) > 1e-9 or min(self.ratios) <= 0:
            raise ValueError("ratios must be three positive numbers summing to 1")
        if self.dev_sample_size <= 0:
            raise ValueError("dev_sample_size must be positive")

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "SplitConfig":
        return cls(
            seed=int(raw["seed"]),
            ratios=tuple(float(item) for item in raw.get("ratios", (0.6, 0.2, 0.2))),
            cluster_fields=tuple(
                str(item) for item in raw.get("cluster_fields", ("apps_involved",))
            ),
            dev_sample_size=int(raw.get("dev_sample_size", 16)),
            dev_strata_fields=tuple(
                str(item)
                for item in raw.get(
                    "dev_strata_fields", ("category", "multi_app", "has_failure_annotation")
                )
            ),
        )


def _field_value(task: Mapping[str, Any], name: str) -> str:
    value = task.get(name)
    if isinstance(value, (list, tuple, set)):
        return "|".join(sorted(str(item) for item in value))
    return str(value)


def cluster_key(task: Mapping[str, Any], fields: Sequence[str]) -> str:
    """Stable cluster identifier from the configured task fields."""

    payload = json.dumps([_field_value(task, name) for name in fields], ensure_ascii=False)
    return "cluster_%s" % hashlib.sha256(payload.encode("utf-8")).hexdigest()[:12]


def assign_splits(tasks: Sequence[Mapping[str, Any]], config: SplitConfig) -> Dict[str, str]:
    """Return ``task_id -> split`` with whole clusters assigned together.

    Clusters are shuffled with the seed, then placed largest-first into the split that is
    furthest below its target share, so big families cannot overflow one split.
    """

    clusters: Dict[str, List[str]] = defaultdict(list)
    for task in tasks:
        clusters[cluster_key(task, config.cluster_fields)].append(str(task["id"]))
    rng = random.Random(config.seed)
    names = sorted(clusters)
    rng.shuffle(names)
    names.sort(key=lambda name: -len(clusters[name]))  # stable among equal sizes
    total = float(len(tasks))
    filled = {name: 0 for name in SPLIT_NAMES}
    assignment: Dict[str, str] = {}
    for name in names:
        target = min(
            SPLIT_NAMES,
            key=lambda split: (filled[split] / total) - config.ratios[SPLIT_NAMES.index(split)],
        )
        for task_id in clusters[name]:
            assignment[task_id] = target
        filled[target] += len(clusters[name])
    return assignment


def enrich_tasks(
    tasks: Iterable[Mapping[str, Any]],
    *,
    task_types: Mapping[str, str],
    task_horizons: Mapping[str, str],
    failure_task_ids: Iterable[str],
) -> List[Dict[str, Any]]:
    """Attach the stratification fields used by splits and the development sample."""

    failed = set(failure_task_ids)
    enriched = []
    for task in tasks:
        task_id = str(task["id"])
        record = dict(task)
        record["task_type"] = task_types.get(task_id, "unknown")
        record["horizon"] = task_horizons.get(task_id, task.get("horizon"))
        record["multi_app"] = len(task.get("apps_involved", [])) > 1
        record["has_failure_annotation"] = task_id in failed
        enriched.append(record)
    return enriched


def select_dev_sample(
    tasks: Sequence[Mapping[str, Any]], assignment: Mapping[str, str], config: SplitConfig
) -> List[str]:
    """Stratified development sample drawn only from ``source-development``.

    Strata are visited round-robin in a seeded order, one task per visit, so rare combinations
    are never skipped in favour of easy tasks.
    """

    rng = random.Random(config.seed + 1)
    eligible = [task for task in tasks if assignment.get(str(task["id"])) == SPLIT_NAMES[0]]
    strata: Dict[Tuple[str, ...], List[str]] = defaultdict(list)
    for task in eligible:
        key = tuple(_field_value(task, name) for name in config.dev_strata_fields)
        strata[key].append(str(task["id"]))
    for members in strata.values():
        rng.shuffle(members)
    order = sorted(strata)
    rng.shuffle(order)
    selected: List[str] = []
    while len(selected) < config.dev_sample_size and any(strata[key] for key in order):
        for key in order:
            if strata[key] and len(selected) < config.dev_sample_size:
                selected.append(strata[key].pop())
    return sorted(selected)


def build_split_record(
    tasks: Sequence[Mapping[str, Any]], config: SplitConfig, *, source_sha256: str
) -> Dict[str, Any]:
    assignment = assign_splits(tasks, config)
    dev_sample = select_dev_sample(tasks, assignment, config)
    counts = Counter(assignment.values())
    chosen = set(dev_sample)
    strata = Counter(
        tuple(_field_value(task, name) for name in config.dev_strata_fields)
        for task in tasks
        if str(task["id"]) in chosen
    )
    return {
        "schema_version": SPLITS_SCHEMA_VERSION,
        "seed": config.seed,
        "ratios": list(config.ratios),
        "cluster_fields": list(config.cluster_fields),
        "cluster_key_status": "provisional_until_task_ir_clusters_exist",
        "source_sha256": source_sha256,
        "split_counts": {name: counts.get(name, 0) for name in SPLIT_NAMES},
        "assignment": dict(sorted(assignment.items())),
        "dev_sample": {
            "size": len(dev_sample),
            "strata_fields": list(config.dev_strata_fields),
            "task_ids": dev_sample,
            "strata_counts": [
                {"stratum": list(key), "count": count} for key, count in sorted(strata.items())
            ],
        },
    }
