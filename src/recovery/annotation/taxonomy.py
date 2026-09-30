"""Open-coding and frozen-taxonomy policy helpers."""

from __future__ import annotations

import itertools
import statistics
from collections import Counter, defaultdict
from typing import Any, Dict, Iterable, Mapping, Sequence, Tuple

from recovery.annotation.records import Adjudication, AnnotationError


def error_types_outside_seed(
    error_types: Iterable[str], taxonomy: Mapping[str, Any]
) -> Tuple[str, ...]:
    frozen = taxonomy.get("frozen")
    mode = taxonomy.get("mode")
    expected_mode = "frozen" if frozen is True else "open_coding"
    if not isinstance(frozen, bool) or mode != expected_mode:
        raise AnnotationError("taxonomy frozen/mode contract 不一致")
    seed = taxonomy.get("seed_labels")
    labels = taxonomy.get("labels")
    if not isinstance(seed, (list, tuple)) or not seed:
        raise AnnotationError("taxonomy seed_labels 不能为空")
    if not isinstance(labels, (list, tuple)) or not labels:
        raise AnnotationError("taxonomy labels 不能为空")
    outside = tuple(sorted(set(error_types) - set(seed)))
    if frozen is True and set(error_types) - set(labels):
        raise AnnotationError("adjudication 含 frozen taxonomy 外标签")
    return outside


def summarize_open_codes(
    adjudications: Sequence[Adjudication], seed_labels: Iterable[str]
) -> Dict[str, Any]:
    seed = set(seed_labels)
    label_counts: Counter[str] = Counter()
    outside_seed_counts: Counter[str] = Counter()
    reversibility_counts: Counter[str] = Counter()
    cooccurrence_counts: Counter[Tuple[str, str]] = Counter()
    horizon_values: Dict[str, list[int]] = defaultdict(list)
    horizon_missing: Counter[str] = Counter()
    multi_label_cases = 0

    for item in adjudications:
        labels = tuple(sorted(item.error_types))
        if len(labels) > 1:
            multi_label_cases += 1
        label_counts.update(labels)
        outside_seed_counts.update(label for label in labels if label not in seed)
        reversibility_counts[item.reversibility.value] += 1
        cooccurrence_counts.update(itertools.combinations(labels, 2))
        for label in labels:
            if item.error_horizon_actions is None:
                horizon_missing[label] += 1
            else:
                horizon_values[label].append(item.error_horizon_actions)

    horizon_summary = {}
    for label in sorted(label_counts):
        values = horizon_values[label]
        horizon_summary[label] = {
            "observed_count": len(values),
            "missing_count": horizon_missing[label],
            "min": min(values) if values else None,
            "median": statistics.median(values) if values else None,
            "max": max(values) if values else None,
            "values": sorted(values),
        }
    return {
        "adjudicated_case_count": len(adjudications),
        "multi_label_case_count": multi_label_cases,
        "error_type_counts": dict(sorted(label_counts.items())),
        "outside_seed_error_type_counts": dict(sorted(outside_seed_counts.items())),
        "reversibility_counts": dict(sorted(reversibility_counts.items())),
        "error_horizon_actions_by_type": horizon_summary,
        "cooccurrence_counts": [
            {"error_types": list(pair), "count": count}
            for pair, count in sorted(cooccurrence_counts.items())
        ],
    }
