from __future__ import annotations

import hashlib
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Union

import yaml

HAZARD_TYPES = (
    "identity_alias",
    "stale_version",
    "cross_app_inconsistency",
    "constraint_drift",
    "distractor_candidates",
    "precondition_gap",
)


@dataclass(frozen=True)
class HazardConfig:
    base_rate_queries: Mapping[str, Sequence[Mapping[str, str]]]
    patches: Mapping[str, Mapping[str, Any]]
    base_rate_multiplier_max: float
    max_hazards_per_task: int
    min_variants_per_task: int

    @classmethod
    def from_yaml(cls, path: Union[str, Path]) -> "HazardConfig":
        raw = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
        if raw.get("schema_version") != "hazards-config/1.0":
            raise ValueError("unsupported hazards config %r" % raw.get("schema_version"))
        return cls(
            base_rate_queries=raw.get("base_rate_queries") or {},
            patches=raw.get("patches") or {},
            base_rate_multiplier_max=float(raw.get("base_rate_multiplier_max", 2.0)),
            max_hazards_per_task=int(raw.get("max_hazards_per_task", 2)),
            min_variants_per_task=int(raw.get("min_variants_per_task", 2)),
        )


def base_rates(config: HazardConfig, database_dir: Union[str, Path]) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    for hazard, queries in config.base_rate_queries.items():
        units = []
        for spec in queries:
            path = Path(database_dir) / ("%s.sqlite" % spec["app"])
            if not path.is_file():
                units.append(
                    {
                        "app": spec["app"],
                        "unit": spec["unit"],
                        "count": None,
                        "note": "database missing",
                    }
                )
                continue
            conn = sqlite3.connect("file:%s?mode=ro" % path, uri=True)
            try:
                count = conn.execute(spec["query"]).fetchone()[0]
            except sqlite3.Error as exc:
                units.append(
                    {
                        "app": spec["app"],
                        "unit": spec["unit"],
                        "count": None,
                        "note": str(exc)[:120],
                    }
                )
                continue
            finally:
                conn.close()
            units.append({"app": spec["app"], "unit": spec["unit"], "count": int(count or 0)})
        out[hazard] = {"count": sum(u["count"] or 0 for u in units), "units": units}
    return out


def injectable(hazard: str, rates: Mapping[str, Mapping[str, Any]]) -> bool:
    return rates.get(hazard, {}).get("count", 0) > 0


def render_patch(
    config: HazardConfig, hazard: str, values: Mapping[str, Any]
) -> List[Dict[str, Any]]:
    spec = config.patches.get(hazard)
    if spec is None:
        raise ValueError("no patch template for hazard %s" % hazard)

    def fill(item: Any) -> Any:
        if isinstance(item, str):
            for key, value in values.items():
                item = item.replace("{%s}" % key, str(value))
            return item
        if isinstance(item, list):
            return [fill(x) for x in item]
        if isinstance(item, dict):
            return {k: fill(v) for k, v in item.items()}
        return item

    return [fill(dict(op)) for op in spec["template"]]


def injection_record(
    config: HazardConfig,
    hazard: str,
    task_id: str,
    base_world_id: str,
    target_edge: Mapping[str, str],
    values: Mapping[str, Any],
    rates: Mapping[str, Mapping[str, Any]],
    target_cell: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    injection_id = (
        "hz-%s"
        % hashlib.sha256(
            ("%s|%s|%s|%s" % (task_id, hazard, target_edge, sorted(values.items()))).encode()
        ).hexdigest()[:10]
    )
    rate = float(rates.get(hazard, {}).get("count", 0))
    status = "pending" if injectable(hazard, rates) else "BASE_RATE_ZERO"
    return {
        "schema_version": "hazard-injection/1.0",
        "injection_id": injection_id,
        "base_world_id": base_world_id,
        "variant_world_id": "%s+%s" % (base_world_id, injection_id),
        "task_id": task_id,
        "hazard_type": hazard,
        "target_edge": {
            "from_node_id": target_edge["from_node_id"],
            "to_node_id": target_edge["to_node_id"],
            "edge_id": target_edge.get("edge_id"),
        },
        "target_paper_type": str(config.patches.get(hazard, {}).get("target_paper_type", "")),
        "target_cell": dict(target_cell or {}),
        "persona_diff": render_patch(config, hazard, values) if hazard in config.patches else [],
        "profile_before": None,
        "profile_after": None,
        "gold_unique": None,
        "seeder_consistent": None,
        "status": status,
        "hazard_provenance": {
            "base_rate_before": rate,
            "base_rate_after": rate + 1 if status == "pending" else rate,
            "base_rate_unit": ",".join(u["unit"] for u in rates.get(hazard, {}).get("units", [])),
        },
        "provenance": {"multiplier_max": config.base_rate_multiplier_max},
    }
