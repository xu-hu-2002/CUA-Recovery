"""Offline preflight: does a hazard patch survive the guest seeder and move its claimed rate unit."""

from __future__ import annotations

import argparse
import hashlib
import json
import sqlite3
import sys
import tempfile
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from derail.gen.hazards import HazardConfig  # noqa: E402

CALENDAR_APP = "hoolicalendar"
CALENDAR_OVERRIDE_PATH = "/app_overrides/calendar/events/-"
OPTION_A_STRATEGY = "base-singleton-title-prefix/1.0"
TITLE_PREFIX_RATE_QUERY = (
    "SELECT COUNT(*) FROM (SELECT substr(title, 1, 12) p FROM events "
    "GROUP BY p HAVING COUNT(*) > 1)"
)


def _query_for(config: HazardConfig, hazard: str, unit: str) -> Optional[str]:
    for spec in config.base_rate_queries.get(hazard, ()):
        if spec.get("unit") == unit:
            return str(spec["query"])
    return None


def _override_value(record: Mapping[str, Any]) -> Optional[Mapping[str, Any]]:
    ops = record.get("persona_diff") or ()
    if len(ops) != 1 or ops[0].get("path") != CALENDAR_OVERRIDE_PATH:
        return None
    value = ops[0].get("value")
    return value if isinstance(value, dict) else None


def calendar_override_survives(value: Mapping[str, Any]) -> bool:
    """Mirror of seed_calendar's two falsy guards on summary and start."""

    return bool(value.get("summary")) and bool(value.get("start"))


def singleton_prefix_titles(db: Path) -> List[str]:
    query = (
        "SELECT MIN(title) FROM events WHERE title IS NOT NULL AND length(title) >= 12 "
        "GROUP BY substr(title, 1, 12) HAVING COUNT(*) = 1 ORDER BY MIN(title)"
    )
    conn = sqlite3.connect("file:%s?mode=ro&immutable=1" % db, uri=True)
    try:
        return [str(row[0]) for row in conn.execute(query)]
    finally:
        conn.close()


def _select_option_a_title(injection_id: str, titles: Sequence[str]) -> str:
    if not titles:
        raise ValueError("frozen BASE world has no singleton-prefix calendar titles")
    return min(titles, key=lambda title: hashlib.sha256(
        (injection_id + "|" + title).encode("utf-8")).hexdigest())


def realize_option_a(record: Mapping[str, Any], db_dir: Path,
                     reference_time: str) -> Dict[str, Any]:
    repaired = json.loads(json.dumps(record))
    source_id = str(repaired["injection_id"])
    title = _select_option_a_title(
        source_id, singleton_prefix_titles(db_dir / "hoolicalendar.sqlite"))
    value = repaired["persona_diff"][0]["value"]
    source_summary = value.get("summary")
    value.update({"summary": title + " (prep)", "start": reference_time,
                  "end": reference_time})
    payload = json.dumps(repaired["persona_diff"], sort_keys=True, separators=(",", ":"))
    new_id = "hz-" + hashlib.sha256((source_id + "|" + payload).encode()).hexdigest()[:10]
    repaired["injection_id"] = new_id
    repaired["variant_world_id"] = "%s+%s" % (repaired["base_world_id"], new_id)
    base_rate = float(_rate(db_dir / "hoolicalendar.sqlite", TITLE_PREFIX_RATE_QUERY))
    repaired["hazard_provenance"].update({"base_rate_before": base_rate,
                                          "base_rate_after": base_rate + 1})
    repaired.setdefault("provenance", {}).update({
        "option_a_strategy": OPTION_A_STRATEGY, "source_injection_id": source_id,
        "source_summary": source_summary, "selected_base_title": title,
        "reference_time": reference_time,
    })
    return repaired


def _rate(db: Path, query: str) -> int:
    conn = sqlite3.connect("file:%s?mode=ro&immutable=1" % db, uri=True)
    try:
        return int(conn.execute(query).fetchone()[0])
    finally:
        conn.close()


def _drop_scratch(scratch: Path) -> None:
    for candidate in (scratch, scratch.with_name(scratch.name + "-wal"),
                      scratch.with_name(scratch.name + "-shm")):
        candidate.unlink(missing_ok=True)


def _rate_with_injected_row(db: Path, query: str, value: Mapping[str, Any]) -> int:
    handle = tempfile.NamedTemporaryFile(prefix="phase5-preflight-", suffix=".sqlite", delete=False)
    handle.close()
    scratch = Path(handle.name)
    scratch.write_bytes(db.read_bytes())
    conn = sqlite3.connect(scratch)
    try:
        conn.execute("PRAGMA journal_mode=DELETE")
        conn.execute(
            "INSERT INTO events (calendar_id, user_email, title, description, location, "
            "start_at, end_at, is_all_day, status, source_app, source_ref, source_kind, "
            "created_by, updated_by, organizer_email) VALUES (1, '', ?, '', '', ?, ?, 0, "
            "'confirmed', 'hoolicalendar', NULL, 'native', '', '', '')",
            (str(value["summary"]), str(value["start"]), str(value.get("end") or value["start"])),
        )
        conn.commit()
        return int(conn.execute(query).fetchone()[0])
    finally:
        conn.close()
        _drop_scratch(scratch)


def classify_distractor(record: Mapping[str, Any], config: HazardConfig,
                        db_dir: Path) -> Dict[str, Any]:
    unit = "same_title_prefix_events"
    query = _query_for(config, "distractor_candidates", unit)
    db = db_dir / ("%s.sqlite" % CALENDAR_APP)
    out: Dict[str, Any] = {
        "modeled": True, "unit": unit, "base_rate": None, "realized_rate": None,
        "survives_seeder_guards": False, "delta": None, "admissible": False, "note": "",
    }
    if query is None or not db.is_file():
        out["note"] = "no %s query or database in %s" % (unit, db_dir)
        return out
    value = _override_value(record)
    out["base_rate"] = _rate(db, query)
    if value is None:
        out["note"] = "persona_diff is not a single calendar override"
        return out
    out["survives_seeder_guards"] = calendar_override_survives(value)
    if not out["survives_seeder_guards"]:
        out["note"] = "dropped by seed_calendar falsy guard (empty start)"
        return out
    out["realized_rate"] = _rate_with_injected_row(db, query, value)
    out["delta"] = out["realized_rate"] - out["base_rate"]
    out["admissible"] = out["delta"] >= 1
    if not out["admissible"]:
        out["note"] = "row is inserted but %s does not change" % unit
    return out


MODELS: Dict[str, Callable[[Mapping[str, Any], HazardConfig, Path], Dict[str, Any]]] = {
    "distractor_candidates": classify_distractor,
}


def classify(record: Mapping[str, Any], config: HazardConfig, db_dir: Path) -> Dict[str, Any]:
    hazard = str(record.get("hazard_type", ""))
    model = MODELS.get(hazard)
    if model is None:
        return {"modeled": False, "admissible": False, "injection_id": record.get("injection_id"),
                "hazard_type": hazard, "status": record.get("status"),
                "note": "no realization model for this hazard type"}
    out = model(record, config, db_dir)
    out.update({"injection_id": record.get("injection_id"), "hazard_type": hazard,
                "status": record.get("status"),
                "recorded_base_rate_before": (record.get("hazard_provenance") or {}).get(
                    "base_rate_before"),
                "recorded_base_rate_after": (record.get("hazard_provenance") or {}).get(
                    "base_rate_after")})
    return out


def read_records(path: Path) -> List[Mapping[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def run(config_path: Path, hazards_path: Path, db_dir: Path, out_path: Path) -> Dict[str, Any]:
    config = HazardConfig.from_yaml(config_path)
    records = read_records(hazards_path)
    results = [classify(r, config, db_dir) for r in records]
    out_path.write_text("".join(json.dumps(r, sort_keys=True) + "\n" for r in results))
    by_status: Dict[str, Dict[str, int]] = {}
    for result in results:
        bucket = by_status.setdefault(
            "%s|%s" % (result["hazard_type"], result["status"]),
            {"records": 0, "admissible": 0, "survives_guards": 0})
        bucket["records"] += 1
        bucket["admissible"] += int(bool(result.get("admissible")))
        bucket["survives_guards"] += int(bool(result.get("survives_seeder_guards")))
    return {
        "schema": "phase5-hazard-preflight/1.0",
        "hazards": str(hazards_path),
        "database_dir": str(db_dir),
        "records": len(records),
        "admissible": sum(int(bool(r.get("admissible"))) for r in results),
        "by_status": by_status,
    }


def repair_option_a(hazards_path: Path, db_dir: Path, reference_time: str,
                    out_path: Path) -> int:
    records = read_records(hazards_path)
    repaired = [realize_option_a(r, db_dir, reference_time)
                if r.get("hazard_type") == "distractor_candidates"
                and r.get("status") == "pending" else dict(r) for r in records]
    out_path.write_text("".join(json.dumps(r, sort_keys=True) + "\n" for r in repaired))
    return len(repaired)


def main(argv: List[str] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path,
                        default=Path("configs/synthesis/hazards_v1.yaml"))
    parser.add_argument("--hazards", type=Path,
                        default=Path("data/synthesis/generation/final_v1/hazards.jsonl"))
    parser.add_argument("--db-dir", type=Path, required=True,
                        help="directory holding the frozen BASE world's *.sqlite files")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--repair-option-a", type=Path,
                        help="write a derived hazard catalog using BASE singleton-prefix titles")
    parser.add_argument("--reference-time",
                        default="2026-09-01T07:24:24.351485+00:00")
    args = parser.parse_args(argv)
    if args.repair_option_a:
        repair_option_a(args.hazards, args.db_dir, args.reference_time,
                        args.repair_option_a)
        args.hazards = args.repair_option_a
    summary = run(args.config, args.hazards, args.db_dir, args.out)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
