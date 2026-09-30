#!/usr/bin/env python3
"""Terminalize the frozen option-A hazard catalog from measured evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from derail.phase5.hazard_records import assert_manifest_safe, terminalize


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def gold_unique(record: dict, bundle: Path) -> tuple[bool, str]:
    candidate_path = bundle / "records" / f"{record['task_id']}.json"
    candidate = json.loads(candidate_path.read_text())
    title = record["persona_diff"][0]["value"]["summary"].casefold()
    collision = any(title in item["instruction"].casefold()
                    for item in candidate["realization"]["instructions"])
    return candidate["gold_execution"]["final_verifier_passed"] is True and not collision, sha256(
        bundle / candidate["verifier_bundle_ref"])


def terminalize_catalog(catalog: Path, preflight: Path, smoke: Path,
                        bundle: Path) -> list[dict]:
    records = read_jsonl(catalog)
    measured = {row["injection_id"]: row for row in read_jsonl(preflight)}
    smoke_record = json.loads(smoke.read_text())
    smoke_ok = (
        smoke_record.get("status") == "completed"
        and smoke_record.get("verdict", {}).get("seeder_consistent") is True
        and smoke_record.get("verdict", {}).get("variant_rate_is_base_plus_one") is True
        and smoke_record.get("gold_uniqueness", {}).get("gold_unique") is True
    )
    if not smoke_ok:
        raise ValueError("representative option-A smoke is not accepted")
    evidence = {"catalog_sha256": sha256(catalog), "preflight_sha256": sha256(preflight),
                "smoke_sha256": sha256(smoke), "smoke_id": smoke_record["probe_id"],
                "sandbox_id": smoke_record["sandbox_id"]}
    output = []
    for record in records:
        row = measured[record["injection_id"]]
        attempt = dict(evidence, method="measured-preflight+representative-real-smoke/1.0")
        if record["hazard_type"] == "distractor_candidates":
            unique, verifier_sha = gold_unique(record, bundle)
            enriched = dict(record, profile_after={"base_rate": row["base_rate"],
                                                   "realized_rate": row["realized_rate"],
                                                   "delta": row["delta"]},
                            seeder_consistent=bool(row["admissible"]), gold_unique=unique)
            attempt["verifier_sha256"] = verifier_sha
            output.append(terminalize(enriched, accepted=bool(row["admissible"] and unique),
                                      reason=None if row["admissible"] and unique
                                      else "OPTION_A_MEASUREMENT_FAILED", attempt=attempt))
            continue
        enriched = dict(record, profile_after={"base_rate": row.get("base_rate")},
                        seeder_consistent=False, gold_unique=False)
        output.append(terminalize(enriched, accepted=False,
                                  reason=str(record.get("status") or "UNMODELED_HAZARD"),
                                  attempt=attempt))
    assert_manifest_safe(output)
    return output


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalog", type=Path, required=True)
    parser.add_argument("--preflight", type=Path, required=True)
    parser.add_argument("--smoke", type=Path, required=True)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    records = terminalize_catalog(args.catalog, args.preflight, args.smoke, args.bundle)
    args.out.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in records))
    print(json.dumps({"records": len(records), "pending": 0,
                      "injected": sum(r["status"] == "injected" for r in records),
                      "rejected": sum(r["status"] == "rejected" for r in records)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
