#!/usr/bin/env python3
"""Validate a derived build and distinguish plumbing checks from release evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from derail.adapters import registry_status
from derail.annotation.taxonomy import error_types_outside_seed
from derail.construction.cases import DepthInstance
from derail.derived.layout import DEPTH_GRID, DerivedBuild, atomic_write_json, sha256_file
from derail.derived.schema import validate_schema
from derail.replay.verification import ReplayVerification
from derail.takeover.history import NativeHistoryArtifact


def _read(path: Path):
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def _ref_valid(reference) -> bool:
    if not isinstance(reference, dict) or not reference.get("uri") or not reference.get("sha256"):
        return False
    path = Path(reference["uri"])
    return path.is_file() and sha256_file(path) == reference["sha256"]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build-dir", type=Path, required=True)
    parser.add_argument("--strict-release", action="store_true")
    args = parser.parse_args()
    repository = Path(__file__).resolve().parents[1]
    build_dir = args.build_dir.resolve()
    manifest = _read(build_dir / "build_manifest.json")
    validate_schema(manifest, "derived_build_manifest.schema.json", repository)
    source = manifest["source_collection"]
    build = DerivedBuild(
        root=build_dir.parent,
        build_id=build_dir.name,
        collection_root=Path(source["uri"]),
        collection_id=source["collection_id"],
    )
    taxonomy = manifest.get("taxonomy", {})
    taxonomy_ref_valid = _ref_valid(taxonomy)
    checks = {
        "depth_grid_frozen": tuple(manifest["depths"]) == DEPTH_GRID,
        "raw_collection_unchanged": build.verify_raw_unchanged(),
        "taxonomy_frozen": (
            taxonomy.get("frozen") is True
            and taxonomy.get("mode") == "frozen"
            and taxonomy.get("version") != "draft"
            and taxonomy_ref_valid
            and bool(taxonomy.get("labels"))
        ),
    }
    source_collection_manifest = _read(Path(source["manifest_uri"]))
    expected_snapshot_sha256 = source_collection_manifest.get("image_sha256", "")
    target_agents = tuple(manifest.get("builder_config", {}).get("target_agents", ()))
    case_reports = []
    for case_path in sorted((build_dir / "cases").glob("*/case.json")):
        case = _read(case_path)
        validate_schema(case, "benchmark_case.schema.json", repository)
        case_id = case["case_id"]
        try:
            outside_seed = error_types_outside_seed(case.get("error_types", ()), taxonomy)
            taxonomy_membership_valid = (
                case.get("taxonomy_version") == taxonomy.get("version")
                and list(outside_seed) == case.get("error_types_outside_seed")
                and case.get("taxonomy_status")
                == (
                    "frozen_membership_validated"
                    if taxonomy.get("frozen") is True
                    else "open_coding"
                )
            )
        except ValueError:
            taxonomy_membership_valid = False
        provenance_valid = (
            _ref_valid(case.get("adjudication_ref"))
            and _ref_valid(case.get("repair_patch_ref"))
            and _ref_valid(case.get("prefix_audit_ref"))
            and _ref_valid(case.get("source_collection_manifest_ref"))
            and _ref_valid(case.get("task_config_ref"))
            and _ref_valid(case.get("rubric_bundle_ref"))
            and _ref_valid(case.get("state_probe_config_ref"))
            and _ref_valid(case.get("taxonomy_ref"))
            and all(_ref_valid(item) for item in case.get("human_annotation_refs", ()))
            and Path(case.get("canonical_original_uri", "")).is_file()
            and sha256_file(Path(case["canonical_original_uri"]))
            == case.get("canonical_original_sha256")
            and Path(case.get("canonical_repaired_uri", "")).is_file()
            and sha256_file(Path(case["canonical_repaired_uri"]))
            == case.get("canonical_repaired_sha256")
            and case.get("instruction_sha256")
            == hashlib.sha256(case.get("instruction", "").encode()).hexdigest()
            and taxonomy_membership_valid
        )
        compatibility_reports = []
        instance_reports = []
        for depth in case["available_depths"]:
            path = case_path.parent / "depths" / ("d%d" % depth) / "instance.json"
            present = path.is_file()
            instance_raw = _read(path) if present else {}
            try:
                if present:
                    validate_schema(instance_raw, "depth_instance.schema.json", repository)
                instance = DepthInstance.from_dict(instance_raw) if present else None
                semantics_ok = bool(
                    instance
                    and instance.case_id == case_id
                    and instance.depth == depth
                    and instance.to_dict() == instance_raw
                )
            except (KeyError, TypeError, ValueError):
                instance = None
                semantics_ok = False
            replay_path = (
                build_dir
                / "replay"
                / case_id
                / ((instance.instance_id if instance else "invalid-instance") + ".json")
            )
            replay_raw = _read(replay_path) if replay_path.is_file() else None
            replay = None
            replay_reasons = []
            if replay_raw is not None and instance is not None:
                try:
                    validate_schema(
                        replay_raw, "replay_verification.schema.json", repository
                    )
                    replay = ReplayVerification.from_dict(replay_raw)
                    if replay.build_id != manifest["build_id"]:
                        replay_reasons.append("build_id_mismatch")
                    if replay.case_id != case_id or replay.instance_id != instance.instance_id:
                        replay_reasons.append("case_or_instance_mismatch")
                    if replay.depth != depth:
                        replay_reasons.append("depth_mismatch")
                    if replay.canonical_actions_sha256 != instance.canonical_actions_sha256:
                        replay_reasons.append("canonical_actions_hash_mismatch")
                    if replay.canonical_trajectory_sha256 != case["canonical_repaired_sha256"]:
                        replay_reasons.append("canonical_trajectory_hash_mismatch")
                    if replay.snapshot_sha256 != expected_snapshot_sha256:
                        replay_reasons.append("qcow2_snapshot_hash_mismatch")
                    if replay.task_id != case["task_id"]:
                        replay_reasons.append("task_id_mismatch")
                    if replay.instruction_sha256 != case["instruction_sha256"]:
                        replay_reasons.append("instruction_hash_mismatch")
                    if replay.task_config_sha256 != case["task_config_ref"]["sha256"]:
                        replay_reasons.append("task_config_hash_mismatch")
                    if replay.state_probe_config_sha256 != case["state_probe_config_ref"]["sha256"]:
                        replay_reasons.append("state_probe_config_hash_mismatch")
                    for log in replay.per_action_log:
                        for prefix in ("observation_before", "observation_after"):
                            if not _ref_valid(
                                {
                                    "uri": log.get(prefix + "_uri"),
                                    "sha256": log.get(prefix + "_sha256"),
                                }
                            ):
                                replay_reasons.append(prefix + "_evidence_invalid")
                                break
                except (KeyError, TypeError, ValueError) as exc:
                    replay_reasons.append("invalid_replay_record:%s" % exc)
            vm_accepted = bool(
                replay and replay.accepted_for_release and not replay_reasons
            )
            instance_reports.append(
                {
                    "depth": depth,
                    "instance_present": present,
                    "depth_semantics_valid": semantics_ok,
                    "replay_evidence_uri": str(replay_path),
                    "real_vm_replay_accepted": vm_accepted,
                    "observed_execution_mode": replay.execution_mode if replay else None,
                    "rejection_reasons": replay_reasons,
                }
            )
            for agent_id in target_agents:
                compatibility_path = (
                    case_path.parent
                    / "compatibility"
                    / agent_id
                    / ((instance.instance_id if instance else "invalid-instance") + ".json")
                )
                supported = False
                compatibility_reason = "missing"
                if compatibility_path.is_file() and replay is not None:
                    try:
                        compatibility_raw = _read(compatibility_path)
                        validate_schema(
                            compatibility_raw, "native_history.schema.json", repository
                        )
                        artifact = NativeHistoryArtifact.from_dict(compatibility_raw)
                        expected_observations = tuple(
                            str(item["observation_before_sha256"])
                            for item in replay.per_action_log
                        )
                        probe_path = Path(artifact.conformance_probe_uri)
                        supported = bool(
                            artifact.release_eligible
                            and artifact.agent_id == agent_id
                            and artifact.instance_id == replay.instance_id
                            and artifact.replay_verification_id == replay.attempt_id
                            and artifact.action_indices == replay.executed_action_indices
                            and artifact.replay_observation_sha256s == expected_observations
                            and probe_path.is_file()
                            and sha256_file(probe_path) == artifact.conformance_probe_sha256
                        )
                        compatibility_reason = "ok" if supported else "binding_or_probe_invalid"
                    except (KeyError, TypeError, ValueError) as exc:
                        compatibility_reason = "invalid:%s" % exc
                compatibility_reports.append(
                    {
                        "agent_id": agent_id,
                        "instance_id": instance.instance_id if instance else None,
                        "evidence_uri": str(compatibility_path),
                        "supported": supported,
                        "reason": compatibility_reason,
                    }
                )
        case_reports.append(
            {
                "case_id": case_id,
                "instances": instance_reports,
                "agent_compatibility": compatibility_reports,
                "provenance_hashes_valid": provenance_valid,
            }
        )

    structural = checks["depth_grid_frozen"] and checks["raw_collection_unchanged"] and all(
        item["instance_present"] and item["depth_semantics_valid"]
        for case in case_reports
        for item in case["instances"]
    ) and all(case["provenance_hashes_valid"] for case in case_reports)
    real_replay_ready = bool(case_reports) and all(
        item["real_vm_replay_accepted"] for case in case_reports for item in case["instances"]
    )
    compatibility_ready = bool(target_agents) and bool(case_reports) and all(
        item["supported"]
        for case in case_reports
        for item in case["agent_compatibility"]
    )
    release_ready = (
        structural and checks["taxonomy_frozen"] and real_replay_ready and compatibility_ready
    )
    report = {
        "build_id": manifest["build_id"],
        "checks": checks,
        "cases": case_reports,
        "structural_validation_passed": structural,
        "real_vm_replay_validation_passed": real_replay_ready,
        "agent_history_conformance_passed": compatibility_ready,
        "release_ready": release_ready,
        "dry_run_policy": "synthetic/dry_run replay never counts as release evidence",
        "native_history_registry": list(registry_status()),
    }
    atomic_write_json(build_dir / "reports" / "validation_report.json", report)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if not structural:
        return 1
    if args.strict_release and not release_ready:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
