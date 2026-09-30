#!/usr/bin/env python3
"""Validate human labels, apply immutable repairs, and create depth case plans."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict

import yaml

from derail.annotation.records import Adjudication, HumanAnnotation
from derail.annotation.rubric_review import (
    HumanRubricReview,
    RubricReviewAdjudication,
    load_rubric_specs,
)
from derail.annotation.taxonomy import error_types_outside_seed
from derail.canonical.mypcbench import load_canonical_jsonl
from derail.construction.cases import build_case_plan
from derail.construction.repair import PrefixAudit, RepairPatch
from derail.derived.layout import DerivedBuild, atomic_write_json, atomic_write_jsonl, sha256_file
from derail.derived.schema import validate_schema


def _read_json(path: Path):
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def _require_failed_trajectory(human_rubric_score: Dict[str, Any]) -> None:
    if human_rubric_score["perfect"]:
        raise RuntimeError(
            "Only failure trajectories may become DERAIL cases; "
            "human rubric review marked every rubric successful"
        )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build-dir", type=Path, required=True)
    parser.add_argument("--trajectory-id", required=True)
    parser.add_argument("--case-id", required=True)
    parser.add_argument("--annotations", type=Path, nargs="+", required=True)
    parser.add_argument("--adjudication", type=Path, required=True)
    parser.add_argument("--rubric-reviews", type=Path, nargs="+", required=True)
    parser.add_argument("--rubric-review-adjudication", type=Path, required=True)
    parser.add_argument("--patches", type=Path)
    parser.add_argument("--prefix-audit", type=Path, required=True)
    parser.add_argument("--state-probe-config", type=Path, required=True)
    args = parser.parse_args()
    repository = Path(__file__).resolve().parents[1]

    build_dir = args.build_dir.resolve()
    manifest = _read_json(build_dir / "build_manifest.json")
    source = manifest["source_collection"]
    build = DerivedBuild(
        root=build_dir.parent,
        build_id=build_dir.name,
        collection_root=Path(source["uri"]),
        collection_id=source["collection_id"],
    )
    if not build.verify_raw_unchanged():
        raise RuntimeError("raw collection tree hash 与 build initialization 不一致")
    canonical_dir = build_dir / "canonical" / args.trajectory_id
    report = _read_json(canonical_dir / "normalization_report.json")
    if not report["case_eligible"]:
        raise RuntimeError("normalization report 标记该 trajectory 不可用于 case")
    steps = load_canonical_jsonl(canonical_dir / "trajectory.jsonl")

    annotation_raw = tuple(_read_json(path) for path in args.annotations)
    for raw in annotation_raw:
        validate_schema(raw, "annotation.schema.json", repository)
    annotations = tuple(HumanAnnotation.from_dict(raw) for raw in annotation_raw)
    for annotation in annotations:
        annotation.validate_against_action_count(len(steps))
        if annotation.source_trajectory_sha256 != report["source_trajectory_sha256"]:
            raise RuntimeError("human annotation source trajectory hash 不匹配")
    adjudication_raw = _read_json(args.adjudication)
    validate_schema(adjudication_raw, "adjudication.schema.json", repository)
    adjudication = Adjudication.from_dict(adjudication_raw)
    adjudication.validate_inputs(annotations)
    taxonomy = manifest["taxonomy"]
    taxonomy_path = Path(taxonomy.get("uri", ""))
    if not taxonomy_path.is_file() or sha256_file(taxonomy_path) != taxonomy.get("sha256"):
        raise RuntimeError("taxonomy URI/hash 无效")
    if adjudication.taxonomy_version != taxonomy["version"]:
        raise RuntimeError("adjudication taxonomy_version 与 build 不一致")
    outside_seed = list(error_types_outside_seed(adjudication.error_types, taxonomy))

    task_provenance = report.get("task_provenance", {})
    required_task_fields = (
        "task_id",
        "instruction",
        "instruction_sha256",
        "pre_command_sha256",
        "task_config_uri",
        "task_config_sha256",
        "rubric_bundle_uri",
        "rubric_bundle_sha256",
    )
    if any(not task_provenance.get(field) for field in required_task_fields):
        raise RuntimeError("normalization report 缺少 task provenance")

    # Human rubric review. The LLM judge's verdicts decide whether this
    # trajectory counts as failed at all, so a case built on a wrong verdict is
    # not a derailment case. Two reviewers must have re-judged every rubric of
    # THIS bundle, and a third must have adjudicated them.
    rubric_bundle_path = Path(task_provenance["rubric_bundle_uri"])
    if (
        not rubric_bundle_path.is_file()
        or sha256_file(rubric_bundle_path) != task_provenance["rubric_bundle_sha256"]
    ):
        raise RuntimeError("rubric bundle URI/hash 无效")
    rubric_specs = load_rubric_specs(_read_json(rubric_bundle_path))
    review_raw = tuple(_read_json(path) for path in args.rubric_reviews)
    for raw in review_raw:
        validate_schema(raw, "rubric_review.schema.json", repository)
    rubric_reviews = tuple(HumanRubricReview.from_dict(raw) for raw in review_raw)
    for review in rubric_reviews:
        review.validate_against_bundle(rubric_specs)
        if review.trajectory_id != args.trajectory_id:
            raise RuntimeError("rubric review trajectory_id 与 case 不一致")
        if review.source_trajectory_sha256 != report["source_trajectory_sha256"]:
            raise RuntimeError("rubric review source trajectory hash 不匹配")
        if review.rubric_bundle_sha256 != task_provenance["rubric_bundle_sha256"]:
            raise RuntimeError("rubric review rubric bundle hash 不匹配")
    rubric_adjudication_raw = _read_json(args.rubric_review_adjudication)
    validate_schema(rubric_adjudication_raw, "rubric_review_adjudication.schema.json", repository)
    rubric_adjudication = RubricReviewAdjudication.from_dict(rubric_adjudication_raw)
    if rubric_adjudication.trajectory_id != args.trajectory_id:
        raise RuntimeError("rubric adjudication trajectory_id 与 case 不一致")
    if rubric_adjudication.source_trajectory_sha256 != report["source_trajectory_sha256"]:
        raise RuntimeError("rubric adjudication source trajectory hash 不匹配")
    if rubric_adjudication.rubric_bundle_sha256 != task_provenance["rubric_bundle_sha256"]:
        raise RuntimeError("rubric adjudication rubric bundle hash 不匹配")
    rubric_adjudication.validate_inputs(rubric_reviews)
    human_rubric_score = rubric_adjudication.weighted_score(rubric_specs)
    _require_failed_trajectory(human_rubric_score)

    patches = []
    if args.patches:
        with args.patches.open(encoding="utf-8") as handle:
            patch_records = [json.loads(line) for line in handle if line.strip()]
            for raw in patch_records:
                validate_schema(raw, "repair_patch.schema.json", repository)
            patches = [RepairPatch.from_dict(raw) for raw in patch_records]
    prefix_audit_raw = _read_json(args.prefix_audit)
    validate_schema(prefix_audit_raw, "prefix_audit.schema.json", repository)
    protocol = yaml.safe_load(
        (repository / "configs/benchmark/derail_v1.yaml").read_text(encoding="utf-8")
    )
    prefix_audit = PrefixAudit.from_dict(
        prefix_audit_raw, int(protocol["prefix_repair"]["min_reviewers"])
    )
    plan = build_case_plan(
        case_id=args.case_id,
        trajectory_id=args.trajectory_id,
        source_trajectory_sha256=report["source_trajectory_sha256"],
        steps=steps,
        adjudication=adjudication,
        prefix_audit=prefix_audit,
        patches=patches,
    )

    existing_case_ids = {item["case_id"] for item in manifest.get("case_index", ())}
    if args.case_id in existing_case_ids or (build_dir / "cases" / args.case_id).exists():
        raise RuntimeError("case_id 已存在，拒绝覆盖: %s" % args.case_id)
    state_probe_config = _read_json(args.state_probe_config.resolve())
    if not isinstance(state_probe_config, dict) or not state_probe_config or any(
        not isinstance(name, str)
        or not name.strip()
        or not isinstance(command, str)
        or not command.strip()
        for name, command in state_probe_config.items()
    ):
        raise RuntimeError("state probe config 必须是 nonempty string->command object")

    annotation_dir = build_dir / "annotations" / args.case_id
    for annotation in annotations:
        atomic_write_json(
            annotation_dir / (annotation.annotation_id + ".json"), annotation.to_dict()
        )
    atomic_write_json(annotation_dir / "adjudication.json", adjudication.to_dict())
    review_dir = annotation_dir / "rubric_review"
    for review in rubric_reviews:
        atomic_write_json(review_dir / (review.review_id + ".json"), review.to_dict())
    atomic_write_json(review_dir / "adjudication.json", rubric_adjudication.to_dict())
    repair_dir = build_dir / "repairs" / args.case_id
    atomic_write_json(repair_dir / "prefix_audit.json", prefix_audit.to_dict())
    atomic_write_jsonl(repair_dir / "repair_patches.jsonl", (patch.to_dict() for patch in patches))
    atomic_write_jsonl(
        repair_dir / "repaired_trajectory.jsonl",
        (step.to_dict() for step in plan.repaired_steps),
    )
    case_dir = build_dir / "cases" / args.case_id
    state_probe_path = case_dir / "state_probe_config.json"
    atomic_write_json(state_probe_path, state_probe_config)
    case_record = plan.to_dict()
    case_record.update(
        {
            "build_id": manifest["build_id"],
            "collection_id": source["collection_id"],
            "source_raw_uri": report["source_trajectory_uri"],
            "canonical_original_uri": report["canonical_trajectory_uri"],
            "canonical_original_sha256": report["canonical_trajectory_sha256"],
            "canonical_repaired_uri": str((repair_dir / "repaired_trajectory.jsonl").resolve()),
            "canonical_repaired_sha256": sha256_file(repair_dir / "repaired_trajectory.jsonl"),
            "human_annotation_refs": [
                {
                    "annotation_id": annotation.annotation_id,
                    "uri": str((annotation_dir / (annotation.annotation_id + ".json")).resolve()),
                    "sha256": sha256_file(annotation_dir / (annotation.annotation_id + ".json")),
                }
                for annotation in annotations
            ],
            "adjudication_ref": {
                "uri": str((annotation_dir / "adjudication.json").resolve()),
                "sha256": sha256_file(annotation_dir / "adjudication.json"),
            },
            "rubric_review_refs": [
                {
                    "review_id": review.review_id,
                    "reviewer_id": review.reviewer_id,
                    "uri": str((review_dir / (review.review_id + ".json")).resolve()),
                    "sha256": sha256_file(review_dir / (review.review_id + ".json")),
                }
                for review in rubric_reviews
            ],
            "rubric_review_adjudication_ref": {
                "adjudication_id": rubric_adjudication.adjudication_id,
                "uri": str((review_dir / "adjudication.json").resolve()),
                "sha256": sha256_file(review_dir / "adjudication.json"),
            },
            "human_rubric_score": human_rubric_score,
            "rubric_review_disagreement_ids": list(
                rubric_adjudication.disagreement_rubric_ids
            ),
            "judge_rubric_disagreement_ids": list(
                rubric_adjudication.judge_disagreement_rubric_ids()
            ),
            "repair_patch_ref": {
                "uri": str((repair_dir / "repair_patches.jsonl").resolve()),
                "sha256": sha256_file(repair_dir / "repair_patches.jsonl"),
            },
            "prefix_audit_ref": {
                "uri": str((repair_dir / "prefix_audit.json").resolve()),
                "sha256": sha256_file(repair_dir / "prefix_audit.json"),
            },
            "source_collection_manifest_ref": {
                "uri": source["manifest_uri"],
                "sha256": source["manifest_sha256"],
            },
            "task_id": task_provenance["task_id"],
            "instruction": task_provenance["instruction"],
            "instruction_sha256": task_provenance["instruction_sha256"],
            "pre_command_sha256": task_provenance["pre_command_sha256"],
            "task_config_ref": {
                "uri": task_provenance["task_config_uri"],
                "sha256": task_provenance["task_config_sha256"],
            },
            "rubric_bundle_ref": {
                "uri": task_provenance["rubric_bundle_uri"],
                "sha256": task_provenance["rubric_bundle_sha256"],
            },
            "state_probe_config_ref": {
                "uri": str(state_probe_path.resolve()),
                "sha256": sha256_file(state_probe_path),
            },
            "taxonomy_ref": {
                "uri": taxonomy["uri"],
                "sha256": taxonomy["sha256"],
            },
            "taxonomy_status": (
                "frozen_membership_validated"
                if taxonomy.get("frozen") is True
                else "open_coding"
            ),
            "error_types_outside_seed": outside_seed,
            "required_compatible_agents": manifest["builder_config"].get(
                "target_agents", []
            ),
            "agent_compatibility_status": "pending_conformance_evidence",
        }
    )
    validate_schema(case_record, "benchmark_case.schema.json", repository)
    atomic_write_json(case_dir / "case.json", case_record)
    for instance in plan.instances:
        validate_schema(instance.to_dict(), "depth_instance.schema.json", repository)
        atomic_write_json(
            case_dir / "depths" / ("d%d" % instance.depth) / "instance.json",
            instance.to_dict(),
        )

    counts = dict(manifest["counts"])
    counts["candidate_cases"] = counts.get("candidate_cases", 0) + 1
    case_index = list(manifest.get("case_index", ()))
    case_index.append(
        {
            "case_id": args.case_id,
            "uri": str((case_dir / "case.json").resolve()),
            "sha256": sha256_file(case_dir / "case.json"),
            "status": "pending_real_vm_replay",
        }
    )
    build.update_manifest(
        {
            "status": "pending_replay",
            "counts": counts,
            "case_index": case_index,
        }
    )
    validate_schema(
        build.read_manifest(), "derived_build_manifest.schema.json", repository
    )
    print(json.dumps(case_record, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
