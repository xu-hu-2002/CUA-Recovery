#!/usr/bin/env python3
"""Turn human cleaning proposals into audited drop patches and the repaired canonical prefix."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import yaml

from recovery.annotation.records import Adjudication, HumanAnnotation
from recovery.canonical.actions import action_to_dict
from recovery.canonical.mypcbench import load_canonical_jsonl
from recovery.construction.cases import eligible_depths
from recovery.construction.repair import PrefixAudit, RepairPatch, apply_repair_patches
from recovery.derived.layout import atomic_write_json, atomic_write_jsonl, sha256_file
from recovery.derived.schema import validate_schema

REPORT_SCHEMA_VERSION = "prefix-repair-report/0.1"


def _read_json(path: Path) -> Dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError("Expected a JSON object: %s" % path)
    return value


def build_consensus(
    *,
    case_id: str,
    adjudicator_id: str,
    adjudication: Any,
    steps: Sequence[Any],
    proposals: Sequence[Tuple[Path, Dict[str, Any]]],
    min_reviewers: int = 1,
) -> Tuple[PrefixAudit, Tuple[RepairPatch, ...], Tuple[int, ...]]:
    """Return (audit, patches, rejected post-root drop indices)."""

    if len(proposals) < max(1, min_reviewers):
        raise RuntimeError(
            "Clean-prefix consensus requires at least %d proposals" % max(1, min_reviewers)
        )
    reviewer_ids = tuple(str(raw["reviewer_id"]) for _, raw in proposals)
    if len(set(reviewer_ids)) != len(reviewer_ids):
        raise RuntimeError("Cleaning proposals must come from different reviewers")
    trajectory_ids = {str(raw["trajectory_id"]) for _, raw in proposals}
    source_hashes = {str(raw["source_trajectory_sha256"]) for _, raw in proposals}
    roots = {int(raw["root_cause_action_index"]) for _, raw in proposals}
    if trajectory_ids != {adjudication.trajectory_id}:
        raise RuntimeError("Cleaning proposal trajectory does not match adjudication")
    if roots != {adjudication.root_cause_action_index}:
        raise RuntimeError("Cleaning proposal root does not match adjudication")
    if len(source_hashes) != 1:
        raise RuntimeError("Cleaning proposals do not share one source trajectory hash")

    root = adjudication.root_cause_action_index
    available_depths, _ = eligible_depths(root, len(steps) - 1, None, require_error_explicit=False)
    if not available_depths:
        raise RuntimeError("No depth is available for the adjudicated root")
    audit_end = min(
        [root + max(available_depths)]
        + [int(raw["audit_end_action_index"]) for _, raw in proposals]
    )
    for path, raw in proposals:
        if raw["review_complete"] is not True:
            raise RuntimeError("Cleaning review is incomplete: %s" % path)
        if audit_end < root:
            raise RuntimeError("Cleaning proposal stops before the root cause: %s" % path)
        if raw["audited_action_indices"][: audit_end + 1] != list(range(audit_end + 1)):
            raise RuntimeError("Cleaning proposal did not review every source action: %s" % path)
    expected_audited = list(range(audit_end + 1))

    rejected = sorted(
        {
            int(item["action_index_global"])
            for _, raw in proposals
            for item in raw["drop_candidates"]
            if int(item["action_index_global"]) >= root
        }
    )
    candidate_maps = []
    for _, raw in proposals:
        candidate_maps.append(
            {
                int(item["action_index_global"]): item
                for item in raw["drop_candidates"]
                if int(item["action_index_global"]) < root
            }
        )
    candidate_sets = {
        tuple(
            sorted(
                (index, int(item["recovery_action_index"]))
                for index, item in items.items()
            )
        )
        for items in candidate_maps
    }
    if len(candidate_sets) != 1:
        raise RuntimeError(
            "Cleaning reviewers disagree on removable actions; adjudicate the proposals manually"
        )

    steps_by_id = {step.action_index_global: step for step in steps}
    candidate_signatures = next(iter(candidate_sets), ())
    candidate_indices = tuple(index for index, _ in candidate_signatures)
    if any(recovery in candidate_indices for _, recovery in candidate_signatures):
        raise RuntimeError("Every clean-prefix recovery step must be retained")
    by_recovery: Dict[int, list[int]] = {}
    for index, recovery in candidate_signatures:
        by_recovery.setdefault(recovery, []).append(index)
    for recovery, indices in by_recovery.items():
        if indices != list(range(min(indices), recovery)):
            raise RuntimeError(
                "Consensus drop candidates must form contiguous ranges ending before recovery"
            )
    patches = []
    for index, recovery_index in candidate_signatures:
        if index == root or index > audit_end or index not in steps_by_id:
            raise RuntimeError("Invalid consensus drop candidate: %d" % index)
        canonical_action = action_to_dict(steps_by_id[index].action)
        reasons = []
        for proposal in candidate_maps:
            candidate = proposal[index]
            if int(candidate["recovery_action_index"]) != recovery_index:
                raise RuntimeError("Cleaning proposal recovery boundary disagrees at step %d" % index)
            if candidate["old_action"] != canonical_action:
                raise RuntimeError("Cleaning proposal action is stale at source step %d" % index)
            reason = str(candidate["reason"]).strip()
            if reason not in reasons:
                reasons.append(reason)
        patches.append(
            RepairPatch(
                patch_id="%s__drop-%d" % (case_id, index),
                case_id=case_id,
                step_id=index,
                root_cause_step=root,
                old_action=steps_by_id[index].action,
                operation="drop",
                reason="Recovered at source step %d. %s" % (
                    recovery_index,
                    " | ".join(reasons),
                ),
                annotator_id=adjudicator_id,
                self_recovered=True,
                persistent_state_effect=False,
                causal_to_root_or_task=False,
                evidence=tuple(
                    "%s#candidate-%d" % (path.resolve(), index) for path, _ in proposals
                ),
            )
        )

    audit = PrefixAudit(
        audit_id="%s__clean-prefix" % case_id,
        case_id=case_id,
        source_trajectory_sha256=next(iter(source_hashes)),
        root_cause_action_index=root,
        audit_end_action_index=audit_end,
        audited_prefix_action_indices=tuple(expected_audited),
        unrelated_error_action_indices=tuple(candidate_indices),
        repair_patch_ids=tuple(patch.patch_id for patch in patches),
        reviewer_ids=reviewer_ids,
        evidence_refs=tuple(str(path.resolve()) for path, _ in proposals),
        rationale=(
            "Independent reviewers agreed that the listed actions were self-recovered detours "
            "without persistent state effects or causal influence. Retained source IDs are unchanged."
        ),
        approved=True,
        min_reviewers=min_reviewers,
    )
    audit.validate_patches(patches)
    return audit, tuple(patches), tuple(rejected)


def load_protocol(repository: Path) -> Dict[str, Any]:
    raw = yaml.safe_load((repository / "configs/benchmark/recovery_v1.yaml").read_text("utf-8"))
    return dict(raw.get("prefix_repair") or {})


def write_repair(
    *,
    out_dir: Path,
    case_id: str,
    label: Any,
    label_uri: str,
    canonical_path: Path,
    steps: Sequence[Any],
    proposal_records: Sequence[Tuple[Path, Dict[str, Any]]],
    adjudicator_id: str,
    min_reviewers: int,
    repository: Path,
    allow_overwrite: bool,
) -> Dict[str, Any]:
    """Build the consensus and write audit, patches, repaired prefix and report."""

    audit, patches, rejected = build_consensus(
        case_id=case_id,
        adjudicator_id=adjudicator_id,
        adjudication=label,
        steps=steps,
        proposals=proposal_records,
        min_reviewers=min_reviewers,
    )
    paths = {
        name: out_dir / name
        for name in ("prefix_audit.json", "repair_patches.jsonl", "trajectory.jsonl", "repair_report.json")
    }
    if not allow_overwrite and any(path.exists() for path in paths.values()):
        raise RuntimeError("Clean-prefix output already exists; refusing to overwrite: %s" % out_dir)
    validate_schema(audit.to_dict(), "prefix_audit.schema.json", repository)
    for patch in patches:
        validate_schema(patch.to_dict(), "repair_patch.schema.json", repository)
    repaired = apply_repair_patches(steps, patches)
    atomic_write_json(paths["prefix_audit.json"], audit.to_dict())
    atomic_write_jsonl(paths["repair_patches.jsonl"], (patch.to_dict() for patch in patches))
    atomic_write_jsonl(paths["trajectory.jsonl"], (step.to_dict() for step in repaired))
    report = {
        "schema_version": REPORT_SCHEMA_VERSION,
        "case_id": case_id,
        "trajectory_id": label.trajectory_id,
        "source_trajectory_sha256": audit.source_trajectory_sha256,
        "canonical_trajectory_uri": str(canonical_path.resolve()),
        "canonical_trajectory_sha256": sha256_file(canonical_path),
        "repaired_trajectory_uri": str(paths["trajectory.jsonl"].resolve()),
        "repaired_trajectory_sha256": sha256_file(paths["trajectory.jsonl"]),
        "index_semantics": "source_action_index_global_with_gaps",
        "root_cause_action_index": label.root_cause_action_index,
        "identifiable_at_action_index": label.identifiable_at_action_index,
        "last_action_index": len(steps) - 1,
        "dropped_action_indices": [patch.step_id for patch in patches],
        "rejected_post_root_drop_indices": list(rejected),
        "label_uri": label_uri,
        "proposal_uris": [str(path.resolve()) for path, _ in proposal_records],
        "reviewer_ids": list(audit.reviewer_ids),
        "min_reviewers": min_reviewers,
        "human_verified": True,
        "status": "repaired" if patches else "unchanged",
        "prefix_state_status": "pending_replay_verification",
    }
    atomic_write_json(paths["repair_report.json"], report)
    return report


def _label_for(labels_dir: Path, trajectory_id: str, source_sha: str) -> Tuple[Any, Path]:
    """The failure label a prefix is repaired for: the adjudication if any, else the single annotation."""

    adjudications = sorted((labels_dir / "adjudications").glob("%s__*.json" % trajectory_id))
    for path in adjudications:
        record = _read_json(path).get("failure_adjudication")
        if record:
            return Adjudication.from_dict(record), path
    annotations = []
    for path in sorted(labels_dir.glob("%s__*.json" % trajectory_id)):
        raw = _read_json(path)
        if raw.get("annotator_role") == "human" and raw.get("source_trajectory_sha256") == source_sha:
            annotations.append((HumanAnnotation.from_dict(raw), path))
    roots = {item.root_cause_action_index for item, _ in annotations}
    if not annotations:
        raise RuntimeError("no human failure annotation")
    if len(roots) != 1:
        raise RuntimeError("human annotations disagree on the root cause; adjudicate first")
    return annotations[0]


def run_build(
    build_dir: Path, labels_dir: Path, repository: Path, protocol: Dict[str, Any], allow_overwrite: bool
) -> Dict[str, Any]:
    min_reviewers = int(protocol.get("min_reviewers", 1))
    out_root = build_dir / str(protocol.get("output_layer", "canonical_repaired"))
    written: List[str] = []
    skipped: List[Dict[str, str]] = []
    for canonical_dir in sorted(p for p in (build_dir / "canonical").iterdir() if p.is_dir()):
        trajectory_id = canonical_dir.name
        report_path = canonical_dir / "normalization_report.json"
        if not report_path.is_file() or not any(labels_dir.glob("%s__*.json" % trajectory_id)):
            continue
        source_sha = str(_read_json(report_path).get("source_trajectory_sha256", ""))
        try:
            label, label_path = _label_for(labels_dir, trajectory_id, source_sha)
            proposals = [
                (path, _read_json(path))
                for path in sorted(
                    (labels_dir / "cleaning_proposals").glob("%s__*.json" % trajectory_id)
                )
            ]
            proposals = [
                (path, raw)
                for path, raw in proposals
                if raw.get("source_trajectory_sha256") == source_sha
                and int(raw.get("root_cause_action_index", -1)) == label.root_cause_action_index
            ]
            for _, raw in proposals:
                validate_schema(raw, "cleaning_proposal.schema.json", repository)
            write_repair(
                out_dir=out_root / trajectory_id,
                case_id=trajectory_id,
                label=label,
                label_uri=str(label_path.resolve()),
                canonical_path=canonical_dir / "trajectory.jsonl",
                steps=load_canonical_jsonl(canonical_dir / "trajectory.jsonl"),
                proposal_records=proposals,
                adjudicator_id=getattr(label, "adjudicator_id", None) or label.annotator_id,
                min_reviewers=min_reviewers,
                repository=repository,
                allow_overwrite=allow_overwrite,
            )
            written.append(trajectory_id)
        except Exception as exc:
            skipped.append({"trajectory_id": trajectory_id, "reason": str(exc)[:300]})
    summary = {"output_root": str(out_root), "written": len(written), "skipped": skipped}
    atomic_write_json(out_root / "repair_summary.json", summary)
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__.split("\n\n")[0], formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--build-dir", type=Path, help="batch mode: repair every labelled trajectory")
    parser.add_argument("--human-labels-dir", type=Path)
    parser.add_argument("--case-id")
    parser.add_argument("--adjudicator-id")
    parser.add_argument(
        "--adjudication", type=Path, help="adjudication or single human annotation JSON"
    )
    parser.add_argument("--canonical-trajectory", type=Path)
    parser.add_argument("--proposals", type=Path, nargs="+")
    parser.add_argument("--out-dir", type=Path)
    parser.add_argument(
        "--min-reviewers",
        type=int,
        help="default: prefix_repair.min_reviewers in configs/benchmark/recovery_v1.yaml",
    )
    parser.add_argument("--allow-overwrite", action="store_true")
    parser.add_argument("--repository", type=Path, default=Path(__file__).resolve().parents[2])
    args = parser.parse_args()

    repository = args.repository.resolve()
    protocol = load_protocol(repository)
    if args.min_reviewers is not None:
        protocol["min_reviewers"] = args.min_reviewers
    min_reviewers = int(protocol.get("min_reviewers", 1))
    if args.build_dir:
        if not args.human_labels_dir:
            parser.error("--build-dir needs --human-labels-dir")
        summary = run_build(
            args.build_dir.resolve(),
            args.human_labels_dir.resolve(),
            repository,
            protocol,
            args.allow_overwrite,
        )
        print("repaired prefixes %d -> %s" % (summary["written"], summary["output_root"]))
        print("skipped           %d (repair_summary.json)" % len(summary["skipped"]))
        return 0

    missing = [
        name
        for name in ("case_id", "adjudicator_id", "adjudication", "canonical_trajectory", "proposals", "out_dir")
        if not getattr(args, name)
    ]
    if missing:
        parser.error("single-case mode needs --%s" % ", --".join(m.replace("_", "-") for m in missing))
    label_raw = _read_json(args.adjudication)
    if "adjudication_id" in label_raw:
        validate_schema(label_raw, "adjudication.schema.json", repository)
        label = Adjudication.from_dict(label_raw)
    else:
        validate_schema(label_raw, "annotation.schema.json", repository)
        label = HumanAnnotation.from_dict(label_raw)
    proposal_records = []
    for path in args.proposals:
        raw = _read_json(path)
        validate_schema(raw, "cleaning_proposal.schema.json", repository)
        proposal_records.append((path, raw))
    report = write_repair(
        out_dir=args.out_dir.resolve(),
        case_id=args.case_id,
        label=label,
        label_uri=str(args.adjudication.resolve()),
        canonical_path=args.canonical_trajectory,
        steps=load_canonical_jsonl(args.canonical_trajectory),
        proposal_records=proposal_records,
        adjudicator_id=args.adjudicator_id,
        min_reviewers=min_reviewers,
        repository=repository,
        allow_overwrite=args.allow_overwrite,
    )
    print("repaired prefix %s" % report["repaired_trajectory_uri"])
    print("drop patches    %d (rejected post-root: %d)" % (
        len(report["dropped_action_indices"]), len(report["rejected_post_root_drop_indices"])))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
