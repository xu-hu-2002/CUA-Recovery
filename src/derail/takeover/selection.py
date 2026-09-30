"""Select valid human-annotated failure trajectories for takeover runs."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

from derail.canonical.mypcbench import load_canonical_jsonl
from derail.construction.cases import eligible_depths
from derail.derived.layout import sha256_file
from derail.takeover.diagnosis import load_human_diagnosis_evidence


def _read_object(path: Path) -> Mapping[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read JSON object: {path}") from exc
    if not isinstance(value, Mapping):
        raise ValueError(f"JSON document is not an object: {path}")
    return value


def _label_key(raw: Mapping[str, Any]) -> tuple:
    return (raw.get("root_cause_action_index"), raw.get("identifiable_at_action_index"))


def _resolve_double_annotation(labels: Path, trajectory_id: str, candidates: list) -> list:
    keys = {_label_key(_read_object(item[0])) for item in candidates}
    if len(keys) == 1:
        return candidates[:1]
    for path in sorted((labels / "adjudications").glob(f"{trajectory_id}__*.json")):
        settled = _read_object(path).get("failure_adjudication")
        if isinstance(settled, Mapping):
            matching = [c for c in candidates if _label_key(_read_object(c[0])) == _label_key(settled)]
            if matching:
                return matching[:1]
    return candidates


def _repaired_prefix(
    repairs_dir: Optional[Path], trajectory_id: str, canonical_sha256: str, root: int
) -> Optional[Mapping[str, Any]]:
    """The human-verified repaired prefix of one trajectory, or ``None`` when absent or stale."""

    if repairs_dir is None:
        return None
    path = repairs_dir.expanduser().resolve() / trajectory_id / "repair_report.json"
    if not path.is_file():
        return None
    report = dict(_read_object(path))
    repaired = Path(str(report.get("repaired_trajectory_uri", "")))
    if not (
        report.get("trajectory_id") == trajectory_id
        and report.get("human_verified") is True
        and report.get("canonical_trajectory_sha256") == canonical_sha256
        and report.get("root_cause_action_index") == root
        and repaired.is_file()
        and sha256_file(repaired) == report.get("repaired_trajectory_sha256")
    ):
        return None
    report["_uri"] = str(path)
    return report


def select_takeover_failures(
    *,
    build_dir: Path,
    human_labels_dir: Path,
    source_agent: str,
    annotator_id: Optional[str] = None,
    trajectory_id_filter: Optional[str] = None,
    trajectory_id_allowlist: Sequence[str] = (),
    depths: Sequence[int] = (),
    require_error_explicit: bool = True,
    repairs_dir: Optional[Path] = None,
    require_repaired_prefix: bool = False,
) -> dict[str, Any]:
    """Return failures with paired human review+annotation, excluding bad rollouts.

    Depth eligibility is ``derail.construction.cases.eligible_depths`` (paper App. C).  When
    ``repairs_dir`` holds a human-verified repaired prefix for a trajectory
    (``<repairs_dir>/<trajectory_id>/repair_report.json``), the item carries its URIs; with
    ``require_repaired_prefix`` a trajectory without one is excluded.
    """

    build = build_dir.expanduser().resolve()
    labels = human_labels_dir.expanduser().resolve()
    canonical_root = build / "canonical"
    if not canonical_root.is_dir() or not labels.is_dir():
        raise ValueError("build canonical directory or human-label directory does not exist")
    requested_depths = tuple(depths)
    allowlist_values = tuple(trajectory_id_allowlist)
    allowed_trajectory_ids = frozenset(allowlist_values)
    if len(allowed_trajectory_ids) != len(allowlist_values):
        raise ValueError("takeover trajectory allowlist must contain unique IDs")
    if any(
        isinstance(depth, bool) or not isinstance(depth, int) or depth < 0
        for depth in requested_depths
    ):
        raise ValueError("takeover depths must be non-negative integers")
    if len(set(requested_depths)) != len(requested_depths):
        raise ValueError("takeover depths must be unique")

    included = []
    excluded = []
    seen_trajectory_ids = set()
    for trajectory_dir in sorted(path for path in canonical_root.iterdir() if path.is_dir()):
        trajectory_id = trajectory_dir.name
        if trajectory_id_filter and trajectory_id != trajectory_id_filter:
            continue
        if allowed_trajectory_ids and trajectory_id not in allowed_trajectory_ids:
            continue
        seen_trajectory_ids.add(trajectory_id)
        trajectory_path = trajectory_dir / "trajectory.jsonl"
        report_path = trajectory_dir / "normalization_report.json"
        task_path = trajectory_dir / "task_config.json"
        if not all(path.is_file() for path in (trajectory_path, report_path, task_path)):
            excluded.append({"trajectory_id": trajectory_id, "reason": "incomplete_canonical"})
            continue
        report = _read_object(report_path)
        source_sha256 = str(report.get("source_trajectory_sha256", ""))
        if report.get("trajectory_id") != trajectory_id:
            excluded.append({"trajectory_id": trajectory_id, "reason": "provenance_mismatch"})
            continue
        if report.get("canonical_trajectory_sha256") != sha256_file(trajectory_path):
            excluded.append({"trajectory_id": trajectory_id, "reason": "provenance_mismatch"})
            continue
        steps = load_canonical_jsonl(trajectory_path)
        if not steps or {step.source_agent for step in steps} != {source_agent}:
            excluded.append({"trajectory_id": trajectory_id, "reason": "source_agent_mismatch"})
            continue

        matching_flags = []
        flagged_annotators = set()
        for flag_path in sorted((labels / "rollout_flags").glob(f"{trajectory_id}__*.json")):
            flag = _read_object(flag_path)
            if (
                flag.get("annotator_role") == "human"
                and flag.get("trajectory_id") == trajectory_id
                and flag.get("source_trajectory_sha256") == source_sha256
                and flag.get("rollout_status") == "needs_rerun"
                and flag.get("reason") == "wrong_rollout"
            ):
                matching_flags.append(
                    {
                        "annotator_id": str(flag.get("annotator_id", "")),
                        "uri": str(flag_path.resolve()),
                    }
                )
                flagged_annotators.add(str(flag.get("annotator_id", "")))

        candidates = []
        for annotation_path in sorted(labels.glob(f"{trajectory_id}__*.json")):
            annotation = _read_object(annotation_path)
            candidate_annotator = str(annotation.get("annotator_id", ""))
            if annotator_id and candidate_annotator != annotator_id:
                continue
            # A flag invalidates only that annotator's own submission.  A
            # different human's failure annotation is disagreement, not a
            # global veto on the trajectory.
            if candidate_annotator in flagged_annotators:
                continue
            review_path = labels / "rubric_scores" / annotation_path.name
            if not review_path.is_file():
                continue
            review = _read_object(review_path)
            if not (
                annotation.get("annotator_role") == "human"
                and annotation.get("trajectory_id") == trajectory_id
                and annotation.get("source_trajectory_sha256") == source_sha256
                and review.get("reviewer_role") == "human"
                and review.get("reviewer_id") == candidate_annotator
                and review.get("trajectory_id") == trajectory_id
                and review.get("source_trajectory_sha256") == source_sha256
                and review.get("task_success") is False
            ):
                continue
            diagnosis = load_human_diagnosis_evidence(
                annotation_path,
                expected_trajectory_id=trajectory_id,
                expected_source_trajectory_sha256=source_sha256,
                maximum_action_index=len(steps) - 1,
            )
            candidates.append((annotation_path, review_path, diagnosis))

        if not candidates:
            if matching_flags:
                excluded.append(
                    {
                        "trajectory_id": trajectory_id,
                        "reason": "wrong_rollout",
                        "rollout_flags": matching_flags,
                    }
                )
            else:
                excluded.append(
                    {"trajectory_id": trajectory_id, "reason": "no_human_failure_annotation"}
                )
            continue
        if len(candidates) > 1:
            # Double-annotated trajectory (paper App. E): take the label the third annotator's
            # adjudication settled on, or either label when both agree on root and t_e.
            candidates = _resolve_double_annotation(labels, trajectory_id, candidates)
        if len(candidates) != 1:
            excluded.append(
                {
                    "trajectory_id": trajectory_id,
                    "reason": "ambiguous_human_failure_annotations",
                    "annotation_uris": [str(item[0].resolve()) for item in candidates],
                }
            )
            continue
        annotation_path, review_path, diagnosis = candidates[0]
        review = _read_object(review_path)
        last_action_index = len(steps) - 1
        identifiable = _read_object(annotation_path).get("identifiable_at_action_index")
        available, skipped_depths = eligible_depths(
            diagnosis.root_cause_action_index,
            last_action_index,
            identifiable,
            requested_depths,
            require_error_explicit=require_error_explicit,
        )
        available_depths = list(available)
        repair = _repaired_prefix(
            repairs_dir, trajectory_id, sha256_file(trajectory_path), diagnosis.root_cause_action_index
        )
        if require_repaired_prefix and repair is None:
            excluded.append({"trajectory_id": trajectory_id, "reason": "no_verified_repaired_prefix"})
            continue
        included.append(
            {
                "trajectory_id": trajectory_id,
                "source_agent": source_agent,
                "annotator_id": diagnosis.annotator_id,
                "root_cause_action_index": diagnosis.root_cause_action_index,
                "last_action_index": last_action_index,
                "available_depths": available_depths,
                "unavailable_depths": [
                    depth for depth in requested_depths if depth not in available_depths
                ],
                "unavailable_depth_reasons": {
                    str(depth): reason for depth, reason in skipped_depths.items()
                },
                "identifiable_at_action_index": identifiable,
                "repaired_trajectory_uri": repair["repaired_trajectory_uri"] if repair else None,
                "repaired_trajectory_sha256": (
                    repair["repaired_trajectory_sha256"] if repair else None
                ),
                "repair_report_uri": repair["_uri"] if repair else None,
                "annotation_uri": str(annotation_path.resolve()),
                "annotation_sha256": diagnosis.annotation_sha256,
                "rubric_review_uri": str(review_path.resolve()),
                "rubric_review_sha256": sha256_file(review_path),
                "human_rubric_scores": dict(review.get("scores", {})),
                "human_task_success": False,
                "perfect_score_policy": "all human rubric scores equal 1",
                "canonical_trajectory_uri": str(trajectory_path.resolve()),
                "normalization_report_uri": str(report_path.resolve()),
                "task_config_uri": str(task_path.resolve()),
                "source_trajectory_sha256": source_sha256,
                "ignored_cross_annotator_wrong_rollout_flags": [
                    flag
                    for flag in matching_flags
                    if flag["annotator_id"] != diagnosis.annotator_id
                ],
            }
        )

    build_manifest = _read_object(build / "build_manifest.json")
    for report in build_manifest.get("normalization_reports", ()):
        if not isinstance(report, Mapping):
            continue
        trajectory_id = str(report.get("trajectory_id", ""))
        if not trajectory_id or trajectory_id in seen_trajectory_ids:
            continue
        if trajectory_id_filter and trajectory_id != trajectory_id_filter:
            continue
        if allowed_trajectory_ids and trajectory_id not in allowed_trajectory_ids:
            continue
        annotation_paths = sorted(labels.glob(f"{trajectory_id}__*.json"))
        if annotator_id:
            annotation_paths = [
                path
                for path in annotation_paths
                if str(_read_object(path).get("annotator_id", "")) == annotator_id
            ]
        if not annotation_paths:
            continue
        excluded.append(
            {
                "trajectory_id": trajectory_id,
                "reason": "canonicalization_rejected",
                "annotation_uris": [str(path.resolve()) for path in annotation_paths],
                "normalization_report_uri": str(report.get("uri", "")),
                "rejection_reason": str(report.get("rejection_reason", "")),
            }
        )

    reasons = Counter(str(item["reason"]) for item in excluded)
    return {
        "selection_policy": {
            "failure": (
                "same human annotator has task_success=false rubric review and "
                "a hash-bound failure annotation"
            ),
            "wrong_rollout_exclusion": (
                "a hash-bound human wrong-rollout flag invalidates only that "
                "annotator's candidate; cross-annotator disagreement is retained"
            ),
            "source_agent": source_agent,
            "annotator_id_filter": annotator_id,
            "trajectory_id_filter": trajectory_id_filter,
            "trajectory_id_allowlist": sorted(allowed_trajectory_ids),
            "requested_depths": list(requested_depths),
            "depth_eligibility": (
                "root + d <= last action and root + d <= identifiable_at (d <= h_e); "
                "unobserved h_e %s" % ("excluded" if require_error_explicit else "kept")
            ),
            "repairs_dir": str(repairs_dir) if repairs_dir else None,
            "require_repaired_prefix": require_repaired_prefix,
            "selection_universe": "build normalization reports for human-labelled trajectories",
        },
        "included_count": len(included),
        "excluded_count": len(excluded),
        "excluded_reason_counts": dict(sorted(reasons.items())),
        "depth_coverage": {
            str(depth): sum(depth in item["available_depths"] for item in included)
            for depth in requested_depths
        },
        "included": included,
        "excluded": excluded,
    }
