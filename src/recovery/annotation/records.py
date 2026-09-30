"""Immutable human annotations and separate adjudication records."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

from recovery.annotation.labels import Reversibility


class AnnotationError(ValueError):
    """An annotation or adjudication violates the benchmark protocol."""


@dataclass(frozen=True)
class HumanAnnotation:
    annotation_id: str
    trajectory_id: str
    source_trajectory_sha256: str
    annotator_id: str
    root_cause_action_index: int
    error_types: Tuple[str, ...]
    reversibility: Reversibility
    rationale: str
    taxonomy_version: str
    schema_version: str = "0.2.0"
    error_horizon_actions: Optional[int] = None
    identifiable_at_action_index: Optional[int] = None

    def __post_init__(self) -> None:
        required = (
            self.annotation_id,
            self.trajectory_id,
            self.source_trajectory_sha256,
            self.annotator_id,
            self.rationale,
            self.taxonomy_version,
        )
        if any(not value.strip() for value in required):
            raise AnnotationError("annotation provenance/rationale 字段不能为空")
        if self.root_cause_action_index < 0:
            raise AnnotationError("root_cause_action_index 不能为负数")
        if not self.error_types or any(not item.strip() for item in self.error_types):
            raise AnnotationError("error_types 至少包含一个非空标签")
        if len(set(self.error_types)) != len(self.error_types):
            raise AnnotationError("error_types 不允许重复")
        if self.error_horizon_actions is None:
            if self.identifiable_at_action_index is not None:
                raise AnnotationError("无 horizon 时 identifiable_at_action_index 必须为空")
        else:
            if self.error_horizon_actions < 0:
                raise AnnotationError("error_horizon_actions 不能为负数")
            expected = self.root_cause_action_index + self.error_horizon_actions
            if self.identifiable_at_action_index != expected:
                raise AnnotationError(
                    "identifiable_at_action_index 必须等于 root + error_horizon_actions"
                )

    def validate_against_action_count(self, action_count: int) -> None:
        if self.root_cause_action_index >= action_count:
            raise AnnotationError("root cause 超出 canonical trajectory")
        if (
            self.identifiable_at_action_index is not None
            and self.identifiable_at_action_index >= action_count
        ):
            raise AnnotationError("error horizon 超出 canonical trajectory")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "annotation_id": self.annotation_id,
            "trajectory_id": self.trajectory_id,
            "source_trajectory_sha256": self.source_trajectory_sha256,
            "annotator_id": self.annotator_id,
            "annotator_role": "human",
            "root_cause_action_index": self.root_cause_action_index,
            "error_horizon_actions": self.error_horizon_actions,
            "identifiable_at_action_index": self.identifiable_at_action_index,
            "error_types": list(self.error_types),
            "reversibility": self.reversibility.value,
            "rationale": self.rationale,
            "taxonomy_version": self.taxonomy_version,
            "schema_version": self.schema_version,
        }

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "HumanAnnotation":
        horizon = raw.get("error_horizon_actions")
        identifiable = raw.get("identifiable_at_action_index")
        return cls(
            annotation_id=str(raw["annotation_id"]),
            trajectory_id=str(raw["trajectory_id"]),
            source_trajectory_sha256=str(raw["source_trajectory_sha256"]),
            annotator_id=str(raw["annotator_id"]),
            root_cause_action_index=int(raw["root_cause_action_index"]),
            error_horizon_actions=int(horizon) if horizon is not None else None,
            identifiable_at_action_index=int(identifiable) if identifiable is not None else None,
            error_types=tuple(str(item) for item in raw["error_types"]),
            reversibility=Reversibility(str(raw["reversibility"])),
            rationale=str(raw["rationale"]),
            taxonomy_version=str(raw["taxonomy_version"]),
            schema_version=str(raw.get("schema_version", "0.2.0")),
        )


@dataclass(frozen=True)
class Adjudication:
    adjudication_id: str
    trajectory_id: str
    input_annotation_ids: Tuple[str, ...]
    adjudicator_id: str
    root_cause_action_index: int
    error_types: Tuple[str, ...]
    reversibility: Reversibility
    resolution_rationale: str
    evidence_refs: Tuple[str, ...]
    taxonomy_version: str
    error_horizon_actions: Optional[int] = None
    identifiable_at_action_index: Optional[int] = None
    disagreement_fields: Tuple[str, ...] = ()
    schema_version: str = "0.2.0"

    def __post_init__(self) -> None:
        if len(self.input_annotation_ids) < 2:
            raise AnnotationError("adjudication 至少引用两份独立 human annotations")
        if len(set(self.input_annotation_ids)) != len(self.input_annotation_ids):
            raise AnnotationError("input_annotation_ids 不允许重复")
        if self.root_cause_action_index < 0 or not self.error_types:
            raise AnnotationError("adjudicated root/error_types 无效")
        if not self.resolution_rationale.strip() or not self.evidence_refs:
            raise AnnotationError("adjudication 必须保留 resolution rationale 和 evidence")
        if self.error_horizon_actions is None:
            if self.identifiable_at_action_index is not None:
                raise AnnotationError("无 horizon 时 identifiable index 必须为空")
        elif self.identifiable_at_action_index != (
            self.root_cause_action_index + self.error_horizon_actions
        ):
            raise AnnotationError("adjudicated identifiable index 必须等于 root + horizon")

    def validate_inputs(self, annotations: Sequence[HumanAnnotation]) -> None:
        indexed = {item.annotation_id: item for item in annotations}
        missing = set(self.input_annotation_ids) - set(indexed)
        if missing:
            raise AnnotationError("adjudication 引用不存在的 annotation: %s" % sorted(missing))
        selected = [indexed[item] for item in self.input_annotation_ids]
        if len({item.annotator_id for item in selected}) < 2:
            raise AnnotationError("adjudication 输入必须来自至少两名 annotator")
        if any(item.trajectory_id != self.trajectory_id for item in selected):
            raise AnnotationError("adjudication 输入 trajectory_id 不一致")
        if any(item.taxonomy_version != self.taxonomy_version for item in selected):
            raise AnnotationError("adjudication 输入 taxonomy_version 不一致")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "adjudication_id": self.adjudication_id,
            "trajectory_id": self.trajectory_id,
            "input_annotation_ids": list(self.input_annotation_ids),
            "adjudicator_id": self.adjudicator_id,
            "root_cause_action_index": self.root_cause_action_index,
            "error_horizon_actions": self.error_horizon_actions,
            "identifiable_at_action_index": self.identifiable_at_action_index,
            "error_types": list(self.error_types),
            "reversibility": self.reversibility.value,
            "disagreement_fields": list(self.disagreement_fields),
            "resolution_rationale": self.resolution_rationale,
            "evidence_refs": list(self.evidence_refs),
            "taxonomy_version": self.taxonomy_version,
            "schema_version": self.schema_version,
        }

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "Adjudication":
        horizon = raw.get("error_horizon_actions")
        identifiable = raw.get("identifiable_at_action_index")
        return cls(
            adjudication_id=str(raw["adjudication_id"]),
            trajectory_id=str(raw["trajectory_id"]),
            input_annotation_ids=tuple(str(item) for item in raw["input_annotation_ids"]),
            adjudicator_id=str(raw["adjudicator_id"]),
            root_cause_action_index=int(raw["root_cause_action_index"]),
            error_horizon_actions=int(horizon) if horizon is not None else None,
            identifiable_at_action_index=int(identifiable) if identifiable is not None else None,
            error_types=tuple(str(item) for item in raw["error_types"]),
            reversibility=Reversibility(str(raw["reversibility"])),
            disagreement_fields=tuple(str(item) for item in raw.get("disagreement_fields", [])),
            resolution_rationale=str(raw["resolution_rationale"]),
            evidence_refs=tuple(str(item) for item in raw["evidence_refs"]),
            taxonomy_version=str(raw["taxonomy_version"]),
            schema_version=str(raw.get("schema_version", "0.2.0")),
        )
