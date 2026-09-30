"""Human-annotation evidence used by the diagnosed takeover condition."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Optional

from derail.derived.layout import sha256_file


_EVIDENCE_MARKER = "Error type evidence:"


@dataclass(frozen=True)
class HumanDiagnosisEvidence:
    annotation_id: str
    trajectory_id: str
    annotator_id: str
    root_cause_action_index: int
    evidence: str
    annotation_uri: str
    annotation_sha256: str
    source_trajectory_sha256: str

    def to_manifest_dict(self) -> dict[str, Any]:
        return {
            "annotation_id": self.annotation_id,
            "trajectory_id": self.trajectory_id,
            "annotator_id": self.annotator_id,
            "root_cause_action_index": self.root_cause_action_index,
            "evidence": self.evidence,
            "annotation_uri": self.annotation_uri,
            "annotation_sha256": self.annotation_sha256,
            "source_trajectory_sha256": self.source_trajectory_sha256,
            "diagnosis_policy": "verbatim_human_error_type_evidence_block",
        }


def _required_text(raw: Mapping[str, Any], name: str) -> str:
    value = raw.get(name)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"human annotation has no non-empty {name}")
    return value.strip()


def load_human_diagnosis_evidence(
    annotation_path: Path,
    *,
    expected_trajectory_id: Optional[str] = None,
    expected_source_trajectory_sha256: Optional[str] = None,
    maximum_action_index: Optional[int] = None,
) -> HumanDiagnosisEvidence:
    """Load and bind the diagnosed prompt to one immutable human label."""

    path = annotation_path.expanduser().resolve()
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read human annotation: {path}") from exc
    if not isinstance(raw, Mapping):
        raise ValueError("human annotation must be a JSON object")
    if raw.get("annotator_role") != "human":
        raise ValueError("diagnosed takeover requires annotator_role=human")

    annotation_id = _required_text(raw, "annotation_id")
    trajectory_id = _required_text(raw, "trajectory_id")
    annotator_id = _required_text(raw, "annotator_id")
    source_sha256 = _required_text(raw, "source_trajectory_sha256")
    if len(source_sha256) != 64 or any(ch not in "0123456789abcdef" for ch in source_sha256):
        raise ValueError("human annotation source_trajectory_sha256 is invalid")

    root = raw.get("root_cause_action_index")
    if isinstance(root, bool) or not isinstance(root, int) or root < 0:
        raise ValueError("human annotation root_cause_action_index is invalid")
    if maximum_action_index is not None and root > maximum_action_index:
        raise ValueError("annotated root cause lies outside the canonical trajectory")
    if expected_trajectory_id is not None and trajectory_id != expected_trajectory_id:
        raise ValueError(
            f"annotation trajectory_id={trajectory_id!r} does not match "
            f"{expected_trajectory_id!r}"
        )
    if (
        expected_source_trajectory_sha256 is not None
        and source_sha256 != expected_source_trajectory_sha256
    ):
        raise ValueError("annotation is not bound to this source trajectory SHA-256")

    rationale = _required_text(raw, "rationale")
    marker_count = rationale.count(_EVIDENCE_MARKER)
    if marker_count != 1:
        raise ValueError(
            "human annotation rationale must contain exactly one "
            f"{_EVIDENCE_MARKER!r} block"
        )
    evidence = rationale.split(_EVIDENCE_MARKER, 1)[1].strip()
    if not evidence:
        raise ValueError("human annotation evidence block is empty")

    return HumanDiagnosisEvidence(
        annotation_id=annotation_id,
        trajectory_id=trajectory_id,
        annotator_id=annotator_id,
        root_cause_action_index=root,
        evidence=evidence,
        annotation_uri=str(path),
        annotation_sha256=sha256_file(path),
        source_trajectory_sha256=source_sha256,
    )
