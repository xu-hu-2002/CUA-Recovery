"""Action-indexed RECOVERY case and depth-instance construction."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

from recovery.annotation.labels import Reversibility
from recovery.annotation.records import Adjudication
from recovery.canonical.trajectory import CanonicalStep
from recovery.construction.repair import PrefixAudit, RepairPatch, apply_repair_patches
from recovery.derived.layout import DEPTH_GRID


class CaseConstructionError(ValueError):
    """A source trajectory cannot instantiate a protocol-valid RECOVERY case."""


SKIP_SHORT_SUFFIX = "source suffix shorter than root + depth"
SKIP_AFTER_EXPLICIT = "takeover point after the step where the error becomes explicit"
SKIP_UNOBSERVED = "error never becomes explicit in the rollout"


def eligible_depths(
    root_cause_action_index: int,
    last_action_index: int,
    identifiable_at_action_index: Optional[int],
    depths: Sequence[int] = DEPTH_GRID,
    *,
    require_error_explicit: bool = True,
) -> Tuple[Tuple[int, ...], Dict[int, str]]:
    root = int(root_cause_action_index)
    available = []
    skipped: Dict[int, str] = {}
    for depth in depths:
        end = root + int(depth)
        if end > last_action_index:
            skipped[depth] = SKIP_SHORT_SUFFIX
        elif identifiable_at_action_index is None:
            if require_error_explicit:
                skipped[depth] = SKIP_UNOBSERVED
            else:
                available.append(depth)
        elif end > int(identifiable_at_action_index):
            skipped[depth] = SKIP_AFTER_EXPLICIT
        else:
            available.append(depth)
    return tuple(available), skipped


def _action_digest(steps: Sequence[CanonicalStep]) -> str:
    payload = [step.to_dict()["action"] for step in steps]
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


@dataclass(frozen=True)
class DepthInstance:
    instance_id: str
    case_id: str
    depth: int
    root_cause_action_index: int
    replay_end_action_index: int
    executed_action_indices: Tuple[int, ...]
    history_action_indices: Tuple[int, ...]
    canonical_actions_sha256: str

    def __post_init__(self) -> None:
        if self.depth not in DEPTH_GRID:
            raise CaseConstructionError("depth must be in %s" % (DEPTH_GRID,))
        if self.replay_end_action_index != self.root_cause_action_index + self.depth:
            raise CaseConstructionError("replay_end_action_index must equal root + depth")
        if not self.executed_action_indices:
            raise CaseConstructionError("cleaned depth instance must not be empty")
        if self.executed_action_indices != tuple(sorted(set(self.executed_action_indices))):
            raise CaseConstructionError("cleaned action indices must be strictly increasing and unique")
        if any(index < 0 or index > self.replay_end_action_index for index in self.executed_action_indices):
            raise CaseConstructionError("cleaned action index is beyond the source prefix depth boundary")
        if self.root_cause_action_index not in self.executed_action_indices:
            raise CaseConstructionError("cleaned prefix must not drop the root-cause action")
        if self.history_action_indices != self.executed_action_indices:
            raise CaseConstructionError("injected history must come from the replay executed actions")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "instance_id": self.instance_id,
            "case_id": self.case_id,
            "depth": self.depth,
            "root_cause_action_index": self.root_cause_action_index,
            "replay_end_action_index": self.replay_end_action_index,
            "executed_action_indices": list(self.executed_action_indices),
            "history_action_indices": list(self.history_action_indices),
            "canonical_actions_sha256": self.canonical_actions_sha256,
            "takeover_semantics": "source_action_index_offset_from_root",
            "replay_verification_status": "pending",
        }

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "DepthInstance":
        return cls(
            instance_id=str(raw["instance_id"]),
            case_id=str(raw["case_id"]),
            depth=int(raw["depth"]),
            root_cause_action_index=int(raw["root_cause_action_index"]),
            replay_end_action_index=int(raw["replay_end_action_index"]),
            executed_action_indices=tuple(int(item) for item in raw["executed_action_indices"]),
            history_action_indices=tuple(int(item) for item in raw["history_action_indices"]),
            canonical_actions_sha256=str(raw["canonical_actions_sha256"]),
        )


@dataclass(frozen=True)
class CasePlan:
    case_id: str
    trajectory_id: str
    source_trajectory_sha256: str
    adjudication_id: str
    root_cause_action_index: int
    error_horizon_actions: Optional[int]
    identifiable_at_action_index: Optional[int]
    error_types: Tuple[str, ...]
    reversibility: Reversibility
    taxonomy_version: str
    available_depths: Tuple[int, ...]
    unavailable_depth_reasons: Mapping[str, str]
    repaired_steps: Tuple[CanonicalStep, ...]
    instances: Tuple[DepthInstance, ...]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "case_id": self.case_id,
            "trajectory_id": self.trajectory_id,
            "source_trajectory_sha256": self.source_trajectory_sha256,
            "adjudication_id": self.adjudication_id,
            "root_cause_action_index": self.root_cause_action_index,
            "error_horizon_actions": self.error_horizon_actions,
            "identifiable_at_action_index": self.identifiable_at_action_index,
            "error_types": list(self.error_types),
            "reversibility": self.reversibility.value,
            "taxonomy_version": self.taxonomy_version,
            "available_depths": list(self.available_depths),
            "unavailable_depth_reasons": dict(self.unavailable_depth_reasons),
            "canonical_repaired_actions_sha256": _action_digest(self.repaired_steps),
            "prefix_state_status": "pending_real_vm_replay",
            "release_eligible": False,
            "instances": [instance.to_dict() for instance in self.instances],
        }


def build_case_plan(
    *,
    case_id: str,
    trajectory_id: str,
    source_trajectory_sha256: str,
    steps: Sequence[CanonicalStep],
    adjudication: Adjudication,
    prefix_audit: PrefixAudit,
    patches: Sequence[RepairPatch],
    require_error_explicit: bool = True,
) -> CasePlan:
    if not case_id.strip() or not trajectory_id.strip() or not source_trajectory_sha256.strip():
        raise CaseConstructionError("case/source provenance must not be empty")
    actual_indices = [step.action_index_global for step in steps]
    if actual_indices != list(range(len(steps))):
        raise CaseConstructionError("canonical action indices must be contiguous from 0")
    if adjudication.trajectory_id != trajectory_id:
        raise CaseConstructionError("adjudication trajectory_id does not match the case source")
    root = adjudication.root_cause_action_index
    if root >= len(steps):
        raise CaseConstructionError("root cause is beyond the trajectory")
    if root == 0 and (
        not steps[0].observation_before_sha256 or not steps[0].observation_before_uri
    ):
        raise CaseConstructionError("root=0 is missing the initial pre-action observation; evidence incomplete")
    if (
        adjudication.identifiable_at_action_index is not None
        and adjudication.identifiable_at_action_index >= len(steps)
    ):
        raise CaseConstructionError("error horizon is beyond the trajectory")
    if any(patch.case_id != case_id for patch in patches):
        raise CaseConstructionError("repair patch case_id mismatch")
    if any(patch.root_cause_step != root for patch in patches):
        raise CaseConstructionError("repair patch root does not match adjudication")
    if prefix_audit.case_id != case_id or prefix_audit.root_cause_action_index != root:
        raise CaseConstructionError("prefix audit case/root does not match adjudication")
    if prefix_audit.source_trajectory_sha256 != source_trajectory_sha256:
        raise CaseConstructionError("prefix audit source trajectory hash mismatch")
    audited, _ = eligible_depths(root, len(steps) - 1, None, require_error_explicit=False)
    available, unavailable = eligible_depths(
        root,
        len(steps) - 1,
        adjudication.identifiable_at_action_index,
        require_error_explicit=require_error_explicit,
    )
    if not available:
        raise CaseConstructionError("trajectory has no usable depth instance: %s" % unavailable)
    audit_end = root + max(audited)
    if prefix_audit.audit_end_action_index > audit_end:
        raise CaseConstructionError(
            "prefix audit end is beyond the maximum usable depth boundary: %d" % audit_end
        )
    prefix_audit.validate_patches(patches)

    repaired = apply_repair_patches(steps[: audit_end + 1], patches)
    repaired_by_id = {step.action_index_global: step for step in repaired}
    if root not in repaired_by_id or repaired_by_id[root].action != steps[root].action:
        raise CaseConstructionError("cleaning must not drop or modify the root-cause action")
    for step in repaired:
        if step.action_index_global >= root and step.action != steps[step.action_index_global].action:
            raise CaseConstructionError("retained actions at and after root must stay unchanged")

    instances = []
    for depth in available:
        replay_end = root + depth
        selected = tuple(
            step for step in repaired if step.action_index_global <= replay_end
        )
        indices = tuple(step.action_index_global for step in selected)
        instances.append(
            DepthInstance(
                instance_id="%s-d%d" % (case_id, depth),
                case_id=case_id,
                depth=depth,
                root_cause_action_index=root,
                replay_end_action_index=replay_end,
                executed_action_indices=indices,
                history_action_indices=indices,
                canonical_actions_sha256=_action_digest(selected),
            )
        )
    return CasePlan(
        case_id=case_id,
        trajectory_id=trajectory_id,
        source_trajectory_sha256=source_trajectory_sha256,
        adjudication_id=adjudication.adjudication_id,
        root_cause_action_index=root,
        error_horizon_actions=adjudication.error_horizon_actions,
        identifiable_at_action_index=adjudication.identifiable_at_action_index,
        error_types=adjudication.error_types,
        reversibility=adjudication.reversibility,
        taxonomy_version=adjudication.taxonomy_version,
        available_depths=available,
        unavailable_depth_reasons={str(depth): reason for depth, reason in unavailable.items()},
        repaired_steps=tuple(repaired),
        instances=tuple(instances),
    )
