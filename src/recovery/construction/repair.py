from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence, Tuple

from recovery.canonical.actions import Action, action_from_dict, action_to_dict
from recovery.canonical.trajectory import CanonicalStep
from recovery.derived.validation import strict_bool


class PrefixRepairError(ValueError):
    pass


@dataclass(frozen=True)
class PrefixAudit:
    """Human approval for every source action through the largest available depth."""

    audit_id: str
    case_id: str
    source_trajectory_sha256: str
    root_cause_action_index: int
    audit_end_action_index: int
    audited_prefix_action_indices: Tuple[int, ...]
    unrelated_error_action_indices: Tuple[int, ...]
    repair_patch_ids: Tuple[str, ...]
    reviewer_ids: Tuple[str, ...]
    evidence_refs: Tuple[str, ...]
    rationale: str
    approved: bool
    schema_version: str = "0.2.0"
    min_reviewers: int = field(default=1, compare=False, repr=False)

    def __post_init__(self) -> None:
        if not self.audit_id or not self.case_id or not self.source_trajectory_sha256:
            raise PrefixRepairError("prefix audit provenance 字段不能为空")
        if self.root_cause_action_index < 0:
            raise PrefixRepairError("prefix audit root 不能为负数")
        if self.audit_end_action_index < self.root_cause_action_index:
            raise PrefixRepairError("prefix audit end 不能早于 root cause")
        expected = tuple(range(self.audit_end_action_index + 1))
        if self.audited_prefix_action_indices != expected:
            raise PrefixRepairError("prefix audit 必须逐一覆盖 clean-prefix 最大 depth 边界内的 actions")
        if any(index >= self.root_cause_action_index for index in self.unrelated_error_action_indices):
            raise PrefixRepairError("prefix repair 只能修改 root cause 之前的 actions")
        if len(set(self.unrelated_error_action_indices)) != len(
            self.unrelated_error_action_indices
        ):
            raise PrefixRepairError("unrelated_error_action_indices 不允许重复")
        if len(set(self.reviewer_ids)) < max(1, self.min_reviewers):
            raise PrefixRepairError(
                "prefix error-free audit 至少需要 %d 名 human reviewers" % max(1, self.min_reviewers)
            )
        if not self.evidence_refs or not self.rationale.strip():
            raise PrefixRepairError("prefix audit 必须包含 evidence 和 rationale")
        if len(set(self.repair_patch_ids)) != len(self.repair_patch_ids):
            raise PrefixRepairError("repair_patch_ids 不允许重复")
        if not self.approved:
            raise PrefixRepairError("未获 human approval 的 prefix 不能构建 case")

    def validate_patches(self, patches: Sequence["RepairPatch"]) -> None:
        patch_indices = tuple(sorted(patch.step_id for patch in patches))
        error_indices = tuple(sorted(self.unrelated_error_action_indices))
        if patch_indices != error_indices:
            raise PrefixRepairError("每个 unrelated prefix error 必须恰好有一个 repair patch")
        if set(self.repair_patch_ids) != {patch.patch_id for patch in patches}:
            raise PrefixRepairError("prefix audit repair_patch_ids 与实际 patches 不一致")
        if any(patch.root_cause_step != self.root_cause_action_index for patch in patches):
            raise PrefixRepairError("prefix audit 与 repair patch 的 root cause 不一致")
        if any(patch.step_id > self.audit_end_action_index for patch in patches):
            raise PrefixRepairError("repair patch 超出 prefix audit 边界")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "audit_id": self.audit_id,
            "case_id": self.case_id,
            "source_trajectory_sha256": self.source_trajectory_sha256,
            "root_cause_action_index": self.root_cause_action_index,
            "audit_end_action_index": self.audit_end_action_index,
            "audited_prefix_action_indices": list(self.audited_prefix_action_indices),
            "unrelated_error_action_indices": list(self.unrelated_error_action_indices),
            "repair_patch_ids": list(self.repair_patch_ids),
            "reviewer_ids": list(self.reviewer_ids),
            "evidence_refs": list(self.evidence_refs),
            "rationale": self.rationale,
            "approved": self.approved,
            "schema_version": self.schema_version,
        }

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any], min_reviewers: int = 1) -> "PrefixAudit":
        return cls(
            audit_id=str(raw["audit_id"]),
            case_id=str(raw["case_id"]),
            source_trajectory_sha256=str(raw["source_trajectory_sha256"]),
            root_cause_action_index=int(raw["root_cause_action_index"]),
            audit_end_action_index=int(raw["audit_end_action_index"]),
            audited_prefix_action_indices=tuple(
                int(item) for item in raw["audited_prefix_action_indices"]
            ),
            unrelated_error_action_indices=tuple(
                int(item) for item in raw["unrelated_error_action_indices"]
            ),
            repair_patch_ids=tuple(str(item) for item in raw["repair_patch_ids"]),
            reviewer_ids=tuple(str(item) for item in raw["reviewer_ids"]),
            evidence_refs=tuple(str(item) for item in raw["evidence_refs"]),
            rationale=str(raw["rationale"]),
            approved=strict_bool(raw["approved"], "approved"),
            schema_version=str(raw.get("schema_version", "0.2.0")),
            min_reviewers=min_reviewers,
        )


@dataclass(frozen=True)
class RepairPatch:
    patch_id: str
    case_id: str
    step_id: int
    root_cause_step: int
    old_action: Action
    reason: str
    annotator_id: str
    operation: str = "replace"
    new_action: Optional[Action] = None
    self_recovered: bool = False
    persistent_state_effect: bool = True
    causal_to_root_or_task: bool = True
    evidence: Tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.patch_id or not self.case_id:
            raise PrefixRepairError("patch_id 和 case_id 不能为空")
        if self.step_id < 0 or self.root_cause_step < 0:
            raise PrefixRepairError("step_id 和 root_cause_step 不能为负数")
        if self.operation not in {"replace", "drop"}:
            raise PrefixRepairError("repair operation 只能是 replace 或 drop")
        if not self.reason.strip():
            raise PrefixRepairError("repair 必须包含人工可审计的 reason")
        if not self.annotator_id.strip():
            raise PrefixRepairError("repair 必须记录 annotator_id")
        if self.operation == "replace":
            if self.step_id >= self.root_cause_step:
                raise PrefixRepairError("replace 只能修改 root cause 之前的动作")
            if self.new_action is None or self.old_action == self.new_action:
                raise PrefixRepairError("replace 必须提供与 old_action 不同的 new_action")
        else:
            if self.step_id >= self.root_cause_step:
                raise PrefixRepairError("不能删除 root-cause action 及其之后的 actions")
            if self.new_action is not None:
                raise PrefixRepairError("drop operation 不能包含 new_action")
            if not self.self_recovered:
                raise PrefixRepairError("只能删除已被 agent 自恢复的 action")
            if self.persistent_state_effect:
                raise PrefixRepairError("不能删除具有持续状态影响的 action")
            if self.causal_to_root_or_task:
                raise PrefixRepairError("不能删除对 root cause 或任务结果具有因果作用的 action")

    def to_dict(self) -> Dict[str, Any]:
        record = {
            "patch_id": self.patch_id,
            "case_id": self.case_id,
            "operation": self.operation,
            "step_id": self.step_id,
            "action_index_global": self.step_id,
            "root_cause_step": self.root_cause_step,
            "root_cause_action_index": self.root_cause_step,
            "old_action": action_to_dict(self.old_action),
            "reason": self.reason,
            "annotator_id": self.annotator_id,
            "evidence": list(self.evidence),
        }
        if self.operation == "replace":
            assert self.new_action is not None
            record["new_action"] = action_to_dict(self.new_action)
        else:
            record.update(
                {
                    "self_recovered": self.self_recovered,
                    "persistent_state_effect": self.persistent_state_effect,
                    "causal_to_root_or_task": self.causal_to_root_or_task,
                }
            )
        return record

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "RepairPatch":
        operation = str(raw.get("operation", "replace"))
        action_index = int(raw.get("action_index_global", raw.get("step_id", -1)))
        root_index = int(
            raw.get("root_cause_action_index", raw.get("root_cause_step", -1))
        )
        if raw.get("step_id") is not None and int(raw["step_id"]) != action_index:
            raise PrefixRepairError("step_id 与 action_index_global 不一致")
        if raw.get("root_cause_step") is not None and int(raw["root_cause_step"]) != root_index:
            raise PrefixRepairError("root_cause_step 与 root_cause_action_index 不一致")
        return cls(
            patch_id=str(raw["patch_id"]),
            case_id=str(raw["case_id"]),
            step_id=action_index,
            root_cause_step=root_index,
            old_action=action_from_dict(raw["old_action"]),
            reason=str(raw["reason"]),
            annotator_id=str(raw["annotator_id"]),
            operation=operation,
            new_action=(
                action_from_dict(raw["new_action"])
                if raw.get("new_action") is not None
                else None
            ),
            self_recovered=strict_bool(raw.get("self_recovered", False), "self_recovered"),
            persistent_state_effect=strict_bool(
                raw.get("persistent_state_effect", True), "persistent_state_effect"
            ),
            causal_to_root_or_task=strict_bool(
                raw.get("causal_to_root_or_task", True), "causal_to_root_or_task"
            ),
            evidence=tuple(str(item) for item in raw.get("evidence", [])),
        )


def apply_repair_patches(
    steps: Sequence[CanonicalStep], patches: Iterable[RepairPatch]
) -> Tuple[CanonicalStep, ...]:
    patch_by_step: Dict[int, RepairPatch] = {}
    for patch in patches:
        if patch.step_id in patch_by_step:
            raise PrefixRepairError("同一步存在多个 repair patch: %d" % patch.step_id)
        patch_by_step[patch.step_id] = patch

    known_step_ids = {step.step_id for step in steps}
    missing = sorted(set(patch_by_step) - known_step_ids)
    if missing:
        raise PrefixRepairError("repair 指向不存在的 step: %s" % missing)

    repaired_steps = []
    earliest_patch = min(patch_by_step) if patch_by_step else None
    for step in steps:
        patch = patch_by_step.get(step.step_id)
        if patch is None:
            if earliest_patch is None or step.step_id < earliest_patch:
                repaired_steps.append(step)
                continue
            repaired_steps.append(
                CanonicalStep(
                    step_id=step.step_id,
                    action=step.action,
                    observation_before_sha256="",
                    observation_after_sha256="",
                    observation_before_uri="",
                    observation_after_uri="",
                    tool_result="pending_replay",
                    source_agent=step.source_agent,
                    source_step_id=step.source_step_id,
                    turn_index=step.turn_index,
                    action_index_within_turn=step.action_index_within_turn,
                    source_action_timestamp=step.source_action_timestamp,
                    source_record_uri=step.source_record_uri,
                    repaired=step.repaired,
                    repair_patch_id=step.repair_patch_id,
                )
            )
            continue
        if step.action != patch.old_action:
            raise PrefixRepairError(
                "step %d 的 old_action 与轨迹不匹配，拒绝应用过期 patch" % step.step_id
            )
        if patch.operation == "drop":
            continue
        assert patch.new_action is not None
        before_is_still_valid = earliest_patch == step.step_id
        repaired_steps.append(
            CanonicalStep(
                step_id=step.step_id,
                action=patch.new_action,
                observation_before_sha256=(
                    step.observation_before_sha256 if before_is_still_valid else ""
                ),
                observation_before_uri=(
                    step.observation_before_uri if before_is_still_valid else ""
                ),
                observation_after_sha256="",
                tool_result="pending_replay",
                source_agent=step.source_agent,
                source_step_id=step.source_step_id,
                turn_index=step.turn_index,
                action_index_within_turn=step.action_index_within_turn,
                source_action_timestamp=step.source_action_timestamp,
                source_record_uri=step.source_record_uri,
                repaired=True,
                repair_patch_id=patch.patch_id,
            )
        )
    return tuple(repaired_steps)


def load_repaired_prefix(path: Path) -> Tuple[CanonicalStep, ...]:
    steps = []
    with Path(path).open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                steps.append(CanonicalStep.from_dict(json.loads(line)))
    indices = [step.action_index_global for step in steps]
    if not steps or indices != sorted(set(indices)):
        raise PrefixRepairError("repaired prefix action indices 必须严格递增: %s" % path)
    return tuple(steps)
