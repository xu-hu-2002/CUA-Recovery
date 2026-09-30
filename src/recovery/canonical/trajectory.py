"""Canonical trajectory action records and JSONL serialization."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Mapping, Optional

from recovery.derived.validation import strict_bool

from .actions import Action, action_from_dict, action_to_dict
from .summaries import summarize_action


@dataclass(frozen=True)
class CanonicalStep:
    step_id: int
    action: Action
    observation_before_sha256: str
    observation_after_sha256: str = ""
    tool_result: str = "ok"
    source_agent: str = ""
    source_step_id: Optional[int] = None
    repaired: bool = False
    repair_patch_id: str = ""
    turn_index: Optional[int] = None
    action_index_within_turn: int = 0
    source_action_timestamp: str = ""
    observation_before_uri: str = ""
    observation_after_uri: str = ""
    source_record_uri: str = ""

    def __post_init__(self) -> None:
        if self.step_id < 0:
            raise ValueError("step_id 不能为负数")
        if self.turn_index is not None and self.turn_index < 0:
            raise ValueError("turn_index 不能为负数")
        if self.action_index_within_turn < 0:
            raise ValueError("action_index_within_turn 不能为负数")
        if self.repaired and not self.repair_patch_id:
            raise ValueError("repaired step 必须记录 repair_patch_id")

    @property
    def action_index_global(self) -> int:
        """Global executed-action index; alias of the legacy ``step_id`` field."""

        return self.step_id

    def to_dict(self) -> Dict[str, Any]:
        return {
            "step_id": self.step_id,
            "action_index_global": self.step_id,
            "turn_index": self.turn_index if self.turn_index is not None else self.step_id,
            "action_index_within_turn": self.action_index_within_turn,
            "observation_before_sha256": self.observation_before_sha256,
            "observation_before_uri": self.observation_before_uri,
            "action": action_to_dict(self.action),
            "action_summary": summarize_action(self.action),
            "observation_after_sha256": self.observation_after_sha256,
            "observation_after_uri": self.observation_after_uri,
            "tool_result": self.tool_result,
            "source_agent": self.source_agent,
            "source_step_id": self.source_step_id,
            "source_action_timestamp": self.source_action_timestamp,
            "source_record_uri": self.source_record_uri,
            "repaired": self.repaired,
            "repair_patch_id": self.repair_patch_id,
        }

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "CanonicalStep":
        step_id = int(raw.get("action_index_global", raw.get("step_id", -1)))
        legacy_step_id = raw.get("step_id")
        if legacy_step_id is not None and int(legacy_step_id) != step_id:
            raise ValueError("step_id 与 action_index_global 不一致")
        return cls(
            step_id=step_id,
            observation_before_sha256=str(raw["observation_before_sha256"]),
            action=action_from_dict(raw["action"]),
            observation_after_sha256=str(raw.get("observation_after_sha256", "")),
            observation_before_uri=str(raw.get("observation_before_uri", "")),
            observation_after_uri=str(raw.get("observation_after_uri", "")),
            tool_result=str(raw.get("tool_result", "ok")),
            source_agent=str(raw.get("source_agent", "")),
            source_step_id=raw.get("source_step_id"),
            turn_index=(int(raw["turn_index"]) if raw.get("turn_index") is not None else None),
            action_index_within_turn=int(raw.get("action_index_within_turn", 0)),
            source_action_timestamp=str(raw.get("source_action_timestamp", "")),
            source_record_uri=str(raw.get("source_record_uri", "")),
            repaired=strict_bool(raw.get("repaired", False), "repaired"),
            repair_patch_id=str(raw.get("repair_patch_id", "")),
        )
