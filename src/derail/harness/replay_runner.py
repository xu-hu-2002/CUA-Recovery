from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Callable, Dict, Mapping, Optional, Sequence

from derail.harness.env_wrapper import DerailEnvWrapper
from derail.world.volatile import VolatileColumns

VERIFICATION_VERSION = "replay-verification/1.0"


def change_signature(row: Mapping[str, Any], volatile: Optional[VolatileColumns] = None) -> tuple:
    new = json.loads(row["new_json"]) if row.get("new_json") else {}
    table = "%s.%s" % (row["db"], row["tbl"])
    if volatile is not None:
        new = {k: v for k, v in new.items() if not volatile.is_volatile(table, k)}
    return (
        row["db"],
        row["tbl"],
        int(row["rowid"]),
        row["op"],
        json.dumps(new, sort_keys=True, ensure_ascii=False),
    )


def step_signature(
    step: Mapping[str, Any], volatile: Optional[VolatileColumns] = None
) -> Dict[str, Any]:
    return {
        "changes": sorted(change_signature(r, volatile) for r in step.get("delta", ())),
        "pages": [(p["app"], p["route"]) for p in step.get("pages", ())],
    }


def compare_steps(
    original: Sequence[Mapping[str, Any]],
    replayed: Sequence[Mapping[str, Any]],
    volatile: Optional[VolatileColumns] = None,
) -> Dict[str, Any]:
    changelog_match = page_match = True
    first_mismatch: Optional[int] = None
    for index, (before, after) in enumerate(zip(original, replayed)):
        a, b = step_signature(before, volatile), step_signature(after, volatile)
        if a["changes"] != b["changes"]:
            changelog_match = False
        if a["pages"] != b["pages"]:
            page_match = False
        if (a != b) and first_mismatch is None:
            first_mismatch = int(before.get("action_index", index))
    if len(original) != len(replayed):
        changelog_match = page_match = False
        first_mismatch = (
            first_mismatch if first_mismatch is not None else min(len(original), len(replayed))
        )
    return {
        "changelog_match": changelog_match,
        "page_sequence_match": page_match,
        "first_mismatch_action_index": first_mismatch,
    }


@dataclass
class ReplayRunner:
    restore: Callable[[], None]
    make_wrapper: Callable[[], DerailEnvWrapper]
    volatile: Optional[VolatileColumns] = None
    required_attempts: int = 3
    execution_mode: str = "vm"

    def replay_once(
        self,
        task_config: Mapping[str, Any],
        actions: Sequence[Mapping[str, Any]],
        pause: float = 2.0,
    ) -> Dict[str, Any]:
        self.restore()
        wrapper = self.make_wrapper()
        wrapper.reset(task_config=dict(task_config))
        for step in actions:
            action = step["action"]
            if action["type"] == "shell":
                wrapper._execute_shell(str(action["raw"]))
            else:
                wrapper.step(action["raw"], pause)
        return wrapper.finish(step_budget_reached=False)

    def verify(
        self,
        *,
        trace: Mapping[str, Any],
        task_config: Mapping[str, Any],
        repaired_prefix_action_indices: Sequence[int],
        root_cause_action_index: int,
        depths: Sequence[int],
        repaired_prefix_ref: str,
        snapshot_sha256: str,
        image_digest: Optional[str] = None,
        pause: float = 2.0,
    ) -> Dict[str, Any]:
        by_index = {int(s["action_index"]): s for s in trace["steps"]}
        prefix = [by_index[i] for i in repaired_prefix_action_indices]
        last_index = max(int(s["action_index"]) for s in trace["steps"])
        results = []
        for depth in depths:
            end = root_cause_action_index + int(depth)
            if end > last_index:
                results.append(
                    {
                        "depth": int(depth),
                        "replay_end_action_index": end,
                        "attempts": [],
                        "passed": False,
                        "code": "DEPTH_UNAVAILABLE",
                        "human_review": "not_required",
                    }
                )
                continue
            tail = [by_index[i] for i in range(root_cause_action_index, end + 1)]
            actions = prefix + tail
            attempts = []
            for attempt in range(1, self.required_attempts + 1):
                replayed = self.replay_once(task_config, actions, pause)
                verdict = compare_steps(actions, replayed["steps"], self.volatile)
                attempts.append(
                    {
                        "attempt_index": attempt,
                        "changelog_match": verdict["changelog_match"],
                        "page_sequence_match": verdict["page_sequence_match"],
                        "state_sha256": None,
                        "first_mismatch_action_index": verdict["first_mismatch_action_index"],
                        "log_ref": None,
                    }
                )
                if not (verdict["changelog_match"] and verdict["page_sequence_match"]):
                    break
            passed = len(attempts) == self.required_attempts and all(
                a["changelog_match"] and a["page_sequence_match"] for a in attempts
            )
            results.append(
                {
                    "depth": int(depth),
                    "replay_end_action_index": end,
                    "attempts": attempts,
                    "passed": passed,
                    "code": None if passed else "PREFIX_REPLAY_MISMATCH",
                    "human_review": "not_required",
                }
            )
        return {
            "schema_version": VERIFICATION_VERSION,
            "case_id": None,
            "rollout_id": str(trace["rollout_id"]),
            "task_id": str(trace["task_id"]),
            "world_id": str(trace["world_id"]),
            "root_cause_action_index": int(root_cause_action_index),
            "repaired_prefix_ref": repaired_prefix_ref,
            "required_attempts": self.required_attempts,
            "depths": results,
            "execution_mode": self.execution_mode,
            "snapshot_sha256": snapshot_sha256,
            "image_digest": image_digest,
            "verification_version": "replay-runner/1.0",
            "provenance": {"prefix_action_indices": list(repaired_prefix_action_indices)},
        }
