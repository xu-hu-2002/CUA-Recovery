from __future__ import annotations

import json
from collections import Counter
from dataclasses import asdict, dataclass
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

LOOP_HEURISTIC_VERSION = "signature_repeat_v1"


@dataclass(frozen=True)
class LoopConfig:
    min_repeats: int = 3
    click_cell_px: int = 40
    exclude_kinds: Tuple[str, ...] = ("wait",)

    def __post_init__(self) -> None:
        if self.min_repeats < 2 or self.click_cell_px < 1:
            raise ValueError("min_repeats must be >= 2 and click_cell_px >= 1")


def action_signature(action: Mapping[str, Any], *, click_cell_px: int = 40) -> str:
    reduced: Dict[str, Any] = {}
    for key in sorted(action):
        if key.startswith("frame_"):
            continue
        value = action[key]
        if key.endswith("_px") and isinstance(value, int) and not isinstance(value, bool):
            reduced[key] = value // click_cell_px
        elif isinstance(value, (list, tuple)):
            reduced[key] = list(value)
        else:
            reduced[key] = value
    return json.dumps(reduced, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


@dataclass(frozen=True)
class ContinuationStats:
    trajectory_length: int
    root_cause_action_index: int
    post_error_steps: int
    terminal_action_kind: str
    terminated_explicitly: bool
    terminate_status: Optional[str]
    early_stop: bool
    loop_detected: bool
    loop_signature: Optional[str]
    loop_repeat_count: int
    loop_start_offset: Optional[int]
    loop_established_offset: Optional[int]
    distinct_signature_ratio: float
    loop_heuristic_version: str = LOOP_HEURISTIC_VERSION

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def compute_continuation(
    actions: Sequence[Mapping[str, Any]],
    root_cause_action_index: int,
    *,
    loop: LoopConfig = LoopConfig(),
    early_stop_window: int = 5,
) -> ContinuationStats:
    length = len(actions)
    if root_cause_action_index < 0 or root_cause_action_index >= length:
        raise ValueError(
            "root cause index %d outside trajectory of %d actions"
            % (root_cause_action_index, length)
        )
    suffix = actions[root_cause_action_index + 1 :]
    post_error_steps = len(suffix)
    terminal = actions[-1]
    terminal_kind = str(terminal.get("kind", ""))
    terminated = terminal_kind == "terminate"

    signatures = [
        action_signature(action, click_cell_px=loop.click_cell_px)
        if action.get("kind") not in loop.exclude_kinds
        else None
        for action in suffix
    ]
    counted = Counter(signature for signature in signatures if signature is not None)
    loop_signature = None
    repeat_count = 0
    start_offset = None
    established_offset = None
    if counted:
        best = max(counted.values())
        loop_signature = next(sig for sig in signatures if sig is not None and counted[sig] == best)
        repeat_count = best
        occurrences = [
            offset for offset, sig in enumerate(signatures, start=1) if sig == loop_signature
        ]
        start_offset = occurrences[0]
        if best >= loop.min_repeats:
            established_offset = occurrences[loop.min_repeats - 1]
    loop_detected = repeat_count >= loop.min_repeats
    considered = [sig for sig in signatures if sig is not None]
    ratio = (len(set(considered)) / float(len(considered))) if considered else 1.0
    return ContinuationStats(
        trajectory_length=length,
        root_cause_action_index=root_cause_action_index,
        post_error_steps=post_error_steps,
        terminal_action_kind=terminal_kind,
        terminated_explicitly=terminated,
        terminate_status=str(terminal.get("status")) if terminated else None,
        early_stop=post_error_steps <= early_stop_window,
        loop_detected=loop_detected,
        loop_signature=loop_signature if loop_detected else None,
        loop_repeat_count=repeat_count,
        loop_start_offset=start_offset if loop_detected else None,
        loop_established_offset=established_offset,
        distinct_signature_ratio=ratio,
    )
