"""Retrospective analysis of the existing failures (execution doc v1.2 section 9.6).

The 680 labelled failures were collected without a change log or a page log; their
``traj.jsonl`` files keep the PyAutoGUI action, the agent's response and a screenshot per
step.  That is enough for the *parameter* detector (typed values vs gold slots) and for the
first automatic-vs-human agreement numbers (E4).  The omission and state detectors need
ledgers the old runs do not have and are not applied.

``trace_from_traj`` converts one ``traj.jsonl`` into a minimal ``rollout-trace/1.0`` (actions,
parameters, thoughts; empty delta / observations); ``retrospective_row`` runs the parameter
detector for one failure and compares with the human root cause (exact, within one action).
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Union

import yaml

from derail.failure_analysis.detectors import parameter_candidates, pick_root_cause
from derail.harness.trace_builder import ActionRecord, TraceBuilder

RUNTIME_CONFIG = Path(__file__).resolve().parents[3] / "configs/collection/mypcbench_runtime.yaml"


def clean_start_step_budget(path: Path = RUNTIME_CONFIG) -> int:
    """Clean-start step budget (paper 05:15), ``max_steps`` of the collection runtime config."""

    return int(yaml.safe_load(path.read_text(encoding="utf-8"))["max_steps"])


def load_traj(path: Union[str, Path]) -> List[Dict[str, Any]]:
    rows = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def trace_from_traj(
    rows: Sequence[Mapping[str, Any]],
    *,
    task_id: str,
    agent: str,
    rollout_id: str,
    world_id: str = "legacy",
    instruction: str = "",
    step_budget: Optional[int] = None,
) -> Dict[str, Any]:
    """Minimal rollout-trace/1.0 from a legacy ``traj.jsonl`` (0-based action indices)."""

    builder = TraceBuilder(
        rollout_id=rollout_id,
        task_id=task_id,
        world_id=world_id,
        agent=agent,
        seed=0,
        step_budget=step_budget if step_budget is not None else clean_start_step_budget(),
    )
    builder.start({})
    declared_complete = declared_infeasible = False
    for index, row in enumerate(rows):
        action = row.get("action")
        text = action if isinstance(action, str) else json.dumps(action, ensure_ascii=False)
        upper = text.strip().upper()
        modality = "cli" if isinstance(action, Mapping) and action.get("command") else "gui"
        if upper == "DONE":
            declared_complete = True
        if upper == "FAIL":
            declared_infeasible = True
        raw = action if isinstance(action, str) else str((action or {}).get("command") or text)
        builder.record_step(
            index,
            ActionRecord(raw=raw, modality=modality, thought=str(row.get("response") or "")[:2000]),
            delta=[],
            trace_records=[],
            screenshot=hashlib.sha256(str(row.get("screenshot_file", "")).encode()).digest()
            if row.get("screenshot_file")
            else None,
            a11y_text=None,
        )
    trace = builder.finish(
        declared_complete=declared_complete,
        declared_infeasible=declared_infeasible,
        budget_exhausted=not (declared_complete or declared_infeasible),
        provenance={"legacy": True, "instruction": instruction},
    )
    trace["instruction"] = instruction
    return trace


def retrospective_row(
    trace: Mapping[str, Any],
    gold: Mapping[str, Any],
    human_root: Optional[int],
    human_type: Optional[str],
    residual_threshold: float = 0.7,
) -> Dict[str, Any]:
    candidates = parameter_candidates(trace, gold)
    root = pick_root_cause(candidates, residual_threshold)
    auto_root = root.action_index if root else None
    return {
        "rollout_id": str(trace["rollout_id"]),
        "task_id": str(trace["task_id"]),
        "agent": str(trace["agent"]),
        "human_root": human_root,
        "human_type": human_type,
        "auto_root": auto_root,
        "auto_pattern": root.evidence_pattern if root else None,
        "candidate_count": len(candidates),
        "exact": auto_root is not None and human_root is not None and auto_root == human_root,
        "within_one": auto_root is not None
        and human_root is not None
        and abs(auto_root - human_root) <= 1,
        "covered": auto_root is not None,
    }


def agreement_summary(rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    total = len(rows)
    covered = [r for r in rows if r["covered"]]
    return {
        "failures": total,
        "covered": len(covered),
        "coverage": round(len(covered) / total, 4) if total else 0.0,
        "exact_of_covered": round(sum(r["exact"] for r in covered) / len(covered), 4)
        if covered
        else 0.0,
        "within_one_of_covered": round(sum(r["within_one"] for r in covered) / len(covered), 4)
        if covered
        else 0.0,
    }
