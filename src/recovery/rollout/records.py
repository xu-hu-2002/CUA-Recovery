"""Minimal traceable records of raw rollouts."""

from __future__ import annotations

from dataclasses import dataclass


SOURCE_BENCHMARKS = frozenset({"mypcbench", "rerail_workflows"})


@dataclass(frozen=True)
class RolloutRecord:
    rollout_id: str
    source_benchmark: str
    task_id: str
    agent_id: str
    snapshot_sha256: str
    trajectory_uri: str
    trajectory_sha256: str
    task_success: bool
    step_count: int

    def __post_init__(self) -> None:
        if self.source_benchmark not in SOURCE_BENCHMARKS:
            raise ValueError("source_benchmark 必须是 %s 之一" % sorted(SOURCE_BENCHMARKS))
        if self.step_count < 0:
            raise ValueError("step_count 不能为负数")
        required = (
            self.rollout_id,
            self.source_benchmark,
            self.task_id,
            self.agent_id,
            self.snapshot_sha256,
            self.trajectory_uri,
            self.trajectory_sha256,
        )
        if any(not value for value in required):
            raise ValueError("RolloutRecord 的 provenance 字段不能为空")
