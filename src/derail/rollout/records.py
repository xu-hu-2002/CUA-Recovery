"""原始 rollout 的最小可追溯记录。"""

from __future__ import annotations

from dataclasses import dataclass


# 与 configs/collection/sources.yaml 的 source_benchmark 一致。
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
        # 显式记录 source，避免后续导入或扩展数据时依赖路径猜测 provenance。
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
