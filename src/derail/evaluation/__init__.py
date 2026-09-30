"""DERAIL evaluation metrics 与结果数据结构。"""

from .metrics import AggregateMetrics, EpisodeResult, PassAtK, aggregate_metrics, pass_at_k, rubric_verdict
from .records import (
    PESR_SUCCESS_THRESHOLD,
    POST_TAKEOVER_ACTION_BUDGET,
    ErrorAwarenessJudgment,
    EvaluationEpisode,
    EvaluationRecordError,
    PostErrorOutcome,
)
from .takeover import TakeoverRunError, TakeoverRunTrace, TakeoverTurn, run_takeover_episode

__all__ = [
    "AggregateMetrics",
    "EpisodeResult",
    "EvaluationEpisode",
    "EvaluationRecordError",
    "ErrorAwarenessJudgment",
    "PassAtK",
    "PostErrorOutcome",
    "PESR_SUCCESS_THRESHOLD",
    "POST_TAKEOVER_ACTION_BUDGET",
    "TakeoverRunError",
    "TakeoverRunTrace",
    "TakeoverTurn",
    "aggregate_metrics",
    "pass_at_k",
    "rubric_verdict",
    "run_takeover_episode",
]
