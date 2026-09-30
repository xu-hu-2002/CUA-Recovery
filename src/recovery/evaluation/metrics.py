from __future__ import annotations

from dataclasses import dataclass
from statistics import mean
from typing import Any, Iterable, Mapping, Optional, Sequence, Tuple

from recovery.derived.layout import DEPTH_GRID


@dataclass(frozen=True)
class EpisodeResult:
    case_id: str
    agent_id: str
    depth: int
    error_aware: bool
    recovered: bool
    valid: bool = True

    def __post_init__(self) -> None:
        if self.depth not in DEPTH_GRID:
            raise ValueError("depth 必须属于 %s" % (DEPTH_GRID,))


@dataclass(frozen=True)
class AggregateMetrics:
    episode_count: int
    error_awareness_rate: float
    post_error_success_rate: float
    total_episode_count: int = 0
    invalid_episode_count: int = 0
    invalid_rate: float = 0.0


def aggregate_metrics(results: Iterable[EpisodeResult]) -> AggregateMetrics:
    all_results: Tuple[EpisodeResult, ...] = tuple(results)
    valid: Tuple[EpisodeResult, ...] = tuple(result for result in all_results if result.valid)
    if not valid:
        raise ValueError("没有 valid episodes，不能计算指标")
    return AggregateMetrics(
        episode_count=len(valid),
        error_awareness_rate=mean(float(result.error_aware) for result in valid),
        post_error_success_rate=mean(float(result.recovered) for result in valid),
        total_episode_count=len(all_results),
        invalid_episode_count=len(all_results) - len(valid),
        invalid_rate=(len(all_results) - len(valid)) / len(all_results),
    )


def rubric_verdict(rubric_results: Sequence[Mapping[str, Any]]) -> Tuple[float, bool]:
    if not rubric_results:
        raise ValueError("没有 rubric 结果，不能计算 ρ 与 V")
    weights = [float(r["weight"]) for r in rubric_results]
    if any(w <= 0 for w in weights):
        raise ValueError("rubric 权重必须为正")
    satisfied = [bool(r["success"]) for r in rubric_results]
    rho = sum(w for w, c in zip(weights, satisfied) if c) / sum(weights)
    return rho, all(satisfied)


@dataclass(frozen=True)
class PassAtK:
    repeats: int
    unit_count: int
    missing_count: int
    solved_count: int
    pass_at_k: float
    rubric_score: float
    incomplete_count: int = 0


def pass_at_k(
    runs: Mapping[str, Sequence[Optional[Tuple[float, bool]]]],
    units: Iterable[str],
    repeats: int,
    missing_counts_as_failure: bool = True,
) -> PassAtK:
    if repeats < 1:
        raise ValueError("repeats 必须 >= 1")
    units = tuple(dict.fromkeys(units))
    unknown = set(runs) - set(units)
    if unknown:
        raise ValueError("runs 含分母以外的单元：%s" % sorted(unknown)[:3])
    solved = missing = incomplete = 0
    scores = []
    for unit in units:
        attempts = list(runs.get(unit, ()))
        if len(attempts) > repeats:
            raise ValueError("%s 有 %d 次 run，超过 repeats=%d" % (unit, len(attempts), repeats))
        valid = [a for a in attempts if a is not None]
        incomplete += int(len(valid) < repeats)
        if not valid:
            missing += 1
            if missing_counts_as_failure:
                scores.append(0.0)
            continue
        success = [rho for rho, passed in valid if passed]
        solved += int(bool(success))
        scores.append(success[0] if success else max(rho for rho, _ in valid))
    if not scores:
        raise ValueError("分母为空，不能计算 Pass@k")
    return PassAtK(
        repeats=repeats,
        unit_count=len(scores),
        missing_count=missing,
        solved_count=solved,
        pass_at_k=solved / len(scores),
        rubric_score=mean(scores),
        incomplete_count=incomplete,
    )
