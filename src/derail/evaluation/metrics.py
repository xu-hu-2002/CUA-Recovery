"""基础 episode-level 指标。

``aggregate_metrics`` 是 v1.2 episode 记录的 micro average；论文口径的 ρ / V /
Pass@k 见文件下半部分（``rubric_verdict``、``pass_at_k``），改动必须同步论文与 tests。
"""

from __future__ import annotations

from dataclasses import dataclass
from statistics import mean
from typing import Any, Iterable, Mapping, Optional, Sequence, Tuple

from derail.derived.layout import DEPTH_GRID


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
    """对 valid episodes 做 micro average。

    无效 episode 不进入分母，但调用者必须在论文中另行报告 invalid rate；这里不允许
    把空集合悄悄当成 0 分。
    """

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


# ---------------------------------------------------------------------------
# Rubric Score、V 与 Pass@k（论文 02_formulation「Rollouts and Evaluation」）
# ---------------------------------------------------------------------------


def rubric_verdict(rubric_results: Sequence[Mapping[str, Any]]) -> Tuple[float, bool]:
    """(ρ, V)：ρ = Σ w_j c_j / Σ w_j；V = 1 当且仅当每条 criterion 都满足。

    V 按逐条 criterion 重算，不用上游 judge 的 ``round(ρ*100) >= 100``（取整会让
    极小权重的失败条目被吞掉）。
    """

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
    unit_count: int          # 进入分母的任务 / state 数
    missing_count: int       # 没有任何有效判分的单元（n/a、infra 失败、缺 run）
    solved_count: int
    pass_at_k: float
    rubric_score: float      # 各单元 "那次 rollout" 的 ρ 的均值
    incomplete_count: int = 0  # 有效 run 少于 repeats 的单元（缺的 run 按失败计）


def pass_at_k(
    runs: Mapping[str, Sequence[Optional[Tuple[float, bool]]]],
    units: Iterable[str],
    repeats: int,
    missing_counts_as_failure: bool = True,
) -> PassAtK:
    """每个单元（任务或 state）最多 ``repeats`` 次 rollout，任一次 V=1 即 solved。

    ``runs[unit]`` 是各次 rollout 的 (ρ, V)，判分缺失的那次记 None。Rubric Score 取
    "那次 rollout" 的 ρ：有成功的 run 取成功那次（按定义 ρ=1）；全部失败时取最高 ρ。
    分母是 ``units`` 全体；没有任何有效判分的单元在 ``missing_counts_as_failure``
    时按失败（ρ=0）留在分母，否则排除并在 ``missing_count`` 里单独报数。
    """

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
