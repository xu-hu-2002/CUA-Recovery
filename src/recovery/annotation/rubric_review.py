"""Human review of the per-rubric verdicts produced by the LLM judge."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple


class RubricReviewError(ValueError):
    """A rubric review or its adjudication violates the benchmark protocol."""


class RubricVerdict(str, Enum):
    SUCCESS = "success"
    FAILURE = "failure"


@dataclass(frozen=True)
class RubricSpec:
    """One rubric as the judge sees it: id, criterion text, weight."""

    rubric_id: str
    criterion: str
    weight: float


def load_rubric_specs(bundle: Mapping[str, Any]) -> Tuple[RubricSpec, ...]:
    grading = bundle.get("grading_manifest")
    rubrics = grading.get("rubrics") if isinstance(grading, Mapping) else None
    if not isinstance(rubrics, (list, tuple)) or not rubrics:
        raise RubricReviewError("rubric bundle 缺少 grading_manifest.rubrics")
    specs: list[RubricSpec] = []
    for index, raw in enumerate(rubrics):
        if not isinstance(raw, Mapping):
            raise RubricReviewError("rubric 条目必须是 object")
        rubric_id = str(raw.get("id") or raw.get("rubric_id") or "R%d" % (index + 1))
        criterion = str(raw.get("criterion") or raw.get("requirement") or "").strip()
        if not criterion:
            raise RubricReviewError("rubric %s 缺少 criterion" % rubric_id)
        try:
            weight = float(raw.get("weight")) if raw.get("weight") is not None else 0.0
        except (TypeError, ValueError):
            weight = 0.0
        specs.append(
            RubricSpec(rubric_id=rubric_id, criterion=criterion, weight=weight if weight > 0 else 1.0)
        )
    ids = [item.rubric_id for item in specs]
    if len(set(ids)) != len(ids):
        raise RubricReviewError("rubric bundle 含重复 rubric id")
    return tuple(specs)


@dataclass(frozen=True)
class RubricJudgment:
    """One person's verdict on one rubric."""

    rubric_id: str
    verdict: RubricVerdict
    rationale: str
    evidence_refs: Tuple[str, ...]

    def __post_init__(self) -> None:
        if not self.rubric_id.strip():
            raise RubricReviewError("rubric_id 不能为空")
        if not self.rationale.strip():
            raise RubricReviewError("rubric %s 缺少 rationale" % self.rubric_id)
        if not self.evidence_refs or any(not item.strip() for item in self.evidence_refs):
            raise RubricReviewError("rubric %s 必须提供 evidence_refs" % self.rubric_id)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "rubric_id": self.rubric_id,
            "verdict": self.verdict.value,
            "rationale": self.rationale,
            "evidence_refs": list(self.evidence_refs),
        }

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "RubricJudgment":
        return cls(
            rubric_id=str(raw["rubric_id"]),
            verdict=RubricVerdict(str(raw["verdict"])),
            rationale=str(raw["rationale"]),
            evidence_refs=tuple(str(item) for item in raw["evidence_refs"]),
        )


def _validate_coverage(
    judgments: Sequence[RubricJudgment], specs: Sequence[RubricSpec], what: str
) -> None:
    seen = [item.rubric_id for item in judgments]
    if len(set(seen)) != len(seen):
        raise RubricReviewError("%s 含重复 rubric_id" % what)
    expected = {item.rubric_id for item in specs}
    missing = sorted(expected - set(seen))
    extra = sorted(set(seen) - expected)
    if missing:
        raise RubricReviewError("%s 未覆盖 rubric: %s" % (what, missing))
    if extra:
        raise RubricReviewError("%s 含 bundle 之外的 rubric: %s" % (what, extra))


@dataclass(frozen=True)
class HumanRubricReview:
    """One reviewer's independent verdicts over every rubric of one trajectory."""

    review_id: str
    trajectory_id: str
    source_trajectory_sha256: str
    rubric_bundle_sha256: str
    reviewer_id: str
    judgments: Tuple[RubricJudgment, ...]
    schema_version: str = "0.1.0"

    def __post_init__(self) -> None:
        required = (
            self.review_id,
            self.trajectory_id,
            self.source_trajectory_sha256,
            self.rubric_bundle_sha256,
            self.reviewer_id,
        )
        if any(not value.strip() for value in required):
            raise RubricReviewError("rubric review provenance 字段不能为空")
        if not self.judgments:
            raise RubricReviewError("rubric review 不能为空")
        seen = [item.rubric_id for item in self.judgments]
        if len(set(seen)) != len(seen):
            raise RubricReviewError("rubric review 含重复 rubric_id")

    def validate_against_bundle(self, specs: Sequence[RubricSpec]) -> None:
        _validate_coverage(self.judgments, specs, "rubric review %s" % self.review_id)

    def verdicts(self) -> Dict[str, RubricVerdict]:
        return {item.rubric_id: item.verdict for item in self.judgments}

    def to_dict(self) -> Dict[str, Any]:
        return {
            "review_id": self.review_id,
            "trajectory_id": self.trajectory_id,
            "source_trajectory_sha256": self.source_trajectory_sha256,
            "rubric_bundle_sha256": self.rubric_bundle_sha256,
            "reviewer_id": self.reviewer_id,
            "reviewer_role": "human",
            "judgments": [item.to_dict() for item in self.judgments],
            "schema_version": self.schema_version,
        }

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "HumanRubricReview":
        return cls(
            review_id=str(raw["review_id"]),
            trajectory_id=str(raw["trajectory_id"]),
            source_trajectory_sha256=str(raw["source_trajectory_sha256"]),
            rubric_bundle_sha256=str(raw["rubric_bundle_sha256"]),
            reviewer_id=str(raw["reviewer_id"]),
            judgments=tuple(RubricJudgment.from_dict(item) for item in raw["judgments"]),
            schema_version=str(raw.get("schema_version", "0.1.0")),
        )


@dataclass(frozen=True)
class RubricReviewAdjudication:
    """Third-party resolution of two independent rubric reviews."""

    adjudication_id: str
    trajectory_id: str
    source_trajectory_sha256: str
    rubric_bundle_sha256: str
    input_review_ids: Tuple[str, ...]
    adjudicator_id: str
    judgments: Tuple[RubricJudgment, ...]
    resolution_rationale: str
    disagreement_rubric_ids: Tuple[str, ...] = ()
    judge_model: Optional[str] = None
    judge_verdicts: Optional[Mapping[str, RubricVerdict]] = None
    schema_version: str = "0.1.0"

    def __post_init__(self) -> None:
        required = (
            self.adjudication_id,
            self.trajectory_id,
            self.source_trajectory_sha256,
            self.rubric_bundle_sha256,
            self.adjudicator_id,
        )
        if any(not value.strip() for value in required):
            raise RubricReviewError("rubric adjudication provenance 字段不能为空")
        if len(self.input_review_ids) < 2:
            raise RubricReviewError("rubric adjudication 至少引用两份独立 review")
        if len(set(self.input_review_ids)) != len(self.input_review_ids):
            raise RubricReviewError("input_review_ids 不允许重复")
        if not self.judgments:
            raise RubricReviewError("rubric adjudication 不能为空")
        if not self.resolution_rationale.strip():
            raise RubricReviewError("rubric adjudication 必须保留 resolution rationale")
        if len(set(self.disagreement_rubric_ids)) != len(self.disagreement_rubric_ids):
            raise RubricReviewError("disagreement_rubric_ids 不允许重复")
        if self.judge_verdicts is not None and not self.judge_model:
            raise RubricReviewError("给出 judge_verdicts 时必须记录 judge_model")

    def validate_inputs(self, reviews: Sequence[HumanRubricReview]) -> None:
        indexed = {item.review_id: item for item in reviews}
        missing = sorted(set(self.input_review_ids) - set(indexed))
        if missing:
            raise RubricReviewError("rubric adjudication 引用不存在的 review: %s" % missing)
        selected = [indexed[item] for item in self.input_review_ids]
        if len({item.reviewer_id for item in selected}) < 2:
            raise RubricReviewError("rubric adjudication 输入必须来自至少两名 reviewer")
        if any(item.trajectory_id != self.trajectory_id for item in selected):
            raise RubricReviewError("rubric adjudication 输入 trajectory_id 不一致")
        if any(item.rubric_bundle_sha256 != self.rubric_bundle_sha256 for item in selected):
            raise RubricReviewError("rubric adjudication 输入 rubric bundle hash 不一致")
        if any(
            item.source_trajectory_sha256 != self.source_trajectory_sha256 for item in selected
        ):
            raise RubricReviewError("rubric adjudication 输入 trajectory hash 不一致")
        actual = disagreement_rubric_ids(selected)
        if tuple(sorted(self.disagreement_rubric_ids)) != actual:
            raise RubricReviewError(
                "disagreement_rubric_ids 与输入 review 的实际分歧不一致: 记录 %s，实际 %s"
                % (tuple(sorted(self.disagreement_rubric_ids)), actual)
            )

    def validate_against_bundle(self, specs: Sequence[RubricSpec]) -> None:
        _validate_coverage(
            self.judgments, specs, "rubric adjudication %s" % self.adjudication_id
        )
        if self.judge_verdicts is not None:
            expected = {item.rubric_id for item in specs}
            if set(self.judge_verdicts) != expected:
                raise RubricReviewError("judge_verdicts 必须覆盖且只覆盖 bundle 的 rubric")

    def verdicts(self) -> Dict[str, RubricVerdict]:
        return {item.rubric_id: item.verdict for item in self.judgments}

    def weighted_score(self, specs: Sequence[RubricSpec]) -> Dict[str, Any]:
        self.validate_against_bundle(specs)
        verdicts = self.verdicts()
        total_weight = sum(item.weight for item in specs) or 1.0
        passed_weight = sum(
            item.weight for item in specs if verdicts[item.rubric_id] is RubricVerdict.SUCCESS
        )
        fraction = max(0.0, min(1.0, passed_weight / total_weight))
        score = int(round(fraction * 100))
        return {
            "weighted_fraction": fraction,
            "score": score,
            "perfect": score >= 100,
            "passed_rubric_count": sum(
                1 for item in specs if verdicts[item.rubric_id] is RubricVerdict.SUCCESS
            ),
            "rubric_count": len(specs),
        }

    def judge_disagreement_rubric_ids(self) -> Tuple[str, ...]:
        if self.judge_verdicts is None:
            return ()
        final = self.verdicts()
        return tuple(
            sorted(
                rubric_id
                for rubric_id, verdict in self.judge_verdicts.items()
                if final.get(rubric_id) is not verdict
            )
        )

    def to_dict(self) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "adjudication_id": self.adjudication_id,
            "trajectory_id": self.trajectory_id,
            "source_trajectory_sha256": self.source_trajectory_sha256,
            "rubric_bundle_sha256": self.rubric_bundle_sha256,
            "input_review_ids": list(self.input_review_ids),
            "adjudicator_id": self.adjudicator_id,
            "judgments": [item.to_dict() for item in self.judgments],
            "disagreement_rubric_ids": list(self.disagreement_rubric_ids),
            "resolution_rationale": self.resolution_rationale,
            "judge_model": self.judge_model,
            "judge_verdicts": (
                {key: value.value for key, value in sorted(self.judge_verdicts.items())}
                if self.judge_verdicts is not None
                else None
            ),
            "schema_version": self.schema_version,
        }
        payload["judge_disagreement_rubric_ids"] = list(self.judge_disagreement_rubric_ids())
        return payload

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "RubricReviewAdjudication":
        judge_verdicts = raw.get("judge_verdicts")
        return cls(
            adjudication_id=str(raw["adjudication_id"]),
            trajectory_id=str(raw["trajectory_id"]),
            source_trajectory_sha256=str(raw["source_trajectory_sha256"]),
            rubric_bundle_sha256=str(raw["rubric_bundle_sha256"]),
            input_review_ids=tuple(str(item) for item in raw["input_review_ids"]),
            adjudicator_id=str(raw["adjudicator_id"]),
            judgments=tuple(RubricJudgment.from_dict(item) for item in raw["judgments"]),
            resolution_rationale=str(raw["resolution_rationale"]),
            disagreement_rubric_ids=tuple(
                str(item) for item in raw.get("disagreement_rubric_ids", ())
            ),
            judge_model=(str(raw["judge_model"]) if raw.get("judge_model") else None),
            judge_verdicts=(
                {str(key): RubricVerdict(str(value)) for key, value in judge_verdicts.items()}
                if isinstance(judge_verdicts, Mapping)
                else None
            ),
            schema_version=str(raw.get("schema_version", "0.1.0")),
        )


def disagreement_rubric_ids(reviews: Sequence[HumanRubricReview]) -> Tuple[str, ...]:
    if len(reviews) < 2:
        raise RubricReviewError("计算分歧至少需要两份 review")
    tables = [item.verdicts() for item in reviews]
    covered = set(tables[0])
    for table in tables[1:]:
        if set(table) != covered:
            raise RubricReviewError("review 覆盖的 rubric 集合不一致，无法比较")
    return tuple(
        sorted(
            rubric_id
            for rubric_id in covered
            if len({table[rubric_id] for table in tables}) > 1
        )
    )


def summarize_rubric_agreement(
    review_groups: Sequence[Sequence[HumanRubricReview]],
    adjudications: Sequence[RubricReviewAdjudication] = (),
) -> Dict[str, Any]:
    item_total = 0
    item_agreed = 0
    trajectory_total = 0
    trajectory_fully_agreed = 0
    for reviews in review_groups:
        if len(reviews) < 2:
            raise RubricReviewError("每条 trajectory 至少需要两份 review 才能算一致性")
        disagreed = disagreement_rubric_ids(reviews)
        covered = len(reviews[0].judgments)
        item_total += covered
        item_agreed += covered - len(disagreed)
        trajectory_total += 1
        trajectory_fully_agreed += 1 if not disagreed else 0

    judge_item_total = 0
    judge_item_agreed = 0
    judge_false_failure = 0
    judge_false_success = 0
    for adjudication in adjudications:
        if adjudication.judge_verdicts is None:
            continue
        final = adjudication.verdicts()
        for rubric_id, judge_verdict in adjudication.judge_verdicts.items():
            judge_item_total += 1
            human_verdict = final.get(rubric_id)
            if human_verdict is judge_verdict:
                judge_item_agreed += 1
            elif judge_verdict is RubricVerdict.FAILURE:
                judge_false_failure += 1
            else:
                judge_false_success += 1

    return {
        "trajectory_count": trajectory_total,
        "rubric_item_count": item_total,
        "inter_reviewer_item_agreement": (item_agreed / item_total) if item_total else None,
        "inter_reviewer_trajectory_agreement": (
            (trajectory_fully_agreed / trajectory_total) if trajectory_total else None
        ),
        "judge_compared_item_count": judge_item_total,
        "judge_human_item_agreement": (
            (judge_item_agreed / judge_item_total) if judge_item_total else None
        ),
        "judge_marked_failure_human_success": judge_false_failure,
        "judge_marked_success_human_failure": judge_false_success,
    }
