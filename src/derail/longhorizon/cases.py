"""Depth instantiation, reversibility stratum, dedup, funnel and yields (manual v0.2 14A.4-6).

This module is data-level: it decides *which* ``(failure, d)`` pairs become case candidates and
how they are counted.  Executing prefix repair and replay stays in ``derail.construction`` and
``derail.replay``; their identifiers are passed in through ``provenance``.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass, field, replace
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from derail.longhorizon.continuation import action_signature
from derail.longhorizon.ontology import Ontology

CASE_SCHEMA_VERSION = "derail-case/0.1"
DEPTH_AVAILABLE = "available"
INSUFFICIENT_POST_ERROR_STEPS = "INSUFFICIENT_POST_ERROR_STEPS"
DUPLICATE_CASE = "DUPLICATE_CASE"
EFFECTIVELY_COMPLETE = "EFFECTIVELY_COMPLETE"
UNRECOVERABLE_BY_CONSTRUCTION = "UNRECOVERABLE_BY_CONSTRUCTION"
CASE_STATUS_CANDIDATE = "candidate"
STRATUM_REVERSIBLE = "reversible"
STRATUM_IRREVERSIBLE = "irreversible"
VERIFICATION_PATHS = ("pending", "auto", "human", "human_and_auto")
# Higher index wins during dedup ("stronger verification path").
VERIFICATION_STRENGTH = {path: rank for rank, path in enumerate(VERIFICATION_PATHS)}


def depth_availability(
    post_error_steps_available: int, depth_grid: Sequence[int]
) -> Dict[int, str]:
    """Map each depth to ``available`` or ``INSUFFICIENT_POST_ERROR_STEPS``.

    A depth is instantiable only when at least ``d`` executed actions follow the root cause; a
    truncated suffix is never padded.
    """

    if post_error_steps_available < 0:
        raise ValueError("post_error_steps_available cannot be negative")
    return {
        int(depth): DEPTH_AVAILABLE
        if post_error_steps_available >= depth
        else INSUFFICIENT_POST_ERROR_STEPS
        for depth in depth_grid
    }


def reversibility_stratum(
    effects_within_prefix: Iterable[Mapping[str, Any]], ontology: Ontology
) -> Tuple[str, bool]:
    """Return ``(stratum, high_consequence)`` from the effects executed up to root + d.

    The stratum is irreversible iff at least one strictly irreversible (R3) effect executed.
    High-consequence effects (R2) set the flag but never move a case into the irreversible
    stratum.
    """

    classes = {str(effect.get("reversibility_class", "")) for effect in effects_within_prefix}
    irreversible = bool(classes & ontology.irreversible_classes)
    high = bool(classes & (ontology.high_consequence_classes - ontology.irreversible_classes))
    return (STRATUM_IRREVERSIBLE if irreversible else STRATUM_REVERSIBLE), high


def stratum_from_human_label(label: str) -> str:
    """Map the human ``reversibility`` label of a source-corpus failure onto a stratum."""

    if label not in (STRATUM_REVERSIBLE, STRATUM_IRREVERSIBLE):
        raise ValueError("unknown reversibility label %r" % label)
    return label


@dataclass(frozen=True)
class CaseCandidate:
    """One ``(failure, depth)`` pair before or after verification."""

    case_id: str
    task_id: str
    source_rollout_id: str
    source_agent: str
    error_depth: int
    root_cause_action_index: int
    paper_type: str
    paper_category: str
    group: str
    reversibility_stratum: str
    post_error_steps_available: int
    root_action_signature: str
    verification_path: str = "pending"
    high_consequence: bool = False
    root_cause_reversibility_class: Optional[str] = None
    downstream_commit_class: Optional[str] = None
    task_horizon_bucket: Optional[str] = None
    takeover_step_budget: Optional[int] = None
    status: str = CASE_STATUS_CANDIDATE
    merged_from: Tuple[str, ...] = ()
    provenance: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.verification_path not in VERIFICATION_PATHS:
            raise ValueError("unknown verification_path %r" % self.verification_path)
        if self.reversibility_stratum not in (STRATUM_REVERSIBLE, STRATUM_IRREVERSIBLE):
            raise ValueError("unknown reversibility stratum %r" % self.reversibility_stratum)
        if self.post_error_steps_available < self.error_depth:
            raise ValueError("post_error_steps_available must be >= error_depth")
        if self.error_depth < 0 or self.root_cause_action_index < 0:
            raise ValueError("depth and root index cannot be negative")

    def to_record(self) -> Dict[str, Any]:
        return {
            "schema_version": CASE_SCHEMA_VERSION,
            "case_id": self.case_id,
            "task_id": self.task_id,
            "source_rollout_id": self.source_rollout_id,
            "source_agent": self.source_agent,
            "verification_path": self.verification_path,
            "error_depth": self.error_depth,
            "root_cause_action_index": self.root_cause_action_index,
            "paper_type": self.paper_type,
            "paper_category": self.paper_category,
            "group": self.group,
            "reversibility_stratum": self.reversibility_stratum,
            "high_consequence": self.high_consequence,
            "root_cause_reversibility_class": self.root_cause_reversibility_class,
            "downstream_commit_class": self.downstream_commit_class,
            "post_error_steps_available": self.post_error_steps_available,
            "root_action_signature": self.root_action_signature,
            "task_horizon_bucket": self.task_horizon_bucket,
            "takeover_step_budget": self.takeover_step_budget,
            "status": self.status,
            "merged_from": list(self.merged_from),
            "provenance": dict(self.provenance),
        }


@dataclass(frozen=True)
class FailureSource:
    """The failure-level facts shared by every depth instance of one rollout."""

    task_id: str
    source_rollout_id: str
    source_agent: str
    root_cause_action_index: int
    root_action: Mapping[str, Any]
    paper_type: str
    paper_category: str
    group: str
    reversibility_stratum: str
    post_error_steps_available: int
    high_consequence: bool = False
    root_cause_reversibility_class: Optional[str] = None
    downstream_commit_class: Optional[str] = None
    task_horizon_bucket: Optional[str] = None
    provenance: Mapping[str, Any] = field(default_factory=dict)


def instantiate_depths(
    source: FailureSource,
    depth_grid: Sequence[int],
    *,
    click_cell_px: int = 40,
    case_id_prefix: str = "derail",
) -> Tuple[List[CaseCandidate], Dict[int, str]]:
    """Create one candidate per available depth; return skipped depths with their code."""

    availability = depth_availability(source.post_error_steps_available, depth_grid)
    signature = action_signature(source.root_action, click_cell_px=click_cell_px)
    candidates = []
    skipped = {}
    for depth, status in availability.items():
        if status != DEPTH_AVAILABLE:
            skipped[depth] = status
            continue
        candidates.append(
            CaseCandidate(
                case_id="%s_%s_%s_d%d"
                % (case_id_prefix, source.source_rollout_id, source.paper_type, depth),
                task_id=source.task_id,
                source_rollout_id=source.source_rollout_id,
                source_agent=source.source_agent,
                error_depth=depth,
                root_cause_action_index=source.root_cause_action_index,
                paper_type=source.paper_type,
                paper_category=source.paper_category,
                group=source.group,
                reversibility_stratum=source.reversibility_stratum,
                post_error_steps_available=source.post_error_steps_available,
                root_action_signature=signature,
                high_consequence=source.high_consequence,
                root_cause_reversibility_class=source.root_cause_reversibility_class,
                downstream_commit_class=source.downstream_commit_class,
                task_horizon_bucket=source.task_horizon_bucket,
                provenance=dict(source.provenance),
            )
        )
    return candidates, skipped


def _keeper_key(candidate: CaseCandidate) -> Tuple[int, int, str]:
    # Stronger verification first, then more post-error steps, then a stable id tie-break.
    return (
        VERIFICATION_STRENGTH[candidate.verification_path],
        candidate.post_error_steps_available,
        candidate.case_id,
    )


def dedup_cases(
    candidates: Sequence[CaseCandidate], *, root_tolerance: int = 1
) -> Tuple[List[CaseCandidate], List[CaseCandidate]]:
    """Merge near-duplicates and return ``(kept, removed)``.

    Two candidates are near-duplicates when they share ``task_id``, ``error_depth``,
    ``paper_type`` and root-action signature and their root indices differ by at most
    ``root_tolerance``.  Closeness is chained (roots 10, 11, 12 form one cluster).  The keeper
    is the strongest-verified, then longest-suffix candidate; every merged id is kept in
    ``merged_from`` and the losers are marked ``DUPLICATE_CASE``.
    """

    groups: Dict[Tuple[str, int, str, str], List[CaseCandidate]] = defaultdict(list)
    for candidate in candidates:
        key = (
            candidate.task_id,
            candidate.error_depth,
            candidate.paper_type,
            candidate.root_action_signature,
        )
        groups[key].append(candidate)
    kept: List[CaseCandidate] = []
    removed: List[CaseCandidate] = []
    for key in sorted(groups):
        members = sorted(groups[key], key=lambda item: (item.root_cause_action_index, item.case_id))
        cluster: List[CaseCandidate] = []
        clusters: List[List[CaseCandidate]] = []
        for member in members:
            if (
                cluster
                and member.root_cause_action_index - cluster[-1].root_cause_action_index
                > root_tolerance
            ):
                clusters.append(cluster)
                cluster = []
            cluster.append(member)
        if cluster:
            clusters.append(cluster)
        for cluster in clusters:
            keeper = max(cluster, key=_keeper_key)
            losers = [item for item in cluster if item is not keeper]
            merged = tuple(sorted({*keeper.merged_from, *(item.case_id for item in losers)}))
            kept.append(replace(keeper, merged_from=merged))
            removed.extend(replace(item, status=DUPLICATE_CASE) for item in losers)
    return kept, removed


@dataclass(frozen=True)
class FunnelRow:
    """One rollout's progress through Runs -> Fail -> Typed -> Repaired -> Verified -> Cases."""

    agent: str
    failed: bool = False
    typed: bool = False
    repaired: bool = False
    verified: bool = False
    case_count: int = 0


def funnel_by_agent(rows: Iterable[FunnelRow]) -> Dict[str, Dict[str, int]]:
    """Aggregate the per-agent funnel table used for the paper's Table 12 columns."""

    table: Dict[str, Counter] = defaultdict(Counter)
    for row in rows:
        counter = table[row.agent]
        counter["runs"] += 1
        counter["failed"] += int(row.failed)
        counter["typed"] += int(row.failed and row.typed)
        counter["repaired"] += int(row.failed and row.typed and row.repaired)
        counter["verified"] += int(row.failed and row.typed and row.repaired and row.verified)
        counter["cases"] += row.case_count
    return {agent: dict(counter) for agent, counter in sorted(table.items())}


def case_yield(
    cases: Sequence[CaseCandidate], depth_grid: Sequence[int], *, by: str = "paper_category"
) -> Dict[str, Dict[int, int]]:
    """Count cases per depth, grouped by a candidate attribute (default: paper category)."""

    counts: Dict[str, Dict[int, int]] = defaultdict(lambda: {int(depth): 0 for depth in depth_grid})
    for case in cases:
        counts[str(getattr(case, by))][case.error_depth] += 1
    return {key: dict(value) for key, value in sorted(counts.items())}


def stratum_counts(
    cases: Sequence[CaseCandidate], depth_grid: Sequence[int]
) -> Dict[str, Dict[str, Dict[int, int]]]:
    """``stratum -> paper_type -> depth -> count`` (the per-stratum table missing in the paper)."""

    table: Dict[str, Dict[str, Dict[int, int]]] = defaultdict(
        lambda: defaultdict(lambda: {int(depth): 0 for depth in depth_grid})
    )
    for case in cases:
        table[case.reversibility_stratum][case.paper_type][case.error_depth] += 1
    return {
        stratum: {paper_type: dict(depths) for paper_type, depths in sorted(inner.items())}
        for stratum, inner in sorted(table.items())
    }
