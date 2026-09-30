"""Phase 1: assemble one tidy row per analysed rollout, and derive fields.

Analysis set
------------
A rollout enters ``build_clean_table`` iff **all** of the following hold:

1. its ``source_agent`` is one of the agents listed under
   ``analysis_set.agent_annotator_whitelist`` in the field mapping;
2. it carries a human rubric review (``rubric_scores`` export) written by
   *that agent's whitelisted annotator* -- reviews of the same agent by any
   other annotator are dropped as noise, per the user's instruction;
3. it is not flagged Wrong Rollout in ``rollout_flags``.

Everything excluded is recorded with a reason in the exclusion log, never
silently dropped. Nothing is imputed: a missing value stays missing and is
counted separately.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from .io import (
    discover_canonical_dirs,
    load_canonical_record,
    load_label_store,
    load_taxonomy,
)
from .paths import Paths

UNMAPPED = "UNMAPPED"


def load_mapping(config_path: Path) -> dict:
    """Read the confirmed field mapping. Every column name used downstream
    originates here, so the pipeline never hard-codes a physical column."""
    with config_path.open(encoding="utf-8") as fh:
        mapping = yaml.safe_load(fh)
    for required in ("fields", "analysis_set", "error_categories", "meta"):
        if required not in mapping:
            raise SystemExit(f"field_mapping.yaml is missing top-level key {required!r}")
    return mapping


def _col(mapping: dict, logical: str, store: str | None = None) -> str:
    """Resolve a logical field name to its physical column for a given store."""
    spec = mapping["fields"][logical]
    col = spec.get("column")
    if isinstance(col, dict):
        if store is None:
            raise ValueError(f"{logical} needs a store to resolve its column")
        return col[store]
    if col is None:
        raise ValueError(f"{logical} has no physical column (store={spec.get('store')})")
    return col


def _snake(label: str) -> str:
    """Normalise a taxonomy label to snake_case without inventing characters."""
    return "_".join(str(label).strip().lower().replace("-", " ").replace("/", " ").split())


def build_category_lookup(mapping: dict) -> dict[str, str]:
    """Invert the configured error_type -> category grouping into label -> category.

    The grouping actually used is selected by ``error_category_source``. Labels
    absent from it are NOT guessed; they surface later as ``UNMAPPED``.
    """
    source = mapping.get("error_category_source", "spec")
    groups = mapping["error_categories"][source]
    lookup: dict[str, str] = {}
    for category, labels in groups.items():
        for label in labels or []:
            key = _snake(label)
            if key in lookup and lookup[key] != category:
                raise ValueError(f"label {key!r} appears in two categories")
            lookup[key] = category
    return lookup


def _agent_of(canon_rec: dict) -> str | None:
    """Return the single source agent of a trajectory, or None if ambiguous/absent."""
    agents = canon_rec.get("source_agents") or []
    return agents[0] if len(agents) == 1 else None


def build_clean_table(
    paths: Paths, mapping: dict
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Return ``(clean, exclusions, diagnostics)``.

    ``clean`` is one row per analysed rollout, keyed by ``trajectory_id``.
    ``exclusions`` lists every rollout considered and rejected, with a reason.
    ``diagnostics`` holds cross-checks that the report and sanity pass consume.
    """
    aset = mapping["analysis_set"]
    whitelist: dict[str, str] = aset["agent_annotator_whitelist"]
    low_n = int(aset.get("low_n_threshold", 10))

    canon = {d.name: load_canonical_record(d) for d in discover_canonical_dirs(paths.builds_dir)}
    rubric = load_label_store(paths.rubric_scores, "rubric_scores")
    failure = load_label_store(paths.human_labels, "failure_annotations")
    flags = load_label_store(paths.rollout_flags, "rollout_flags")

    c_task_success = _col(mapping, "task_score")
    c_scores = _col(mapping, "rubric_scores")
    c_reviewer = _col(mapping, "annotator", "rubric_scores")
    c_annotator = _col(mapping, "annotator", "failure_annotations")
    c_root = _col(mapping, "root_cause_step")
    c_clear = _col(mapping, "clear_failure_step")
    c_depth = _col(mapping, "failure_depth")
    c_types = _col(mapping, "error_types")
    c_rev = _col(mapping, "reversibility")
    c_len = _col(mapping, "trajectory_length")
    c_task_id = _col(mapping, "task_id")

    flagged = {r["_trajectory_id_from_filename"] for r in flags}
    fail_by_key = {
        (r["_trajectory_id_from_filename"], r[c_annotator]): r for r in failure
    }

    rows, excl = [], []
    for rec in rubric:
        tid = rec["_trajectory_id_from_filename"]
        reviewer = rec[c_reviewer]
        cr = canon.get(tid)
        if cr is None:
            excl.append(dict(trajectory_id=tid, annotator=reviewer, agent=None,
                             reason="no_canonical_record"))
            continue
        agent = _agent_of(cr)
        if agent is None:
            excl.append(dict(trajectory_id=tid, annotator=reviewer, agent=None,
                             reason="source_agent_absent_or_ambiguous"))
            continue
        if agent not in whitelist:
            excl.append(dict(trajectory_id=tid, annotator=reviewer, agent=agent,
                             reason="agent_not_in_analysis_set"))
            continue
        if reviewer != whitelist[agent]:
            excl.append(dict(trajectory_id=tid, annotator=reviewer, agent=agent,
                             reason=f"annotator_not_whitelisted_for_agent"
                                    f" (expected {whitelist[agent]})"))
            continue
        if aset.get("exclude_wrong_rollout", True) and tid in flagged:
            excl.append(dict(trajectory_id=tid, annotator=reviewer, agent=agent,
                             reason="flagged_wrong_rollout"))
            continue

        scores: dict = rec[c_scores]
        n_def = cr["n_rubrics_defined"]
        task_score = int(bool(rec[c_task_success]))

        fa = fail_by_key.get((tid, reviewer))
        root = fa[c_root] if fa else None
        clear = fa[c_clear] if fa else None
        depth_stored = fa[c_depth] if fa else None
        types_raw = fa[c_types] if fa else None
        rev = fa[c_rev] if fa else None

        traj_len = cr[c_len]
        # 0-based indexing: the last observable action index is traj_len - 1.
        base = int(mapping["meta"].get("step_index_base", 0))
        last_index = traj_len - 1 + base

        rows.append(
            {
                "trajectory_id": tid,
                "task_id": cr[c_task_id],
                "agent": agent,
                "annotator": reviewer,
                "task_category": cr["category"],
                "difficulty": cr["difficulty"],
                "app": cr["app"],
                "trajectory_length": traj_len,
                "task_score": task_score,
                "n_rubrics_defined": n_def,
                "n_rubrics_scored": len(scores),
                "n_rubrics_passed": sum(int(v) for v in scores.values()),
                "n_rubrics_failed": sum(1 for v in scores.values() if int(v) == 0),
                "rubric_scores_json": scores,
                "rubric_weights": cr["rubric_weights"],
                "has_failure_annotation": fa is not None,
                "root_cause_step": root,
                "clear_failure_step": clear,
                "failure_depth_stored": depth_stored,
                "error_types_raw": types_raw,
                "reversibility": rev,
                "last_action_index": last_index,
                "terminal_action_kind": cr["terminal_action_kind"],
                "terminal_action_status": cr["terminal_action_status"],
            }
        )

    clean = pd.DataFrame(rows)
    exclusions = pd.DataFrame(excl)
    if clean.empty:
        raise SystemExit("analysis set is empty; check the whitelist in field_mapping.yaml")

    # --- primary key -------------------------------------------------------
    assert clean["trajectory_id"].is_unique, (
        "trajectory_id is not unique after whitelist filtering; the whitelist was "
        "supposed to leave exactly one annotator per agent"
    )
    for agent, grp in clean.groupby("agent"):
        assert grp["annotator"].nunique() == 1, f"{agent} still has >1 annotator"

    # --- rubric derivations ------------------------------------------------
    assert (clean["n_rubrics_scored"] == clean["n_rubrics_defined"]).all(), (
        "a rubric_scores export disagrees with its task_config rubric count"
    )
    assert (clean["n_rubrics_defined"] > 0).all(), "a task defines zero rubrics"
    clean["rubric_pass_ratio"] = clean["n_rubrics_passed"] / clean["n_rubrics_defined"]
    assert clean["rubric_pass_ratio"].between(0, 1).all()

    clean["recomputed_task_score"] = (
        clean["n_rubrics_passed"] == clean["n_rubrics_defined"]
    ).astype(int)

    clean["weighted_rubric_score"] = [
        _weighted(s, w) for s, w in zip(clean["rubric_scores_json"], clean["rubric_weights"])
    ]

    # --- error types -------------------------------------------------------
    lookup = build_category_lookup(mapping)
    norm = mapping.get("label_normalization") or {}
    renames: dict = {_snake(k): _snake(v) for k, v in (norm.get("rename") or {}).items()}
    drops: set = {_snake(x) for x in (norm.get("drop") or [])}

    normalized, norm_log = [], []
    for tid, agent, raw in zip(clean["trajectory_id"], clean["agent"],
                               clean["error_types_raw"]):
        if not isinstance(raw, list):
            normalized.append(None)
            continue
        original = sorted({_snake(t) for t in raw})
        kept = set()
        for label in original:
            if label in drops:
                norm_log.append(dict(trajectory_id=tid, agent=agent, action="drop",
                                     original_label=label, new_label=""))
                continue
            if label in renames:
                norm_log.append(dict(trajectory_id=tid, agent=agent, action="rename",
                                     original_label=label, new_label=renames[label]))
                kept.add(renames[label])
                continue
            kept.add(label)
        if original and not kept:
            norm_log.append(dict(trajectory_id=tid, agent=agent,
                                 action="emptied_by_normalization",
                                 original_label="|".join(original), new_label=""))
        normalized.append(sorted(kept))

    clean["error_types"] = normalized
    # A rollout whose labels were all dropped has no analysable root cause left.
    # It is NOT given an empty-set interpretation and NOT counted as a zero; it
    # is flagged so the error-type analysis can exclude it and report it.
    clean["error_types_emptied_by_normalization"] = [
        bool(isinstance(raw, list) and raw and not new)
        for raw, new in zip(clean["error_types_raw"], clean["error_types"])
    ]
    clean["n_error_types"] = [
        len(v) if v else np.nan for v in clean["error_types"]
    ]
    clean["error_categories"] = [
        sorted({lookup.get(t, UNMAPPED) for t in v}) if v else None
        for v in clean["error_types"]
    ]
    clean.attrs["label_normalization_log"] = pd.DataFrame(
        norm_log, columns=["trajectory_id", "agent", "action",
                           "original_label", "new_label"])

    # --- depth -------------------------------------------------------------
    clean = _derive_depth(clean, mapping)

    diagnostics = {
        "low_n_threshold": low_n,
        "unmapped_labels": sorted(
            {t for v in clean["error_types"] if v for t in v if t not in lookup}
        ),
        "category_source": mapping.get("error_category_source", "spec"),
        "n_flagged_wrong_rollout_total": len(flagged),
        "n_flagged_wrong_rollout_removed": int(
            (exclusions["reason"] == "flagged_wrong_rollout").sum()
        ) if not exclusions.empty else 0,
    }
    return clean, exclusions, diagnostics


def _weighted(scores: dict, weights: list | None) -> float:
    """Weighted rubric score = sum(score_i * weight_i) / sum(weight_i).

    Returns NaN when weights are absent or sum to zero; never substitutes an
    unweighted score, because that would silently change the metric.
    """
    if not weights or len(weights) != len(scores):
        return float("nan")
    if any(w is None for w in weights):
        return float("nan")
    total = float(sum(weights))
    if total == 0:
        return float("nan")
    ordered = [scores[f"R{i}"] for i in range(1, len(weights) + 1)]
    return float(sum(s * w for s, w in zip(ordered, weights)) / total)


def _derive_depth(clean: pd.DataFrame, mapping: dict) -> pd.DataFrame:
    """Derive failure depth and its censoring status.

    Computed only for rows with ``task_score == 0`` and a non-missing
    ``root_cause_step``; every other row gets NaN and a status explaining why.

    Statuses
    --------
    observed          clear_failure_step present -> depth = clear - root
    right_censored    clear_failure_step is null ("failure never becomes clear")
                      -> censor_time = last_action_index - root_cause_step
    not_applicable    task_score == 1 (no failure to locate)
    missing_root      task_score == 0 but no root cause recorded
    """
    depth, status, censor = [], [], []
    for _, r in clean.iterrows():
        if r["task_score"] == 1:
            depth.append(np.nan); status.append("not_applicable"); censor.append(np.nan); continue
        root = r["root_cause_step"]
        if root is None or (isinstance(root, float) and np.isnan(root)):
            depth.append(np.nan); status.append("missing_root"); censor.append(np.nan); continue
        clear = r["clear_failure_step"]
        if clear is None or (isinstance(clear, float) and np.isnan(clear)):
            depth.append(np.nan)
            status.append("right_censored")
            censor.append(float(r["last_action_index"] - root))
        else:
            depth.append(float(clear - root))
            status.append("observed")
            censor.append(np.nan)
    clean["failure_depth"] = depth
    clean["depth_status"] = status
    clean["censor_time"] = censor

    # lifelines needs a single time column plus an event indicator.
    clean["depth_time"] = clean["failure_depth"].where(
        clean["depth_status"] == "observed", clean["censor_time"]
    )
    clean["depth_event"] = np.where(
        clean["depth_status"] == "observed", 1,
        np.where(clean["depth_status"] == "right_censored", 0, np.nan),
    )

    edges = mapping["depth_bins"]["edges"]
    labels = mapping["depth_bins"]["labels"]
    clean["depth_bin"] = [
        _bin_depth(d, s, c, edges, labels)
        for d, s, c in zip(clean["failure_depth"], clean["depth_status"], clean["censor_time"])
    ]
    return clean


def _bin_depth(depth, status, censor, edges: list[int], labels: list[str]):
    """Assign an ordinal depth bin, resolving censored rows only when the
    censoring time already forces the answer.

    A row censored at time c is known to have depth >= c. If c falls in the top
    bin, the bin is determined; otherwise the bin is genuinely unknown and is
    labelled ``CENSORED_UNRESOLVED`` rather than being guessed.
    """
    if status == "observed":
        idx = int(np.searchsorted(edges, depth, side="right")) - 1
        return labels[min(max(idx, 0), len(labels) - 1)]
    if status == "right_censored":
        if censor is not None and not np.isnan(censor) and censor >= edges[-1]:
            return labels[-1]
        return "CENSORED_UNRESOLVED"
    return None


def explode_error_types(clean: pd.DataFrame, mapping: dict) -> pd.DataFrame:
    """Long-format label table: one row per (rollout, error_type).

    Analysis set: rows with ``task_score == 0`` that carry a failure annotation
    **and still have at least one label after normalisation**. Rollouts whose
    every label was dropped are excluded here and reported separately as
    label-missing, rather than being counted as a rollout with zero errors.

    Multi-label by construction, so rollout-normalised rates across labels sum
    to more than 1 and that is correct.
    """
    lookup = build_category_lookup(mapping)
    sub = clean[(clean["task_score"] == 0) & clean["has_failure_annotation"]
                & ~clean["error_types_emptied_by_normalization"]]
    out = []
    for _, r in sub.iterrows():
        for label in r["error_types"] or []:
            out.append(
                {
                    "trajectory_id": r["trajectory_id"],
                    "agent": r["agent"],
                    "task_id": r["task_id"],
                    "task_category": r["task_category"],
                    "error_type": label,
                    "error_category": lookup.get(label, UNMAPPED),
                    "failure_depth": r["failure_depth"],
                    "depth_time": r["depth_time"],
                    "depth_event": r["depth_event"],
                    "depth_bin": r["depth_bin"],
                    "depth_status": r["depth_status"],
                }
            )
    return pd.DataFrame(out)


def consistency_checks(clean: pd.DataFrame) -> pd.DataFrame:
    """Collect every internal contradiction as one tidy table; fix nothing."""
    issues = []

    m = clean["task_score"] != clean["recomputed_task_score"]
    for _, r in clean[m].iterrows():
        issues.append(dict(
            issue="task_score_vs_all_rubrics_passed", trajectory_id=r["trajectory_id"],
            agent=r["agent"],
            detail=f"task_score={r['task_score']} but "
                   f"{r['n_rubrics_passed']}/{r['n_rubrics_defined']} rubrics passed"))

    m = (clean["task_score"] == 1) & clean["has_failure_annotation"]
    for _, r in clean[m].iterrows():
        issues.append(dict(issue="success_with_failure_annotation",
                           trajectory_id=r["trajectory_id"], agent=r["agent"], detail=""))

    m = (clean["task_score"] == 0) & ~clean["has_failure_annotation"]
    for _, r in clean[m].iterrows():
        issues.append(dict(issue="failure_without_failure_annotation",
                           trajectory_id=r["trajectory_id"], agent=r["agent"], detail=""))

    m = clean["failure_depth"] < 0
    for _, r in clean[m].iterrows():
        issues.append(dict(issue="negative_depth", trajectory_id=r["trajectory_id"],
                           agent=r["agent"],
                           detail=f"clear={r['clear_failure_step']} root={r['root_cause_step']}"))

    m = clean["root_cause_step"].notna() & (clean["root_cause_step"] > clean["last_action_index"])
    for _, r in clean[m].iterrows():
        issues.append(dict(issue="root_cause_step_out_of_range",
                           trajectory_id=r["trajectory_id"], agent=r["agent"],
                           detail=f"root={r['root_cause_step']} last={r['last_action_index']}"))

    obs = clean["depth_status"] == "observed"
    m = obs & (clean["failure_depth"] != clean["failure_depth_stored"])
    for _, r in clean[m].iterrows():
        issues.append(dict(issue="recomputed_depth_differs_from_stored",
                           trajectory_id=r["trajectory_id"], agent=r["agent"],
                           detail=f"recomputed={r['failure_depth']} "
                                  f"stored={r['failure_depth_stored']}"))

    return pd.DataFrame(issues, columns=["issue", "trajectory_id", "agent", "detail"])


def annotation_coverage(paths: Paths, mapping: dict, clean: pd.DataFrame) -> pd.DataFrame:
    """Compare annotated vs un-annotated rollouts per agent (selection-bias probe).

    Human labelling covered only part of each agent's canonical build. If the
    labelled subset differs systematically from the unlabelled remainder, every
    rate computed on it is a property of the labelling process as much as of the
    agent. This table quantifies that gap using the agent's own terminate status
    -- the one outcome-adjacent signal available for *unlabelled* rollouts.

    It imputes nothing about the unlabelled rows; it only describes them.
    """
    whitelist = mapping["analysis_set"]["agent_annotator_whitelist"]
    canon = {d.name: load_canonical_record(d) for d in discover_canonical_dirs(paths.builds_dir)}
    labelled = set(clean["trajectory_id"])
    rows = []
    for tid, rec in canon.items():
        agent = _agent_of(rec)
        if agent not in whitelist:
            continue
        status = rec["terminal_action_status"]
        rows.append({
            "agent": agent,
            "trajectory_id": tid,
            "subset": "annotated" if tid in labelled else "not_annotated",
            "self_report": status if status is not None else "no_terminate",
        })
    df = pd.DataFrame(rows)
    out = []
    for (agent, subset), g in df.groupby(["agent", "subset"]):
        n = len(g)
        row = {"agent": agent, "subset": subset, "n": n}
        for level in ("success", "failure", "no_terminate"):
            k = int((g["self_report"] == level).sum())
            row[f"n_{level}"] = k
            row[f"pct_{level}"] = k / n if n else float("nan")
        out.append(row)
    return pd.DataFrame(out).sort_values(["agent", "subset"])
