"""Phase 2: statistics. Every rate in the pipeline is produced by ``rate_with_ci``.

Conventions enforced here
-------------------------
* A rate is never reported without its numerator, denominator and 95% CI.
* Any denominator below ``low_n`` carries a ``[LOW_N]`` marker.
* Missing values are counted and reported; they are never imputed and never
  folded into a category.
* All resampling uses ``SEED``.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict

import numpy as np
import pandas as pd
from scipy import stats as sps
from statsmodels.stats.multitest import multipletests
from statsmodels.stats.proportion import proportion_confint

SEED = 42
LOW_N_MARK = "[LOW_N]"


@dataclass(frozen=True)
class Rate:
    """A proportion with everything needed to judge it."""

    k: int
    n: int
    rate: float
    ci_low: float
    ci_high: float
    low_n: bool

    @property
    def flag(self) -> str:
        return LOW_N_MARK if self.low_n else ""


def rate_with_ci(k: int, n: int, low_n: int = 10, method: str = "wilson") -> Rate:
    """The single entry point for every proportion in this pipeline.

    Returns k, n, k/n and a 95% CI. With ``n == 0`` the rate and both bounds are
    NaN -- an undefined proportion is reported as undefined, not as zero.
    """
    k, n = int(k), int(n)
    if k > n:
        raise ValueError(f"numerator {k} exceeds denominator {n}")
    if n == 0:
        return Rate(k, n, float("nan"), float("nan"), float("nan"), True)
    lo, hi = proportion_confint(k, n, alpha=0.05, method=method)
    return Rate(k, n, k / n, float(lo), float(hi), n < low_n)


def rate_row(k: int, n: int, low_n: int = 10, **extra) -> dict:
    """``rate_with_ci`` flattened into a tidy-CSV row, with the LOW_N marker."""
    r = rate_with_ci(k, n, low_n)
    row = dict(extra)
    row.update(numerator=r.k, denominator=r.n, rate=r.rate,
               ci_low=r.ci_low, ci_high=r.ci_high, low_n_flag=r.flag)
    return row


def bh_fdr(pvalues: list[float]) -> list[float]:
    """Benjamini-Hochberg q-values; NaN p-values pass through as NaN."""
    p = np.asarray(pvalues, dtype=float)
    ok = ~np.isnan(p)
    q = np.full_like(p, np.nan)
    if ok.sum():
        q[ok] = multipletests(p[ok], method="fdr_bh")[1]
    return q.tolist()


# --------------------------------------------------------------------------
# 1. task score
# --------------------------------------------------------------------------
def task_score_by_agent(clean: pd.DataFrame, low_n: int = 10) -> pd.DataFrame:
    """Task success rate per agent plus an overall row.

    Analysis set: the whole clean table (every row already passed the whitelist
    and wrong-rollout filters).
    """
    rows = []
    for agent, g in clean.groupby("agent"):
        rows.append(rate_row(int(g["task_score"].sum()), len(g), low_n,
                             agent=agent, metric="task_score"))
    rows.append(rate_row(int(clean["task_score"].sum()), len(clean), low_n,
                         agent="OVERALL", metric="task_score"))
    return pd.DataFrame(rows)


def task_score_by_group(clean: pd.DataFrame, group: str, low_n: int = 10) -> pd.DataFrame:
    """Task success rate split by agent x ``group`` (e.g. task_category)."""
    rows = []
    for (agent, key), g in clean.groupby(["agent", group], dropna=False):
        rows.append(rate_row(int(g["task_score"].sum()), len(g), low_n,
                             agent=agent, group_field=group, group_value=key))
    for key, g in clean.groupby(group, dropna=False):
        rows.append(rate_row(int(g["task_score"].sum()), len(g), low_n,
                             agent="OVERALL", group_field=group, group_value=key))
    return pd.DataFrame(rows)


def unpaired_group_test(clean: pd.DataFrame, value_col: str, low_n: int = 10) -> pd.DataFrame:
    """Chi-square across agents plus pairwise Fisher with BH-FDR.

    Unpaired because the agents do not share a common task set; see the report's
    data-hygiene section for the coverage numbers that justify this choice.
    """
    agents = sorted(clean["agent"].unique())
    table = np.array([
        [int((clean[clean.agent == a][value_col] == 1).sum()),
         int((clean[clean.agent == a][value_col] == 0).sum())]
        for a in agents
    ])
    rows = []
    chi2, p, dof, expected = sps.chi2_contingency(table)
    n = table.sum()
    # Cramer's V for a k x 2 table.
    v = float(np.sqrt((chi2 / n) / (min(table.shape) - 1))) if n and min(table.shape) > 1 else np.nan
    rows.append(dict(comparison="omnibus", test="chi2_contingency", statistic=chi2,
                     dof=dof, p_value=p, q_value=np.nan, effect_size_name="cramers_v",
                     effect_size=v, min_expected_count=float(expected.min()),
                     note="expected counts below 5 make the chi-square approximation unreliable"
                          if expected.min() < 5 else ""))
    pw = []
    for i in range(len(agents)):
        for j in range(i + 1, len(agents)):
            sub = table[[i, j], :]
            odds, pp = sps.fisher_exact(sub)
            pw.append(dict(comparison=f"{agents[i]} vs {agents[j]}", test="fisher_exact",
                           statistic=float(odds), dof=np.nan, p_value=float(pp),
                           effect_size_name="odds_ratio", effect_size=float(odds),
                           min_expected_count=float(sub.min()),
                           note=LOW_N_MARK if sub.sum(axis=1).min() < low_n else ""))
    for row, q in zip(pw, bh_fdr([r["p_value"] for r in pw])):
        row["q_value"] = q
    rows.extend(pw)
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# 2. rubric score
# --------------------------------------------------------------------------
def _bootstrap_ci(values: np.ndarray, n_resamples: int, seed: int) -> tuple[float, float]:
    """Percentile bootstrap CI of the mean. NaN if fewer than 2 observations."""
    values = values[~np.isnan(values)]
    if len(values) < 2:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(values), size=(n_resamples, len(values)))
    means = values[idx].mean(axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def rubric_score_by_agent(clean: pd.DataFrame, low_n: int = 10,
                          n_resamples: int = 10000, seed: int = SEED) -> pd.DataFrame:
    """MACRO and MICRO rubric scores per agent, plus overall.

    MACRO: mean over rollouts of each rollout's rubric_pass_ratio. Every rollout
    counts once regardless of how many rubrics its task defines.
    MICRO: pooled passed rubrics / pooled defined rubrics. Tasks with more
    rubrics pull harder. The two differ whenever rubric counts vary by task,
    which they do here (3..17 rubrics).
    """
    rows = []
    groups = [(a, g) for a, g in clean.groupby("agent")] + [("OVERALL", clean)]
    for agent, g in groups:
        v = g["rubric_pass_ratio"].to_numpy(dtype=float)
        lo, hi = _bootstrap_ci(v, n_resamples, seed)
        micro = rate_with_ci(int(g["n_rubrics_passed"].sum()),
                             int(g["n_rubrics_defined"].sum()), low_n)
        w = g["weighted_rubric_score"].to_numpy(dtype=float)
        wv = w[~np.isnan(w)]
        rho, rho_p = (sps.spearmanr(v, w, nan_policy="omit") if len(wv) > 2
                      else (np.nan, np.nan))
        rows.append(dict(
            agent=agent, n_rollouts=len(g),
            macro_mean=float(np.nanmean(v)), macro_sd=float(np.nanstd(v, ddof=1)) if len(v) > 1 else np.nan,
            macro_median=float(np.nanmedian(v)),
            macro_q1=float(np.nanpercentile(v, 25)), macro_q3=float(np.nanpercentile(v, 75)),
            macro_boot_ci_low=lo, macro_boot_ci_high=hi,
            micro_numerator=micro.k, micro_denominator=micro.n, micro_rate=micro.rate,
            micro_ci_low=micro.ci_low, micro_ci_high=micro.ci_high,
            weighted_mean=float(np.nanmean(w)) if len(wv) else np.nan,
            weighted_sd=float(np.nanstd(w, ddof=1)) if len(wv) > 1 else np.nan,
            weighted_median=float(np.nanmedian(w)) if len(wv) else np.nan,
            weighted_q1=float(np.nanpercentile(w, 25)) if len(wv) else np.nan,
            weighted_q3=float(np.nanpercentile(w, 75)) if len(wv) else np.nan,
            weighted_boot_ci_low=_bootstrap_ci(w, n_resamples, seed)[0],
            weighted_boot_ci_high=_bootstrap_ci(w, n_resamples, seed)[1],
            weighted_minus_macro_mean=float(np.nanmean(w) - np.nanmean(v)) if len(wv) else np.nan,
            weighted_minus_macro_maxabs=float(np.nanmax(np.abs(w - v))) if len(wv) else np.nan,
            weighted_n_missing=int(np.isnan(w).sum()),
            spearman_macro_vs_weighted=float(rho) if not np.isnan(float(rho)) else np.nan,
            spearman_p=float(rho_p) if rho_p == rho_p else np.nan,
            low_n_flag=LOW_N_MARK if len(g) < low_n else "",
        ))
    return pd.DataFrame(rows)


def rubric_ratio_distribution(clean: pd.DataFrame, low_n: int = 10) -> pd.DataFrame:
    """Histogram of rubric_pass_ratio in fixed 0.1 bins, per agent.

    Reported as counts rather than a smoothed density so that bimodality (mass
    piled at 0 and at 1) stays visible instead of being averaged away.
    """
    edges = np.round(np.arange(0, 1.0001, 0.1), 3)
    rows = []
    for agent, g in list(clean.groupby("agent")) + [("OVERALL", clean)]:
        v = g["rubric_pass_ratio"].to_numpy(dtype=float)
        counts, _ = np.histogram(v, bins=edges)
        # np.histogram puts 1.0 in the last bin already (right-closed final bin).
        for i, c in enumerate(counts):
            rows.append(rate_row(int(c), len(v), low_n, agent=agent,
                                 bin_low=edges[i], bin_high=edges[i + 1]))
    return pd.DataFrame(rows)


def rubric_by_agent_matrix(clean: pd.DataFrame, low_n: int = 10) -> pd.DataFrame:
    """Pass rate of each rubric slot R1..Rn, per agent.

    Rubric ids are positional within a task's bundle, so R3 means different
    criteria across tasks. This table is therefore a slot-level view, not a
    criterion-level one, and must not be read as "rubric 3 is hard".
    """
    rows = []
    for agent, g in list(clean.groupby("agent")) + [("OVERALL", clean)]:
        slot_pass, slot_total = {}, {}
        for scores in g["rubric_scores_json"]:
            for rid, val in scores.items():
                slot_total[rid] = slot_total.get(rid, 0) + 1
                slot_pass[rid] = slot_pass.get(rid, 0) + int(val)
        for rid in sorted(slot_total, key=lambda x: int(x[1:])):
            rows.append(rate_row(slot_pass[rid], slot_total[rid], low_n,
                                 agent=agent, rubric_id=rid))
    return pd.DataFrame(rows)


def weighted_ratio_distribution(clean: pd.DataFrame, low_n: int = 10) -> pd.DataFrame:
    """Histogram of weighted_rubric_score in fixed 0.1 bins, per agent.

    Reported alongside the unweighted distribution so that a difference in shape,
    not only in mean, is visible.
    """
    edges = np.round(np.arange(0, 1.0001, 0.1), 3)
    rows = []
    for agent, g in list(clean.groupby("agent")) + [("OVERALL", clean)]:
        v = g["weighted_rubric_score"].to_numpy(dtype=float)
        v = v[~np.isnan(v)]
        counts, _ = np.histogram(v, bins=edges)
        for i, c in enumerate(counts):
            rows.append(rate_row(int(c), len(v), low_n, agent=agent,
                                 bin_low=edges[i], bin_high=edges[i + 1]))
    return pd.DataFrame(rows)


def rubric_weight_profile(clean: pd.DataFrame) -> pd.DataFrame:
    """Describe the rubric weight vectors themselves.

    Establishes whether weighting can carry information at all: if every task
    weighted its rubrics uniformly, the weighted score would be identical to the
    unweighted pass ratio and would add nothing.
    """
    rows = []
    for _, r in clean.iterrows():
        w = r["rubric_weights"]
        if not w or any(x is None for x in w):
            rows.append(dict(trajectory_id=r["trajectory_id"], agent=r["agent"],
                             task_id=r["task_id"], n_rubrics=r["n_rubrics_defined"],
                             weights_present=False, uniform=None, weight_sum=np.nan,
                             weight_min=np.nan, weight_max=np.nan, weight_ratio=np.nan))
            continue
        w = np.asarray(w, dtype=float)
        rows.append(dict(trajectory_id=r["trajectory_id"], agent=r["agent"],
                         task_id=r["task_id"], n_rubrics=r["n_rubrics_defined"],
                         weights_present=True,
                         uniform=bool(np.allclose(w, w[0])),
                         weight_sum=float(w.sum()),
                         weight_min=float(w.min()), weight_max=float(w.max()),
                         weight_ratio=float(w.max() / w.min()) if w.min() > 0 else np.nan))
    return pd.DataFrame(rows)


def weighted_vs_unweighted_gap(clean: pd.DataFrame, top: int = 15) -> pd.DataFrame:
    """Rollouts where weighting moves the score the most.

    These are the only rows where the weighted metric can change a conclusion;
    if the list is short and the gaps are small, the two metrics are effectively
    interchangeable for this dataset.
    """
    d = clean.assign(gap=(clean["weighted_rubric_score"] - clean["rubric_pass_ratio"]))
    d = d[d["gap"].notna()].copy()
    d["abs_gap"] = d["gap"].abs()
    cols = ["trajectory_id", "agent", "task_id", "n_rubrics_defined",
            "n_rubrics_passed", "rubric_pass_ratio", "weighted_rubric_score",
            "gap", "task_score"]
    return d.sort_values("abs_gap", ascending=False).head(top)[cols]


# --------------------------------------------------------------------------
# 3. error types ("root cause")
# --------------------------------------------------------------------------
def error_analysis_set(clean: pd.DataFrame) -> pd.DataFrame:
    """The denominator for every error-type statistic.

    Failed rollouts that carry a failure annotation **and** still have at least
    one label after normalisation. Rollouts emptied by label normalisation are
    excluded here and counted by ``label_missing_by_agent`` instead, so they
    never silently deflate a rate.
    """
    return clean[(clean["task_score"] == 0) & clean["has_failure_annotation"]
                 & ~clean["error_types_emptied_by_normalization"]]


def label_missing_by_agent(clean: pd.DataFrame, low_n: int = 10) -> pd.DataFrame:
    """Failed rollouts left with no error label after normalisation, per agent."""
    failed = clean[(clean["task_score"] == 0) & clean["has_failure_annotation"]]
    rows = []
    for agent, g in list(failed.groupby("agent")) + [("OVERALL", failed)]:
        k = int(g["error_types_emptied_by_normalization"].sum())
        rows.append(rate_row(k, len(g), low_n, agent=agent,
                             metric="failed_rollouts_with_no_label_after_normalization"))
    return pd.DataFrame(rows)


def error_type_rates(long: pd.DataFrame, clean: pd.DataFrame, level: str,
                     low_n: int = 10) -> pd.DataFrame:
    """Rollout-normalised and label-normalised distribution of ``level``.

    Analysis set: failed rollouts (``task_score == 0``) carrying a failure
    annotation.

    ROLLOUT-NORMALIZED = rollouts containing label L / failed rollouts. Labels
    are multi-select, so the column sums to more than 1 by design.
    LABEL-NORMALIZED  = occurrences of L / total label occurrences; sums to 1.
    """
    failed = error_analysis_set(clean)
    rows = []
    scopes = [(a, long[long.agent == a], failed[failed.agent == a])
              for a in sorted(long["agent"].unique())]
    scopes.append(("OVERALL", long, failed))
    for agent, sub, denom_df in scopes:
        n_failed = len(denom_df)
        total_labels = len(sub)
        for value, g in sub.groupby(level):
            n_rollouts = g["trajectory_id"].nunique()
            row = rate_row(n_rollouts, n_failed, low_n, agent=agent,
                           level=level, value=value)
            row["label_occurrences"] = len(g)
            row["label_normalized"] = len(g) / total_labels if total_labels else np.nan
            row["total_label_occurrences"] = total_labels
            rows.append(row)
    return pd.DataFrame(rows)


def label_cardinality(clean: pd.DataFrame, low_n: int = 10) -> pd.DataFrame:
    """How many error labels a failed rollout carries, per agent.

    Denominator excludes rollouts emptied by label normalisation.
    """
    failed = error_analysis_set(clean)
    rows = []
    for agent, g in list(failed.groupby("agent")) + [("OVERALL", failed)]:
        v = g["n_error_types"].to_numpy(dtype=float)
        rows.append(dict(agent=agent, n_failed_rollouts=len(g),
                         mean=float(np.nanmean(v)),
                         sd=float(np.nanstd(v, ddof=1)) if len(v) > 1 else np.nan,
                         median=float(np.nanmedian(v)),
                         q1=float(np.nanpercentile(v, 25)), q3=float(np.nanpercentile(v, 75)),
                         min=float(np.nanmin(v)), max=float(np.nanmax(v)),
                         low_n_flag=LOW_N_MARK if len(g) < low_n else ""))
    return pd.DataFrame(rows)


def error_level_homogeneity(long: pd.DataFrame, clean: pd.DataFrame, level: str,
                            low_n: int = 10) -> pd.DataFrame:
    """Per-value cross-agent proportion test with BH-FDR, plus an omnibus test.

    Each value of ``level`` is tested as present/absent across agents on the
    failed-rollout denominator. Fisher is used when any expected count is below
    5, which is common here.
    """
    failed = error_analysis_set(clean)
    agents = sorted(failed["agent"].unique())
    denom = {a: int((failed.agent == a).sum()) for a in agents}
    rows = []
    for value in sorted(long[level].unique()):
        present = {a: long[(long.agent == a) & (long[level] == value)]["trajectory_id"].nunique()
                   for a in agents}
        table = np.array([[present[a], denom[a] - present[a]] for a in agents])
        if table.sum() == 0 or (table.sum(axis=0) == 0).any():
            continue
        try:
            chi2, p, dof, expected = sps.chi2_contingency(table)
            min_exp = float(expected.min())
        except ValueError:
            chi2, p, dof, min_exp = np.nan, np.nan, np.nan, np.nan
        if min_exp < 5 and len(agents) == 2:
            _, p = sps.fisher_exact(table)
        n = table.sum()
        v = float(np.sqrt((chi2 / n) / (min(table.shape) - 1))) if n and chi2 == chi2 else np.nan
        rows.append(dict(level=level, value=value, test="chi2_contingency",
                         statistic=chi2, dof=dof, p_value=p, cramers_v=v,
                         min_expected_count=min_exp,
                         counts="; ".join(f"{a}:{present[a]}/{denom[a]}" for a in agents),
                         note="expected<5, chi-square unreliable" if min_exp == min_exp and min_exp < 5 else ""))
    df = pd.DataFrame(rows)
    if not df.empty:
        df["q_value"] = bh_fdr(df["p_value"].tolist())
    return df


def top_labels_rank_correlation(long: pd.DataFrame, level: str) -> pd.DataFrame:
    """Spearman rho between agents' label-frequency rankings.

    A uniformly high rho indicates the error mix tracks the task set rather than
    the policy; this is descriptive only.
    """
    agents = sorted(long["agent"].unique())
    counts = {a: long[long.agent == a][level].value_counts() for a in agents}
    labels = sorted(set().union(*[set(c.index) for c in counts.values()]))
    rows = []
    for i in range(len(agents)):
        for j in range(i + 1, len(agents)):
            a, b = agents[i], agents[j]
            va = [counts[a].get(l, 0) for l in labels]
            vb = [counts[b].get(l, 0) for l in labels]
            rho, p = sps.spearmanr(va, vb)
            rows.append(dict(agent_a=a, agent_b=b, n_labels=len(labels),
                             spearman_rho=float(rho), p_value=float(p)))
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# 4. error depth
# --------------------------------------------------------------------------
def depth_sample_composition(clean: pd.DataFrame, low_n: int = 10) -> pd.DataFrame:
    """Where every failed rollout ends up in the depth analysis, per agent.

    Depth is defined only for failed rollouts with a recorded root cause. The
    remaining rows are reported by reason, never dropped without a count.
    """
    failed = clean[clean["task_score"] == 0]
    rows = []
    for agent, g in list(failed.groupby("agent")) + [("OVERALL", failed)]:
        rows.append(dict(
            agent=agent, n_failed=len(g),
            n_observed=int((g["depth_status"] == "observed").sum()),
            n_right_censored=int((g["depth_status"] == "right_censored").sum()),
            n_missing_root=int((g["depth_status"] == "missing_root").sum()),
            n_censor_time_missing=int(
                ((g["depth_status"] == "right_censored") & g["censor_time"].isna()).sum()),
            low_n_flag=LOW_N_MARK if len(g) < low_n else ""))
    return pd.DataFrame(rows)


def depth_descriptives(clean: pd.DataFrame, group_cols: list[str], low_n: int = 10) -> pd.DataFrame:
    """Descriptive statistics of OBSERVED (uncensored) depth by group.

    Censored rows are excluded here by necessity and counted in the companion
    composition table; the Kaplan-Meier summary is what uses them.
    """
    obs = clean[clean["depth_status"] == "observed"]
    rows = []
    grouped = list(obs.groupby(group_cols, dropna=False)) if group_cols else []
    for key, g in grouped:
        key = key if isinstance(key, tuple) else (key,)
        v = g["failure_depth"].to_numpy(dtype=float)
        row = dict(zip(group_cols, key))
        row.update(n_observed=len(v), mean=float(np.mean(v)),
                   sd=float(np.std(v, ddof=1)) if len(v) > 1 else np.nan,
                   median=float(np.median(v)),
                   q1=float(np.percentile(v, 25)), q3=float(np.percentile(v, 75)),
                   min=float(np.min(v)), max=float(np.max(v)),
                   low_n_flag=LOW_N_MARK if len(v) < low_n else "")
        rows.append(row)
    return pd.DataFrame(rows)


def depth_km_summary(clean: pd.DataFrame, group_col: str, low_n: int = 10) -> pd.DataFrame:
    """Kaplan-Meier summary of depth by group, treating "failure never becomes
    clear" as right censoring.

    event = the failure became clear; time = depth. A median that the survival
    curve never reaches is reported as ``not reached`` rather than as the
    largest observed value.
    """
    from lifelines import KaplanMeierFitter

    sub = clean[clean["depth_status"].isin(["observed", "right_censored"])]
    rows = []
    for key, g in list(sub.groupby(group_col, dropna=False)) + [("OVERALL", sub)]:
        t = g["depth_time"].to_numpy(dtype=float)
        e = g["depth_event"].to_numpy(dtype=float)
        ok = ~np.isnan(t) & ~np.isnan(e)
        t, e = t[ok], e[ok]
        if len(t) == 0:
            continue
        kmf = KaplanMeierFitter().fit(t, e)
        med = kmf.median_survival_time_
        ci = kmf.confidence_interval_
        try:
            from lifelines.utils import median_survival_times
            mci = median_survival_times(ci)
            lo = float(mci.iloc[0, 0]); hi = float(mci.iloc[0, 1])
        except Exception:
            lo = hi = float("nan")
        rows.append(dict(
            group_field=group_col, group_value=key, n=len(t),
            n_events=int(e.sum()), n_censored=int((e == 0).sum()),
            km_median="not reached" if not np.isfinite(med) else f"{med:g}",
            km_median_ci_low="not reached" if not np.isfinite(lo) else f"{lo:g}",
            km_median_ci_high="not reached" if not np.isfinite(hi) else f"{hi:g}",
            n_excluded_time_missing=int((~ok).sum()),
            low_n_flag=LOW_N_MARK if len(t) < low_n else ""))
    return pd.DataFrame(rows)


def depth_logrank(clean: pd.DataFrame, group_col: str) -> pd.DataFrame:
    """Multivariate log-rank test of depth across groups (censoring-aware)."""
    from lifelines.statistics import multivariate_logrank_test

    sub = clean[clean["depth_status"].isin(["observed", "right_censored"])].copy()
    sub = sub[sub["depth_time"].notna() & sub["depth_event"].notna()]
    if sub[group_col].nunique() < 2:
        return pd.DataFrame([dict(group_field=group_col, test="multivariate_logrank",
                                  statistic=np.nan, p_value=np.nan,
                                  note="fewer than two groups")])
    res = multivariate_logrank_test(sub["depth_time"], sub[group_col], sub["depth_event"])
    return pd.DataFrame([dict(group_field=group_col, test="multivariate_logrank",
                              statistic=float(res.test_statistic),
                              p_value=float(res.p_value),
                              n=len(sub), note="")])


def depth_bins_by_group(clean: pd.DataFrame, group_col: str, mapping: dict,
                        low_n: int = 10) -> pd.DataFrame:
    """Ordinal depth-bin proportions per group.

    Continuous depth is the difference of two noisy annotated step indices, so
    the binned view is the more robust one; ``CENSORED_UNRESOLVED`` is kept as
    its own bin instead of being assigned.
    """
    labels = list(mapping["depth_bins"]["labels"]) + ["CENSORED_UNRESOLVED"]
    sub = clean[clean["depth_bin"].notna()]
    rows = []
    for key, g in list(sub.groupby(group_col, dropna=False)) + [("OVERALL", sub)]:
        for lab in labels:
            rows.append(rate_row(int((g["depth_bin"] == lab).sum()), len(g), low_n,
                                 group_field=group_col, group_value=key, depth_bin=lab))
    return pd.DataFrame(rows)


def depth_by_error_type(long: pd.DataFrame, level: str, low_n: int = 10) -> pd.DataFrame:
    """Median depth per error label, censoring-aware, sorted latest-first.

    A rollout with k distinct labels contributes to k rows, so these groups
    overlap; the table describes labels, not a partition of rollouts.

    Rows are de-duplicated on (trajectory_id, level) first. Without that, a
    rollout carrying two labels of the same category would enter that category's
    Kaplan-Meier fit twice and inflate its denominator.
    """
    from lifelines import KaplanMeierFitter

    long = long.drop_duplicates(subset=["trajectory_id", level])
    rows = []
    for value, g in long.groupby(level):
        t = g["depth_time"].to_numpy(dtype=float)
        e = g["depth_event"].to_numpy(dtype=float)
        ok = ~np.isnan(t) & ~np.isnan(e)
        t, e = t[ok], e[ok]
        obs = g["failure_depth"].dropna().to_numpy(dtype=float)
        med = np.nan
        if len(t):
            med = KaplanMeierFitter().fit(t, e).median_survival_time_
        rows.append(dict(
            level=level, value=value, n_rollouts=g["trajectory_id"].nunique(),
            n_in_km=len(t), n_events=int(e.sum()) if len(e) else 0,
            km_median_depth="not reached" if len(t) and not np.isfinite(med)
            else (f"{med:g}" if len(t) else ""),
            observed_median_depth=float(np.median(obs)) if len(obs) else np.nan,
            observed_mean_depth=float(np.mean(obs)) if len(obs) else np.nan,
            low_n_flag=LOW_N_MARK if g["trajectory_id"].nunique() < low_n else ""))
    df = pd.DataFrame(rows)
    return df.sort_values("observed_median_depth", ascending=False, na_position="last")


def coverage_matrix(clean: pd.DataFrame) -> pd.DataFrame:
    """task_id x agent coverage; decides paired vs unpaired design."""
    m = pd.crosstab(clean["task_id"], clean["agent"])
    m["n_agents_covering"] = (m > 0).sum(axis=1)
    return m.reset_index()


def missingness(clean: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """Per-agent missing count and rate for each requested column."""
    rows = []
    for agent, g in list(clean.groupby("agent")) + [("OVERALL", clean)]:
        for c in columns:
            if c not in g.columns:
                rows.append(dict(agent=agent, column=c, n_rows=len(g),
                                 n_missing=len(g), missing_rate=1.0,
                                 note="column absent from clean table"))
                continue
            n_missing = int(g[c].isna().sum())
            rows.append(dict(agent=agent, column=c, n_rows=len(g), n_missing=n_missing,
                             missing_rate=n_missing / len(g) if len(g) else np.nan, note=""))
    return pd.DataFrame(rows)
