"""Phase 3 figures. matplotlib only, one figure per call, no explicit colours,
300 dpi PNG. No seaborn.

Every figure is a direct rendering of a table written alongside it; none of them
smooths, interpolates or extrapolates.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

DPI = 300


def _save(fig, path: Path) -> Path:
    fig.tight_layout()
    fig.savefig(path, dpi=DPI)
    plt.close(fig)
    return path


def task_score_forest(task_tbl: pd.DataFrame, out: Path) -> Path:
    """Forest plot of task success rate with Wilson 95% CI, one row per agent."""
    d = task_tbl.sort_values("rate", na_position="first").reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(8, 0.6 * len(d) + 2))
    y = np.arange(len(d))
    ax.errorbar(d["rate"], y,
                xerr=[d["rate"] - d["ci_low"], d["ci_high"] - d["rate"]],
                fmt="o", capsize=4)
    labels = [f"{r.agent} ({r.numerator}/{r.denominator}) {r.low_n_flag}".strip()
              for r in d.itertuples()]
    ax.set_yticks(y); ax.set_yticklabels(labels)
    ax.set_xlim(0, 1); ax.set_xlabel("task success rate (Wilson 95% CI)")
    ax.set_title("Task score by agent")
    ax.grid(axis="x", alpha=0.3)
    return _save(fig, out)


def rubric_ratio_hist(clean: pd.DataFrame, out: Path) -> Path:
    """Faceted histogram of rubric_pass_ratio, one panel per agent.

    Counts in fixed 0.1 bins so that piling at 0 and 1 (bimodality) is visible.
    """
    agents = sorted(clean["agent"].unique())
    fig, axes = plt.subplots(len(agents), 1, figsize=(8, 2.4 * len(agents)), sharex=True)
    if len(agents) == 1:
        axes = [axes]
    edges = np.round(np.arange(0, 1.0001, 0.1), 3)
    for ax, agent in zip(axes, agents):
        v = clean[clean.agent == agent]["rubric_pass_ratio"].to_numpy(dtype=float)
        ax.hist(v, bins=edges)
        ax.set_ylabel("rollouts")
        ax.set_title(f"{agent}  (n={len(v)}, mean={np.nanmean(v):.3f})")
        ax.grid(axis="y", alpha=0.3)
    axes[-1].set_xlabel("rubric pass ratio")
    fig.suptitle("Rubric pass ratio distribution by agent", y=1.0)
    return _save(fig, out)


def error_type_stacked(long: pd.DataFrame, out: Path) -> Path:
    """Stacked bar of error-category share per agent (label-normalised)."""
    pivot = (long.groupby(["agent", "error_category"]).size()
             .unstack(fill_value=0))
    shares = pivot.div(pivot.sum(axis=1), axis=0)
    fig, ax = plt.subplots(figsize=(9, 5))
    bottom = np.zeros(len(shares))
    for cat in shares.columns:
        ax.bar(shares.index, shares[cat], bottom=bottom, label=cat)
        bottom += shares[cat].to_numpy()
    ax.set_ylabel("share of error-label occurrences")
    ax.set_title("Error category composition by agent (label-normalised)")
    ax.legend(loc="center left", bbox_to_anchor=(1.0, 0.5))
    ax.grid(axis="y", alpha=0.3)
    return _save(fig, out)


def error_type_bars(rates: pd.DataFrame, out: Path) -> Path:
    """Rollout-normalised error-label rate per agent, grouped bars with CI."""
    d = rates[(rates.level == "error_type") & (rates.agent != "OVERALL")]
    order = (rates[(rates.level == "error_type") & (rates.agent == "OVERALL")]
             .sort_values("rate", ascending=False)["value"].tolist())
    agents = sorted(d["agent"].unique())
    fig, ax = plt.subplots(figsize=(11, 6))
    width = 0.8 / len(agents)
    x = np.arange(len(order))
    for i, agent in enumerate(agents):
        sub = d[d.agent == agent].set_index("value").reindex(order)
        vals = sub["rate"].to_numpy(dtype=float)
        lo = np.nan_to_num(vals - sub["ci_low"].to_numpy(dtype=float))
        hi = np.nan_to_num(sub["ci_high"].to_numpy(dtype=float) - vals)
        ax.bar(x + i * width, np.nan_to_num(vals), width, label=agent,
               yerr=[lo, hi], capsize=2)
    ax.set_xticks(x + 0.4 - width / 2)
    ax.set_xticklabels(order, rotation=45, ha="right")
    ax.set_ylabel("P(label present | failed rollout)")
    ax.set_title("Error type distribution, rollout-normalised (Wilson 95% CI)")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    return _save(fig, out)


def depth_km(clean: pd.DataFrame, group_col: str, out: Path, title: str) -> Path:
    """Kaplan-Meier curves of failure depth by group.

    y = P(failure not yet clear); censoring = "failure never becomes clear".
    """
    from lifelines import KaplanMeierFitter

    sub = clean[clean["depth_status"].isin(["observed", "right_censored"])]
    fig, ax = plt.subplots(figsize=(9, 5.5))
    plotted = 0
    for key, g in sub.groupby(group_col, dropna=False):
        t = g["depth_time"].to_numpy(dtype=float)
        e = g["depth_event"].to_numpy(dtype=float)
        ok = ~np.isnan(t) & ~np.isnan(e)
        if ok.sum() == 0:
            continue
        kmf = KaplanMeierFitter().fit(t[ok], e[ok],
                                      label=f"{key} (n={int(ok.sum())}, "
                                            f"cens={int((e[ok] == 0).sum())})")
        kmf.plot_survival_function(ax=ax, ci_show=True)
        plotted += 1
    ax.set_xlabel("failure depth (actions from root cause to clear failure)")
    ax.set_ylabel("P(failure not yet clear)")
    ax.set_title(title)
    ax.grid(alpha=0.3)
    if plotted == 0:
        ax.text(0.5, 0.5, "no eligible rows", ha="center", transform=ax.transAxes)
    return _save(fig, out)


def depth_hist(clean: pd.DataFrame, out: Path) -> Path:
    """Histogram of observed (uncensored) depth, one panel per agent."""
    obs = clean[clean["depth_status"] == "observed"]
    agents = sorted(obs["agent"].unique())
    fig, axes = plt.subplots(len(agents), 1, figsize=(9, 2.4 * len(agents)), sharex=True)
    if len(agents) == 1:
        axes = [axes]
    for ax, agent in zip(axes, agents):
        v = obs[obs.agent == agent]["failure_depth"].to_numpy(dtype=float)
        ax.hist(v, bins=30)
        ax.set_ylabel("rollouts")
        ax.set_title(f"{agent}  (n={len(v)}, median={np.median(v):.0f})" if len(v)
                     else f"{agent}  (n=0)")
        ax.grid(axis="y", alpha=0.3)
    axes[-1].set_xlabel("failure depth (observed only)")
    fig.suptitle("Observed failure depth by agent", y=1.0)
    return _save(fig, out)


def depth_bins_bar(bins_tbl: pd.DataFrame, out: Path, title: str) -> Path:
    """Stacked proportion bars of the ordinal depth bins per group."""
    d = bins_tbl[bins_tbl.group_value != "OVERALL"]
    pivot = d.pivot_table(index="group_value", columns="depth_bin",
                          values="rate", aggfunc="first").fillna(0)
    fig, ax = plt.subplots(figsize=(9, 5))
    bottom = np.zeros(len(pivot))
    for col in pivot.columns:
        ax.bar(pivot.index.astype(str), pivot[col], bottom=bottom, label=col)
        bottom += pivot[col].to_numpy()
    ax.set_ylabel("share of rollouts")
    ax.set_title(title)
    ax.legend(loc="center left", bbox_to_anchor=(1.0, 0.5))
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
    ax.grid(axis="y", alpha=0.3)
    return _save(fig, out)


def rubric_heatmap(matrix_tbl: pd.DataFrame, out: Path) -> Path:
    """Heatmap of rubric-slot pass rate (rows) by agent (columns).

    Slots are positional within each task's bundle, so a row mixes criteria.
    """
    d = matrix_tbl[matrix_tbl.agent != "OVERALL"]
    pivot = d.pivot_table(index="rubric_id", columns="agent", values="rate")
    pivot = pivot.reindex(sorted(pivot.index, key=lambda x: int(x[1:])))
    fig, ax = plt.subplots(figsize=(1.8 * len(pivot.columns) + 3, 0.4 * len(pivot) + 3))
    im = ax.imshow(pivot.to_numpy(dtype=float), aspect="auto", vmin=0, vmax=1)
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns, rotation=45, ha="right")
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index)
    n = d.pivot_table(index="rubric_id", columns="agent", values="denominator")
    n = n.reindex(pivot.index)
    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            v = pivot.iat[i, j]
            if v == v:
                ax.text(j, i, f"{v:.2f}\nn={int(n.iat[i, j])}", ha="center",
                        va="center", fontsize=7)
    fig.colorbar(im, ax=ax, label="pass rate")
    ax.set_title("Rubric-slot pass rate by agent")
    return _save(fig, out)


def coverage_bar(cov: pd.DataFrame, out: Path) -> Path:
    """Annotated vs un-annotated self-report composition, per agent.

    This is the selection-bias figure: a large gap between the two bars of an
    agent means its labelled subset is not representative of its build.
    """
    agents = sorted(cov["agent"].unique())
    levels = ["success", "failure", "no_terminate"]
    fig, ax = plt.subplots(figsize=(10, 5))
    xs, labels = [], []
    pos = 0
    for agent in agents:
        for subset in ("annotated", "not_annotated"):
            r = cov[(cov.agent == agent) & (cov.subset == subset)]
            if r.empty:
                continue
            bottom = 0.0
            for lev in levels:
                v = float(r[f"pct_{lev}"].iloc[0])
                ax.bar(pos, v, bottom=bottom,
                       label=lev if pos == 0 else "_nolegend_")
                bottom += v
            xs.append(pos)
            labels.append(f"{agent}\n{subset}\nn={int(r['n'].iloc[0])}")
            pos += 1
        pos += 0.6
    ax.set_xticks(xs); ax.set_xticklabels(labels, fontsize=7)
    ax.set_ylabel("share of rollouts (agent self-report)")
    ax.set_title("Annotation coverage: labelled vs unlabelled rollouts")
    ax.legend(loc="center left", bbox_to_anchor=(1.0, 0.5))
    ax.grid(axis="y", alpha=0.3)
    return _save(fig, out)


def weighted_vs_unweighted_scatter(clean: pd.DataFrame, out: Path) -> Path:
    """Weighted vs unweighted rubric score, one point per rollout.

    The diagonal is where weighting changes nothing; vertical distance from it
    is the effect of the task's weight vector on that rollout.
    """
    fig, ax = plt.subplots(figsize=(7, 7))
    for agent, g in clean.groupby("agent"):
        ax.scatter(g["rubric_pass_ratio"], g["weighted_rubric_score"],
                   s=18, alpha=0.6, label=f"{agent} (n={len(g)})")
    ax.plot([0, 1], [0, 1], linestyle="--", linewidth=1)
    ax.set_xlabel("unweighted rubric pass ratio")
    ax.set_ylabel("weighted rubric score")
    ax.set_title("Weighted vs unweighted rubric score (dashed = no effect)")
    ax.set_xlim(-0.02, 1.02); ax.set_ylim(-0.02, 1.02)
    ax.legend(); ax.grid(alpha=0.3)
    return _save(fig, out)


def weighted_ratio_hist(clean: pd.DataFrame, out: Path) -> Path:
    """Histogram of the weighted rubric score, one panel per agent."""
    agents = sorted(clean["agent"].unique())
    fig, axes = plt.subplots(len(agents), 1, figsize=(8, 2.4 * len(agents)), sharex=True)
    if len(agents) == 1:
        axes = [axes]
    edges = np.round(np.arange(0, 1.0001, 0.1), 3)
    for ax, agent in zip(axes, agents):
        v = clean[clean.agent == agent]["weighted_rubric_score"].to_numpy(dtype=float)
        v = v[~np.isnan(v)]
        ax.hist(v, bins=edges)
        ax.set_ylabel("rollouts")
        ax.set_title(f"{agent}  (n={len(v)}, mean={np.mean(v):.3f})" if len(v)
                     else f"{agent}  (n=0)")
        ax.grid(axis="y", alpha=0.3)
    axes[-1].set_xlabel("weighted rubric score")
    fig.suptitle("Weighted rubric score distribution by agent", y=1.0)
    return _save(fig, out)
