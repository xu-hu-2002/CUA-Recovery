from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def _fmt_rate(row) -> str:
    if row["denominator"] == 0 or not np.isfinite(row["rate"]):
        return f"— (0/0) {row.get('low_n_flag', '')}".strip()
    return (f"{row['rate']:.3f} ({int(row['numerator'])}/{int(row['denominator'])}, "
            f"95% CI {row['ci_low']:.3f}–{row['ci_high']:.3f}) "
            f"{row.get('low_n_flag', '')}").strip()


def _md_table(df: pd.DataFrame, cols: list[str] | None = None, floatfmt: str = "{:.3f}") -> str:
    d = df[cols] if cols else df
    if d.empty:
        return "_(empty)_\n"
    def cell(v):
        if isinstance(v, float):
            return "" if not np.isfinite(v) else floatfmt.format(v)
        return str(v)
    head = "| " + " | ".join(map(str, d.columns)) + " |"
    sep = "| " + " | ".join("---" for _ in d.columns) + " |"
    body = "\n".join("| " + " | ".join(cell(v) for v in r) + " |"
                     for r in d.itertuples(index=False))
    return f"{head}\n{sep}\n{body}\n"


def methodology_warnings(clean: pd.DataFrame, exclusions: pd.DataFrame,
                         coverage: pd.DataFrame, diagnostics: dict) -> str:
    pairs = (clean.groupby("agent")["annotator"].unique()
             .apply(lambda x: ", ".join(sorted(x))).to_dict())
    mapping_lines = "\n".join(f"  - `{a}` → `{v}`" for a, v in sorted(pairs.items()))

    excl_lines = ""
    if not exclusions.empty:
        counts = exclusions["reason"].value_counts()
        excl_lines = "\n".join(f"  - {r}: {c} rows" for r, c in counts.items())
    else:
        excl_lines = "  - (none)"

    cov_lines = []
    for agent, g in coverage.groupby("agent"):
        a = g[g.subset == "annotated"]
        u = g[g.subset == "not_annotated"]
        if a.empty or u.empty:
            continue
        cov_lines.append(
            f"  - `{agent}`: annotated n={int(a['n'].iloc[0])} "
            f"(self-report success {a['pct_success'].iloc[0]:.1%}, "
            f"no_terminate {a['pct_no_terminate'].iloc[0]:.1%}) vs "
            f"un-annotated n={int(u['n'].iloc[0])} "
            f"(success {u['pct_success'].iloc[0]:.1%}, "
            f"no_terminate {u['pct_no_terminate'].iloc[0]:.1%})")

    n_obs = int((clean["depth_status"] == "observed").sum())
    n_cens = int((clean["depth_status"] == "right_censored").sum())

    return f"""## 0.5 Methodological warnings

**Read before any result below.**

**Single annotator per agent, by construction.** Each agent in this dataset is
annotated by exactly one annotator:

{mapping_lines}

Labels written by any other annotator on these agents were removed as noise at
the user's instruction. Consequently every cross-agent difference is a composite
of an agent effect and an annotator effect, and the two are **not identifiable
under this design**. No cross-agent comparison below may be read as a difference
in model capability.

**No repeated annotation, therefore no inter-annotator agreement.** Because a
single annotator covers each agent, no κ / Krippendorff α / ICC can be computed.
The measurement error of every label is unknown.

**Labels are LLM-assisted, not independent human judgement.** The annotation
process used an LLM (codex) to pre-generate suggested labels which a human then
reviewed. Labels are therefore review decisions on machine proposals, and may
inherit the proposer's systematic biases.

**Error depth is a difference of two annotated step indices.** Its measurement
error is the sum of the errors of both endpoints. Without repeated annotation
its signal-to-noise ratio cannot be estimated. For this reason the binned
ordinal view of depth is more trustworthy than any regression on the continuous
value, and both are reported.

**The annotated subset is not a random sample of each agent's build.** Coverage
and self-report composition:

{chr(10).join(cov_lines) if cov_lines else "  - (not computed)"}

Where the two rows of an agent differ materially, that agent's rates describe
the labelled subset, not the agent's build.

**Rows excluded, with reasons:**

{excl_lines}

**Fields requested but absent from the data** (not computed, not substituted):
`human_verdict` (Perfect/Partial/Failure), `perfect_success`, and `confidence`.
The UI never recorded them, so the verdict distribution and the
confidence-restricted sensitivity analysis are omitted entirely.

**Depth censoring:** {n_obs} rollouts have an observed depth and {n_cens} are
right-censored ("the overall failure never becomes clear"). Censored rows are
retained via Kaplan-Meier and are never dropped or imputed.

**Unmapped error labels** under the `{diagnostics['category_source']}` grouping:
{', '.join('`' + l + '`' for l in diagnostics['unmapped_labels']) or '(none)'}.
These are reported as `UNMAPPED` rather than being assigned to a category.
"""


def _fact_task_score(clean: pd.DataFrame, ts: pd.DataFrame) -> str:
    agree = int((clean["task_score"] == clean["recomputed_task_score"]).sum())
    d = ts[ts.agent != "OVERALL"].sort_values("rate", ascending=False)
    hi, lo = d.iloc[0], d.iloc[-1]
    return (
        f"`task_score` equals \"all rubrics passed\" in {agree}/{len(clean)} rows "
        f"({agree / len(clean):.1%}). Task score is therefore a deterministic "
        f"function of the rubric scores in this dataset, not an independent "
        f"judgement, and sections 1 and 2 are not two independent measurements.\n\n"
        f"Observed range across agents: `{hi.agent}` {hi.rate:.3f} "
        f"({int(hi.numerator)}/{int(hi.denominator)}) to `{lo.agent}` {lo.rate:.3f} "
        f"({int(lo.numerator)}/{int(lo.denominator)}). The Wilson intervals of "
        f"these two do not overlap. Per the warning block, this gap is an agent "
        f"effect and an annotator effect combined; it is not attributable to "
        f"either alone.\n")


def _fact_macro_micro(rb: pd.DataFrame) -> str:
    lines = []
    for r in rb[rb.agent != "OVERALL"].itertuples():
        gap = r.macro_mean - r.micro_rate
        direction = "above" if gap > 0 else "below"
        lines.append(
            f"- `{r.agent}`: MACRO {r.macro_mean:.3f}, MICRO {r.micro_rate:.3f} "
            f"({int(r.micro_numerator)}/{int(r.micro_denominator)} rubrics); MACRO is "
            f"{abs(gap):.3f} {direction} MICRO.")
    return ("\n".join(lines) +
            "\n\nMACRO exceeding MICRO means the rollouts that scored well tend to "
            "belong to tasks defining fewer rubrics; the reverse means the opposite. "
            "Rubric counts per task range across the dataset, so the two figures "
            "answer different questions and neither is a correction of the other.\n")


def _fact_bimodality(dist: pd.DataFrame) -> str:
    lines = []
    for agent, g in dist.groupby("agent"):
        g = g.sort_values("bin_low")
        rates = g["rate"].to_numpy(dtype=float)
        if len(rates) < 3:
            continue
        lowest, highest, interior = rates[0], rates[-1], rates[1:-1]
        extremes = lowest + highest
        bimodal = extremes > 0.5 and lowest > interior.max() and highest > interior.max()
        peak_only_top = highest > 0.5
        verdict = ("bimodal (mass piles at both ends)" if bimodal
                   else "single-peaked at the top decile" if peak_only_top
                   else "not bimodal by this rule")
        lines.append(
            f"- `{agent}`: bottom decile {lowest:.1%}, top decile {highest:.1%}, "
            f"largest interior decile {interior.max():.1%} -> **{verdict}**.")
    return ("\n".join(lines) +
            "\n\nRule applied: bimodal iff the two extreme deciles together exceed "
            "50% of the mass *and* each individually exceeds every interior decile. "
            "Where a distribution is bimodal or piled at one end, its mean is a poor "
            "summary and the histogram above should be read instead.\n")


def _fact_error_types(rates: pd.DataFrame, card: pd.DataFrame) -> str:
    lines = []
    for agent, g in rates[(rates.level == "error_type") & (rates.agent != "OVERALL")].groupby("agent"):
        top = g.sort_values("rate", ascending=False).head(3)
        c = card[card.agent == agent]
        items = "; ".join(
            f"`{r.value}` {r.rate:.1%} ({int(r.numerator)}/{int(r.denominator)}){r.low_n_flag}"
            for r in top.itertuples())
        mean_c = float(c["mean"].iloc[0]) if not c.empty else float("nan")
        lines.append(f"- `{agent}` top three: {items}. Mean labels per failed "
                     f"rollout: {mean_c:.2f}.")
    return ("\n".join(lines) +
            "\n\nLabels are multi-select, so these rates are not mutually exclusive "
            "and do not sum to 1. Differences in mean label count per rollout also "
            "mean the agents' label totals are not on a common scale.\n")


def _fact_rank_corr(rc: pd.DataFrame) -> str:
    lines = [f"- `{r.agent_a}` vs `{r.agent_b}`: ρ = {r.spearman_rho:.3f} "
             f"(p = {r.p_value:.3f}, {int(r.n_labels)} labels)"
             for r in rc.itertuples()]
    rhos = rc["spearman_rho"].to_numpy(dtype=float)
    if np.nanmax(rhos) < 0.5:
        verdict = ("All pairwise ρ are below 0.5 and none reaches significance, so "
                   "the label rankings do **not** agree across agents. The error mix "
                   "therefore does not look like a pure property of the shared task "
                   "set.")
    else:
        verdict = ("Rank agreement is substantial for at least one pair, which is "
                   "consistent with the error mix partly tracking the task set.")
    return ("\n".join(lines) + "\n\n" + verdict +
            " This comparison cannot separate a genuine difference in error mix from "
            "a difference in labelling habit between the three annotators, because "
            "each agent has exactly one annotator.\n")


def _fact_depth(km: pd.DataFrame, comp: pd.DataFrame, lr: pd.DataFrame) -> str:
    lines = []
    for r in km[km.group_value != "OVERALL"].itertuples():
        c = comp[comp.agent == r.group_value]
        cens = int(c["n_right_censored"].iloc[0]) if not c.empty else 0
        lines.append(
            f"- `{r.group_value}`: KM median depth {r.km_median} "
            f"(95% CI {r.km_median_ci_low}–{r.km_median_ci_high}), "
            f"n = {int(r.n)}, censored = {cens}{(' ' + r.low_n_flag) if r.low_n_flag else ''}")
    p = float(lr["p_value"].iloc[0]) if not lr.empty else float("nan")
    return ("\n".join(lines) +
            f"\n\nLog-rank test across agents: p = {p:.3g}. "
            "Depth counts actions between two human-annotated indices, so it is "
            "measured on each agent's own action stream; agents whose rollouts run "
            "longer have more room for a large depth, and this table does not "
            "normalise for that.\n")


def _fact_weights(prof: pd.DataFrame) -> str:
    present = prof[prof["weights_present"]]
    n = len(present)
    if n == 0:
        return "No rubric weights are present, so no weighted score is computable.\n"
    uni = int(present["uniform"].sum())
    non = n - uni
    sums = present["weight_sum"]
    off = int((~np.isclose(sums, 1.0, atol=1e-3)).sum())
    ratio = present.loc[~present["uniform"], "weight_ratio"]
    rmax = float(ratio.max()) if len(ratio) else float("nan")
    return (
        f"Weight vectors are present for {n} of {len(prof)} rollouts. "
        f"**{non} use non-uniform weights** and {uni} weight their rubrics "
        f"uniformly. Because most tasks weight rubrics unevenly, the weighted "
        f"score is not algebraically identical to the unweighted ratio and can "
        f"carry independent information.\n\n"
        f"Among non-uniform tasks the heaviest rubric outweighs the lightest by "
        f"up to {rmax:.2f}×. Weight sums deviate from 1.0 by more than 1e-3 in "
        f"{off} rollouts, so the score is normalised by Σ(weight) rather than "
        f"assuming the weights already sum to one.\n")


def _fact_weighted_gap(rb: pd.DataFrame) -> str:
    d = rb[rb.agent != "OVERALL"]
    max_shift = float(np.nanmax(np.abs(d["weighted_minus_macro_mean"])))
    max_row = float(np.nanmax(d["weighted_minus_macro_maxabs"]))
    min_rho = float(np.nanmin(d["spearman_macro_vs_weighted"]))
    macro_order = d.sort_values("macro_mean", ascending=False)["agent"].tolist()
    weight_order = d.sort_values("weighted_mean", ascending=False)["agent"].tolist()
    same = macro_order == weight_order
    return (
        f"\nThe largest shift in any agent's mean is {max_shift:.4f}; the largest "
        f"shift for any single rollout is {max_row:.4f}. Spearman ρ between the "
        f"weighted and unweighted scores is at least {min_rho:.3f} for every "
        f"agent. The agent ordering is "
        f"{'**unchanged**' if same else '**changed**'} by weighting "
        f"({' > '.join(macro_order)} unweighted; {' > '.join(weight_order)} "
        f"weighted).\n\n"
        f"Weighting therefore moves the decimals but not the conclusions in this "
        f"dataset. Both figures are reported so that neither is presented as the "
        f"single authoritative rubric score.\n")


def build_report(tables: dict[str, pd.DataFrame], figures: dict[str, Path],
                 clean: pd.DataFrame, exclusions: pd.DataFrame,
                 diagnostics: dict) -> str:
    parts: list[str] = []
    A = parts.append

    A("# DERAIL human-annotation statistics\n")
    A("Scope: task score, rubric score, root-cause (error type) distribution, and "
      "error depth, for the three agents configured in `field_mapping.yaml`. "
      "Reversibility, clean prefix, sole-blocker and co-occurrence analyses were "
      "explicitly out of scope for this run and are not computed.\n")

    A("## 0. Data hygiene and sample size\n")
    A("### 0.1 Analysis set\n")
    A(_md_table(tables["sample_size"]))
    A("### 0.2 Annotation coverage (labelled vs unlabelled)\n")
    A(_md_table(tables["annotation_coverage"]))
    A("### 0.3 task × agent coverage — paired or unpaired?\n")
    cov = tables["coverage_overlap"]
    A(_md_table(cov))
    n_all = int(tables["coverage_matrix"]["n_agents_covering"].eq(
        clean["agent"].nunique()).sum())
    A(f"\n**Conclusion: UNPAIRED design.** Only {n_all} task_ids are covered by all "
      f"{clean['agent'].nunique()} agents, out of "
      f"{tables['coverage_matrix'].shape[0]} distinct task_ids in the analysis set. "
      "Paired tests (Cochran's Q, McNemar) are therefore not applicable; all "
      "cross-agent testing below uses chi-square with pairwise Fisher and "
      "Benjamini-Hochberg FDR correction.\n")
    A("### 0.4 Missingness by field\n")
    A(_md_table(tables["missingness"]))

    A(methodology_warnings(clean, exclusions, tables["annotation_coverage"], diagnostics))

    A("\n## 1. Task score\n")
    A("### 1.1 Task success rate by agent\n")
    A(_md_table(tables["task_score_by_agent"],
                ["agent", "numerator", "denominator", "rate", "ci_low", "ci_high",
                 "low_n_flag"]))
    ts = tables["task_score_by_agent"]
    lines = [f"- `{r.agent}`: {_fmt_rate(r._asdict())}" for r in ts.itertuples()]
    A("\n".join(lines) + "\n")
    A(_fact_task_score(clean, ts))
    A(f"\n![task score forest]({figures['task_forest'].name})\n")
    A("### 1.2 Cross-agent test (unpaired)\n")
    A(_md_table(tables["task_score_test"]))
    A("### 1.3 Task success by task category\n")
    A(_md_table(tables["task_score_by_task_category"],
                ["agent", "group_value", "numerator", "denominator", "rate",
                 "ci_low", "ci_high", "low_n_flag"]))

    A("\n## 2. Rubric score\n")
    A("### 2.1 MACRO and MICRO\n")
    A("MACRO averages each rollout's pass ratio, so every rollout weighs the same. "
      "MICRO pools rubrics, so tasks defining more rubrics weigh more. They differ "
      "here because rubric counts per task range from 3 to 17.\n")
    A(_md_table(tables["rubric_score_by_agent"],
                ["agent", "n_rollouts", "macro_mean", "macro_sd", "macro_median",
                 "macro_q1", "macro_q3", "macro_boot_ci_low", "macro_boot_ci_high",
                 "micro_numerator", "micro_denominator", "micro_rate",
                 "micro_ci_low", "micro_ci_high", "low_n_flag"]))
    A("### 2.2 Weighted rubric score\n")
    A("Not a recorded label: computed as Σ(score·weight)/Σ(weight) from "
      "`task_config.json:grading.rubrics[*].weight`. No human ever reviewed such "
      "a number, so it is a derived quantity, not an annotation.\n")
    A(_fact_weights(tables["rubric_weight_profile"]))
    A(_md_table(tables["rubric_score_by_agent"],
                ["agent", "weighted_mean", "weighted_sd", "weighted_median",
                 "weighted_q1", "weighted_q3", "weighted_boot_ci_low",
                 "weighted_boot_ci_high", "weighted_n_missing"]))
    A("\n**Agreement with the unweighted MACRO ratio:**\n")
    A(_md_table(tables["rubric_score_by_agent"],
                ["agent", "macro_mean", "weighted_mean", "weighted_minus_macro_mean",
                 "weighted_minus_macro_maxabs", "spearman_macro_vs_weighted",
                 "spearman_p"]))
    A(_fact_weighted_gap(tables["rubric_score_by_agent"]))
    A("\n**Distribution of the weighted score:**\n")
    A(_md_table(tables["weighted_ratio_distribution"],
                ["agent", "bin_low", "bin_high", "numerator", "denominator",
                 "rate", "ci_low", "ci_high", "low_n_flag"]))
    A(f"\n![weighted distribution]({figures['weighted_hist'].name})\n")
    A(f"\n![weighted vs unweighted]({figures['weighted_scatter'].name})\n")
    A("\n**Rollouts where weighting moves the score the most:**\n")
    A(_md_table(tables["weighted_vs_unweighted_gap"]))
    A("### 2.3 Distribution of rubric pass ratio\n")
    A(_md_table(tables["rubric_ratio_distribution"],
                ["agent", "bin_low", "bin_high", "numerator", "denominator",
                 "rate", "ci_low", "ci_high", "low_n_flag"]))
    A(f"\n![rubric distribution]({figures['rubric_hist'].name})\n")
    A("**Is the distribution bimodal?**\n")
    A(_fact_bimodality(tables["rubric_ratio_distribution"]))
    A("### 2.4 Rubric-slot pass rate by agent\n")
    A("Rubric ids are positional inside each task's bundle, so a row pools "
      "different criteria across tasks. Read it as a slot-position view only.\n")
    A(f"\n![rubric heatmap]({figures['rubric_heatmap'].name})\n")

    A("\n## 3. Root cause: error type distribution\n")
    A("Analysis set: rollouts with `task_score == 0` carrying a failure "
      "annotation. Labels are multi-select, so rollout-normalised rates sum to "
      "more than 1 across labels; that is correct and is not rescaled.\n")
    A("### 3.0 Label normalisation applied\n")
    A("Edits applied to the raw labels before any counting, per "
      "`label_normalization` in `field_mapping.yaml`:\n")
    nlog = tables["label_normalization_log"]
    if not nlog.empty:
        summary = (nlog.groupby(["action", "original_label", "new_label"])
                   .size().reset_index(name="n_rollouts"))
        A(_md_table(summary))
        A("\nAffected rollouts, in full:\n")
        A(_md_table(nlog))
    else:
        A("_(none configured)_\n")
    A("\n**Rollouts left with no label after normalisation** are excluded from "
      "the denominators in 3.1–3.5 and reported here instead. They are not "
      "counted as rollouts with zero errors.\n")
    A(_md_table(tables["label_missing_by_agent"],
                ["agent", "numerator", "denominator", "rate", "ci_low", "ci_high",
                 "low_n_flag"]))
    A("### 3.1 Error type, rollout-normalised and label-normalised\n")
    A(_md_table(tables["error_types_by_agent"],
                ["agent", "value", "numerator", "denominator", "rate", "ci_low",
                 "ci_high", "label_occurrences", "label_normalized", "low_n_flag"]))
    A(f"\n![error types]({figures['error_bars'].name})\n")
    A(_fact_error_types(tables["error_types_by_agent"], tables["label_cardinality"]))
    A("### 3.2 Error category\n")
    A(_md_table(tables["error_categories_by_agent"],
                ["agent", "value", "numerator", "denominator", "rate", "ci_low",
                 "ci_high", "label_occurrences", "label_normalized", "low_n_flag"]))
    A(f"\n![error categories]({figures['error_stacked'].name})\n")
    A("### 3.3 Labels per failed rollout\n")
    A(_md_table(tables["label_cardinality"]))
    A("### 3.4 Cross-agent homogeneity per label (BH-FDR)\n")
    A(_md_table(tables["error_type_homogeneity"]))
    A("### 3.5 Rank correlation of label frequency between agents\n")
    A(_md_table(tables["error_rank_correlation"]))
    A(_fact_rank_corr(tables["error_rank_correlation"]))

    A("\n## 4. Error depth\n")
    A("Depth = `clear_failure_step − root_cause_step`, both 0-based action "
      "indices. Computed only for failed rollouts with a recorded root cause. "
      "Rows where the failure never becomes clear are right-censored and enter "
      "the Kaplan-Meier estimate rather than being discarded.\n")
    A("**Measurement caveat, required reading:** continuous depth is the "
      "difference of two noisy annotated step indices, so their measurement "
      "variances add. The binned ordinal analysis in 4.5 is therefore more robust "
      "than any regression on the continuous value, and should be preferred when "
      "the two disagree.\n")
    A("### 4.1 Sample composition\n")
    A(_md_table(tables["depth_sample_composition"]))
    A("### 4.2 Observed depth by agent\n")
    A(_md_table(tables["depth_by_agent"]))
    A(f"\n![depth histogram]({figures['depth_hist'].name})\n")
    A("### 4.3 Kaplan-Meier (censoring-aware) by agent\n")
    A(_md_table(tables["depth_km_by_agent"]))
    A(_md_table(tables["depth_logrank_agent"]))
    A(_fact_depth(tables["depth_km_by_agent"], tables["depth_sample_composition"],
                  tables["depth_logrank_agent"]))
    A(f"\n![depth KM by agent]({figures['depth_km_agent'].name})\n")
    A("### 4.4 Depth by error category and by task category\n")
    A(_md_table(tables["depth_by_error_category"]))
    A(_md_table(tables["depth_km_by_error_category"]))
    A(f"\n![depth KM by error category]({figures['depth_km_errcat'].name})\n")
    A(_md_table(tables["depth_by_task_category"]))
    A(_md_table(tables["depth_km_by_task_category"]))
    A("### 4.5 Ordinal depth bins\n")
    A(_md_table(tables["depth_bins_by_agent"],
                ["group_value", "depth_bin", "numerator", "denominator", "rate",
                 "ci_low", "ci_high", "low_n_flag"]))
    A(f"\n![depth bins]({figures['depth_bins'].name})\n")
    A("### 4.6 Depth by individual error type (latest-surfacing first)\n")
    A("A rollout carrying k labels contributes to k rows, so these groups overlap "
      "and do not partition the failed rollouts.\n")
    A(_md_table(tables["depth_by_error_type"]))

    A("\n## Appendix: internal inconsistencies\n")
    A(_md_table(tables["inconsistencies"]))
    A("\n## Appendix: excluded rows\n")
    A(_md_table(exclusions.groupby("reason").size().reset_index(name="n_rows")
                if not exclusions.empty else pd.DataFrame()))
    return "\n".join(parts)


def build_sanity(tables: dict[str, pd.DataFrame], clean: pd.DataFrame,
                 exclusions: pd.DataFrame, checks: list[dict]) -> str:
    parts = ["# Sanity checks\n",
             "Each row is a check that was actually executed. `PASS`/`FAIL` is "
             "computed, not asserted by hand.\n"]
    df = pd.DataFrame(checks)
    parts.append(_md_table(df))
    parts.append("\n## Minimum cell frequencies of the reported tests\n")
    parts.append(_md_table(tables["task_score_test"],
                           ["comparison", "test", "p_value", "q_value",
                            "min_expected_count", "note"]))
    parts.append("\n## Internal inconsistencies (full list)\n")
    parts.append(_md_table(tables["inconsistencies"]))
    parts.append("\n## Sensitivity analysis\n")
    parts.append(
        "The requested confidence-restricted re-run (`confidence == 高`) **cannot "
        "be performed**: the annotation UI never collected an annotator confidence "
        "field, so no such subset exists. No substitute stratifier was invented.\n")
    return "\n".join(parts)
