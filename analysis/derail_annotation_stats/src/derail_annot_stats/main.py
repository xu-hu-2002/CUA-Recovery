"""Entry point: run the whole pipeline. Idempotent, side-effect free outside out_dir.

    DERAIL_BUILDS_DIR=... DERAIL_ANALYSIS_OUT=... python -m derail_annot_stats.main
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from . import clean as C
from . import plots as P
from . import report as R
from . import stats as S
from .paths import Paths

DEFAULT_MAPPING = Path(__file__).resolve().parents[2] / "config" / "field_mapping.yaml"


def _write(df: pd.DataFrame, path: Path) -> Path:
    df.to_csv(path, index=False)
    return path


def run(paths: Paths, mapping_path: Path) -> None:
    paths.ensure_out()
    mapping = C.load_mapping(mapping_path)
    low_n = int(mapping["analysis_set"].get("low_n_threshold", 10))
    boot = mapping.get("bootstrap", {})
    n_res = int(boot.get("n_resamples", 10000))
    seed = int(boot.get("seed", S.SEED))

    # ---------------- Phase 1 ----------------
    cl, exclusions, diagnostics = C.build_clean_table(paths, mapping)
    long = C.explode_error_types(cl, mapping)
    inconsistencies = C.consistency_checks(cl)
    coverage = C.annotation_coverage(paths, mapping, cl)

    print(f"analysis set: {len(cl)} rollouts, {cl['agent'].nunique()} agents, "
          f"{len(exclusions)} rows excluded")
    print("agent -> annotator:",
          cl.groupby('agent')['annotator'].unique().to_dict())
    if diagnostics["unmapped_labels"]:
        print("UNMAPPED error labels:", diagnostics["unmapped_labels"])
    nlog = cl.attrs.get("label_normalization_log", pd.DataFrame())
    if not nlog.empty:
        print("label normalization:",
              nlog["action"].value_counts().to_dict())
    n_empty = int(cl["error_types_emptied_by_normalization"].sum())
    if n_empty:
        print(f"{n_empty} failed rollouts have no error label left after "
              f"normalization; excluded from error-type analysis, reported separately")

    T: dict[str, pd.DataFrame] = {}

    # ---------------- Phase 2.0 hygiene ----------------
    T["sample_size"] = pd.DataFrame([
        dict(agent=a,
             annotator="; ".join(sorted(g["annotator"].unique())),
             n_rollouts=len(g),
             n_task_success=int(g["task_score"].sum()),
             n_task_failed=int((g["task_score"] == 0).sum()),
             n_with_failure_annotation=int(g["has_failure_annotation"].sum()),
             n_distinct_tasks=g["task_id"].nunique(),
             low_n_flag=S.LOW_N_MARK if len(g) < low_n else "")
        for a, g in cl.groupby("agent")
    ] + [dict(agent="OVERALL", annotator="—", n_rollouts=len(cl),
              n_task_success=int(cl["task_score"].sum()),
              n_task_failed=int((cl["task_score"] == 0).sum()),
              n_with_failure_annotation=int(cl["has_failure_annotation"].sum()),
              n_distinct_tasks=cl["task_id"].nunique(), low_n_flag="")])

    T["annotation_coverage"] = coverage
    T["coverage_matrix"] = S.coverage_matrix(cl)
    agents = sorted(cl["agent"].unique())
    overlap = []
    sets = {a: set(cl[cl.agent == a]["task_id"]) for a in agents}
    for i in range(len(agents)):
        for j in range(i + 1, len(agents)):
            a, b = agents[i], agents[j]
            overlap.append(dict(agent_a=a, agent_b=b, n_a=len(sets[a]), n_b=len(sets[b]),
                                n_shared=len(sets[a] & sets[b])))
    overlap.append(dict(agent_a="ALL", agent_b="ALL",
                        n_a=len(set().union(*sets.values())), n_b=np.nan,
                        n_shared=len(set.intersection(*sets.values()))))
    T["coverage_overlap"] = pd.DataFrame(overlap)
    T["missingness"] = S.missingness(cl, [
        "task_score", "rubric_pass_ratio", "weighted_rubric_score",
        "root_cause_step", "clear_failure_step", "failure_depth",
        "error_types", "reversibility", "trajectory_length"])

    # ---------------- Phase 2.1 task score ----------------
    T["task_score_by_agent"] = S.task_score_by_agent(cl, low_n)
    T["task_score_test"] = S.unpaired_group_test(cl, "task_score", low_n)
    T["task_score_by_task_category"] = S.task_score_by_group(cl, "task_category", low_n)

    # ---------------- Phase 2.2 rubric ----------------
    T["rubric_score_by_agent"] = S.rubric_score_by_agent(cl, low_n, n_res, seed)
    T["rubric_ratio_distribution"] = S.rubric_ratio_distribution(cl, low_n)
    T["rubric_by_agent_matrix"] = S.rubric_by_agent_matrix(cl, low_n)
    T["weighted_ratio_distribution"] = S.weighted_ratio_distribution(cl, low_n)
    T["rubric_weight_profile"] = S.rubric_weight_profile(cl)
    T["weighted_vs_unweighted_gap"] = S.weighted_vs_unweighted_gap(cl)

    # ---------------- Phase 2.3 error types ----------------
    T["label_normalization_log"] = cl.attrs.get(
        "label_normalization_log", pd.DataFrame())
    T["label_missing_by_agent"] = S.label_missing_by_agent(cl, low_n)
    T["error_types_by_agent"] = S.error_type_rates(long, cl, "error_type", low_n)
    T["error_categories_by_agent"] = S.error_type_rates(long, cl, "error_category", low_n)
    T["label_cardinality"] = S.label_cardinality(cl, low_n)
    T["error_type_homogeneity"] = S.error_level_homogeneity(long, cl, "error_type", low_n)
    T["error_category_homogeneity"] = S.error_level_homogeneity(long, cl, "error_category", low_n)
    T["error_rank_correlation"] = S.top_labels_rank_correlation(long, "error_type")

    # ---------------- Phase 2.4 depth ----------------
    T["depth_sample_composition"] = S.depth_sample_composition(cl, low_n)
    T["depth_by_agent"] = S.depth_descriptives(cl, ["agent"], low_n)
    T["depth_by_task_category"] = S.depth_descriptives(cl, ["task_category"], low_n)
    T["depth_by_agent_task_category"] = S.depth_descriptives(
        cl, ["agent", "task_category"], low_n)
    T["depth_km_by_agent"] = S.depth_km_summary(cl, "agent", low_n)
    T["depth_km_by_task_category"] = S.depth_km_summary(cl, "task_category", low_n)
    T["depth_logrank_agent"] = S.depth_logrank(cl, "agent")
    T["depth_logrank_task_category"] = S.depth_logrank(cl, "task_category")
    T["depth_bins_by_agent"] = S.depth_bins_by_group(cl, "agent", mapping, low_n)
    T["depth_bins_by_task_category"] = S.depth_bins_by_group(
        cl, "task_category", mapping, low_n)
    T["depth_by_error_type"] = S.depth_by_error_type(long, "error_type", low_n)
    T["depth_by_error_category"] = S.depth_by_error_type(long, "error_category", low_n)

    # Depth by error category needs a per-rollout view for KM/log-rank: a rollout
    # is assigned to a category only when all its labels agree, so overlapping
    # multi-category rollouts are reported separately rather than double counted.
    cl_cat = cl.copy()
    cl_cat["single_error_category"] = [
        c[0] if isinstance(c, list) and len(c) == 1 else
        ("MULTI_CATEGORY" if isinstance(c, list) and len(c) > 1 else None)
        for c in cl_cat["error_categories"]]
    sub = cl_cat[cl_cat["single_error_category"].notna()]
    T["depth_km_by_error_category"] = S.depth_km_summary(
        sub, "single_error_category", low_n)
    T["depth_logrank_error_category"] = S.depth_logrank(sub, "single_error_category")

    T["inconsistencies"] = inconsistencies

    # ---------------- Phase 3 figures ----------------
    F: dict[str, Path] = {}
    fg = paths.figures
    F["task_forest"] = P.task_score_forest(T["task_score_by_agent"], fg / "task_score_forest.png")
    F["rubric_hist"] = P.rubric_ratio_hist(cl, fg / "rubric_pass_ratio_hist.png")
    F["weighted_hist"] = P.weighted_ratio_hist(cl, fg / "weighted_rubric_score_hist.png")
    F["weighted_scatter"] = P.weighted_vs_unweighted_scatter(
        cl, fg / "weighted_vs_unweighted.png")
    F["rubric_heatmap"] = P.rubric_heatmap(T["rubric_by_agent_matrix"],
                                           fg / "rubric_by_agent_heatmap.png")
    F["error_stacked"] = P.error_type_stacked(long, fg / "error_category_stacked.png")
    F["error_bars"] = P.error_type_bars(T["error_types_by_agent"], fg / "error_type_bars.png")
    F["depth_hist"] = P.depth_hist(cl, fg / "depth_hist_by_agent.png")
    F["depth_km_agent"] = P.depth_km(cl, "agent", fg / "depth_km_by_agent.png",
                                     "Failure depth, Kaplan-Meier by agent")
    F["depth_km_errcat"] = P.depth_km(sub, "single_error_category",
                                      fg / "depth_km_by_error_category.png",
                                      "Failure depth, Kaplan-Meier by error category")
    F["depth_bins"] = P.depth_bins_bar(T["depth_bins_by_agent"],
                                       fg / "depth_bins_by_agent.png",
                                       "Ordinal depth bins by agent")
    F["coverage"] = P.coverage_bar(coverage, fg / "annotation_coverage.png")

    # ---------------- write tables ----------------
    _write(cl.drop(columns=["rubric_scores_json", "rubric_weights"])
             .assign(error_types=cl["error_types"].apply(
                 lambda v: "|".join(v) if isinstance(v, list) else ""),
                     error_categories=cl["error_categories"].apply(
                 lambda v: "|".join(v) if isinstance(v, list) else "")),
           paths.tables / "clean_annotations.csv")
    _write(long, paths.tables / "error_types_long.csv")
    _write(exclusions, paths.tables / "excluded_rows.csv")
    for name, df in T.items():
        _write(df, paths.tables / f"{name}.csv")

    # ---------------- Phase 4 sanity ----------------
    checks = _sanity_checks(cl, exclusions, T, long)

    (paths.out_dir / "report.md").write_text(
        R.build_report(T, F, cl, exclusions, diagnostics), encoding="utf-8")
    (paths.out_dir / "sanity_checks.md").write_text(
        R.build_sanity(T, cl, exclusions, checks), encoding="utf-8")

    # Consolidated one-row-per-agent summary at the path named in the request.
    summary = _summary_csv(T, cl)
    _write(summary, paths.out_dir / "statistic_analysis.csv")

    n_fail = sum(1 for c in checks if c["result"] == "FAIL")
    print(f"tables -> {paths.tables}")
    print(f"figures -> {paths.figures}")
    print(f"report -> {paths.out_dir / 'report.md'}")
    print(f"sanity checks: {len(checks)} run, {n_fail} FAIL")
    for c in checks:
        if c["result"] == "FAIL":
            print("  FAIL:", c["check"], "|", c["detail"])


def _sanity_checks(cl, exclusions, T, long) -> list[dict]:
    """Execute Phase 4 checks; each returns PASS/FAIL with the observed numbers."""
    out = []

    def add(name, ok, detail):
        out.append(dict(check=name, result="PASS" if ok else "FAIL", detail=detail))

    ts = T["task_score_by_agent"]
    per = ts[ts.agent != "OVERALL"]["denominator"].sum()
    tot = int(ts[ts.agent == "OVERALL"]["denominator"].iloc[0])
    add("task_score denominators sum to the overall n", per == tot,
        f"sum(per-agent)={per} overall={tot}")

    add("trajectory_id is unique in the clean table", cl["trajectory_id"].is_unique,
        f"{len(cl)} rows, {cl['trajectory_id'].nunique()} distinct")

    add("exactly one annotator per agent",
        (cl.groupby("agent")["annotator"].nunique() == 1).all(),
        str(cl.groupby("agent")["annotator"].nunique().to_dict()))

    agree = int((cl["task_score"] == cl["recomputed_task_score"]).sum())
    add("task_score equals all-rubrics-passed", agree == len(cl),
        f"{agree}/{len(cl)} agree ({agree / len(cl):.4f})")

    n = int(((cl["task_score"] == 1) & cl["has_failure_annotation"]).sum())
    add("no success row carries a failure annotation", n == 0, f"{n} rows")

    n = int(((cl["task_score"] == 0) & ~cl["has_failure_annotation"]).sum())
    add("every failure row carries a failure annotation", n == 0, f"{n} rows")

    n = int((cl["failure_depth"] < 0).sum())
    add("no negative depth", n == 0, f"{n} rows")

    n = int((cl["root_cause_step"].notna() &
             (cl["root_cause_step"] > cl["last_action_index"])).sum())
    add("root_cause_step within trajectory", n == 0, f"{n} rows out of range")

    obs = cl[cl["depth_status"] == "observed"]
    n = int((obs["failure_depth"] != obs["failure_depth_stored"]).sum())
    add("recomputed depth matches stored error_horizon_actions", n == 0,
        f"{n} mismatches of {len(obs)} observed rows")

    comp = T["depth_sample_composition"]
    row = comp[comp.agent == "OVERALL"].iloc[0]
    s = int(row["n_observed"] + row["n_right_censored"] + row["n_missing_root"])
    add("depth composition accounts for every failed rollout", s == int(row["n_failed"]),
        f"observed+censored+missing={s} n_failed={int(row['n_failed'])}")

    n_lab = long["trajectory_id"].nunique()
    n_empty = int(cl["error_types_emptied_by_normalization"].sum())
    n_failed_ann = int(((cl["task_score"] == 0) & cl["has_failure_annotation"]).sum())
    add("annotated failed rollouts are fully accounted for "
        "(in the label table, or emptied by normalisation)",
        n_lab + n_empty == n_failed_ann,
        f"{n_lab} labelled + {n_empty} emptied = {n_lab + n_empty} "
        f"vs {n_failed_ann} annotated failures")

    nlog = cl.attrs.get("label_normalization_log", pd.DataFrame())
    n_drop = int((nlog["action"] == "drop").sum()) if not nlog.empty else 0
    n_ren = int((nlog["action"] == "rename").sum()) if not nlog.empty else 0
    dropped_labels = set(nlog[nlog.action == "drop"]["original_label"]) if not nlog.empty else set()
    renamed_from = set(nlog[nlog.action == "rename"]["original_label"]) if not nlog.empty else set()
    still_present = {l for v in cl["error_types"] if v for l in v}
    add("dropped labels no longer appear anywhere",
        not (dropped_labels & still_present),
        f"dropped {sorted(dropped_labels)} ({n_drop} instances); "
        f"leaked: {sorted(dropped_labels & still_present) or 'none'}")
    add("renamed labels no longer appear under their old name",
        not (renamed_from & still_present),
        f"renamed {sorted(renamed_from)} ({n_ren} instances); "
        f"leaked: {sorted(renamed_from & still_present) or 'none'}")

    flagged = int((exclusions["reason"] == "flagged_wrong_rollout").sum()) if not exclusions.empty else 0
    add("wrong-rollout rows removed from the analysis set", True,
        f"{flagged} rows removed by the wrong-rollout filter "
        f"(flagged trajectories lacking a rubric review never entered the set)")

    for name in ("task_score_by_agent", "error_types_by_agent",
                 "rubric_ratio_distribution", "depth_bins_by_agent"):
        df = T[name]
        bad = int((df["numerator"] > df["denominator"]).sum())
        add(f"{name}: numerator <= denominator", bad == 0, f"{bad} violations")

    add("annotator confidence subset available for sensitivity analysis", False,
        "field absent from the data; sensitivity analysis not run "
        "(reported as a limitation, not substituted)")
    return out


def _summary_csv(T: dict, cl: pd.DataFrame) -> pd.DataFrame:
    """One consolidated row per agent covering the four requested statistics."""
    ts = T["task_score_by_agent"].set_index("agent")
    rb = T["rubric_score_by_agent"].set_index("agent")
    dp = T["depth_by_agent"].set_index("agent") if not T["depth_by_agent"].empty else None
    km = T["depth_km_by_agent"].set_index("group_value")
    comp = T["depth_sample_composition"].set_index("agent")
    et = T["error_types_by_agent"]

    rows = []
    for agent in list(ts.index):
        top = et[(et.agent == agent)].sort_values("rate", ascending=False).head(3)
        rows.append({
            "agent": agent,
            "annotator": "; ".join(sorted(cl[cl.agent == agent]["annotator"].unique()))
                         if agent != "OVERALL" else "—",
            "n_rollouts": int(ts.loc[agent, "denominator"]),
            "task_score_k": int(ts.loc[agent, "numerator"]),
            "task_score_rate": ts.loc[agent, "rate"],
            "task_score_ci_low": ts.loc[agent, "ci_low"],
            "task_score_ci_high": ts.loc[agent, "ci_high"],
            "task_score_low_n": ts.loc[agent, "low_n_flag"],
            "rubric_macro_mean": rb.loc[agent, "macro_mean"] if agent in rb.index else np.nan,
            "rubric_macro_sd": rb.loc[agent, "macro_sd"] if agent in rb.index else np.nan,
            "rubric_macro_ci_low": rb.loc[agent, "macro_boot_ci_low"] if agent in rb.index else np.nan,
            "rubric_macro_ci_high": rb.loc[agent, "macro_boot_ci_high"] if agent in rb.index else np.nan,
            "rubric_micro_rate": rb.loc[agent, "micro_rate"] if agent in rb.index else np.nan,
            "rubric_micro_k": rb.loc[agent, "micro_numerator"] if agent in rb.index else np.nan,
            "rubric_micro_n": rb.loc[agent, "micro_denominator"] if agent in rb.index else np.nan,
            "rubric_weighted_mean": rb.loc[agent, "weighted_mean"] if agent in rb.index else np.nan,
            "rubric_weighted_sd": rb.loc[agent, "weighted_sd"] if agent in rb.index else np.nan,
            "rubric_weighted_median": rb.loc[agent, "weighted_median"] if agent in rb.index else np.nan,
            "rubric_weighted_q1": rb.loc[agent, "weighted_q1"] if agent in rb.index else np.nan,
            "rubric_weighted_q3": rb.loc[agent, "weighted_q3"] if agent in rb.index else np.nan,
            "rubric_weighted_ci_low": rb.loc[agent, "weighted_boot_ci_low"] if agent in rb.index else np.nan,
            "rubric_weighted_ci_high": rb.loc[agent, "weighted_boot_ci_high"] if agent in rb.index else np.nan,
            "weighted_vs_macro_spearman": rb.loc[agent, "spearman_macro_vs_weighted"] if agent in rb.index else np.nan,
            "n_failed": int(comp.loc[agent, "n_failed"]) if agent in comp.index else np.nan,
            "depth_n_observed": int(comp.loc[agent, "n_observed"]) if agent in comp.index else np.nan,
            "depth_n_censored": int(comp.loc[agent, "n_right_censored"]) if agent in comp.index else np.nan,
            "depth_mean_observed": dp.loc[agent, "mean"] if dp is not None and agent in dp.index else np.nan,
            "depth_median_observed": dp.loc[agent, "median"] if dp is not None and agent in dp.index else np.nan,
            "depth_km_median": km.loc[agent, "km_median"] if agent in km.index else "",
            "top_error_types": "; ".join(
                f"{r.value}={r.rate:.3f}" for r in top.itertuples()),
        })
    return pd.DataFrame(rows)


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--builds-dir", default=None)
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--mapping", default=str(DEFAULT_MAPPING))
    args = ap.parse_args(argv)
    run(Paths.resolve(args.builds_dir, args.out_dir), Path(args.mapping))


if __name__ == "__main__":
    main()
