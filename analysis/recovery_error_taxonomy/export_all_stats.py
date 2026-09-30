#!/usr/bin/env python3
"""Export every dataset statistic, along every dimension, into one zip."""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import hashlib
import importlib.util
import json
import re
import shutil
import statistics
import sys
import zipfile
from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path

import yaml

ROOT_ENTRY_RE = re.compile(r"^-\s+([a-z][a-z0-9_]+)\s+\[([^\]]+)\]\s*:", re.MULTILINE)
ACTION_REF_RE = re.compile(r"#(\d+)")
TASK_CATEGORY_RE = re.compile(r"-([a-z_]+)-f\d+$")
HORIZON_BINS = ("0", "1", "2", "3", "4", "5+")
SCORE_BINS = [(i / 10, (i + 1) / 10) for i in range(10)]
OUTCOMES = ("false_completion", "budget_exhausted", "fail_to_terminate", "other_stop")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def ratio(num: float, den: float) -> float | str:
    return round(num / den, 6) if den else ""


def summary(values: list[float]) -> dict:
    if not values:
        return {"n": 0, "mean": "", "median": "", "p25": "", "p75": "", "max": ""}
    q = statistics.quantiles(values, n=4, method="inclusive") if len(values) > 1 else [values[0]] * 3
    return {"n": len(values), "mean": round(statistics.fmean(values), 3), "median": statistics.median(values),
            "p25": q[0], "p75": q[2], "max": max(values)}


def split_labels(value: str) -> list[str]:
    return [item for item in (value or "").split("|") if item]


def load_continuation(repo: Path):
    path = repo / "src/recovery/longhorizon/continuation.py"
    spec = importlib.util.spec_from_file_location("recovery_continuation", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class Tables:
    def __init__(self) -> None:
        self.items: list[tuple[str, str, list[str], list[dict]]] = []

    def add(self, name: str, description: str, keys: list[str], rows: list[dict]) -> None:
        self.items.append((name, description, keys, rows))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--repo", type=Path, required=True)
    ap.add_argument("--labels", type=Path, required=True)
    ap.add_argument("--source-dir", type=Path, required=True)
    ap.add_argument("--taxonomy", type=Path, required=True)
    ap.add_argument("--benchmark-config", type=Path, required=True)
    ap.add_argument("--precheck-config", type=Path, required=True)
    ap.add_argument("--exclude-type", action="append", default=[],
                    help="error type kept in the data but dropped from the paper taxonomy")
    ap.add_argument("--stale", action="append", default=[], metavar="LABEL=PATH",
                    help="older file or directory to copy under stale_snapshots/LABEL/")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    taxonomy = yaml.safe_load(args.taxonomy.read_text(encoding="utf-8"))
    categories: dict[str, list[str]] = taxonomy["paper_categories"]
    type_order = [t for types in categories.values() for t in types]
    category_of = {t: c for c, types in categories.items() for t in types}
    renames = taxonomy.get("renamed_labels") or {}
    excluded = set(args.exclude_type)
    unknown_excluded = excluded - set(type_order)
    if unknown_excluded:
        raise SystemExit(f"--exclude-type not in taxonomy: {sorted(unknown_excluded)}")
    depths = [int(d) for d in yaml.safe_load(args.benchmark_config.read_text(encoding="utf-8"))["depths"]]
    precheck = yaml.safe_load(args.precheck_config.read_text(encoding="utf-8"))
    cont = load_continuation(args.repo.resolve())
    loop_cfg = cont.LoopConfig(**(precheck.get("loop_detection") or {}))
    early_window = int(precheck["early_stop_window_steps"])

    rows = list(csv.DictReader(args.labels.open(encoding="utf-8")))
    models = sorted({r["model"] for r in rows})
    groups_all = ["overall", "open", "closed", *models]
    audit = Counter()
    records = []
    for r in rows:
        m = TASK_CATEGORY_RE.search(r["trajectory_id"])
        if not m:
            raise SystemExit(f"cannot parse task category from {r['trajectory_id']}")
        rec = {
            "id": r["trajectory_id"], "model": r["model"], "open_closed": r["source_group"],
            "annotator": r["annotator"], "state": r["state"], "n_a_reason": r["n_a_reason"],
            "task_category": m.group(1), "labels": split_labels(r["error_types"]),
            "dropped": split_labels(r["dropped_labels"]),
            "wrs": float(r["weighted_rubric_score"]) if r["weighted_rubric_score"] not in ("", None) else None,
            "primary": r.get("primary_error_type", ""),
        }
        rec["groups"] = ["overall", rec["open_closed"], rec["model"]]
        unknown = set(rec["labels"]) - set(type_order)
        if unknown:
            raise SystemExit(f"{rec['id']}: labels outside the taxonomy {sorted(unknown)}")
        if rec["state"] == "failure":
            ann = json.loads(Path(r["annotation_path"]).read_text(encoding="utf-8"))
            root = int(ann["root_cause_action_index"])
            ident = ann.get("identifiable_at_action_index")
            rec["root"], rec["ident"] = root, ident
            rec["horizon"] = None if ident is None else int(ident) - root
            if ident is not None and ann.get("error_horizon_actions") not in (None, rec["horizon"]):
                audit["horizon_field_disagrees_with_indices"] += 1
            rec["reversibility"] = ann.get("reversibility") or "unrecorded"
            suppressed = {"premature_completion"} if "hit_budget_limit" in rec["labels"] else set()
            entries = []
            for raw_type, refs in ROOT_ENTRY_RE.findall(ann.get("rationale", "")):
                if raw_type in renames:
                    audit[f"rationale_rename_{raw_type}"] += 1
                t = renames.get(raw_type, raw_type)
                if t in category_of and t not in suppressed:
                    entries.append((t, {int(x) for x in ACTION_REF_RE.findall(refs)}))
            roots = {t for t, acts in entries if root + 1 in acts}
            mode = "one_based"
            if not roots:
                roots, mode = {t for t, acts in entries if root in acts}, "zero_based_fallback"
            if not roots:
                mode = "unresolved"
            rec["root_types"], rec["root_mode"] = roots, mode
            labels = set(rec["labels"])
            rec["outcome"] = ("false_completion" if "premature_completion" in labels else
                              "budget_exhausted" if "hit_budget_limit" in labels else
                              "fail_to_terminate" if "fail_to_terminate" in labels else "other_stop")
            try:
                traj = [json.loads(line) for line in Path(r["trajectory_path"]).read_text(encoding="utf-8").splitlines() if line.strip()]
                traj.sort(key=lambda s: int(s["action_index_global"]))
                rec["cont"] = cont.compute_continuation([s["action"] for s in traj], root,
                                                        loop=loop_cfg, early_stop_window=early_window)
            except (ValueError, KeyError, OSError) as exc:
                rec["cont"] = None
                audit["continuation_errors"] += 1
                audit[f"continuation_error:{type(exc).__name__}"] += 1
        records.append(rec)

    def members(group: str, state: str | None = None) -> list[dict]:
        return [x for x in records if group in x["groups"] and (state is None or x["state"] == state)]

    T = Tables()

    na_reasons = sorted({x["n_a_reason"] for x in records if x["state"] == "n/a"})
    out = []
    for g in groups_all:
        mem = members(g)
        eff = [x for x in mem if x["state"] in ("failure", "perfect_pass")]
        fail = [x for x in mem if x["state"] == "failure"]
        pp = [x for x in mem if x["state"] == "perfect_pass"]
        wrs = [x["wrs"] for x in eff if x["wrs"] is not None]
        labels_total = sum(len(x["labels"]) for x in fail)
        row = {"group": g, "raw_n": len(mem), "n_a": sum(x["state"] == "n/a" for x in mem)}
        for reason in na_reasons:
            row[f"n_a_{reason}"] = sum(x["state"] == "n/a" and x["n_a_reason"] == reason for x in mem)
        row.update({"effective_n": len(eff), "perfect_pass": len(pp), "failure": len(fail),
                    "success_rate": ratio(len(pp), len(eff)), "failure_rate": ratio(len(fail), len(eff)),
                    "weighted_rubric_mean": round(statistics.fmean(wrs), 4) if wrs else "",
                    "weighted_rubric_median": statistics.median(wrs) if wrs else "",
                    "error_labels_total": labels_total,
                    "error_labels_total_excluding_dropped_types": sum(
                        1 for x in fail for t in x["labels"] if t not in excluded),
                    "labels_per_failure_mean": round(labels_total / len(fail), 3) if fail else ""})
        out.append(row)
    T.add("T01_corpus_by_group", "Corpus size, states, success rate, rubric score, label totals",
          ["group"], out)

    out = []
    for g in groups_all:
        eff_n = sum(x["state"] in ("failure", "perfect_pass") for x in members(g))
        fail = members(g, "failure")
        lab = Counter(t for x in fail for t in x["labels"])
        tot, tot_kept = sum(lab.values()), sum(v for t, v in lab.items() if t not in excluded)
        for t in type_order:
            out.append({"group": g, "category": category_of[t], "error_type": t,
                        "dropped_from_paper": int(t in excluded), "failures_with_type": lab[t],
                        "rate_over_effective_n": ratio(lab[t], eff_n), "rate_over_failures": ratio(lab[t], len(fail)),
                        "share_of_labels": ratio(lab[t], tot),
                        "share_of_labels_excluding_dropped": "" if t in excluded else ratio(lab[t], tot_kept)})
    T.add("T02_error_type_by_group", "Label incidence per error type (a failure counts once per type)",
          ["group", "category", "error_type"], out)

    out = []
    for g in groups_all:
        fail = members(g, "failure")
        lab = Counter(t for x in fail for t in x["labels"])
        tot, tot_kept = sum(lab.values()), sum(v for t, v in lab.items() if t not in excluded)
        for c, types in categories.items():
            in_cat = sum(any(category_of[t] == c for t in x["labels"]) for x in fail)
            n_lab = sum(lab[t] for t in types)
            n_kept = sum(lab[t] for t in types if t not in excluded)
            out.append({"group": g, "category": c, "failures_with_category": in_cat,
                        "rate_over_failures": ratio(in_cat, len(fail)), "labels_in_category": n_lab,
                        "share_of_labels": ratio(n_lab, tot), "share_of_labels_excluding_dropped": ratio(n_kept, tot_kept)})
    T.add("T03_error_category_by_group", "Label incidence per error category", ["group", "category"], out)

    out = []
    for g in groups_all:
        fail = members(g, "failure")
        prim = Counter(x["primary"] for x in fail if x["primary"])
        for c, types in categories.items():
            n_cat = sum(prim[t] for t in types)
            out.append({"group": g, "category": c, "error_type": "", "failures": n_cat,
                        "share_of_root_causes": ratio(n_cat, len(fail))})
            for t in types:
                out.append({"group": g, "category": c, "error_type": t, "failures": prim[t],
                            "share_of_root_causes": ratio(prim[t], len(fail))})
    T.add("T03b_primary_root_cause_by_group", "One primary type per failure (default paper basis)",
          ["group", "category", "error_type"], out)

    out, out_s = [], []
    for g in groups_all:
        fail = members(g, "failure")
        k = Counter(len(x["labels"]) for x in fail)
        for n in sorted(k):
            out.append({"group": g, "n_labels": n, "failures": k[n], "share": ratio(k[n], len(fail))})
        out_s.append({"group": g, **summary([len(x["labels"]) for x in fail])})
    T.add("T04_labels_per_failure", "How many error types each failure carries", ["group", "n_labels"], out)
    T.add("T04b_labels_per_failure_summary", "Summary of labels per failure", ["group"], out_s)

    out = []
    for g in groups_all:
        fail = members(g, "failure")
        single = Counter(t for x in fail for t in set(x["labels"]))
        pair = Counter(p for x in fail for p in combinations(sorted(set(x["labels"]), key=type_order.index), 2))
        for a, b in combinations(type_order, 2):
            if pair[(a, b)]:
                out.append({"group": g, "type_a": a, "type_b": b, "failures_with_both": pair[(a, b)],
                            "share_of_failures": ratio(pair[(a, b)], len(fail)),
                            "p_b_given_a": ratio(pair[(a, b)], single[a]), "p_a_given_b": ratio(pair[(a, b)], single[b])})
    T.add("T05_type_cooccurrence", "Pairs of error types on the same failure (nonzero pairs only)",
          ["group", "type_a", "type_b"], out)

    out, out_c, out_r = [], [], []
    for g in groups_all:
        fail = members(g, "failure")
        resolved = [x for x in fail if x["root_types"]]
        inc, frac = Counter(), Counter()
        cinc, cfrac = Counter(), Counter()
        for x in resolved:
            for t in x["root_types"]:
                inc[t] += 1
                frac[t] += 1 / len(x["root_types"])
                cfrac[category_of[t]] += 1 / len(x["root_types"])
            for c in {category_of[t] for t in x["root_types"]}:
                cinc[c] += 1
        for t in type_order:
            out.append({"group": g, "category": category_of[t], "error_type": t,
                        "failures_with_type_in_root": inc[t], "fractional_root_mass": round(frac[t], 3),
                        "share_of_resolved_roots": ratio(frac[t], len(resolved))})
        for c in categories:
            out_c.append({"group": g, "category": c, "failures_with_category_in_root": cinc[c],
                          "fractional_root_mass": round(cfrac[c], 3), "share_of_resolved_roots": ratio(cfrac[c], len(resolved))})
        modes = Counter(x["root_mode"] for x in fail)
        card = [len(x["root_types"]) for x in resolved]
        out_r.append({"group": g, "failures": len(fail), "resolved": len(resolved),
                      "unresolved": modes["unresolved"], "one_based": modes["one_based"],
                      "zero_based_fallback": modes["zero_based_fallback"],
                      "multi_type_roots": sum(c > 1 for c in card),
                      "root_types_mean": round(statistics.fmean(card), 3) if card else ""})
    T.add("T06_root_cause_type_by_group", "Root-cause error types (labels whose evidence cites the root action)",
          ["group", "category", "error_type"], out)
    T.add("T06b_root_cause_category_by_group", "Root-cause categories", ["group", "category"], out_c)
    T.add("T06c_root_cause_resolution_by_group", "How root-cause types were resolved from the rationale",
          ["group"], out_r)

    out = []
    for g in groups_all:
        trans = Counter()
        for x in members(g, "failure"):
            for rt in x["root_types"]:
                for t in set(x["labels"]) - x["root_types"]:
                    trans[(rt, t)] += 1
        for (rt, t), n in sorted(trans.items(), key=lambda kv: (type_order.index(kv[0][0]), type_order.index(kv[0][1]))):
            out.append({"group": g, "root_type": rt, "cooccurring_type": t, "failures": n})
    T.add("T07_root_type_to_cooccurring_type", "Root-cause type followed by the other labels on the same failure",
          ["group", "root_type", "cooccurring_type"], out)

    out, out_o = [], []
    for g in groups_all:
        fail = members(g, "failure")
        cell = Counter()
        per_cat = Counter()
        for x in fail:
            for c in {category_of[t] for t in x["root_types"]}:
                cell[(c, x["outcome"])] += 1
                per_cat[c] += 1
        for c in categories:
            for o in OUTCOMES:
                out.append({"group": g, "root_category": c, "outcome": o, "failures": cell[(c, o)],
                            "share_within_root_category": ratio(cell[(c, o)], per_cat[c])})
        for o in OUTCOMES:
            sub = [x for x in fail if x["outcome"] == o]
            loops = [x for x in sub if x["cont"] and x["cont"].loop_detected]
            out_o.append({"group": g, "outcome": o, "failures": len(sub), "share": ratio(len(sub), len(fail)),
                          "loop_rate": ratio(len(loops), sum(1 for x in sub if x["cont"]))})
    T.add("T08_root_category_to_outcome", "Root-cause category x terminal outcome", ["group", "root_category", "outcome"], out)
    T.add("T08b_outcome_by_group", "Terminal outcome of failures, with loop rate", ["group", "outcome"], out_o)

    def horizon_bin(h: int) -> str:
        return "5+" if h >= 5 else str(max(h, 0))

    def horizon_row(sub: list[dict]) -> dict:
        hs = [x["horizon"] for x in sub if x["horizon"] is not None]
        s = summary(hs)
        return {"failures": len(sub), "uncensored": len(hs), "censored": len(sub) - len(hs),
                "median": s["median"], "mean": s["mean"], "p25": s["p25"], "p75": s["p75"], "max": s["max"],
                "share_ge_3": ratio(sum(h >= 3 for h in hs), len(hs))}

    out, out_s, out_t, out_c = [], [], [], []
    for g in groups_all:
        fail = members(g, "failure")
        bins = Counter(horizon_bin(x["horizon"]) for x in fail if x["horizon"] is not None)
        unc = sum(bins.values())
        for b in HORIZON_BINS:
            out.append({"group": g, "horizon_bin": b, "failures": bins[b], "share_of_uncensored": ratio(bins[b], unc)})
        out.append({"group": g, "horizon_bin": "censored", "failures": len(fail) - unc, "share_of_uncensored": ""})
        out_s.append({"group": g, **horizon_row(fail)})
        for t in type_order:
            out_t.append({"group": g, "category": category_of[t], "error_type": t,
                          **horizon_row([x for x in fail if t in x["labels"]])})
        for c in categories:
            out_c.append({"group": g, "root_category": c,
                          **horizon_row([x for x in fail if any(category_of[t] == c for t in x["root_types"])])})
    T.add("T09_error_horizon_bins", "Actions from root cause to first identifiable step", ["group", "horizon_bin"], out)
    T.add("T09b_error_horizon_summary", "Error horizon summary", ["group"], out_s)
    T.add("T10_error_horizon_by_label_type", "Error horizon of failures carrying each type",
          ["group", "category", "error_type"], out_t)
    T.add("T10b_error_horizon_by_root_category", "Error horizon by root-cause category", ["group", "root_category"], out_c)

    def cont_row(sub: list[dict]) -> dict:
        ok = [x["cont"] for x in sub if x["cont"]]
        loops = [c for c in ok if c.loop_detected]
        s = summary([c.post_error_steps for c in ok])
        return {"failures": len(sub), "computed": len(ok), "post_error_steps_median": s["median"],
                "post_error_steps_mean": s["mean"], "post_error_steps_p25": s["p25"], "post_error_steps_p75": s["p75"],
                "loop_rate": ratio(len(loops), len(ok)),
                "loop_repeat_count_median": statistics.median([c.loop_repeat_count for c in loops]) if loops else "",
                "loop_start_offset_median": statistics.median([c.loop_start_offset for c in loops]) if loops else "",
                "terminated_explicitly_rate": ratio(sum(c.terminated_explicitly for c in ok), len(ok)),
                "early_stop_rate": ratio(sum(c.early_stop for c in ok), len(ok))}

    out, out_t, out_k = [], [], []
    for g in groups_all:
        fail = members(g, "failure")
        out.append({"group": g, **cont_row(fail)})
        for t in type_order:
            out_t.append({"group": g, "category": category_of[t], "error_type": t,
                          **cont_row([x for x in fail if t in x["labels"]])})
        kinds = Counter()
        for x in fail:
            c = x["cont"]
            if c and c.loop_detected:
                try:
                    kinds[json.loads(c.loop_signature).get("kind", "unknown")] += 1
                except (TypeError, ValueError):
                    kinds["unparsed"] += 1
        n_loops = sum(kinds.values())
        for k, n in kinds.most_common():
            out_k.append({"group": g, "loop_action_kind": k, "loops": n, "share_of_loops": ratio(n, n_loops)})
    T.add("T11_post_error_continuation", "Steps after the root cause, loops, termination", ["group"], out)
    T.add("T12_continuation_by_label_type", "Post-error continuation by error type",
          ["group", "category", "error_type"], out_t)
    T.add("T12b_loop_action_kind", "Action kind of the repeated signature in detected loops",
          ["group", "loop_action_kind"], out_k)

    out = []
    for g in groups_all:
        fail = [x for x in members(g, "failure") if x["cont"]]
        for d in depths:
            cand = [x for x in fail if x["cont"].post_error_steps >= d]
            vis = [x for x in cand if x["ident"] is not None and x["ident"] <= x["root"] + d]
            est = [x for x in cand if x["cont"].loop_detected and x["cont"].loop_established_offset is not None
                   and x["cont"].loop_established_offset <= d]
            out.append({"group": g, "depth": d, "failures": len(fail), "candidates": len(cand),
                        "share_of_failures": ratio(len(cand), len(fail)), "evidence_visible_at_takeover": len(vis),
                        "share_evidence_visible": ratio(len(vis), len(cand)),
                        "loop_established_by_takeover": len(est), "share_loop_established": ratio(len(est), len(cand))})
    T.add("T13_depth_candidate_pool", "Failures with at least d post-error steps, and what the takeover point shows",
          ["group", "depth"], out)

    task_cats = sorted({x["task_category"] for x in records})
    out, out_t = [], []
    for g in groups_all:
        for tc in task_cats:
            mem = [x for x in members(g) if x["task_category"] == tc]
            eff = [x for x in mem if x["state"] in ("failure", "perfect_pass")]
            fail = [x for x in mem if x["state"] == "failure"]
            pp = [x for x in mem if x["state"] == "perfect_pass"]
            wrs = [x["wrs"] for x in eff if x["wrs"] is not None]
            out.append({"group": g, "task_category": tc, "raw_n": len(mem), "n_a": len(mem) - len(eff),
                        "effective_n": len(eff), "perfect_pass": len(pp), "failure": len(fail),
                        "success_rate": ratio(len(pp), len(eff)),
                        "weighted_rubric_mean": round(statistics.fmean(wrs), 4) if wrs else ""})
            lab = Counter(t for x in fail for t in x["labels"])
            for t in type_order:
                out_t.append({"group": g, "task_category": tc, "category": category_of[t], "error_type": t,
                              "failures_with_type": lab[t], "rate_over_category_failures": ratio(lab[t], len(fail))})
    T.add("T14_task_category_by_group", "Success and failure per MyPCBench task category", ["group", "task_category"], out)
    T.add("T15_error_type_by_task_category", "Error types within each task category",
          ["group", "task_category", "category", "error_type"], out_t)

    out, out_t = [], []
    for g in groups_all:
        fail = members(g, "failure")
        rev = Counter(x["reversibility"] for x in fail)
        for k in sorted(rev):
            out.append({"group": g, "reversibility": k, "failures": rev[k], "share": ratio(rev[k], len(fail))})
        for t in type_order:
            sub = [x for x in fail if t in x["labels"]]
            rv = Counter(x["reversibility"] for x in sub)
            for k in sorted(rv):
                out_t.append({"group": g, "category": category_of[t], "error_type": t, "reversibility": k,
                              "failures": rv[k], "share_within_type": ratio(rv[k], len(sub))})
    T.add("T16_reversibility_by_group", "Reversibility of the failure's side effects", ["group", "reversibility"], out)
    T.add("T16b_reversibility_by_label_type", "Reversibility per error type",
          ["group", "category", "error_type", "reversibility"], out_t)

    out, out_s = [], []
    for g in groups_all:
        for scope in ("effective", "failure", "perfect_pass"):
            mem = [x for x in members(g) if (x["state"] in ("failure", "perfect_pass") if scope == "effective" else x["state"] == scope)]
            vals = [x["wrs"] for x in mem if x["wrs"] is not None]
            for lo, hi in SCORE_BINS:
                n = sum(lo <= v < hi for v in vals)
                out.append({"group": g, "state_scope": scope, "score_bin": f"[{lo:.1f},{hi:.1f})", "trajectories": n,
                            "share": ratio(n, len(vals))})
            n1 = sum(v >= 1.0 for v in vals)
            out.append({"group": g, "state_scope": scope, "score_bin": "=1.0", "trajectories": n1, "share": ratio(n1, len(vals))})
            out_s.append({"group": g, "state_scope": scope, **summary(vals)})
    T.add("T17_weighted_rubric_score_bins", "Weighted rubric score distribution", ["group", "state_scope", "score_bin"], out)
    T.add("T17b_weighted_rubric_score_summary", "Weighted rubric score summary", ["group", "state_scope"], out_s)

    out, out_a = [], []
    for mdl in models:
        mem = members(mdl)
        dropped = Counter(t for x in mem for t in x["dropped"])
        out.append({"model": mdl, "trajectories_with_dropped_labels": sum(bool(x["dropped"]) for x in mem)})
        for t, n in dropped.most_common():
            out.append({"model": mdl, "dropped_label": t, "occurrences": n})
        for ann, n in Counter(x["annotator"] for x in mem).most_common():
            out_a.append({"model": mdl, "annotator": ann, "trajectories": n})
    T.add("T18_label_normalization_audit", "Labels removed during normalization", ["model", "dropped_label"], out)
    T.add("T19_annotators", "Primary annotator per model", ["model", "annotator"], out_a)

    stamp = dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    build = args.out.with_suffix("")
    if build.exists():
        shutil.rmtree(build)
    (build / "tables").mkdir(parents=True)
    long_rows = []
    for name, _desc, keys, table in T.items:
        cols = list(dict.fromkeys(k for row in table for k in row))
        with (build / "tables" / f"{name}.csv").open("w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=cols)
            w.writeheader()
            w.writerows(table)
        for row in table:
            key_vals = [f"{k}={row[k]}" for k in keys if k in row and row[k] != ""]
            for col in cols:
                if col in keys or col not in row:
                    continue
                long_rows.append({"table": name, "key_1": key_vals[0] if key_vals else "",
                                  "key_2": key_vals[1] if len(key_vals) > 1 else "",
                                  "key_3": key_vals[2] if len(key_vals) > 2 else "",
                                  "key_4": key_vals[3] if len(key_vals) > 3 else "",
                                  "metric": col, "value": row[col]})
    with (build / "ALL_STATS_long.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=["table", "key_1", "key_2", "key_3", "key_4", "metric", "value"])
        w.writeheader()
        w.writerows(long_rows)

    (build / "source").mkdir()
    for f in sorted(args.source_dir.iterdir()):
        if f.is_file():
            shutil.copy2(f, build / "source" / f.name)
    stale_lines = []
    for spec in args.stale:
        label, _, raw = spec.partition("=")
        src = Path(raw)
        dest = build / "stale_snapshots" / label
        dest.mkdir(parents=True, exist_ok=True)
        if src.is_dir():
            shutil.copytree(src, dest / src.name, dirs_exist_ok=True)
        else:
            shutil.copy2(src, dest / src.name)
        mtime = dt.datetime.fromtimestamp(src.stat().st_mtime, dt.timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
        stale_lines.append(f"| `stale_snapshots/{label}/{src.name}` | `{raw}` | {mtime} |")

    failures = [x for x in records if x["state"] == "failure"]
    readme = [
        "# RECOVERY dataset statistics, all dimensions", "",
        f"Generated {stamp} by `analysis/recovery_error_taxonomy/export_all_stats.py`.", "",
        "## Inputs", "",
        f"- Label table: `{args.labels}` (sha256 `{sha256(args.labels)}`), {len(records)} trajectories, "
        f"{len(failures)} failures, {len(models)} models.",
        f"- Taxonomy: `{args.taxonomy}` (version `{taxonomy.get('taxonomy_version')}`), "
        f"{len(type_order)} types in {len(categories)} categories.",
        f"- Depth grid: `{args.benchmark_config}` -> {depths}.",
        f"- Loop detection: `{args.precheck_config}` -> {precheck.get('loop_detection')}, "
        f"early-stop window {early_window}; heuristic `{cont.LOOP_HEURISTIC_VERSION}`.",
        f"- Types kept in the data but dropped from the paper: {sorted(excluded) or 'none'}.", "",
        "## Groups", "",
        "Every table is broken down by `group`: `overall`, `open`, `closed`, and each model id "
        f"({', '.join(models)}).", "",
        "## Denominators", "",
        "- `effective_n` = raw trajectories minus `n/a` (wrong rollout, unreviewed).",
        "- Error-type counts are label incidence: a failure counts once per type it carries, so types do "
        "not sum to the number of failures.",
        "- Root-cause types are the labels whose rationale evidence cites the root-cause action; a failure "
        "with k root types contributes 1/k to each (`fractional_root_mass`).",
        "- Error horizon = identifiable action index - root-cause action index; failures with no "
        "identifiable action are `censored` and excluded from medians and shares.",
        "- Terminal outcome precedence: premature_completion > hit_budget_limit > fail_to_terminate > other.",
        "- `T13` candidates are failures with at least d post-error steps. They are a candidate pool, not "
        "the exported benchmark case counts, which additionally require a repaired, replayable prefix.", "",
        "## Tables", "",
        "| File | Content | Key columns | Rows |", "| --- | --- | --- | --- |",
        *[f"| `tables/{n}.csv` | {d} | {', '.join(k)} | {len(t)} |" for n, d, k, t in T.items], "",
        f"`ALL_STATS_long.csv` holds all {len(long_rows)} numbers above as (table, key_1..key_4, metric, value).", "",
        "## Audit", "",
        *([f"- {k}: {v}" for k, v in sorted(audit.items())] or ["- no anomalies"]), "",
        "## Source and stale snapshots", "",
        f"`source/` is a verbatim copy of `{args.source_dir}` (the analyze.py outputs used here).", "",
    ]
    if stale_lines:
        readme += ["Stale snapshots were produced before this label table and use older denominators. "
                   "They are included only for comparison.", "",
                   "| In zip | Original path | Modified |", "| --- | --- | --- |", *stale_lines, ""]
    (build / "README.md").write_text("\n".join(readme), encoding="utf-8")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(args.out, "w", zipfile.ZIP_DEFLATED) as zf:
        for f in sorted(build.rglob("*")):
            if f.is_file():
                zf.write(f, f.relative_to(build.parent))
    print(f"[export] {len(T.items)} tables, {len(long_rows)} numbers -> {args.out}")
    print(f"[export] audit: {dict(audit) or 'clean'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
