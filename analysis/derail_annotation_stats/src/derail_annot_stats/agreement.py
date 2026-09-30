"""Annotation protocol of paper App. E: the double-annotation sample, human-human agreement
and Human-LLM agreement, reported separately.

* ``double_annotation_plan``  per configured agent, ``fraction`` of its trajectories sampled
  with a fixed seed, stratified by task category (largest-remainder allocation).  The plan is
  written for the annotation UI with ``--write-plan`` (normally
  ``<human_labels>/protocol/double_annotation_plan.json``), which flags those trajectories.
* ``human_human_agreement``  trajectories reviewed by two annotators: first vs. second review
  of ``task_success`` (the full-completion judgment), with how many disagreements a third
  annotator adjudicated (``human_labels/adjudications/``).
* ``human_llm_agreement``  the LLM judge's ``passed`` (``osworld_full_traj_result.json`` next to
  the source ``traj.jsonl``) against the human ``task_success`` -- the adjudicated verdict when
  one exists, else the first review.  Unit ``task`` aggregates repeats as Pass@k.

Agreement, Cohen's kappa, precision, recall and F1 take "task passes" as the positive class;
for Human-LLM the human verdict is the reference.

    DERAIL_ANALYSIS_ENTRY=agreement analysis/derail_annotation_stats/run.sh "$DERAIL_ROOT" \
        [--write-plan PATH]
"""

from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Sequence

import pandas as pd

from .clean import load_mapping
from .io import discover_canonical_dirs, load_canonical_record, load_label_store
from .main import DEFAULT_MAPPING
from .paths import Paths

PLAN_SCHEMA_VERSION = "double-annotation-plan/0.1"


def binary_agreement(reference: Sequence[bool], prediction: Sequence[bool]) -> dict:
    """Agreement, Cohen's kappa, precision, recall and F1 of two binary label lists."""
    n = len(reference)
    if n == 0:
        return {"n": 0}
    tp = sum(r and p for r, p in zip(reference, prediction))
    tn = sum(not r and not p for r, p in zip(reference, prediction))
    fp = sum(p and not r for r, p in zip(reference, prediction))
    fn = sum(r and not p for r, p in zip(reference, prediction))
    po = (tp + tn) / n
    ref_pos, pred_pos = (tp + fn) / n, (tp + fp) / n
    pe = ref_pos * pred_pos + (1 - ref_pos) * (1 - pred_pos)
    kappa = (po - pe) / (1 - pe) if pe < 1 else float("nan")
    precision = tp / (tp + fp) if tp + fp else float("nan")
    recall = tp / (tp + fn) if tp + fn else float("nan")
    f1 = 2 * tp / (2 * tp + fp + fn) if tp + fp + fn else float("nan")
    return {"n": n, "reference_pass_rate": ref_pos, "prediction_pass_rate": pred_pos,
            "agreement": po, "kappa": kappa, "precision": precision, "recall": recall,
            "f1": f1, "disagreements": fp + fn}


def _allocate(sizes: dict[str, int], total: int) -> dict[str, int]:
    """Largest-remainder split of ``total`` over strata proportional to ``sizes``."""
    population = sum(sizes.values())
    exact = {k: total * v / population for k, v in sizes.items()} if population else {}
    alloc = {k: int(x) for k, x in exact.items()}
    for k in sorted(exact, key=lambda k: (-(exact[k] - alloc[k]), k))[: total - sum(alloc.values())]:
        alloc[k] += 1
    return alloc


def double_annotation_plan(trajectories: Iterable[dict], cfg: dict) -> list[dict]:
    """Stratified per-agent sample of trajectories for a second, independent annotation."""
    stratum_key = cfg.get("stratify_by", "task_category")
    fraction, seed = float(cfg["fraction"]), int(cfg["seed"])
    chosen = []
    for agent in cfg["agents"]:
        strata: dict[str, list[dict]] = defaultdict(list)
        for t in trajectories:
            if t["agent"] == agent:
                strata[str(t.get(stratum_key) or "uncategorized")].append(t)
        size = sum(len(v) for v in strata.values())
        alloc = _allocate({k: len(v) for k, v in strata.items()}, round(fraction * size))
        for stratum in sorted(strata):
            members = sorted(strata[stratum], key=lambda t: t["trajectory_id"])
            rng = random.Random("%d:%s:%s" % (seed, agent, stratum))
            for t in sorted(rng.sample(members, alloc[stratum]), key=lambda t: t["trajectory_id"]):
                chosen.append({"trajectory_id": t["trajectory_id"], "agent": agent,
                               stratum_key: stratum, "task_id": t.get("task_id")})
    return chosen


def _trajectories(paths: Paths) -> dict[str, dict]:
    out = {}
    for d in discover_canonical_dirs(paths.builds_dir):
        if any(part.startswith("_") for part in d.relative_to(paths.builds_dir).parts):
            continue  # _backups and other superseded trees
        rec = load_canonical_record(d)
        agents = rec.get("source_agents") or []
        if len(agents) != 1:
            continue
        out[rec["trajectory_id"]] = {"trajectory_id": rec["trajectory_id"], "agent": agents[0],
                                     "task_id": rec.get("task_id"),
                                     "task_category": rec.get("category"),
                                     "source_trajectory_uri": rec.get("source_trajectory_uri"),
                                     "source_trajectory_sha256": rec.get("source_trajectory_sha256")}
    return out


def _reviews(paths: Paths, trajectories: dict[str, dict]) -> dict[str, list[dict]]:
    """Hash-bound human rubric reviews per trajectory, earliest submission first, excluding
    trajectories flagged as wrong rollouts."""
    flagged = {r["_trajectory_id_from_filename"]
               for r in load_label_store(paths.rollout_flags, "rollout_flags")
               if r.get("rollout_status") == "needs_rerun"}
    by_tid: dict[str, list[dict]] = defaultdict(list)
    for r in load_label_store(paths.rubric_scores, "rubric_scores"):
        tid = r["_trajectory_id_from_filename"]
        t = trajectories.get(tid)
        if (t is None or tid in flagged or r.get("reviewer_role", "human") != "human"
                or not isinstance(r.get("task_success"), bool)
                or r.get("source_trajectory_sha256") != t["source_trajectory_sha256"]):
            continue
        by_tid[tid].append(r)
    for items in by_tid.values():
        items.sort(key=lambda r: (str(r.get("submitted_at", "")), str(r.get("reviewer_id"))))
    return by_tid


def _adjudicated(paths: Paths) -> dict[str, bool]:
    out = {}
    for r in load_label_store(paths.human_labels / "adjudications", "adjudications"):
        if isinstance(r.get("task_success"), bool):
            out[r["trajectory_id"]] = r["task_success"]
    return out


def _judge_passed(trajectory: dict, cfg: dict) -> bool | None:
    source = trajectory.get("source_trajectory_uri")
    if not source:
        return None
    path = Path(source).parent / cfg["judge_result_file"]
    try:
        value = json.loads(path.read_text(encoding="utf-8")).get(cfg["judge_pass_field"])
    except (OSError, json.JSONDecodeError):
        return None
    return value if isinstance(value, bool) else None


def human_human_agreement(trajectories: dict, reviews: dict, adjudicated: dict) -> pd.DataFrame:
    pairs: dict[str, list[tuple[bool, bool, bool]]] = defaultdict(list)
    for tid, items in reviews.items():
        first_per_reviewer: dict[str, dict] = {}
        for r in items:
            first_per_reviewer.setdefault(str(r["reviewer_id"]), r)
        distinct = list(first_per_reviewer.values())
        if len(distinct) >= 2:
            first, second = distinct[0]["task_success"], distinct[1]["task_success"]
            pairs[trajectories[tid]["agent"]].append((first, second, tid in adjudicated))
    rows = []
    for agent in sorted(pairs) + (["TOTAL"] if pairs else []):
        items = [x for a in sorted(pairs) for x in pairs[a]] if agent == "TOTAL" else pairs[agent]
        stats = binary_agreement([x[0] for x in items], [x[1] for x in items])
        stats["adjudicated_disagreements"] = sum(1 for a, b, adj in items if a != b and adj)
        rows.append({"agent": agent, **stats})
    return pd.DataFrame(rows)


def human_llm_agreement(trajectories: dict, reviews: dict, adjudicated: dict, cfg: dict) -> pd.DataFrame:
    units: dict[tuple, list[tuple[bool, bool]]] = defaultdict(list)
    missing = defaultdict(int)
    for tid, items in reviews.items():
        t = trajectories[tid]
        judge = _judge_passed(t, cfg)
        if judge is None:
            missing[t["agent"]] += 1
            continue
        human = adjudicated.get(tid, items[0]["task_success"])
        key = (t["agent"], t["task_id"]) if cfg.get("unit", "task") == "task" else (t["agent"], tid)
        units[key].append((human, judge))
    by_agent: dict[str, list[tuple[bool, bool]]] = defaultdict(list)
    for (agent, _), verdicts in units.items():
        by_agent[agent].append((any(h for h, _ in verdicts), any(j for _, j in verdicts)))
    rows = []
    for agent in sorted(by_agent) + (["TOTAL"] if by_agent else []):
        items = [x for a in sorted(by_agent) for x in by_agent[a]] if agent == "TOTAL" else by_agent[agent]
        stats = binary_agreement([h for h, _ in items], [j for _, j in items])
        rows.append({"agent": agent, "unit": cfg.get("unit", "task"),
                     "reviewed_without_judge_result": sum(missing.values()) if agent == "TOTAL"
                     else missing[agent], **stats})
    return pd.DataFrame(rows).rename(columns={"reference_pass_rate": "human_pass_rate",
                                              "prediction_pass_rate": "llm_pass_rate"})


def run(paths: Paths, mapping_path: Path, write_plan: Path | None) -> None:
    paths.ensure_out()
    protocol = load_mapping(mapping_path)["annotation_protocol"]
    trajectories = _trajectories(paths)
    plan = double_annotation_plan(trajectories.values(), protocol["double_annotation"])
    pd.DataFrame(plan).to_csv(paths.tables / "double_annotation_plan.csv", index=False)
    if write_plan:
        write_plan.parent.mkdir(parents=True, exist_ok=True)
        write_plan.write_text(json.dumps({"schema_version": PLAN_SCHEMA_VERSION,
                                          **protocol["double_annotation"],
                                          "trajectories": plan}, indent=2) + "\n", encoding="utf-8")
    reviews = _reviews(paths, trajectories)
    adjudicated = _adjudicated(paths)
    hh = human_human_agreement(trajectories, reviews, adjudicated)
    hl = human_llm_agreement(trajectories, reviews, adjudicated, protocol["human_llm_agreement"])
    hh.to_csv(paths.tables / "human_human_agreement.csv", index=False)
    hl.to_csv(paths.tables / "human_llm_agreement.csv", index=False)
    print(f"double-annotation plan: {len(plan)} trajectories")
    print("human-human agreement:\n" + (hh.to_string(index=False) if len(hh) else "  (no pairs)"))
    print("human-LLM agreement:\n" + (hl.to_string(index=False) if len(hl) else "  (no data)"))


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--builds-dir", default=None)
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--mapping", default=str(DEFAULT_MAPPING))
    ap.add_argument("--write-plan", type=Path, default=None,
                    help="also write the plan JSON the annotation UI reads")
    args = ap.parse_args(argv)
    run(Paths.resolve(args.builds_dir, args.out_dir), Path(args.mapping), args.write_plan)


if __name__ == "__main__":
    main()
