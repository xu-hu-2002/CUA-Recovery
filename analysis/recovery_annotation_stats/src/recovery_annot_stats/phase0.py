"""Phase 0: schema discovery over the raw RECOVERY annotation exports."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import pandas as pd

from .io import (
    discover_canonical_dirs,
    load_canonical_record,
    load_label_store,
    load_taxonomy,
)
from .paths import Paths

MAX_UNIQUE_TO_LIST = 30
SEED = 42

LOGICAL_FIELDS = [
    "task_id",
    "agent",
    "annotator",
    "rollout_valid",
    "wrong_rollout_reason",
    "agent_self_report",
    "human_verdict",
    "perfect_success",
    "rubric_scores",
    "weighted_rubric_score",
    "task_score",
    "root_cause_step",
    "clear_failure_step",
    "failure_depth",
    "error_types",
    "reversibility",
    "clean_prefix_range",
    "trajectory_length",
    "confidence",
]


def _describe_value(v):
    if isinstance(v, list):
        return f"<list len={len(v)}>"
    if isinstance(v, dict):
        return f"<dict n={len(v)} keys={sorted(v)[:6]}{'...' if len(v) > 6 else ''}>"
    return repr(v)


def profile_store(records: list[dict], store: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    n = len(records)
    col_rows, val_rows = [], []
    all_cols: list[str] = []
    for rec in records:
        for k in rec:
            if k not in all_cols:
                all_cols.append(k)

    for col in all_cols:
        present = [r[col] for r in records if col in r]
        non_null = [v for v in present if v is not None and v != ""]
        rendered = Counter(_describe_value(v) for v in present)
        py_types = sorted({type(v).__name__ for v in present})
        col_rows.append(
            {
                "store": store,
                "column": col,
                "n_files": n,
                "n_present": len(present),
                "n_non_null": len(non_null),
                "non_null_rate": (len(non_null) / n) if n else float("nan"),
                "n_unique_rendered": len(rendered),
                "python_types": "|".join(py_types),
            }
        )
        if len(rendered) <= MAX_UNIQUE_TO_LIST:
            for value, count in rendered.most_common():
                val_rows.append(
                    {"store": store, "column": col, "value": value, "count": count}
                )

    return pd.DataFrame(col_rows), pd.DataFrame(val_rows)


def key_uniqueness(records: list[dict], store: str) -> list[dict]:
    out = []
    trials = {
        "trajectory_id": lambda r: (r["_trajectory_id_from_filename"],),
        "trajectory_id+annotator": lambda r: (
            r["_trajectory_id_from_filename"],
            r["_annotator_id_from_filename"],
        ),
    }
    for name, fn in trials.items():
        keys = [fn(r) for r in records]
        counts = Counter(keys)
        dups = {k: c for k, c in counts.items() if c > 1}
        out.append(
            {
                "store": store,
                "candidate_key": name,
                "n_rows": len(keys),
                "n_distinct": len(counts),
                "is_unique": len(counts) == len(keys),
                "n_duplicated_keys": len(dups),
                "duplicated_examples": "; ".join(
                    "|".join(map(str, k)) + f" x{c}" for k, c in list(dups.items())[:8]
                ),
            }
        )
    return out


def guess_mapping(
    stores: dict[str, list[dict]], canonical: list[dict]
) -> pd.DataFrame:
    label_cols = {s: {k for r in recs for k in r} for s, recs in stores.items()}
    canon_cols = {k for r in canonical for k in r}

    def has(store: str, col: str) -> bool:
        return col in label_cols.get(store, set())

    rows = [
        dict(
            logical_field="task_id",
            source="canonical/normalization_report.json:task_provenance.task_id"
            " (also task_config.json:id)",
            confidence="high" if "task_id" in canon_cols else "ABSENT",
            note="Bare task id such as 'aggregation-f003'. The label exports key on"
            " trajectory_id, which is '<agent>-r1-<vm>-<task_id>'; task_id is the"
            " cross-agent join key.",
        ),
        dict(
            logical_field="agent",
            source="canonical/annotation_task.json:actions[*].source_agent",
            confidence="high",
            note="Preferred over parsing the trajectory_id prefix because it is"
            " recorded per action by the builder. Verify it is single-valued per"
            " trajectory before collapsing.",
        ),
        dict(
            logical_field="annotator",
            source="rubric_scores:reviewer_id / human_labels:annotator_id"
            " (and the '__<annotator>' file-name suffix)",
            confidence="high",
            note="Two stores use different column names for the same concept"
            " (reviewer_id vs annotator_id).",
        ),
        dict(
            logical_field="rollout_valid",
            source="presence of a file in human_labels/rollout_flags/",
            confidence="medium",
            note="Not a column. Validity is encoded by file existence: a rollout is"
            " Wrong Rollout iff a rollout_flags export exists for it. Everything"
            " else that carries a rubric review is Valid.",
        ),
        dict(
            logical_field="wrong_rollout_reason",
            source="rollout_flags:reason (+ rollout_status)",
            confidence="medium",
            note="Enumerated, not free text. Check the observed value domain before"
            " treating it as a reason taxonomy.",
        ),
        dict(
            logical_field="agent_self_report",
            source="canonical/annotation_task.json:actions[-1].action"
            " {kind, status}",
            confidence="medium",
            note="Self-report exists only when the final action is a 'terminate';"
            " its 'status' is the agent's own success/failure claim. Trajectories"
            " that end on a non-terminate action made no claim at all, which is a"
            " distinct state from a missing label.",
        ),
        dict(
            logical_field="human_verdict",
            source="(none)",
            confidence="ABSENT",
            note="No Perfect/Partial/Failure/unknown field exists in any store. The"
            " UI records only a boolean task_success.",
        ),
        dict(
            logical_field="perfect_success",
            source="(none)",
            confidence="ABSENT",
            note="No such checkbox is present in the exports.",
        ),
        dict(
            logical_field="rubric_scores",
            source="rubric_scores:scores (dict R1..Rn -> 0/1)",
            confidence="high" if has("rubric_scores", "scores") else "ABSENT",
            note="Rubric count varies by task; rubric ids are positional against"
            " task_config.json:grading.rubrics.",
        ),
        dict(
            logical_field="weighted_rubric_score",
            source="derivable from rubric_scores:scores x"
            " task_config.json:grading.rubrics[*].weight",
            confidence="medium",
            note="Not stored. Weights exist in the task config, so the weighted"
            " score can be computed, but no human ever reviewed such a number.",
        ),
        dict(
            logical_field="task_score",
            source="rubric_scores:task_success (bool)",
            confidence="high" if has("rubric_scores", "task_success") else "ABSENT",
            note="Human judgement of overall task success.",
        ),
        dict(
            logical_field="root_cause_step",
            source="human_labels:root_cause_action_index",
            confidence="high",
            note="0-based action_index_global, NOT 1-based. Also duplicated in"
            " cleaning_proposals:root_cause_action_index, which allows a"
            " consistency check.",
        ),
        dict(
            logical_field="clear_failure_step",
            source="human_labels:identifiable_at_action_index",
            confidence="high",
            note="0-based. null encodes 'the overall failure never becomes clear by"
            " the end of the trajectory' (UI checkbox 'noHorizon'), i.e."
            " right-censored, not missing.",
        ),
        dict(
            logical_field="failure_depth",
            source="human_labels:error_horizon_actions",
            confidence="high",
            note="Already stored, and the writer enforces"
            " error_horizon_actions == identifiable - root, so recomputation is a"
            " schema check rather than an independent measurement. null when"
            " right-censored.",
        ),
        dict(
            logical_field="error_types",
            source="human_labels:error_types (list[str])",
            confidence="high",
            note="Free open-coding vocabulary; label->category comes from"
            " human_labels/taxonomy/open_coded_labels.json, whose categories"
            " disagree with the spec's grouping for some labels.",
        ),
        dict(
            logical_field="reversibility",
            source="human_labels:reversibility",
            confidence="high",
            note="Observed domain is reversible/irreversible only.",
        ),
        dict(
            logical_field="clean_prefix_range",
            source="cleaning_proposals:{audited_action_indices, drop_candidates,"
            " root_cause_action_index}",
            confidence="low",
            note="No '#X-#Y' range field exists. A clean prefix has to be"
            " reconstructed as the audited actions before the root cause minus the"
            " dropped ones; the exact definition is an analysis choice, not a"
            " recorded label.",
        ),
        dict(
            logical_field="trajectory_length",
            source="canonical/normalization_report.json:canonical_action_count"
            " (cross-checkable against trajectory.jsonl line count)",
            confidence="high",
            note="Number of canonical actions, so the last valid index is"
            " length - 1 under 0-based indexing.",
        ),
        dict(
            logical_field="confidence",
            source="(none)",
            confidence="ABSENT",
            note="The UI never collected an annotator confidence rating, so the"
            " requested confidence-restricted sensitivity analysis cannot be run.",
        ),
    ]
    assert {r["logical_field"] for r in rows} == set(LOGICAL_FIELDS), (
        "mapping table must cover exactly the requested logical fields"
    )
    return pd.DataFrame(rows)[["logical_field", "source", "confidence", "note"]]


def run(paths: Paths) -> None:
    paths.ensure_out()

    stores = {
        "failure_annotations": load_label_store(paths.human_labels, "failure_annotations"),
        "rubric_scores": load_label_store(paths.rubric_scores, "rubric_scores"),
        "cleaning_proposals": load_label_store(
            paths.cleaning_proposals, "cleaning_proposals"
        ),
        "rollout_flags": load_label_store(paths.rollout_flags, "rollout_flags"),
        "drafts": load_label_store(paths.drafts, "drafts"),
    }
    canon_dirs = discover_canonical_dirs(paths.builds_dir)
    canonical = [load_canonical_record(d) for d in canon_dirs]
    taxonomy = load_taxonomy(paths.taxonomy_file)

    print("=" * 78)
    print("PHASE 0 - SCHEMA DISCOVERY")
    print("=" * 78)
    print(f"builds_dir : {paths.builds_dir}")
    print(f"out_dir    : {paths.out_dir}")
    print()
    print("--- 1. file and row counts ---")
    for store, recs in stores.items():
        print(f"  {store:22s} files={len(recs):5d}")
    print(f"  {'canonical trajectories':22s} dirs ={len(canonical):5d}")
    print()

    col_frames, val_frames, key_rows = [], [], []
    for store, recs in stores.items():
        if not recs:
            print(f"--- 2. store '{store}' is EMPTY, skipped ---\n")
            continue
        cols, vals = profile_store(recs, store)
        col_frames.append(cols)
        if not vals.empty:
            val_frames.append(vals)
        key_rows.extend(key_uniqueness(recs, store))

        print(f"--- 2. store '{store}' : {len(recs)} files ---")
        for _, r in cols.iterrows():
            if r["column"].startswith("_"):
                continue
            print(
                f"  {r['column']:32s} present={r['n_present']:4d}/{r['n_files']:<4d}"
                f" non_null={r['non_null_rate']:6.3f} uniq={r['n_unique_rendered']:5d}"
                f" types={r['python_types']}"
            )
            sub = vals[vals["column"] == r["column"]] if not vals.empty else vals
            if len(sub) and len(sub) <= MAX_UNIQUE_TO_LIST:
                for _, v in sub.iterrows():
                    print(f"        {v['value'][:88]:90s} : {v['count']}")
        print()

    canon_df = pd.DataFrame(canonical)
    canon_prof, canon_vals = profile_store(canonical, "canonical")
    col_frames.append(canon_prof)
    if not canon_vals.empty:
        val_frames.append(canon_vals)
    print(f"--- 2. store 'canonical' : {len(canonical)} trajectories ---")
    for _, r in canon_prof.iterrows():
        if r["column"].startswith("_"):
            continue
        print(
            f"  {r['column']:32s} present={r['n_present']:4d}/{r['n_files']:<4d}"
            f" non_null={r['non_null_rate']:6.3f} uniq={r['n_unique_rendered']:5d}"
            f" types={r['python_types']}"
        )
    print()

    print("--- 3. first 3 records of each store (pretty JSON) ---")
    for store, recs in stores.items():
        print(f"  ### {store}")
        for rec in recs[:3]:
            print(json.dumps(rec, indent=2, ensure_ascii=False, default=str)[:2400])
    print("  ### canonical")
    for rec in canonical[:3]:
        print(json.dumps(rec, indent=2, ensure_ascii=False, default=str)[:2400])
    print()

    print("--- 4. primary-key uniqueness ---")
    key_df = pd.DataFrame(key_rows)
    for _, r in key_df.iterrows():
        print(
            f"  {r['store']:22s} key={r['candidate_key']:24s} rows={r['n_rows']:5d}"
            f" distinct={r['n_distinct']:5d} unique={bool(r['is_unique'])}"
            f" dup_keys={r['n_duplicated_keys']}"
        )
        if r["duplicated_examples"]:
            print(f"        duplicates: {r['duplicated_examples']}")
    print()

    print("--- 5. proposed logical field mapping (CONFIRM BEFORE PHASE 1) ---")
    mapping = guess_mapping(stores, canonical)
    for _, r in mapping.iterrows():
        print(f"  [{r['confidence']:>6s}] {r['logical_field']:22s} <- {r['source']}")
        print(f"           {r['note']}")
    print()

    print("--- taxonomy export ---")
    labels = taxonomy.get("labels", {})
    print(f"  present={taxonomy.get('_present')} n_labels={len(labels)}")
    for name, meta in sorted(labels.items()):
        print(f"    {name:34s} category={meta.get('category')}")
    print()

    tdir = paths.tables
    pd.concat(col_frames, ignore_index=True).to_csv(
        tdir / "phase0_columns.csv", index=False
    )
    if val_frames:
        pd.concat(val_frames, ignore_index=True).to_csv(
            tdir / "phase0_value_counts.csv", index=False
        )
    key_df.to_csv(tdir / "phase0_key_check.csv", index=False)
    mapping.to_csv(tdir / "phase0_field_mapping_guess.csv", index=False)
    canon_df.to_csv(tdir / "phase0_canonical_index.csv", index=False)
    print(f"wrote Phase 0 tables to {tdir}")


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--builds-dir", default=None)
    ap.add_argument("--out-dir", default=None)
    args = ap.parse_args(argv)
    run(Paths.resolve(args.builds_dir, args.out_dir))


if __name__ == "__main__":
    main()
