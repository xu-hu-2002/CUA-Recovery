"""Loading layer: read the raw DERAIL annotation exports into flat records.

This module performs **no** interpretation beyond parsing JSON and attaching
provenance (which file each value came from). All semantic mapping happens in
``clean.py`` and is driven by ``config/field_mapping.yaml`` so that no logical
field name is hard-coded here.

Four independent human-label stores are produced by the annotation UI, each keyed
by ``<trajectory_id>__<annotator_id>``:

* ``human_labels/<key>.json``                     failure root cause + error taxonomy
* ``human_labels/rubric_scores/<key>.json``       per-rubric 0/1 and task success
* ``human_labels/cleaning_proposals/<key>.json``  clean-prefix / drop review
* ``human_labels/rollout_flags/<key>.json``       invalid-rollout flag

Plus a per-trajectory canonical record under
``<agent_build>/canonical/<trajectory_id>/`` supplying trajectory length, task
identity, rubric definitions and the agent's own terminate status.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterator

# The UI joins a trajectory id and an annotator id with this separator when it
# names an export file. It is a file-format constant, not a tunable.
KEY_SEP = "__"


def _read_json(path: Path) -> Any:
    with path.open(encoding="utf-8") as fh:
        return json.load(fh)


def _read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def split_key(stem: str) -> tuple[str, str]:
    """Split ``<trajectory_id>__<annotator_id>`` into its two parts.

    Raises if the separator is absent, rather than silently inventing an
    annotator, because annotator identity drives the confounding warning in the
    report.
    """
    if KEY_SEP not in stem:
        raise ValueError(f"export file name has no '{KEY_SEP}' separator: {stem!r}")
    traj, annot = stem.rsplit(KEY_SEP, 1)
    return traj, annot


def _iter_store(directory: Path) -> Iterator[tuple[str, str, Path, dict]]:
    """Yield ``(trajectory_id, annotator_id, path, payload)`` for one label store.

    Sub-directories are skipped, so calling this on ``human_labels/`` returns only
    the top-level failure annotations and not the nested stores.
    """
    if not directory.is_dir():
        return
    for path in sorted(directory.glob("*.json")):
        if not path.is_file():
            continue
        traj, annot = split_key(path.stem)
        yield traj, annot, path, _read_json(path)


def load_label_store(directory: Path, store: str) -> list[dict]:
    """Load one annotation-UI export directory into a list of flat dicts.

    Every returned record carries ``_store``, ``_file`` and the parsed
    ``_trajectory_id`` / ``_annotator_id`` taken from the file name, so that a
    disagreement between file name and in-file id becomes detectable rather than
    being silently resolved.
    """
    records: list[dict] = []
    for traj, annot, path, payload in _iter_store(directory, ):
        if not isinstance(payload, dict):
            raise ValueError(f"{path} does not contain a JSON object")
        rec = dict(payload)
        rec["_store"] = store
        rec["_file"] = str(path)
        rec["_trajectory_id_from_filename"] = traj
        rec["_annotator_id_from_filename"] = annot
        records.append(rec)
    return records


def discover_canonical_dirs(builds_dir: Path) -> list[Path]:
    """Find every canonical trajectory directory under the builds root.

    Agent builds normally look like ``<agent>_human_label_traj/canonical/<id>/``.
    One agent (the Claude Opus mix) shards by VM and nests one level deeper, so
    both depths are searched. Returns directories, sorted, deduplicated.
    """
    found = set()
    for pattern in ("*/canonical/*", "*/*/canonical/*"):
        for path in builds_dir.glob(pattern):
            if path.is_dir():
                found.add(path)
    return sorted(found)


def load_canonical_record(traj_dir: Path) -> dict:
    """Read one canonical trajectory directory into a flat provenance record.

    Pulls together the three JSON files the builder writes plus the action stream:

    * ``normalization_report.json`` -> canonical action count, task provenance
    * ``task_config.json``          -> category, difficulty, app, rubric definitions
    * ``annotation_task.json``      -> action list (source agent, terminate status)
    * ``trajectory.jsonl``          -> action stream, used only to cross-check length

    Missing optional files leave their fields as ``None``; nothing is imputed.
    """
    rec: dict[str, Any] = {
        "trajectory_id": traj_dir.name,
        "_canonical_dir": str(traj_dir),
        "_build_dir": None,
        "canonical_action_count": None,
        "jsonl_action_count": None,
        "task_id": None,
        "category": None,
        "difficulty": None,
        "app": None,
        "instruction": None,
        "rubric_ids": None,
        "rubric_weights": None,
        "rubric_criteria": None,
        "n_rubrics_defined": None,
        "source_agents": None,
        "terminal_action_kind": None,
        "terminal_action_status": None,
        "terminal_answer_len": None,
        "case_eligible": None,
        "source_trajectory_uri": None,
        "source_trajectory_sha256": None,
        "annotation_gate_passed": None,
        "normalization_gate_passed": None,
    }

    parts = traj_dir.parts
    if "canonical" in parts:
        rec["_build_dir"] = str(Path(*parts[: parts.index("canonical")]))

    nrep = traj_dir / "normalization_report.json"
    if nrep.is_file():
        d = _read_json(nrep)
        rec["canonical_action_count"] = d.get("canonical_action_count")
        rec["case_eligible"] = d.get("case_eligible")
        rec["source_trajectory_uri"] = d.get("source_trajectory_uri")
        rec["source_trajectory_sha256"] = d.get("source_trajectory_sha256")
        prov = d.get("task_provenance") or {}
        rec["task_id"] = prov.get("task_id")
        rec["instruction"] = prov.get("instruction")

    tcfg = traj_dir / "task_config.json"
    if tcfg.is_file():
        d = _read_json(tcfg)
        rec["category"] = d.get("category")
        rec["difficulty"] = d.get("difficulty")
        rec["app"] = d.get("app")
        if rec["task_id"] is None:
            rec["task_id"] = d.get("id")
        rubrics = ((d.get("grading") or {}).get("rubrics")) or []
        # The UI numbers rubrics R1..Rn in bundle order; reproduce that ordering
        # so rubric ids line up with the keys stored in rubric_scores exports.
        rec["rubric_ids"] = [f"R{i}" for i in range(1, len(rubrics) + 1)]
        rec["rubric_weights"] = [r.get("weight") for r in rubrics]
        rec["rubric_criteria"] = [r.get("criterion") for r in rubrics]
        rec["n_rubrics_defined"] = len(rubrics)

    atask = traj_dir / "annotation_task.json"
    if atask.is_file():
        d = _read_json(atask)
        rec["annotation_gate_passed"] = d.get("annotation_gate_passed")
        rec["normalization_gate_passed"] = d.get("normalization_gate_passed")
        actions = d.get("actions") or []
        agents = sorted({a.get("source_agent") for a in actions if a.get("source_agent")})
        rec["source_agents"] = agents
        if actions:
            last = actions[-1].get("action") or {}
            rec["terminal_action_kind"] = last.get("kind")
            rec["terminal_action_status"] = last.get("status")
            answer = last.get("answer")
            rec["terminal_answer_len"] = len(answer) if isinstance(answer, str) else None

    tj = traj_dir / "trajectory.jsonl"
    if tj.is_file():
        rec["jsonl_action_count"] = len(_read_jsonl(tj))

    return rec


def load_taxonomy(path: Path) -> dict:
    """Load the open-coding taxonomy export (label -> category), if present."""
    if not path.is_file():
        return {"labels": {}, "deleted_labels": {}, "_present": False}
    d = _read_json(path)
    d["_present"] = True
    return d
