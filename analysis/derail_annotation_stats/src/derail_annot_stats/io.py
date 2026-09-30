from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterator

KEY_SEP = "__"


def _read_json(path: Path) -> Any:
    with path.open(encoding="utf-8") as fh:
        return json.load(fh)


def _read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def split_key(stem: str) -> tuple[str, str]:
    if KEY_SEP not in stem:
        raise ValueError(f"export file name has no '{KEY_SEP}' separator: {stem!r}")
    traj, annot = stem.rsplit(KEY_SEP, 1)
    return traj, annot


def _iter_store(directory: Path) -> Iterator[tuple[str, str, Path, dict]]:
    if not directory.is_dir():
        return
    for path in sorted(directory.glob("*.json")):
        if not path.is_file():
            continue
        traj, annot = split_key(path.stem)
        yield traj, annot, path, _read_json(path)


def load_label_store(directory: Path, store: str) -> list[dict]:
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
    found = set()
    for pattern in ("*/canonical/*", "*/*/canonical/*"):
        for path in builds_dir.glob(pattern):
            if path.is_dir():
                found.add(path)
    return sorted(found)


def load_canonical_record(traj_dir: Path) -> dict:
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
    if not path.is_file():
        return {"labels": {}, "deleted_labels": {}, "_present": False}
    d = _read_json(path)
    d["_present"] = True
    return d
