#!/usr/bin/env python3
"""Expand the frozen takeover design into auditable runnable groups."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Mapping

from derail.adapters import NativeHistoryUnavailableError, get_registration
from derail.derived.layout import atomic_write_json, sha256_file, sha256_json
from derail.takeover.selection import select_takeover_failures


def read_object(path: Path) -> Mapping[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def completed_keys(experiment: Mapping[str, object]) -> set[tuple[int, str]]:
    return {
        (int(item["depth"]), str(item["condition"]))
        for item in experiment.get("completed", ())
    }


def select_source(experiment: Mapping[str, object], repository: Path) -> dict | None:
    build = repository / str(experiment["build_dir"])
    if not (build / "canonical").is_dir():
        return None
    selection_manifest = experiment.get("selection_manifest")
    if selection_manifest:
        selection = dict(read_object(repository / str(selection_manifest)))
        policy = selection.get("selection_policy", {})
        if policy.get("source_agent") != experiment["source_agent"]:
            raise ValueError("selection manifest source agent does not match experiment")
        if policy.get("annotator_id_filter") != experiment["annotator"]:
            raise ValueError("selection manifest annotator does not match experiment")
        return selection
    return select_takeover_failures(
        build_dir=build,
        human_labels_dir=repository / "takeovewr_annotation" / "human_labels",
        source_agent=str(experiment["source_agent"]),
        annotator_id=str(experiment["annotator"]),
        depths=tuple(int(value) for value in experiment["depths"]),
    )


def adapter_is_implemented(target_agent: str) -> bool:
    try:
        return get_registration(target_agent).renderer_implemented
    except NativeHistoryUnavailableError:
        return False


def group_status(
    selection: dict | None,
    target_agent: str,
    depth: int,
    is_complete: bool,
    source_records_ready: bool,
) -> str:
    if is_complete:
        return "completed"
    if selection is None:
        return "blocked_data"
    if not source_records_ready:
        return "blocked_source_records"
    if not adapter_is_implemented(target_agent):
        return "blocked_adapter"
    if selection["depth_coverage"].get(str(depth), 0) == 0:
        return "unavailable_depth"
    return "ready"


def expand_experiment(
    experiment: Mapping[str, object],
    repository: Path,
    reasoning_coverage: Mapping[str, object],
) -> list[dict]:
    selection = select_source(experiment, repository)
    target_agent = str(experiment["target_agent"])
    complete = completed_keys(experiment)
    candidate_ids = [item["trajectory_id"] for item in selection["included"]] if selection else []
    selection_sha256 = sha256_json(selection) if selection else ""
    groups = []
    for depth in experiment["depths"]:
        for condition in experiment["conditions"]:
            key = (int(depth), str(condition))
            groups.append(
                {
                    "source_agent": experiment["source_agent"],
                    "target_agent": target_agent,
                    "annotator": experiment["annotator"],
                    "build_dir": experiment["build_dir"],
                    "depth": key[0],
                    "condition": key[1],
                    "reasoning_policy": "preserve_as_recorded",
                    "reasoning_step_coverage": float(
                        reasoning_coverage.get(str(experiment["source_agent"]), 0.0)
                    ),
                    "eligible_rollouts": selection["depth_coverage"].get(str(depth), 0) if selection else 0,
                    "candidate_trajectory_ids": candidate_ids,
                    "trajectory_ids": [
                        item["trajectory_id"]
                        for item in selection["included"]
                        if key[0] in item["available_depths"]
                    ] if selection else [],
                    "selection_sha256": selection_sha256,
                    "status": group_status(
                        selection,
                        target_agent,
                        key[0],
                        key in complete,
                        bool(experiment.get("source_records_ready", True)),
                    ),
                }
            )
    return groups


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repository", type=Path, default=Path(__file__).resolve().parents[2])
    args = parser.parse_args()
    config = read_object(args.config)
    source_judge_policy = config.get("source_judge_policy", {})
    if source_judge_policy.get("authority") != "human_annotation":
        raise ValueError("source failure judge authority must be human_annotation")
    if source_judge_policy.get("raw_rollout_role") != "prefix_replay_evidence_only":
        raise ValueError("raw rollout must not override the human source judge")
    if source_judge_policy.get("model_judge_role") != "takeover_outcome_scoring_only":
        raise ValueError("model judge must not redefine source failure annotations")
    reasoning = config.get("reasoning_policy", {})
    reasoning_coverage = reasoning.get("source_step_coverage", {})
    if reasoning.get("mode") != "preserve_as_recorded":
        raise ValueError("takeover reasoning policy must preserve recorded source content")
    if reasoning.get("missing") != "not_recorded":
        raise ValueError("missing source reasoning must be marked not_recorded")
    groups = [
        group
        for experiment in config["experiments"]
        for group in expand_experiment(
            experiment, args.repository.resolve(), reasoning_coverage
        )
    ]
    payload = {
        "schema_version": config["schema_version"],
        "config_uri": str(args.config.resolve()),
        "config_sha256": sha256_file(args.config),
        "judge": config["judge"],
        "source_judge_policy": source_judge_policy,
        "depth_policy": config["depth_policy"],
        "group_count": len(groups),
        "pending_group_count": sum(group["status"] != "completed" for group in groups),
        "groups": groups,
    }
    atomic_write_json(args.output, payload)
    print(json.dumps({"groups": len(groups), "output": str(args.output.resolve())}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
