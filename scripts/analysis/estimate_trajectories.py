#!/usr/bin/env python3
"""Estimate expected trajectory counts for the rollout plan."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import yaml

REPO_ROOT = Path(os.environ.get("DERAIL_REPO_ROOT", Path(__file__).resolve().parents[2]))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--accepted", type=Path, required=True)
    parser.add_argument("--bundle", type=Path)
    parser.add_argument(
        "--budget", type=Path, default=REPO_ROOT / "configs/benchmark/rollout_budget_v1.yaml"
    )
    args = parser.parse_args()
    budget = yaml.safe_load(args.budget.read_text(encoding="utf-8"))
    seeds = json.loads((args.accepted / "manifest.json").read_text(encoding="utf-8"))["accepted"]
    generated, world_variants = 0, float(budget["world_variants_per_task_default"])
    manifest_path = next(
        (
            args.bundle / name
            for name in ("generation_bundle.json", "manifest.json")
            if args.bundle and (args.bundle / name).is_file()
        ),
        None,
    )
    if manifest_path is not None:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        generated = int(manifest.get("selected", manifest.get("tasks", 0)))
        records = [
            json.loads(line)
            for line in (args.bundle / "candidates.jsonl").read_text(encoding="utf-8").splitlines()
        ]
        selected = [r for r in records if r.get("selected", True)]
        if selected:
            world_variants = sum(r["variants"]["total"] for r in selected) / len(selected)
    tasks = seeds + generated
    variants = world_variants * float(budget["realizations_per_task"])
    base = tasks * variants * int(budget["rollout_models"])
    failures = base * float(budget["failure_rate"])
    takeover = failures * float(budget["usable_takeover_depths"]) * int(budget["takeover_models"])
    total = base + takeover
    rows = [
        ("accepted seed tasks", seeds),
        ("selected generated tasks", generated),
        ("world variants per task (mean)", round(world_variants, 2)),
        ("instruction realizations per task", budget["realizations_per_task"]),
        ("rollout models", budget["rollout_models"]),
        ("base rollouts", round(base)),
        ("failure rate (assumed)", budget["failure_rate"]),
        ("usable takeover depths per failure (assumed)", budget["usable_takeover_depths"]),
        ("takeover models", budget["takeover_models"]),
        ("takeover trajectories", round(takeover)),
        ("total trajectories", round(total)),
        ("target", budget["target_trajectories"]),
        ("meets target", total >= float(budget["target_trajectories"])),
        (
            "GPU hours (assumed s/step x steps)",
            round(
                total
                * float(budget["steps_per_trajectory"])
                * float(budget["seconds_per_step"])
                / 3600
            ),
        ),
    ]
    width = max(len(r[0]) for r in rows)
    for label, value in rows:
        print("%s  %s" % (label.ljust(width), value))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
