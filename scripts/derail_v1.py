#!/usr/bin/env python3
"""One command face for the v1.2 pipeline (brief section 5b).

    python scripts/derail_v1.py <stage> [stage arguments...]

Stages and the script each forwards to:

  schema-graph   build_schema_graph.py        files          build_file_inventory.py
  extract        extract_task_ir_v1.py        fix            fix_task_ir_v1.py
  gold           gold_interpret.py            rubric-check   check_task_ir_rubrics.py
  profiles       build_latent_profiles.py     reference-time check_reference_time.py
  analyze        analyze_failures_v1.py       accept         accept_task_ir_v1.py
  generate       generate_tasks_v1.py         cases          build_cases_v1.py
  rollout        run_rollout_v1.py            samples        build_training_v1.py
  recover        recovery_gen_v1.py

Order (brief section 5): schema-graph, files -> extract -> fix (<= 4 rounds) -> rubric-check ->
accept -> profiles -> generate -> cases -> [VM] rollout -> analyze -> samples -> recover.
Takeover evaluation is not a stage here: it runs through scripts/rock/run_takeover.sh
(scripts/10_run_takeover_rollout.py, configs/takeover/takeover.yaml).

Every stage prints what it reads and writes and leaves a manifest in its output directory.
Long stages belong in tmux; put the environment variables inside the tmux command string.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

STAGES = {
    "schema-graph": "build_schema_graph.py",
    "files": "build_file_inventory.py",
    "extract": "extract_task_ir_v1.py",
    "fix": "fix_task_ir_v1.py",
    "gold": "gold_interpret.py",
    "rubric-check": "check_task_ir_rubrics.py",
    "profiles": "build_latent_profiles.py",
    "reference-time": "check_reference_time.py",
    "analyze": "analyze_failures_v1.py",
    "accept": "accept_task_ir_v1.py",
    "generate": "generate_tasks_v1.py",
    "cases": "build_cases_v1.py",
    "rollout": "run_rollout_v1.py",
    "samples": "build_training_v1.py",
    "recover": "recovery_gen_v1.py",
}


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv or argv[0] in ("-h", "--help") or argv[0] not in STAGES:
        print(__doc__)
        return 0 if argv and argv[0] in ("-h", "--help") else 2
    script = Path(__file__).resolve().parent / STAGES[argv[0]]
    python = os.environ.get("PYTHON_BIN", sys.executable)
    return subprocess.call([python, str(script)] + argv[1:])


if __name__ == "__main__":
    raise SystemExit(main())
