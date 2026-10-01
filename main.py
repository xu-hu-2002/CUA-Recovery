#!/usr/bin/env python3
"""Usage: python main.py <stage> <command> [args...]"""

import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent

COMMANDS = {
    "setup": {
        "third_party": ("setup_third_party.sh", "Clone pinned upstream repos into third_party/ and apply patches."),
    },
    "synthesis": {
        "build_schema_graph": ("synthesis/build_schema_graph.py", "Build the schema-graph record over the app databases."),
        "build_file_inventory": ("synthesis/build_file_inventory.py", "Build the file inventory from the persona home tree."),
        "build_world_graph": ("synthesis/build_world_graph.py", "Build the world-graph record from the persona seed."),
        "check_reference_time": ("synthesis/check_reference_time.py", "Verify the pinned reference time against the databases."),
        "extract_task_ir_v1": ("synthesis/extract_task_ir_v1.py", "Task IR v1 extraction for seed tasks."),
        "run_task_ir_v1_extraction": ("synthesis/run_task_ir_v1_extraction.sh", "One approved Task IR v1 extraction run."),
        "fix_task_ir_v1": ("synthesis/fix_task_ir_v1.py", "One fix round of Task IR v1 extraction."),
        "run_task_ir_v1_fix": ("synthesis/run_task_ir_v1_fix.sh", "One approved Task IR v1 fix round."),
        "gold_interpret": ("synthesis/gold_interpret.py", "Run the gold interpreter over Task IRs."),
        "check_task_ir_rubrics": ("synthesis/check_task_ir_rubrics.py", "Gold-execute Task IRs and check rubric expectations."),
        "build_review_sheet": ("synthesis/build_review_sheet.py", "Review sheet for tasks the rubric check could not decide."),
        "accept_task_ir_v1": ("synthesis/accept_task_ir_v1.py", "Freeze validated Task IRs into an accepted set."),
        "build_latent_profiles": ("synthesis/build_latent_profiles.py", "Static detectability profiles for Task IRs."),
        "generate_tasks_v1": ("synthesis/generate_tasks_v1.py", "Compose, verify, profile and select generated tasks."),
        "run_phase4_batch": ("synthesis/run_phase4_batch.sh", "Unattended fix -> check -> accept -> generate batch."),
        "merge_generation_bundles": ("synthesis/merge_generation_bundles.py", "Merge selected tasks of several generation bundles."),
        "realize_tasks_v1": ("synthesis/realize_tasks_v1.py", "Realize instructions for selected generated tasks."),
        "run_realization_v1": ("synthesis/run_realization_v1.sh", "One approved realization run."),
        "freeze_source_splits": ("synthesis/freeze_source_splits.py", "Freeze the train/test split of composed workflows."),
    },
    "collection": {
        "serve_open_source": ("collection/serve_open_source.sh", "Start/stop frozen vLLM endpoints for open-source agents."),
        "collect_all": ("collection/collect_all.sh", "Preset entry for the full MyPCBench collection."),
        "collect_trajectories": ("collection/collect_trajectories.sh", "Collect raw trajectories for one agent."),
        "install_vm_infra": ("collection/install_vm_infra.py", "Install the in-VM recovery infrastructure."),
        "run_rollout_v1": ("collection/run_rollout_v1.py", "Run a batch of (task, world, agent, seed) rollouts."),
    },
    "judge": {
        "judge_rubrics": ("judge/judge_rubrics.sh", "Rubric-judge collected raw trajectories."),
        "run_judge": ("judge/run_judge.sh", "Unified MyPCBench judge entry."),
        "full_traj_judge": ("judge/full_traj_judge.py", "Full-trajectory rubric judge with final-state data."),
        "run_takeover_judge": ("judge/run_takeover_judge.sh", "Judge takeover cells (EAR, then rubric)."),
        "error_awareness": ("judge/error_awareness.py", "Error-Awareness Rate (EAR) for takeover cells."),
        "rubric_csv": ("judge/rubric_csv.py", "Per-rubric CSV for one takeover cell."),
        "summarize_takeover": ("judge/summarize_takeover.py", "Pass@k and rubric score per depth x condition."),
        "run_with_timeout": ("judge/run_with_timeout.py", "Run a command with a hard timeout."),
    },
    "benchmark": {
        "prepare_annotations": ("benchmark/prepare_annotations.py", "Initialize a derived build and normalize a task for annotation."),
        "serve_annotation_ui": ("benchmark/serve_annotation_ui.py", "Local annotation UI for failure labelling."),
        "prepare_clean_prefix": ("benchmark/prepare_clean_prefix.py", "Turn cleaning proposals into audited prefix patches."),
        "select_takeover_failures": ("benchmark/select_takeover_failures.py", "List valid failure trajectories for takeover."),
        "export_error_depth_slices": ("benchmark/export_error_depth_slices.py", "Export takeover prefixes per agent and depth."),
        "analyze_failures_v1": ("benchmark/analyze_failures_v1.py", "Automatic failure analysis over rollout traces."),
    },
    "takeover": {
        "run": ("takeover/run.sh", "Launch prefix-takeover rollouts."),
        "prepare_source": ("takeover/prepare_source.py", "Canonicalize labelled trajectories for one source."),
        "build_native_history": ("takeover/build_native_history.py", "Build a target agent's native history."),
        "stage_inputs": ("takeover/stage_inputs.py", "Stage hash-audited takeover inputs."),
        "preflight_history": ("takeover/preflight_history.py", "Validate takeover history shape and token budget."),
        "run_rollout": ("takeover/run_rollout.py", "Run one prefix-takeover rollout."),
        "bundle_prefix": ("takeover/bundle_prefix.py", "Prepend the replayed prefix to rubric_bundle.json."),
    },
    "train": {
        "build_training_v1": ("train/build_training_v1.py", "Build training inputs from traces and analyses."),
        "recovery_gen_v1": ("train/recovery_gen_v1.py", "Recovery generation (VM/GPU side)."),
    },
}


def usage():
    lines = [__doc__]
    for stage, commands in COMMANDS.items():
        lines.append(f"\n{stage}:")
        lines += [f"  {name:<34}{text}" for name, (_, text) in commands.items()]
    return "\n".join(lines)


def main(argv):
    if len(argv) < 2 or argv[1] not in COMMANDS.get(argv[0], {}):
        print(usage())
        return 0 if argv[:1] in (["-h"], ["--help"]) else 2
    script = ROOT / "scripts" / COMMANDS[argv[0]][argv[1]][0]
    runner = "bash" if script.suffix == ".sh" else sys.executable
    env = dict(os.environ, PYTHONPATH=os.pathsep.join(filter(None, [str(ROOT / "src"), os.environ.get("PYTHONPATH")])))
    return subprocess.call([runner, str(script), *argv[2:]], env=env)


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
