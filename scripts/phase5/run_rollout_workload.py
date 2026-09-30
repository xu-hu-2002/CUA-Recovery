#!/usr/bin/env python3
"""Run a frozen Phase 5 workload through the instrumented MyPCBench environment."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

REPOSITORY = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPOSITORY / "src"))

from derail.harness.control_client import DerailControlClient  # noqa: E402
from derail.harness.env_wrapper import DerailEnvWrapper  # noqa: E402
from derail.harness.trace_builder import TraceBuilder  # noqa: E402
from derail.failure_analysis.retrospective import clean_start_step_budget  # noqa: E402
from derail.phase5.verifier import verify_changelog  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workload", type=Path, required=True)
    parser.add_argument("--batch", required=True)
    parser.add_argument("--combination-id", action="append", default=[])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--step-budget", type=int, default=clean_start_step_budget(),
                        help="default: max_steps in configs/collection/mypcbench_runtime.yaml")
    parser.add_argument("--pause-after-action-s", type=float, default=2.0)
    return parser.parse_args()


def required_env(name: str) -> str:
    value = os.environ.get(name)
    if not value:
        raise SystemExit(f"environment variable {name} is required")
    return value


def load_rows(path: Path, wanted: set[str]) -> tuple[dict, list[dict]]:
    workload = json.loads(path.read_text(encoding="utf-8"))
    if workload.get("schema") != "phase5-rollout-workload/1.0":
        raise SystemExit("unsupported Phase 5 workload")
    rows = workload["combinations"]
    if wanted:
        rows = [row for row in rows if row["combination_id"] in wanted]
    if len(rows) != len(wanted) and wanted:
        raise SystemExit("one or more requested combination IDs are absent")
    if any(row["hazard_ref"] is not None for row in rows):
        raise SystemExit("hazard workload requires a frozen world materializer")
    return workload, rows


def result_path(root: Path, model: str, batch: str, row: dict, seed: int) -> Path:
    return (
        root / "results/raw/single" / model / "derail_phase5" / batch
        / row["task_id"] / row["world_variant_id"] / row["instruction_id"] / f"seed_{seed}"
    )


def write_screenshot(target: Path, action_index: int, data: bytes) -> None:
    path = target / "screenshots" / f"action_{action_index:04d}.png"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)


def rollout_id(row: dict, model: str, seed: int, image_digest: str) -> str:
    payload = f"{row['combination_id']}:{model}:{seed}:{image_digest}"
    return "phase5-" + hashlib.sha256(payload.encode()).hexdigest()[:20]


def task_config(row: dict) -> dict:
    return {"id": row["task_id"], "instruction": row["instruction"]}


def run_one(raw_env, get_agent, control, workload: dict, row: dict, args, root: Path) -> dict:
    model = workload["model"]
    image_digest = required_env("DERAIL_IMAGE_DIGEST")
    target = result_path(root, model, args.batch, row, args.seed)
    target.mkdir(parents=True, exist_ok=True)
    current_id = rollout_id(row, model, args.seed, image_digest)
    wrapper = DerailEnvWrapper(
        raw_env,
        control,
        lambda: TraceBuilder(current_id, row["task_id"], row["world_variant_id"], model,
                             args.seed, args.step_budget, image_digest=image_digest,
                             instruction=row["instruction"]),
        screenshot_sink=lambda index, data: write_screenshot(target, index, data),
    )
    agent = get_agent(model, env=wrapper)
    observation = wrapper.reset(task_config=task_config(row))
    if observation.get("screenshot"):
        write_screenshot(target, -1, observation["screenshot"])
    execute_loop(agent, wrapper, observation, row["instruction"], args)
    gold = json.loads(Path(row["gold_lineage_path"]).read_text(encoding="utf-8"))
    verdict = verify_changelog(gold, wrapper.builder.steps)
    trace = wrapper.finish(
        step_budget_reached=wrapper.actions_taken >= args.step_budget,
        final_verifier=verdict["passed"],
        verifier_kind=verdict["kind"],
        provenance=provenance(workload, row),
    )
    write_result(target, trace, verdict)
    return terminal_row(row, trace, verdict, target)


def execute_loop(agent, wrapper, observation: dict, instruction: str, args) -> None:
    done = False
    while not done and wrapper.actions_taken < args.step_budget:
        response, actions = agent.predict(instruction, observation)
        wrapper.note_thought(str(response)[:2000] if response else None)
        for action in actions or []:
            observation, _, done, _ = wrapper.step(action, args.pause_after_action_s)
            if done or wrapper.actions_taken >= args.step_budget:
                break


def provenance(workload: dict, row: dict) -> dict:
    return {
        "source_manifest_sha256": workload["source_manifest_sha256"],
        "combination_id": row["combination_id"],
        "instruction_id": row["instruction_id"],
        "hazard_ref": row["hazard_ref"],
        "task_config_sha256": row["task_config_sha256"],
        "verifier_sha256": row["verifier_sha256"],
        "gold_lineage_sha256": row["gold_lineage_sha256"],
    }


def write_result(target: Path, trace: dict, verdict: dict) -> None:
    (target / "rollout_trace.json").write_text(json.dumps(trace, ensure_ascii=False) + "\n")
    (target / "verifier_result.json").write_text(json.dumps(verdict, indent=2) + "\n")


def terminal_row(row: dict, trace: dict, verdict: dict, target: Path) -> dict:
    return {
        "combination_id": row["combination_id"],
        "rollout_id": trace["rollout_id"],
        "result_path": str(target),
        "steps": trace["outcome"]["steps_taken"],
        "verifier_passed": verdict["passed"],
        "status": "completed",
    }


def write_terminal(
    root: Path, workload: dict, args, rows: list[dict], errors: list[dict], started: float
) -> None:
    target = root / "results/raw/single" / workload["model"] / "derail_phase5" / args.batch
    payload = {
        "schema": "phase5-shard-terminal/1.0",
        "model": workload["model"],
        "source_manifest_sha256": workload["source_manifest_sha256"],
        "shard_count": workload["shard_count"],
        "shard_index": workload["shard_index"],
        "completed": rows,
        "errors": errors,
        "seconds": round(time.time() - started, 1),
    }
    (target / "terminal_manifest.json").write_text(json.dumps(payload, indent=2) + "\n")


def main() -> int:
    args = parse_args()
    workload, rows = load_rows(args.workload, set(args.combination_id))
    root = Path(required_env("DERAIL_PHASE5_OUT"))
    mypcbench = Path(required_env("DERAIL_MYPCBENCH_ROOT"))
    sys.path.insert(0, str(mypcbench / "agent-harness"))
    from env import MyPCBenchEnv  # type: ignore
    from run_mypcbench import get_agent  # type: ignore

    control = DerailControlClient(required_env("DERAIL_CONTROL_API_URL"))
    started, completed, errors = time.time(), [], []
    raw_env = MyPCBenchEnv()
    try:
        for row in rows:
            try:
                completed.append(run_one(raw_env, get_agent, control, workload, row, args, root))
            except Exception as exc:
                errors.append({
                    "combination_id": row["combination_id"],
                    "type": type(exc).__name__,
                    "message": str(exc)[:1000],
                })
    finally:
        raw_env.close()
    write_terminal(root, workload, args, completed, errors, started)
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
