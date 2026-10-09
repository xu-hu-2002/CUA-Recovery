#!/usr/bin/env python3
"""Run a batch of (task, world, agent, seed) rollouts with the RECOVERY ledgers."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import yaml

REPO_ROOT = Path(os.environ.get("RECOVERY_REPO_ROOT", Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(REPO_ROOT / "src"))

from recovery.harness.control_client import ControlApiError, RecoveryControlClient  # noqa: E402
from recovery.harness.env_wrapper import RecoveryEnvWrapper  # noqa: E402
from recovery.harness.trace_builder import TraceBuilder  # noqa: E402
from recovery.rollout.state_probe import EnvironmentHooks, install  # noqa: E402
from recovery.rollout.tasks import load_source_tasks  # noqa: E402


def _env(name: str, required: bool = True) -> str:
    value = os.environ.get(name, "")
    if required and not value:
        raise SystemExit("environment variable %s is required" % name)
    return value


def preflight(control: RecoveryControlClient, config: dict, image_digest: str) -> dict:
    try:
        status = control.status()
    except ControlApiError as exc:
        raise SystemExit("control API not reachable: %s" % exc)
    missing = [
        app for app, info in status.get("databases", {}).items() if info.get("triggers", 0) == 0
    ]
    if config["preflight"].get("require_triggers_installed") and missing:
        raise SystemExit(
            "triggers missing on: %s (run: python main.py collection install_vm_infra deploy)" % ", ".join(missing)
        )
    if config["preflight"].get("require_tracer_files") and not status.get("trace_files"):
        raise SystemExit("no tracer files under the trace directory; is the drop-in installed?")
    print(
        "image_digest=%s databases=%d trace_files=%s"
        % (image_digest, len(status.get("databases", {})), status.get("trace_files")),
        file=sys.stderr,
    )
    return status


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "--config", type=Path, default=REPO_ROOT / "configs/harness/rollout_v1.yaml"
    )
    parser.add_argument("--agent", action="append", default=[], help="RECOVERY agent id(s)")
    parser.add_argument("--task-id", action="append", default=[])
    parser.add_argument("--seed", type=int, action="append", default=[])
    parser.add_argument("--task-source", help="task source name (mypcbench | rerail_workflows)")
    parser.add_argument("--tasks-file", type=Path, help="task file in that source format (e.g. a shard)")
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    if config.get("schema_version") != "rollout-config/1.0":
        raise SystemExit("unsupported rollout config")
    mypcbench_root = Path(_env(config["mypcbench_root_env"]))
    control_url = _env(config["control_api_url_env"])
    image_digest = _env(config["image_digest_env"])
    out = Path(_env(config["output_dir_env"]))
    runtime = yaml.safe_load((REPO_ROOT / config["runtime_config"]).read_text(encoding="utf-8"))
    agents = args.agent or list(config.get("agents", []))
    seeds = args.seed or list(range(int(runtime["repeats"])))
    task_source = args.task_source or runtime["task_source"]
    if not agents:
        raise SystemExit("no agents: pass --agent or list them in the config")
    sys.path.insert(0, str(mypcbench_root / "agent-harness"))
    from env import MyPCBenchEnv  # type: ignore
    from run_mypcbench import get_agent  # type: ignore

    install(
        MyPCBenchEnv,
        EnvironmentHooks.from_config(
            REPO_ROOT / "configs/environments" / ("%s.yaml" % runtime["environment"])
        ),
    )
    tasks = load_source_tasks(task_source, args.tasks_file)
    wanted = set(args.task_id or config.get("task_ids", []))
    if wanted:
        tasks = [t for t in tasks if t["id"] in wanted]
    print(
        "reading %s (%d tasks)\nwriting %s" % (task_source, len(tasks), out),
        file=sys.stderr,
    )
    control = RecoveryControlClient(control_url)
    preflight(control, config, image_digest)
    out.mkdir(parents=True, exist_ok=True)
    frame = tuple(config.get("frame", (1280, 800)))
    budget = int(runtime["max_steps"])
    task_timeout = int(runtime["task_timeout"])
    jobs = [(task, agent, seed) for task in tasks for agent in agents for seed in seeds]
    started = time.time()
    results = []
    raw_env = MyPCBenchEnv()
    try:
        for index, (task, agent_id, seed) in enumerate(jobs, 1):
            tick = time.time()
            rollout_id = "%s__%s__s%d__%s" % (
                task["id"],
                agent_id,
                seed,
                hashlib.sha256(
                    ("%s%s%d%s" % (task["id"], agent_id, seed, image_digest)).encode()
                ).hexdigest()[:8],
            )
            wrapper = RecoveryEnvWrapper(
                raw_env,
                control,
                lambda: TraceBuilder(
                    rollout_id=rollout_id,
                    task_id=task["id"],
                    world_id=config["world_id"],
                    agent=agent_id,
                    seed=seed,
                    step_budget=budget,
                    sampling_round=int(config.get("sampling_round", 1)),
                    image_digest=image_digest,
                    frame=frame,
                ),
            )
            agent = get_agent(agent_id, env=wrapper)
            target = out / task["id"] / agent_id / ("seed_%d" % seed)
            target.mkdir(parents=True, exist_ok=True)
            raw_env.recovery_probe_dir = target
            obs = wrapper.reset(task_config=task)
            done, steps = False, 0
            while not done and steps < budget and time.time() - tick < task_timeout:
                response, actions = agent.predict(task["instruction"], obs)
                wrapper.note_thought(str(response)[:2000] if response else None)
                for action in actions or []:
                    obs, _, done, _ = wrapper.step(
                        action, float(config.get("pause_after_action_s", 2.0))
                    )
                    if done:
                        break
                steps = wrapper.actions_taken
            trace = wrapper.finish(
                step_budget_reached=steps >= budget,
                provenance={"agent_config": agent_id, "seed": seed},
            )
            (target / "rollout_trace.json").write_text(
                json.dumps(trace, ensure_ascii=False) + "\n", encoding="utf-8"
            )
            results.append(
                {
                    "rollout_id": rollout_id,
                    "task_id": task["id"],
                    "agent": agent_id,
                    "seed": seed,
                    "steps": trace["outcome"]["steps_taken"],
                    "outcome": trace["outcome"],
                }
            )
            elapsed = time.time() - started
            print(
                "[%d/%d] %s %s seed=%d steps=%d %.0fs | elapsed %dm eta %dm"
                % (
                    index,
                    len(jobs),
                    task["id"],
                    agent_id,
                    seed,
                    trace["outcome"]["steps_taken"],
                    time.time() - tick,
                    elapsed // 60,
                    (elapsed / index * (len(jobs) - index)) // 60,
                ),
                file=sys.stderr,
            )
    finally:
        raw_env.close()
    (out / "manifest.json").write_text(
        json.dumps(
            {
                "config": str(args.config),
                "runtime_config": config["runtime_config"],
                "task_source": task_source,
                "world_id": config["world_id"],
                "image_digest": image_digest,
                "agents": agents,
                "seeds": seeds,
                "tasks": len(tasks),
                "rollouts": results,
                "seconds": round(time.time() - started, 1),
            },
            indent=1,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
