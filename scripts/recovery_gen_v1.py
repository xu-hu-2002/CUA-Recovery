#!/usr/bin/env python3
"""Recovery generation for WATCHDOG (paper 03:28, appendix F:97-101).  VM/GPU side.

    DERAIL_RECOVERY_OUT=<dir> python scripts/recovery_gen_v1.py --cases <cases.jsonl> \
        --gold-dir <gold_lineage dir> --ir-dir <task_ir dir> --task-dir <task config dir> \
        --qcow2 <image> --live-db-dir <pulled /data/*.sqlite> [--files-root <pulled home>]

For every recovery case of ``scripts/build_training_v1.py`` (format ``recovery``): each
attempt resets the VM, replays the prefix ``0..c`` and injects ``H_c`` as the agent's own
history through the takeover runner (``PrefixTakeoverAgent``, reasoning stripped as in
evaluation), with the hint of the current level as the takeover prompt; the base agent then
continues for at most ``max_recovery_steps``.  The task's frozen final verifier decides on
the databases the harness pulls into ``--live-db-dir`` after the continuation.  The accepted,
leak-checked continuation becomes one ``(l, H_c, r*)`` sample (deduplicated).  Settings:
``recovery`` section of ``configs/train/sft_v1.yaml``.  Fake-testable core:
``derail.train.recovery_gen``.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
import time
from dataclasses import replace
from pathlib import Path

import yaml

REPO_ROOT = Path(os.environ.get("DERAIL_REPO_ROOT", Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(REPO_ROOT / "src"))

from derail.canonical.actions import action_from_dict  # noqa: E402
from derail.canonical.mypcbench import load_canonical_jsonl  # noqa: E402
from derail.canonical.trajectory import CanonicalStep  # noqa: E402
from derail.derived.layout import sha256_file  # noqa: E402
from derail.gen.verifiers import verify_final_state, world_sources  # noqa: E402
from derail.ir.gold_interpreter import GoldInterpreter, InterpreterConfig, WorldCopy  # noqa: E402
from derail.ir.model import load_task_ir  # noqa: E402
from derail.takeover.diagnosis import HumanDiagnosisEvidence  # noqa: E402
from derail.rollout.state_probe import EnvironmentHooks  # noqa: E402
from derail.takeover.protocol import load_takeover_config  # noqa: E402
from derail.train.build_samples import BuildConfig, target_loss_weights  # noqa: E402
from derail.train.recovery_gen import (  # noqa: E402
    RecoveryConfig,
    generate_recovery,
    recovery_sample,
    sample_key,
)


def _env(name: str) -> str:
    value = os.environ.get(name, "")
    if not value:
        raise SystemExit("environment variable %s is required" % name)
    return value


def _jsonl(path: Path):
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def _takeover_runner():
    """The per-episode takeover runner (target construction helpers are reused from it)."""

    spec = importlib.util.spec_from_file_location(
        "takeover_runner", REPO_ROOT / "scripts" / "10_run_takeover_rollout.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def canonical_steps(case):
    """The prefix to replay: the source's canonical trajectory when the trace records it,
    else ``H_c`` itself (the trace's canonical actions)."""

    uri = (case.get("provenance") or {}).get("canonical_trajectory_uri")
    if uri:
        return load_canonical_jsonl(Path(uri))
    return tuple(
        CanonicalStep(
            step_id=int(h["action_index"]),
            action=action_from_dict(h["action"]["canonical"]),
            observation_before_sha256="",
            source_agent=str(case["agent"]),
        )
        for h in case["input"]["history"]
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--cases", type=Path, required=True)
    parser.add_argument("--gold-dir", type=Path, required=True)
    parser.add_argument("--ir-dir", type=Path, required=True)
    parser.add_argument(
        "--task-dir", type=Path, required=True, help="MyPCBench task configs <workflow_id>.json"
    )
    parser.add_argument("--qcow2", type=Path, required=True)
    parser.add_argument("--qcow2-sha256", default="")
    parser.add_argument("--config", type=Path, default=REPO_ROOT / "configs/train/sft_v1.yaml")
    parser.add_argument(
        "--takeover-config", type=Path, help="default: configs/takeover/takeover.yaml"
    )
    parser.add_argument(
        "--interpreter-config",
        type=Path,
        default=REPO_ROOT / "configs/synthesis/gold_interpreter_v1.yaml",
    )
    parser.add_argument(
        "--extractor-config",
        type=Path,
        default=REPO_ROOT / "configs/synthesis/task_ir_v1_extractor.yaml",
        help="v1.app_databases: app id -> database stem",
    )
    parser.add_argument("--base-agent", help="default: recovery.base_agent")
    parser.add_argument("--teacher-agent", help="default: recovery.teacher.agent")
    parser.add_argument("--attempts-per-level", type=int)
    parser.add_argument("--max-recovery-steps", type=int)
    parser.add_argument("--leak-policy", choices=("drop", "resample", "redact", "off"))
    parser.add_argument(
        "--live-db-dir",
        type=Path,
        required=True,
        help="where the harness pulls the VM's /data/*.sqlite after a continuation",
    )
    parser.add_argument("--files-root", type=Path, help="pulled home file tree (file verifiers)")
    parser.add_argument("--persona", default="michael_scott")
    parser.add_argument("--world", default="scranton-office")
    parser.add_argument("--backend", choices=("qemu", "docker"), default="qemu")
    parser.add_argument("--client-password", default="password")
    args = parser.parse_args()
    out = Path(_env("DERAIL_RECOVERY_OUT"))
    config = RecoveryConfig.from_yaml(args.config)
    overrides = {
        "base_agent": args.base_agent,
        "teacher_agent": args.teacher_agent,
        "attempts_per_level": args.attempts_per_level,
        "max_recovery_steps": args.max_recovery_steps,
        "leak_policy": args.leak_policy,
    }
    config = replace(config, **{k: v for k, v in overrides.items() if v is not None})
    loss = target_loss_weights(BuildConfig.from_yaml(args.config, REPO_ROOT))
    takeover = load_takeover_config(args.takeover_config)
    environment_hooks = EnvironmentHooks.from_config(Path(takeover["environment_config"]))
    app_databases = yaml.safe_load(args.extractor_config.read_text(encoding="utf-8"))["v1"][
        "app_databases"
    ]
    interpreter = GoldInterpreter(InterpreterConfig.from_yaml(args.interpreter_config))
    runner = _takeover_runner()
    from env import MyPCBenchEnv  # type: ignore  # on sys.path via the runner
    from run_mypcbench import get_agent  # type: ignore

    from derail.mypcbench.agent_config import load_config
    from derail.mypcbench.takeover_agent import PrefixTakeoverAgent

    cases = _jsonl(args.cases)
    out.mkdir(parents=True, exist_ok=True)
    samples_path = out / "recovery_samples.jsonl"
    seen = {sample_key(s) for s in _jsonl(samples_path)}
    done = {r["sample_id"] for r in _jsonl(out / "attempts.jsonl")}
    todo = [c for c in cases if c["sample_id"] not in done]
    print("reading %d cases (%d to do)\nwriting %s" % (len(cases), len(todo), out), file=sys.stderr)
    qcow2 = args.qcow2.expanduser().resolve()
    qcow2_sha256 = args.qcow2_sha256.strip() or sha256_file(qcow2)
    env = MyPCBenchEnv(
        persona=args.persona,
        world=args.world,
        screen_size=(1280, 800),
        client_password=args.client_password,
        backend=args.backend,
        qcow2_path=str(qcow2),
    )

    def make_target(agent_id):
        live = runner._live_target_agent_id(agent_id)
        document = load_config(live).document
        target = get_agent(
            agent_type=str(document["agent_type"]),
            model=runner._runtime_model_name(
                agent_id, str(document.get("checkpoint") or document.get("model") or "")
            ),
            screen_size=(1280, 800),
            client_password=args.client_password,
            env=env,
        )
        target = runner.wrap_qwen35_takeover_target(target, agent_id)
        target = runner.wrap_claude_takeover_target(target, agent_id)
        target = runner.wrap_openai_takeover_target(
            target,
            agent_id,
            context_screenshots=int(takeover["history"]["openai_context_screenshots"]),
        )
        runner._assert_bash_environment_bound(target, env)
        return target

    started = time.time()
    try:
        for index, case in enumerate(todo, 1):
            tick = time.time()
            workflow = str(case["workflow_id"])
            task_ir = load_task_ir(args.ir_dir / ("%s.json" % workflow), REPO_ROOT)
            gold = json.loads(
                (args.gold_dir / ("%s.gold_lineage.json" % workflow)).read_text(encoding="utf-8")
            )
            task = json.loads((args.task_dir / ("%s.json" % workflow)).read_text(encoding="utf-8"))
            steps = canonical_steps(case)
            state = {"agent": None, "pending": [], "attempt": 0}

            def restore(hint, teacher):
                # Replay 0..c and inject H_c through the takeover runner; the hinted condition
                # puts the hint verbatim into the takeover prompt of the first post-prefix
                # turn.  The evidence record only binds the root-cause index.
                state["attempt"] += 1
                agent_id = config.teacher_agent if teacher == "external" else config.base_agent
                diagnosis = HumanDiagnosisEvidence(
                    annotation_id="%s#hint%d" % (case["sample_id"], state["attempt"]),
                    trajectory_id=str(case["rollout_id"]),
                    annotator_id="rerail-hint",
                    root_cause_action_index=int(case["root_cause_action_index"]),
                    evidence="",
                    annotation_uri="",
                    annotation_sha256="",
                    source_trajectory_sha256="",
                )
                state["agent"] = PrefixTakeoverAgent(
                    target_agent=make_target(agent_id),
                    environment=env,
                    task_config=task,
                    canonical_steps=steps,
                    diagnosis=diagnosis,
                    condition="hinted",
                    hint=hint,
                    source_agent=str(case["agent"]),
                    target_agent_id=agent_id,
                    qcow2_path=qcow2,
                    qcow2_sha256=qcow2_sha256,
                    qcow2_hash_preverified=True,
                    artifact_dir=out / "_attempts" / case["sample_id"] / str(state["attempt"]),
                    depth=int(case["depth"]),
                    strip_reasoning=bool(takeover["history"]["strip_reasoning"]),
                    evocua_cross_agent=bool(takeover["history"]["evocua_cross_agent"]),
                    replay_verification=takeover["replay_verification"],
                    environment_hooks=environment_hooks,
                )
                state["pending"] = []
                observation = env.reset(task_config=task)
                state["agent"].reset()
                return observation

            def policy(instruction, history, observation, hint):
                # H_c and the hint are already in the takeover agent's context.
                if not state["pending"]:
                    response, actions = state["agent"].predict(instruction, observation)
                    state["pending"] = [(str(response), a) for a in (actions or [])]
                    if not state["pending"]:
                        return None
                response, action = state["pending"].pop(0)
                if not isinstance(action, str):
                    return dict(action, thought=response)
                kind = "done" if action.strip().upper() in ("DONE", "FAIL") else "pyautogui"
                return {"type": kind, "raw": action, "thought": response}

            def step(action):
                raw = action.get("raw", action) if isinstance(action, dict) else action
                observation, _, done, _ = env.step(raw)
                if done:
                    state["pending"] = []
                return observation

            def verify():
                sources = world_sources(args.live_db_dir, app_databases)
                with WorldCopy.open(
                    workflow,
                    sources,
                    out / "_tmp" / case["sample_id"],
                    files_root=args.files_root,
                ) as world:
                    return verify_final_state(
                        task_ir, gold, world, interpreter, config.observation_ops
                    )["recovered"]

            result = generate_recovery(
                case,
                restore=restore,
                policy=policy,
                step=step,
                verify=verify,
                config=config,
                teacher_policy=policy if config.teacher_agent else None,
            )
            sample = recovery_sample(case, result, loss)
            if sample is not None and sample_key(sample) not in seen:  # Dedup(Q)
                seen.add(sample_key(sample))
                with samples_path.open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(sample, ensure_ascii=False) + "\n")
            with (out / "attempts.jsonl").open("a", encoding="utf-8") as handle:
                handle.write(
                    json.dumps(
                        {
                            "sample_id": case["sample_id"],
                            "accepted_level": result.hint_level,
                            "teacher": result.accepted.teacher if result.accepted else None,
                            "attempts": [
                                {
                                    "level": a.level,
                                    "attempt": a.attempt,
                                    "teacher": a.teacher,
                                    "verified": a.verified,
                                    "leaks": a.leaks,
                                    "steps": len(a.steps),
                                }
                                for a in result.attempts
                            ],
                        }
                    )
                    + "\n"
                )
            elapsed = time.time() - started
            print(
                "[%d/%d] %s level=%s %.0fs | elapsed %dm eta %dm"
                % (
                    index,
                    len(todo),
                    case["sample_id"],
                    result.hint_level,
                    time.time() - tick,
                    elapsed // 60,
                    (elapsed / index * (len(todo) - index)) // 60,
                ),
                file=sys.stderr,
            )
    finally:
        env.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
