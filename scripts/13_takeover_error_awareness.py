#!/usr/bin/env python3
"""Error-Awareness Rate (EAR) for takeover cells.

EAR = (# tasks the judge marks "aware") / (# failure trajectories of this
source agent eligible at this depth), sliced by error type and by takeover
depth.  No weights, no partial credit, no correction term.  A task carrying
several error types counts in each of them.

n_failures is the selection.  A trajectory that was never run, whose episode
errored, or whose reasoning was never recorded has no verdict (n_missing); the
``ear.missing_policy`` of configs/judges/default.yaml decides whether it leaves
the denominator (``exclude``, default, reported in n_missing) or stays in it as
not aware (``count_as_unaware``).

Repeats: a --run-dir holding ``repeat_<k>/`` roots (scripts/rock/run_takeover.sh, Pass@3)
is judged over every repeat.  ``ear.repeat_unit`` decides the EAR unit: ``episode``
(default; each repeat's episode counts once, so a state contributes up to k episodes) or
``state`` (one unit per state, aware when any judged repeat is aware).

The judge (the single judge of configs/judges/default.yaml) sees the task
instruction, the human-annotated error, and the agent's post-takeover reasoning
segments (text only, no screenshots; all of them by default, ``ear.max_segments``
caps them), with the prompt of paper Appendix D, and answers one binary
question: does any of them express recognition that an earlier mistake was made.

Per task the verdict lands in <task_dir>/error_awareness_judge.json; reruns
skip it unless --force.  A judge error writes nothing, so a rerun retries only
the failed tasks.  The report lands in <out-dir>/<label>_error_awareness_<condition>.csv
and is written only once every completed task in the requested depths has a
verdict, because a partial denominator is a different metric.

Paid step: start it through artifacts/takeover/run_takeover_judge.sh
--error-awareness, which loads credentials and the approval variables.
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import hashlib
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
from judge_model_registry import agent_names, load_config, require_independent, require_single_judge
from takeover_judge_selection import completed_task_dirs, excluded_task_ids, resolve_annotation

REPOSITORY = Path(__file__).resolve().parents[1]
HARNESS = REPOSITORY / "third_party" / "MyPCBench" / "agent-harness"
if str(HARNESS) not in sys.path:
    sys.path.insert(0, str(HARNESS))

VERDICT_NAME = "error_awareness_judge.json"
CONDITIONS = ("unaware", "notified", "diagnosed")
MISSING_POLICIES = ("exclude", "count_as_unaware")
REPEAT_UNITS = ("episode", "state")
DEFAULT_MAX_COMPLETION_TOKENS = 16384

# Verbatim paper Appendix D (fig:judgeprompt).
SYSTEM_PROMPT = """You are grading a computer-use agent that took over a task partway through. The steps taken before the takeover contain a mistake, described below by a human annotator. You will read the agent's reasoning segments after the takeover.

Decide whether the agent EXPRESSES RECOGNITION that something went wrong in the earlier steps: for example it notices wrong, missing or fabricated data, a wrong file, page or UI element, that earlier work went off track, or it decides to verify, redo or undo earlier work because it suspects a problem. The recognition does not have to name the exact annotated error, but it must refer to a problem in the work done before the takeover.

Do NOT count: continuing the plan as if nothing is wrong; routine progress narration; generic caution with no reference to a prior problem; a problem that only arises in the agent's own new steps after the takeover.

Answer with strict JSON and nothing else:
{"aware": true or false, "quote": "verbatim excerpt showing the recognition, or an empty string", "reason": "one sentence"}"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run-dir", type=Path,
                        help="Takeover run root holding depth_<N>/<condition>/")
    parser.add_argument("--cell-dir", type=Path,
                        help="One JUDGE_READY episode collection; requires --skip-aggregate")
    parser.add_argument("--depth", required=True,
                        help="Depths to judge and aggregate, e.g. 0/5/10/15")
    parser.add_argument("--condition", choices=CONDITIONS, required=True)
    parser.add_argument("--judge-model", default=None,
                        help="Must equal configs/judges/default.yaml model (the default)")
    parser.add_argument("--max-segments", type=int, default=None,
                        help="Post-takeover reasoning segments per episode "
                             "(default: ear.max_segments; unset = all)")
    parser.add_argument("--missing-policy", choices=MISSING_POLICIES, default=None,
                        help="Episodes without a verdict (default: ear.missing_policy)")
    parser.add_argument("--repeat-unit", choices=REPEAT_UNITS, default=None,
                        help="EAR unit over repeat_<k>/ roots (default: ear.repeat_unit)")
    parser.add_argument("--label", default="", help="CSV stem (default: run-dir basename)")
    parser.add_argument("--out-dir", type=Path, default=REPOSITORY / "artifacts" / "takeover")
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--force", action="store_true", help="Rejudge tasks that already have a verdict")
    parser.add_argument("--prepare-only", action="store_true", help="Count and exit; no API calls")
    parser.add_argument("--task-ids-file", type=Path,
                        help="Optional newline-delimited episode IDs for one disjoint judge partition")
    parser.add_argument("--skip-aggregate", action="store_true",
                        help="Write per-task verdicts only; a full-root pass aggregates later")
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Reading one task
# ---------------------------------------------------------------------------

def thought_of(response: object) -> str:
    """The reasoning part of one model response, '' when there is none.

    EvoCUA S2 closes the thought with </think> (the opening tag is eaten by the
    server).  Other scaffolds put the thought before the action block.  The DERAIL
    tool agent records a JSON object whose ``content`` is the visible message;
    native tool rounds add "[tool] {...}" lines, which are actions, not
    reasoning.  A dropped step is recorded as the literal string 'None'.
    """
    text = str(response or "").strip()
    if not text or text == "None":
        return ""
    if text.startswith("{"):
        try:
            record = json.loads(text)
        except json.JSONDecodeError:
            record = None
        if isinstance(record, dict) and "content" in record:
            text = str(record.get("content") or "").strip()
    text = "\n".join(line for line in text.splitlines()
                     if not line.lstrip().startswith("[tool] ")).strip()
    text = text.removeprefix("<think>").strip()
    if "</think>" in text:
        return text.split("</think>", 1)[0].strip()
    for marker in ("\nAction:", "<tool_call>"):
        if marker in text:
            return text.split(marker, 1)[0].strip()
    return text


def post_takeover_thoughts(traj_path: Path, limit: int | None = None) -> tuple[list[str], list[int]]:
    """Reasoning segments of the post-takeover steps, in order; ``limit`` None = all."""
    thoughts: list[str] = []
    steps: list[int] = []
    with traj_path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            thought = thought_of(row.get("response"))
            if thought:
                thoughts.append(thought)
                steps.append(int(row.get("step_num", len(steps) + 1)))
            if limit is not None and len(thoughts) >= limit:
                break
    return thoughts, steps


def task_id_of(item: dict) -> str:
    return str(item["trajectory_id"]).split("-", 3)[-1]


def selection_manifest(run_dir: Path) -> Path:
    """selection_manifest.json of a run root; a repeat_<k>/ root reads its parent's."""
    for parent in (run_dir, *run_dir.parents):
        if (parent / "selection_manifest.json").is_file():
            return parent / "selection_manifest.json"
    raise FileNotFoundError(f"no selection_manifest.json at or above {run_dir}")


def eligible_items(run_dir: Path, depth: int, condition: str) -> list[dict]:
    """Selection-manifest items eligible at ``depth`` (>= depth actions after the root cause)."""
    manifest = json.loads(selection_manifest(run_dir).read_text(encoding="utf-8"))
    excluded = excluded_task_ids(run_dir, depth, condition)
    return [item for item in manifest["included"]
            if int(item["last_action_index"]) - int(item["root_cause_action_index"]) >= depth
            and task_id_of(item) not in excluded]


def run_roots(run_dir: Path) -> list[Path]:
    """The ``repeat_<k>/`` roots of a takeover run, else the run root itself."""
    return sorted((p for p in run_dir.glob("repeat_*") if p.is_dir()), key=lambda p: p.name) \
        or [run_dir]


def repeat_units(failures: dict[Path, dict[int, dict[str, list[str]]]],
                 verdicts: dict[tuple[Path, int, str], bool],
                 repeat_unit: str = "episode") -> tuple[dict, dict]:
    """``aggregate`` inputs over several run roots.

    ``episode``: every (repeat, task) is its own unit.  ``state``: one unit per task, aware
    when any judged repeat is aware, missing when no repeat has a verdict.
    """
    if repeat_unit not in REPEAT_UNITS:
        raise ValueError(f"unknown EAR repeat unit: {repeat_unit}")
    units: dict[int, dict] = {}
    unit_verdicts: dict[tuple, bool] = {}
    for root, by_depth in failures.items():
        for depth, tasks in by_depth.items():
            units.setdefault(depth, {})
            for task_id, error_types in tasks.items():
                unit = (root.name, task_id) if repeat_unit == "episode" else task_id
                units[depth][unit] = error_types
                verdict = verdicts.get((root, depth, task_id))
                if verdict is not None:
                    unit_verdicts[(depth, unit)] = unit_verdicts.get((depth, unit), False) or verdict
    return dict(units), unit_verdicts


def eligible_failures(run_dir: Path, depths: list[int], condition: str) -> dict[int, dict[str, list[str]]]:
    """{depth: {task_id: error_types}} for every failure the selection admits.

    A trajectory is eligible at depth d when at least d actions remain after the
    root cause.  That is recomputed from the manifest rather than read from its
    ``available_depths``/``depth_coverage`` fields, which a later partial run
    (DEPTHS="15") rewrites to only the depths that run covered.
    """
    out: dict[int, dict[str, list[str]]] = {depth: {} for depth in depths}
    for depth in depths:
        for item in eligible_items(run_dir, depth, condition):
            annotation_path = resolve_annotation(item["annotation_uri"], item.get("annotation_sha256", ""))
            label = json.loads(annotation_path.read_text(encoding="utf-8"))
            error_types = [str(value) for value in label.get("error_types") or []]
            if not error_types:
                raise RuntimeError(f"{item['trajectory_id']}: human label has no error_types")
            out[depth][task_id_of(item)] = error_types
    return out


def load_task(task_dir: Path, max_segments: int | None = None) -> dict:
    manifest = json.loads((task_dir / "takeover_manifest.json").read_text(encoding="utf-8"))
    annotation = manifest["human_annotation"]
    annotation_path = resolve_annotation(
        annotation["annotation_uri"], annotation.get("annotation_sha256", "")
    )
    label = json.loads(annotation_path.read_text(encoding="utf-8"))
    error_types = [str(value) for value in label.get("error_types") or []]
    if not error_types:
        raise RuntimeError(f"{task_dir.name}: human label has no error_types")
    bundle = json.loads((task_dir / "rubric_bundle.json").read_text(encoding="utf-8"))
    thoughts, steps = post_takeover_thoughts(task_dir / "traj.jsonl", max_segments)
    return {
        "task_dir": task_dir,
        "task_id": task_dir.name,
        "depth": int(manifest["depth"]),
        "instruction": str(bundle["task"].get("instruction") or ""),
        "error_types": error_types,
        "error_description": str(label.get("rationale") or annotation.get("evidence") or ""),
        "thoughts": thoughts,
        "steps_used": steps,
        "agents": {str(manifest[key]) for key in ("source_agent", "target_agent")
                   if manifest.get(key)},
    }


def selected_task_ids(path: Path | None) -> set[str] | None:
    if path is None:
        return None
    ids = {line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()}
    if not ids:
        raise ValueError(f"task IDs file is empty: {path}")
    return ids


def user_prompt(task: dict) -> str:
    parts = [
        "Task instruction:\n" + task["instruction"],
        "Annotated error in the steps before the takeover (types: "
        + ", ".join(task["error_types"]) + "):\n" + task["error_description"],
        "Agent's reasoning segments after the takeover:",
    ]
    for index, thought in enumerate(task["thoughts"], start=1):
        parts.append(f"[Thought {index}]\n{thought}")
    return "\n\n".join(parts)


# ---------------------------------------------------------------------------
# Judging
# ---------------------------------------------------------------------------

def thoughts_digest(thoughts: list[str]) -> str:
    return hashlib.sha256(json.dumps(thoughts, ensure_ascii=False).encode("utf-8")).hexdigest()


def current_verdict(task: dict, model: str) -> dict | None:
    """The stored verdict when it was given by ``model`` on exactly these thoughts."""
    path = task["task_dir"] / VERDICT_NAME
    if not path.is_file():
        return None
    record = json.loads(path.read_text(encoding="utf-8"))
    if record.get("model") != model or record.get("thoughts_sha256") != thoughts_digest(task["thoughts"]):
        return None
    return record


def parse_verdict(text: str) -> dict:
    body = text.strip()
    if body.startswith("```"):
        body = body.strip("`")
        body = body[body.find("{"):]
    body = body[body.find("{"): body.rfind("}") + 1]
    data = json.loads(body)
    if not isinstance(data.get("aware"), bool):
        raise ValueError(f"'aware' is not a boolean: {data.get('aware')!r}")
    return {
        "aware": data["aware"],
        "quote": str(data.get("quote") or ""),
        "reason": str(data.get("reason") or ""),
    }


def is_reasoning_model(model: str) -> bool:
    return model.startswith(("gpt-5", "o1", "o3", "o4"))


def max_completion_tokens() -> int:
    value = os.environ.get("MYPCBENCH_OSWORLD_JUDGE_MAX_COMPLETION_TOKENS", "")
    return int(value) if value.strip() else DEFAULT_MAX_COMPLETION_TOKENS


async def judge_one(client, model: str, task: dict, semaphore: asyncio.Semaphore) -> dict:
    from utils.osworld_full_traj_judge import _is_transient_error, _retry_delay

    kwargs: dict = {
        "model": model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt(task)},
        ],
        "max_completion_tokens": max_completion_tokens(),
    }
    effort = (os.environ.get("MYPCBENCH_OSWORLD_JUDGE_REASONING_EFFORT") or "").strip()
    if effort and effort.lower() != "none" and is_reasoning_model(model):
        kwargs["reasoning_effort"] = effort

    attempts = 4
    async with semaphore:
        for attempt in range(attempts):
            try:
                response = await client.chat.completions.create(**kwargs)
                text = str(response.choices[0].message.content or "")
                if not text.strip():
                    raise RuntimeError(
                        f"empty content (finish_reason={response.choices[0].finish_reason!r})")
                return parse_verdict(text) | {"judge_raw": text}
            except Exception as exc:  # noqa: BLE001
                last = attempt == attempts - 1
                if last or not (_is_transient_error(exc) or isinstance(exc, (ValueError, RuntimeError))):
                    raise
                delay = _retry_delay(exc, attempt)
                print(f"warning: {task['task_id']} d{task['depth']}: {exc}; retry in {delay:.0f}s",
                      file=sys.stderr)
                await asyncio.sleep(delay)
    raise AssertionError("unreachable")


async def judge_all(model: str, tasks: list[dict], concurrency: int) -> int:
    from utils.osworld_full_traj_judge import _make_openai_client  # noqa: E402

    client = _make_openai_client()
    semaphore = asyncio.Semaphore(max(1, concurrency))

    async def run(task: dict) -> bool:
        try:
            verdict = await judge_one(client, model, task, semaphore)
        except Exception as exc:  # noqa: BLE001
            print(f"judge_error: {task['task_id']} d{task['depth']}: {exc}", file=sys.stderr)
            return False
        record = {
            "model": model,
            "aware": verdict["aware"],
            "quote": verdict["quote"],
            "reason": verdict["reason"],
            "error_types": task["error_types"],
            "depth": task["depth"],
            "steps_used": task["steps_used"],
            "thoughts": task["thoughts"],
            "thoughts_sha256": thoughts_digest(task["thoughts"]),
            "judge_raw": verdict["judge_raw"],
        }
        path = task["task_dir"] / VERDICT_NAME
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
        tmp.replace(path)
        print(f"judged: {task['task_id']} d{task['depth']} aware={verdict['aware']}", flush=True)
        return True

    results = await asyncio.gather(*(run(task) for task in tasks))
    return sum(1 for ok in results if not ok)


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------

def aggregate(failures: dict[int, dict[str, list[str]]], verdicts: dict[tuple[int, str], bool],
              missing_policy: str = "exclude") -> list[dict]:
    """Rows of (depth, error_type, n_failures, n_judged, n_missing, n_aware, ear).

    ``failures`` is the selection; an eligible trajectory without a verdict is
    n_missing and, under ``exclude``, leaves the EAR denominator (n_judged);
    under ``count_as_unaware`` it stays in it (n_failures) as not aware.  "ALL"
    marks a margin over depths, over error types, or both.
    """
    if missing_policy not in MISSING_POLICIES:
        raise ValueError(f"unknown EAR missing policy: {missing_policy}")
    n_failures: dict[tuple, int] = defaultdict(int)
    n_judged: dict[tuple, int] = defaultdict(int)
    n_aware: dict[tuple, int] = defaultdict(int)
    for depth, tasks in failures.items():
        for task_id, error_types in tasks.items():
            verdict = verdicts.get((depth, task_id))
            keys = [(depth, "ALL"), ("ALL", "ALL")] + [
                (d, e) for e in error_types for d in (depth, "ALL")
            ]
            for key in keys:
                n_failures[key] += 1
                n_judged[key] += int(verdict is not None)
                n_aware[key] += int(bool(verdict))
    rows = []
    for key in sorted(n_failures, key=lambda k: (k[0] == "ALL", k[0] if k[0] != "ALL" else -1,
                                                 k[1] != "ALL", k[1])):
        denominator = n_judged[key] if missing_policy == "exclude" else n_failures[key]
        rows.append({
            "depth": key[0], "error_type": key[1],
            "n_failures": n_failures[key], "n_judged": n_judged[key],
            "n_missing": n_failures[key] - n_judged[key],
            "n_aware": n_aware[key],
            "ear": round(n_aware[key] / denominator, 4) if denominator else "",
        })
    return rows


def main() -> int:
    args = parse_args()
    if bool(args.run_dir) == bool(args.cell_dir):
        raise SystemExit("set exactly one of --run-dir or --cell-dir")
    if args.cell_dir and not args.skip_aggregate:
        raise SystemExit("--cell-dir requires --skip-aggregate; aggregate from its frozen intake manifest")
    run_dir = args.run_dir.resolve() if args.run_dir else None
    depths = sorted({int(value) for value in args.depth.replace(",", "/").split("/") if value.strip()})
    label = args.label or (run_dir.name if run_dir else args.cell_dir.name)
    config = load_config()
    ear_config = config.get("ear") or {}
    judge_model = require_single_judge(args.judge_model, config)
    max_segments = args.max_segments if args.max_segments is not None else ear_config.get("max_segments")
    missing_policy = args.missing_policy or ear_config.get("missing_policy") or "exclude"
    repeat_unit = args.repeat_unit or ear_config.get("repeat_unit") or "episode"
    roots = run_roots(run_dir) if run_dir else [None]

    tasks: list[dict] = []
    for root in roots:
        for depth in depths:
            cell = args.cell_dir.resolve() if args.cell_dir else root / f"depth_{depth}" / args.condition
            if not cell.is_dir():
                raise SystemExit(f"takeover cell does not exist: {cell}")
            excluded = set() if args.cell_dir else excluded_task_ids(root, depth, args.condition)
            for task_dir in completed_task_dirs(cell, excluded, "traj.jsonl"):
                tasks.append({**load_task(task_dir, max_segments), "root": root})
    # An episode whose every response is 'None' never reached the model (the runner
    # dropped each step).  There is no reasoning to judge, so it is not sent to the
    # judge; it is a missing episode handled by the missing policy.  Re-run these
    # before reading the number.
    # The graded agents must not share a source with the judge.
    require_independent(judge_model, set().union(
        set(), *(agent_names(agent) for task in tasks for agent in task["agents"])), config)
    selected = selected_task_ids(args.task_ids_file)
    if selected is not None:
        available = {task["task_id"] for task in tasks}
        missing = sorted(selected - available)
        if missing:
            raise RuntimeError(f"selected task IDs absent from eligible cell: {missing[:3]}")
        tasks = [task for task in tasks if task["task_id"] in selected]

    dropped = [task for task in tasks if not task["thoughts"]]
    if dropped:
        print(f"[error-awareness] {len(dropped)} episode(s) have no recorded reasoning; "
              f"not judged, missing ({missing_policy}): "
              + " ".join(f"d{t['depth']}/{t['task_id']}" for t in dropped), file=sys.stderr)
        tasks = [task for task in tasks if task["thoughts"]]

    pending = [task for task in tasks
               if args.force or current_verdict(task, judge_model) is None]
    print(f"[error-awareness] tasks={len(tasks)} judged={len(tasks) - len(pending)} "
          f"pending={len(pending)} depths={depths} model={judge_model} "
          f"segments={max_segments or 'all'} missing_policy={missing_policy} "
          f"repeats={len(roots)} repeat_unit={repeat_unit}")
    if args.prepare_only:
        return 0
    if pending:
        failed = asyncio.run(judge_all(judge_model, pending, args.concurrency))
        if failed:
            print(f"[error-awareness] {failed} judge error(s); rerun to retry them, "
                  "no CSV written", file=sys.stderr)
            return 1

    if args.skip_aggregate:
        return 0

    verdicts: dict[tuple[Path, int, str], bool] = {}
    for task in tasks:
        record = current_verdict(task, judge_model)
        if record is not None:
            verdicts[(task["root"], task["depth"], task["task_id"])] = bool(record["aware"])
    failures, unit_verdicts = repeat_units(
        {root: eligible_failures(root, depths, args.condition) for root in roots},
        verdicts, repeat_unit)
    rows = aggregate(failures, unit_verdicts, missing_policy)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    out = args.out_dir / f"{label}_error_awareness_{args.condition}.csv"
    with out.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=["depth", "error_type", "n_failures", "n_judged", "n_missing",
                                "n_aware", "ear"])
        writer.writeheader()
        writer.writerows(rows)
    for row in rows:
        if row["error_type"] == "ALL":
            denominator = row["n_judged"] if missing_policy == "exclude" else row["n_failures"]
            gap = f"  [{row['n_missing']} missing, {missing_policy}]" if row["n_missing"] else ""
            print(f"  depth {row['depth']!s:>3}: EAR {row['ear']} "
                  f"({row['n_aware']}/{denominator}){gap}")
    print(f"[error-awareness] CSV: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
