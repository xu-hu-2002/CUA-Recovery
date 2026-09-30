#!/usr/bin/env bash
set -u

SOURCE_AGENT="${SOURCE_AGENT:-evocua_32b}"
TARGET_AGENT="${TARGET_AGENT:-rerail_35b_a3b}"
RUN_TAG="${RUN_TAG:-takeover_v1}"
CONDITIONS="${CONDITIONS:-unaware notified diagnosed}"
DEPTHS="${DEPTHS:-0 5 10 15 20 25}"
MAX_STEPS="${MAX_STEPS:-}"
RUN_JUDGE="${RUN_JUDGE:-1}"
REFRESH="${REFRESH:-15}"
BAR_WIDTH="${BAR_WIDTH:-26}"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
if [[ -z "$MAX_STEPS" ]]; then
  MAX_STEPS="$(python3 -c 'import sys, yaml; print(yaml.safe_load(open(sys.argv[1]))["max_steps"])' \
    "$REPO_ROOT/configs/takeover/takeover.yaml")"
fi
OUTPUT_ROOT="${OUTPUT_ROOT:-$REPO_ROOT/artifacts/model_outputs/takeover/$RUN_TAG/${SOURCE_AGENT}_to_${TARGET_AGENT}}"
if [[ "$OUTPUT_ROOT" != /* ]]; then OUTPUT_ROOT="$REPO_ROOT/$OUTPUT_ROOT"; fi

default_session="takeover-${RUN_TAG}-${SOURCE_AGENT}-to-${TARGET_AGENT}"
SESSION_NAME="${SESSION_NAME:-$(printf '%s' "$default_session" | sed 's/[^A-Za-z0-9_.-]/_/g')}"

usage() {
  cat <<'EOF'
Usage:
  bash scripts/takeover/watch_takeover.sh              # create/attach tmux dashboard
  bash scripts/takeover/watch_takeover.sh --detach     # create it without attaching
  bash scripts/takeover/watch_takeover.sh --once       # print one snapshot
  bash scripts/takeover/watch_takeover.sh --foreground # refresh in this terminal
  bash scripts/takeover/watch_takeover.sh --kill       # stop the dashboard session

The same SOURCE_AGENT, TARGET_AGENT, RUN_TAG, CONDITIONS, and DEPTHS overrides used
by scripts/takeover/run.sh may be supplied here. REFRESH defaults to 15 seconds.
EOF
}

render_once() {
  python3 - "$OUTPUT_ROOT" "$SOURCE_AGENT" "$TARGET_AGENT" "$RUN_TAG" \
    "$CONDITIONS" "$DEPTHS" "$MAX_STEPS" "$RUN_JUDGE" "$BAR_WIDTH" <<'PY'
import json
import math
import pathlib
import statistics
import sys
import time

root = pathlib.Path(sys.argv[1])
source, target, run_tag = sys.argv[2:5]
conditions = sys.argv[5].split()
depths = [int(value) for value in sys.argv[6].split()]
max_steps = int(sys.argv[7])
run_judge = sys.argv[8] == "1"
width = max(10, int(sys.argv[9]))
now = time.time()


def bar(done, total):
    filled = min(width, done * width // total) if total else 0
    return "█" * filled + "░" * (width - filled)


def duration(seconds):
    if seconds is None or not math.isfinite(seconds):
        return "-"
    seconds = max(0, int(seconds))
    days, seconds = divmod(seconds, 86400)
    hours, seconds = divmod(seconds, 3600)
    minutes, seconds = divmod(seconds, 60)
    if days:
        return f"{days}d{hours:02d}h"
    if hours:
        return f"{hours}h{minutes:02d}m"
    if minutes:
        return f"{minutes}m{seconds:02d}s"
    return f"{seconds}s"


def timestamp(epoch):
    return time.strftime("%m-%d %H:%M", time.localtime(epoch))


def read_epoch(name):
    path = root / name
    try:
        return float(path.read_text().strip())
    except (OSError, ValueError):
        return None


def active_row(task_dir, launch, traj, depth, condition):
    steps = 0
    last_update = launch.stat().st_mtime
    if traj.exists():
        try:
            lines = [line for line in traj.read_text(encoding="utf-8").splitlines() if line.strip()]
            steps = len(lines)
            if lines:
                try:
                    steps = int(json.loads(lines[-1]).get("step_num", steps))
                except (json.JSONDecodeError, TypeError, ValueError):
                    pass
            last_update = max(last_update, traj.stat().st_mtime)
        except OSError:
            pass
    trajectory_id = task_dir.name
    try:
        trajectory_id = str(json.loads(launch.read_text()).get("trajectory_id") or task_dir.name)
    except (OSError, json.JSONDecodeError):
        pass
    return (depth, condition, trajectory_id, steps, now - last_update, now - launch.stat().st_mtime)


selection = root / "selection.tsv"
coverage = {depth: 0 for depth in depths}
if selection.exists():
    try:
        for raw in selection.read_text(encoding="utf-8").splitlines():
            fields = raw.split("\t")
            if len(fields) < 8:
                continue
            available = {int(value) for value in fields[7].split(",") if value}
            for depth in depths:
                coverage[depth] += depth in available
    except (OSError, ValueError):
        pass

planned = sum(coverage.values()) * len(conditions)
groups = []
all_durations = []
active = []
completed = runner_errors = judged = started = bundles = 0
agent_done = agent_failed = timed_out = predict_crashed = judged_perfect = 0
newest_activity = 0.0
active_run_started_at = read_epoch(".takeover_active_run_started_at")

for depth in depths:
    for condition in conditions:
        group_root = root / f"depth_{depth}" / condition
        task_dirs = []
        try:
            task_dirs = [path for path in group_root.iterdir() if path.is_dir()]
        except OSError:
            pass

        g_started = g_done = g_runner_errors = g_judged = g_bundles = 0
        g_agent_done = g_agent_failed = g_timed_out = g_predict_crashed = 0
        g_judged_perfect = 0
        for task_dir in task_dirs:
            launch = task_dir / "takeover_launch.json"
            result = task_dir / "result.txt"
            bundle = task_dir / "rubric_bundle.json"
            judge = task_dir / "rubric_judge_result.json"
            traj = task_dir / "traj.jsonl"
            if launch.exists():
                g_started += 1
                newest_activity = max(newest_activity, launch.stat().st_mtime)
            if bundle.exists():
                g_bundles += 1
            if judge.exists():
                g_judged += 1
                newest_activity = max(newest_activity, judge.stat().st_mtime)
                try:
                    raw_judge = json.loads(judge.read_text()).get("result")
                    if (
                        isinstance(raw_judge, (int, float))
                        and not isinstance(raw_judge, bool)
                        and float(raw_judge) >= 99.9
                    ):
                        g_judged_perfect += 1
                except (OSError, json.JSONDecodeError, TypeError, ValueError):
                    pass
            if result.exists():
                g_done += 1
                newest_activity = max(newest_activity, result.stat().st_mtime)
                try:
                    value = float(result.read_text().strip())
                except (OSError, ValueError):
                    value = 0.0
                if value != 1.0 or (task_dir / "incomplete.txt").exists():
                    g_runner_errors += 1
                    if (
                        launch.exists()
                        and launch.stat().st_mtime > result.stat().st_mtime
                        and (
                            active_run_started_at is None
                            or launch.stat().st_mtime >= active_run_started_at
                        )
                    ):
                        active.append(active_row(task_dir, launch, traj, depth, condition))
                terminal_actions = set()
                if traj.exists():
                    try:
                        for line in traj.read_text(encoding="utf-8").splitlines():
                            if not line.strip():
                                continue
                            action = json.loads(line).get("action")
                            if action in {"DONE", "FAIL", "PREDICT_CRASH"}:
                                terminal_actions.add(action)
                    except (OSError, json.JSONDecodeError, TypeError, ValueError):
                        pass
                if "PREDICT_CRASH" in terminal_actions:
                    g_predict_crashed += 1
                elif "FAIL" in terminal_actions:
                    g_agent_failed += 1
                elif "DONE" in terminal_actions:
                    g_agent_done += 1
                else:
                    g_timed_out += 1
                if launch.exists():
                    elapsed = result.stat().st_mtime - launch.stat().st_mtime
                    if elapsed >= 0:
                        all_durations.append(elapsed)
            elif launch.exists() and (
                active_run_started_at is None
                or launch.stat().st_mtime >= active_run_started_at
            ):
                row = active_row(task_dir, launch, traj, depth, condition)
                active.append(row)
                newest_activity = max(newest_activity, now - row[4])

        total = coverage.get(depth, 0)
        groups.append(
            (depth, condition, total, g_started, g_done, g_runner_errors, g_judged)
        )
        started += g_started
        completed += g_done
        runner_errors += g_runner_errors
        bundles += g_bundles
        judged += g_judged
        agent_done += g_agent_done
        agent_failed += g_agent_failed
        timed_out += g_timed_out
        predict_crashed += g_predict_crashed
        judged_perfect += g_judged_perfect

started_at = read_epoch(".takeover_started_at")
paused_at = read_epoch(".takeover_paused_at")
if started_at is None:
    launch_times = []
    for launch in root.glob("depth_*/*/*/takeover_launch.json") if root.exists() else []:
        try:
            launch_times.append(launch.stat().st_mtime)
        except OSError:
            pass
    started_at = min(launch_times) if launch_times else None

print()
print(f"  DERAIL takeover   {run_tag}   {source} → {target}")
print(f"  output: {root}")
if started_at:
    print(f"  elapsed {duration(now - started_at)}   refresh {time.strftime('%H:%M:%S')}   budget {max_steps} steps")
else:
    print(f"  waiting to start   refresh {time.strftime('%H:%M:%S')}   budget {max_steps} steps")
if paused_at:
    print(f"  PAUSED {timestamp(paused_at)}   partial task directories {len(active)}")
print()

if not selection.exists():
    print("  Waiting for selection.tsv. Start scripts/takeover/run.sh first.")
    raise SystemExit(0)
if planned == 0:
    print("  selection.tsv contains no episodes for the requested DEPTHS.")
    raise SystemExit(0)

label_width = max(16, max((len(f"d{d}/{c}") for d in depths for c in conditions), default=16))
print("  " + "─" * 86)
for depth, condition, total, g_started, g_done, g_runner_errors, g_judged in groups:
    label = f"d{depth}/{condition}"
    note = f"  runner error {g_runner_errors}" if g_runner_errors else ""
    judge_note = f"  judge {g_judged}/{total}" if g_judged else ""
    print(
        f"  {label:<{label_width}} [{bar(g_done, total)}] "
        f"{g_done:>3}/{total:<3} {g_done * 100 // total if total else 0:>3}%"
        f"  start {g_started:>3}{judge_note}{note}"
    )
print("  " + "─" * 86)
print(
    f"  rollout {completed}/{planned} ({completed * 100 // planned}%)"
    f"   started {started}/{planned}   running {0 if paused_at else len(active)}"
)
print(
    f"  outcomes DONE {agent_done}   agent FAIL {agent_failed}   timeout {timed_out}"
    f"   predict crash {predict_crashed}"
)
print(
    f"  quality  runner error {runner_errors}   judged perfect {judged_perfect}/{judged}"
)

average = statistics.fmean(all_durations) if all_durations else None
if completed >= planned:
    print("  rollout complete")
elif paused_at:
    print("  rollout paused; ETA unavailable until resume")
elif average is not None:
    remaining_seconds = average * max(0, planned - completed)
    print(
        f"  rollout mean {duration(average)}/episode   remaining ≈ {duration(remaining_seconds)}"
        f"   ETA {timestamp(now + remaining_seconds)}"
    )
else:
    print("  rollout ETA: waiting for the first completed episode")

for depth, condition, trajectory_id, steps, idle, running_for in active[:3]:
    state = "paused" if paused_at else "active"
    warning = "  ⚠ no update" if not paused_at and idle >= 300 else ""
    timing = (
        f"captured after {duration(running_for)}"
        if paused_at
        else f"running {duration(running_for)}   idle {duration(idle)}"
    )
    print(
        f"  {state} d{depth}/{condition}  {trajectory_id}"
        f"   step {steps}/{max_steps}   {timing}{warning}"
    )
if len(active) > 3:
    print(f"  ... plus {len(active) - 3} other active directories")

print()
if run_judge:
    print(f"  judge   {judged}/{planned} ({judged * 100 // planned}%)   bundles {bundles}/{planned}")
    judge_started = read_epoch(".judge_started_at")
    if judged >= planned:
        print("  judge complete; PESR summary should be available in takeover_comparison.json")
    elif judge_started and judged:
        judge_rate = (now - judge_started) / judged
        judge_remaining = judge_rate * max(0, planned - judged)
        print(
            f"  judge mean {duration(judge_rate)}/episode   remaining ≈ {duration(judge_remaining)}"
            f"   ETA {timestamp(now + judge_remaining)}"
        )
    elif completed >= planned:
        print("  judge ETA: waiting for the first judged episode")
    else:
        print("  judge queued after all rollouts; its duration is not included in rollout ETA")
else:
    print("  judge disabled for this run (RUN_JUDGE=0)")

if newest_activity:
    idle = now - newest_activity
    if idle >= 900 and completed < planned:
        print(f"\n  ⚠ No artifact update for {duration(idle)}; inspect the rollout process/logs.")
print()
PY
}

watch_foreground() {
  while true; do
    clear 2>/dev/null || printf '\n\n'
    render_once
    sleep "$REFRESH"
  done
}

mode="tmux"
for argument in "$@"; do
  case "$argument" in
    --once) mode="once" ;;
    --foreground) mode="foreground" ;;
    --detach) mode="detach" ;;
    --kill) mode="kill" ;;
    -h|--help) usage; exit 0 ;;
    *) printf 'Unknown argument: %s\n' "$argument" >&2; usage >&2; exit 2 ;;
  esac
done

case "$mode" in
  once) render_once; exit ;;
  foreground) watch_foreground; exit ;;
  kill)
    if command -v tmux >/dev/null 2>&1 && tmux has-session -t "$SESSION_NAME" 2>/dev/null; then
      tmux kill-session -t "$SESSION_NAME"
      printf 'Stopped tmux session: %s\n' "$SESSION_NAME"
    else
      printf 'No tmux session named %s\n' "$SESSION_NAME"
    fi
    exit
    ;;
esac

command -v tmux >/dev/null 2>&1 || { printf 'tmux is not installed. Use --foreground instead.\n' >&2; exit 1; }

if ! tmux has-session -t "$SESSION_NAME" 2>/dev/null; then
  printf -v watch_command \
    'env SOURCE_AGENT=%q TARGET_AGENT=%q RUN_TAG=%q CONDITIONS=%q DEPTHS=%q MAX_STEPS=%q RUN_JUDGE=%q REFRESH=%q BAR_WIDTH=%q OUTPUT_ROOT=%q bash %q --foreground' \
    "$SOURCE_AGENT" "$TARGET_AGENT" "$RUN_TAG" "$CONDITIONS" "$DEPTHS" "$MAX_STEPS" \
    "$RUN_JUDGE" "$REFRESH" "$BAR_WIDTH" "$OUTPUT_ROOT" "${BASH_SOURCE[0]}"
  tmux new-session -d -x 120 -y 40 -s "$SESSION_NAME" -c "$REPO_ROOT" "$watch_command"
  tmux set-option -t "$SESSION_NAME" remain-on-exit on >/dev/null
fi

if [[ "$mode" == "detach" ]]; then
  printf 'Dashboard running in tmux session: %s\n' "$SESSION_NAME"
  printf 'Attach with: tmux attach -t %q\n' "$SESSION_NAME"
elif [[ -n "${TMUX:-}" ]]; then
  tmux switch-client -t "$SESSION_NAME"
else
  tmux attach-session -t "$SESSION_NAME"
fi
