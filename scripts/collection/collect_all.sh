#!/usr/bin/env bash

set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
MODELS_LOCK="${REPO_ROOT}/configs/models.lock.yaml"
ROOT_DOTENV="${REPO_ROOT}/.env"

DEFAULT_AGENTS=(
  qwen3_5_35b_a3b
)

export PYTHON_BIN="${PYTHON_BIN:-python3}"

export COLLECTION_ID="${COLLECTION_ID:-v1}"

die() {
  printf 'error: %s\n' "$*" >&2
  exit 1
}

# shellcheck source=../lib/collection_config.sh
source "${SCRIPT_DIR}/../lib/collection_config.sh"

COLLECT_CONFIG="${COLLECT_CONFIG:-${REPO_ROOT}/configs/collection/mypcbench_runtime.yaml}"
COLLECT_ENV_OVERRIDES=()
COLLECT_ENV_FORWARD=()

collection_config_value() {
  "$PYTHON_BIN" - "$COLLECT_CONFIG" "$1" <<'PY'
import sys
from pathlib import Path

path, want = Path(sys.argv[1]), sys.argv[2]
if not path.is_file():
    raise SystemExit(f"Collection config not found: {path} (override with COLLECT_CONFIG=)")
for raw in path.read_text(encoding="utf-8").splitlines():
    line = raw.strip()
    if not line or line.startswith("#") or ":" not in line:
        continue
    key, text = line.split(":", 1)
    if key.strip() != want:
        continue
    value = text.split("#", 1)[0].strip().strip('"').strip("'")
    if value:
        print(value)
        raise SystemExit(0)
    break
raise SystemExit(f"{path}: missing usable key {want}")
PY
}

resolve_protocol_param() {
  local var="$1" key="$2" from_config
  from_config="$(collection_config_value "$key")" \
    || die "Failed to read ${key} from ${COLLECT_CONFIG}"
  if [[ -n "${!var:-}" && "${!var}" != "$from_config" ]]; then
    COLLECT_ENV_OVERRIDES+=("${var}=${!var} (config: ${from_config})")
    if [[ "${ALLOW_CONFIG_OVERRIDE:-0}" == "1" ]]; then
      COLLECT_ENV_FORWARD+=("${var}=${!var}")
      return 0
    fi
  fi
  printf -v "$var" '%s' "$from_config"
}

resolve_protocol_param REPEATS repeats
resolve_protocol_param MAX_STEPS max_steps
resolve_protocol_param TASK_TIMEOUT task_timeout

TASKS_FILE="${TASKS_FILE:-${REPO_ROOT}/third_party/MyPCBench/tasks/final/all_tasks_with_grading.json}"
TASK_COUNT="$(
  "$PYTHON_BIN" - "$TASKS_FILE" <<'PY' 2>/dev/null || true
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
tasks = json.loads(path.read_text(encoding="utf-8"))
print(len(tasks))
PY
)"
[[ "$TASK_COUNT" =~ ^[0-9]+$ ]] || TASK_COUNT=0

AVG_STEPS_PER_EPISODE=40

agent_num_vms() {
  agent_vm_count "$REPO_ROOT" "$1"
}

agent_step_seconds() {
  agent_eta_step_seconds "$REPO_ROOT" "$1"
}

export_agent_env() {
  [[ "$(agent_family "$REPO_ROOT" "$1" || true)" == qwen35 ]] || return 0
  export MYPCBENCH_QWEN_MAX_TOKENS="${MYPCBENCH_QWEN_MAX_TOKENS:-4096}"
  export MYPCBENCH_QWEN_HISTORY_N="${MYPCBENCH_QWEN_HISTORY_N:-100}"
  export MYPCBENCH_QWEN_CONTEXT_POLICY="${MYPCBENCH_QWEN_CONTEXT_POLICY:-tokenize_oldest_first_v1}"
}

CONFIRMED=0
RESUME=0
AGENTS=()
for argument in "$@"; do
  case "$argument" in
    --confirm) CONFIRMED=1 ;;
    --resume) RESUME=1 ;;
    -h|--help) sed -n '2,20p' "${BASH_SOURCE[0]}" | sed 's/^# \?//'; exit 0 ;;
    -*) die "Unknown argument: ${argument} (only --confirm and --resume are accepted)" ;;
    *) AGENTS+=("$argument") ;;
  esac
done
(( ${#AGENTS[@]} )) || AGENTS=("${DEFAULT_AGENTS[@]}")

RUN_ROOT="${REPO_ROOT}/artifacts/raw_rollouts/mypcbench/${COLLECTION_ID}"
COLLECTION_LOG_DIR="${REPO_ROOT}/artifacts/collection_logs"
COLLECTION_LOG="${COLLECTION_LOG_DIR}/${COLLECTION_ID}.log"

existing_episodes() {
  local dir="${RUN_ROOT}/$1"
  [[ -d "$dir" ]] || { printf '0\n'; return 0; }
  local n
  n="$(find "$dir" -name result.txt 2>/dev/null | wc -l || true)"
  printf '%s\n' "${n:-0}"
}

printf '================ MyPCBench formal collection ================\n'
printf '  COLLECTION_ID : %s\n' "$COLLECTION_ID"
printf '  Config        : %s\n' "${COLLECT_CONFIG#"${REPO_ROOT}/"}"
if (( TASK_COUNT > 0 )); then
  printf '  Per agent     : %s tasks × %s repeats = %s episodes\n' \
    "$TASK_COUNT" "$REPEATS" "$((TASK_COUNT * REPEATS))"
else
  printf '  Per agent     : ? tasks (cannot read %s) × %s repeats\n' "$TASKS_FILE" "$REPEATS"
fi
printf '  MAX_STEPS     : %s\n' "$MAX_STEPS"
printf '  TASK_TIMEOUT  : %ss\n' "$TASK_TIMEOUT"
printf '  Python        : %s\n' "$PYTHON_BIN"
printf '  Output        : artifacts/raw_rollouts/mypcbench/%s/<agent>/\n' "$COLLECTION_ID"
if (( ${#COLLECT_ENV_OVERRIDES[@]} > 0 )); then
  if [[ "${ALLOW_CONFIG_OVERRIDE:-0}" == "1" ]]; then
    printf '  ★ Env overrides: %s\n' "${COLLECT_ENV_OVERRIDES[*]}"
  else
    printf '  ⚠ These environment variables conflict with config; config wins (collect_trajectories.sh ignores them too): %s\n' \
      "${COLLECT_ENV_OVERRIDES[*]}"
    printf '    Set ALLOW_CONFIG_OVERRIDE=1 to apply them; for formal runs, edit the config.\n'
  fi
fi
printf '\n  %-20s %-8s %-10s %-9s %s\n' agent NUM_VMS step ETA existing
total_hours=0
stale_total=0
for agent_id in "${AGENTS[@]}"; do
  num_vms="$(agent_num_vms "$agent_id")"
  step_s="$(agent_step_seconds "$agent_id")" \
    || die "${agent_id} has no eta_step_seconds; add it to configs/agents/${agent_id}.yaml"
  hours="$(awk -v e="$((TASK_COUNT * REPEATS))" -v s="$step_s" -v v="$num_vms" \
    -v n="$AVG_STEPS_PER_EPISODE" 'BEGIN{printf "%.1f", e*n*s/v/3600}')"
  total_hours="$(awk -v a="$total_hours" -v b="$hours" 'BEGIN{printf "%.1f", a+b}')"
  stale="$(existing_episodes "$agent_id")"
  stale_total=$((stale_total + stale))
  if (( stale > 0 )); then
    note="$( (( RESUME )) && printf '%s, kept (skip)' "$stale" || printf '%s, will be deleted and rerun' "$stale" )"
  else
    note="-"
  fi
  printf '  %-20s %-8s %-10s ~%-8s %s\n' "$agent_id" "$num_vms" "${step_s}s" "${hours}h" "$note"
done
printf '  %-20s %-8s %-10s ~%-8s\n' '(total)' '' '' "${total_hours}h"
cat <<EOF

  ETA assumes ${AVG_STEPS_PER_EPISODE} steps per episode on average; rough guide only.
  collect_trajectories.sh starts each model, waits for ready and stops it when done; agents run sequentially.
  The real OPENAI_API_KEY from .env is used for NPC auto-replies inside the VM.
EOF
if (( stale_total > 0 )); then
  if (( RESUME )); then
    printf '\n  --resume: keep %s finished episodes and only run the missing ones.\n' "$stale_total"
  else
    printf '\n  ⚠ Will delete %s existing episodes and rerun (REPEATS=%s, rerun overwrites).\n' \
      "$stale_total" "$REPEATS"
    printf '    To keep them and only run the missing ones, use --resume.\n'
  fi
fi
printf '===========================================================\n'

grep -q '^formal_collection_authorized: true' "$MODELS_LOCK" || die \
  "formal_collection_authorized is still false in configs/models.lock.yaml.
     After checking revision, runtime image and endpoint gate, set it to true manually and rerun."

if ! "$PYTHON_BIN" - "$ROOT_DOTENV" <<'PY'
import sys
from pathlib import Path
path = Path(sys.argv[1])
if not path.is_file():
    raise SystemExit(1)
for raw in path.read_text(encoding="utf-8").splitlines():
    line = raw.strip()
    if not line or line.startswith("#"):
        continue
    if line.startswith("export "):
        line = line[7:].lstrip()
    separator = "=" if "=" in line else (":" if ":" in line else None)
    if separator is None:
        continue
    key, value = line.split(separator, 1)
    if key.strip() == "OPENAI_API_KEY" and value.strip().strip("\"'"):
        raise SystemExit(0)
raise SystemExit(1)
PY
then
  die "OPENAI_API_KEY in .env is empty.
     The full 184-task collection needs a working key: MyPCBench injects it into
     BuzzChat/WorkBuzz in the VM for NPC auto-replies; chat tasks cannot run without it.
     Fill it in and rerun this command."
fi

if (( CONFIRMED == 0 )); then
  printf '\nThis is a preview; nothing was started. Add --confirm when ready:\n'
  printf '  bash scripts/collection/collect_all.sh --confirm %s\n' "${AGENTS[*]}"
  exit 0
fi

if [[ -z "${TMUX:-}" && "${RECOVERY_TMUX:-1}" != "0" ]]; then
  session="recovery-collect-${COLLECTION_ID}"
  session="$(tr -c '[:alnum:]_-' '-' <<< "$session" | sed 's/-$//')"
  tmux has-session -t "=${session}" 2>/dev/null && die \
    "tmux session already exists: ${session} (remove a finished one with tmux kill-session -t =${session})"
  local_flags=(--confirm)
  (( RESUME )) && local_flags+=(--resume)

  mkdir -p "$COLLECTION_LOG_DIR"
  : > "$COLLECTION_LOG"
  detached_env=(
    RECOVERY_TMUX=0
    PYTHON_BIN="$PYTHON_BIN"
    COLLECTION_ID="$COLLECTION_ID"
    COLLECT_CONFIG="$COLLECT_CONFIG"
  )
  [[ -n "${ALLOW_CONFIG_OVERRIDE:-}" ]] \
    && detached_env+=("ALLOW_CONFIG_OVERRIDE=${ALLOW_CONFIG_OVERRIDE}")
  (( ${#COLLECT_ENV_FORWARD[@]} )) && detached_env+=("${COLLECT_ENV_FORWARD[@]}")
  setsid env "${detached_env[@]}" \
    bash "${BASH_SOURCE[0]}" "${local_flags[@]}" "${AGENTS[@]}" \
    </dev/null >>"$COLLECTION_LOG" 2>&1 &
  collection_pid=$!
  printf '%s\n' "$collection_pid" > "${COLLECTION_LOG_DIR}/${COLLECTION_ID}.pid"
  disown "$collection_pid" 2>/dev/null || true

  printf -v log_text '%q ' tail -n +1 -f "$COLLECTION_LOG"
  tmux new-session -d -s "$session" -n collection -c "$REPO_ROOT" "$log_text"
  tmux new-window -d -t "$session" -n monitor -c "$REPO_ROOT" "watch -n 2 nvidia-smi"
  printf '\nStarted (PID %s), fully detached from this terminal; Ctrl-C will not reach it.\n\n' "$collection_pid"
  printf '  Logs    : tmux attach -t %s\n' "$session"
  printf '  Log file: %s\n' "$COLLECTION_LOG"
  printf '  Abort   : pkill -TERM -f "run_parallel_tasks.py.*%s"\n' "$COLLECTION_ID"
  printf '            (cleans up automatically when done; no need to stop it)\n'
  exit 0
fi

export RECOVERY_TMUX=0
export FORMAL_COLLECTION=1
export RECOVERY_OPENAI_API_APPROVED=1
export RECOVERY_OPENAI_API_PURPOSE=mypcbench_npc_replies

mkdir -p "$RUN_ROOT"

for agent_id in "${AGENTS[@]}"; do
  agent_vms="$(agent_num_vms "$agent_id")"
  export_agent_env "$agent_id"
  if (( RESUME == 0 )) && [[ -d "${RUN_ROOT}/${agent_id}" ]]; then
    stale="$(existing_episodes "$agent_id")"
    printf '[RECOVERY] Clearing old results of %s (%s episodes) for rerun\n' "$agent_id" "$stale"
    rm -rf "${RUN_ROOT:?}/${agent_id:?}"
  fi
  printf '\n[RECOVERY] ===== %s (NUM_VMS=%s) =====\n' "$agent_id" "$agent_vms"
  bash "${SCRIPT_DIR}/collect_trajectories.sh" "$agent_id"

  valid=0; empty=0
  while IFS= read -r d; do
    if compgen -G "${d}/step_*.png" >/dev/null; then valid=$((valid+1)); else empty=$((empty+1)); fi
  done < <(find "${RUN_ROOT}/${agent_id}" -name result.txt -printf '%h\n' 2>/dev/null)
  printf '[RECOVERY] %s check: valid %s, empty %s\n' "$agent_id" "$valid" "$empty"
  if (( empty > valid )); then
    die "${agent_id} has more empty (${empty}) than valid (${valid}) episodes; the endpoint likely died mid-run.
     Aborting the queue; remaining agents will not run. Check ${COLLECTION_LOG} and artifacts/serving_logs/."
  fi
done

printf '\n[RECOVERY] All done: artifacts/raw_rollouts/mypcbench/%s\n' "$COLLECTION_ID"
