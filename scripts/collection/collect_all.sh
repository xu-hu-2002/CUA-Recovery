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
  printf '错误：%s\n' "$*" >&2
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
    raise SystemExit(f"找不到采集配置：{path}（用 COLLECT_CONFIG= 覆盖）")
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
raise SystemExit(f"{path}: 缺少可用的键 {want}")
PY
}

resolve_protocol_param() {
  local var="$1" key="$2" from_config
  from_config="$(collection_config_value "$key")" \
    || die "读取 ${COLLECT_CONFIG} 的 ${key} 失败"
  if [[ -n "${!var:-}" && "${!var}" != "$from_config" ]]; then
    COLLECT_ENV_OVERRIDES+=("${var}=${!var}（config 为 ${from_config}）")
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
  case "$1" in
    qwen3_5_35b_a3b)
      export MYPCBENCH_QWEN_MAX_TOKENS="${MYPCBENCH_QWEN_MAX_TOKENS:-4096}"
      export MYPCBENCH_QWEN_HISTORY_N="${MYPCBENCH_QWEN_HISTORY_N:-100}"
      export MYPCBENCH_QWEN_CONTEXT_POLICY="${MYPCBENCH_QWEN_CONTEXT_POLICY:-tokenize_oldest_first_v1}"
      ;;
  esac
}

CONFIRMED=0
RESUME=0
AGENTS=()
for argument in "$@"; do
  case "$argument" in
    --confirm) CONFIRMED=1 ;;
    --resume) RESUME=1 ;;
    -h|--help) sed -n '2,20p' "${BASH_SOURCE[0]}" | sed 's/^# \?//'; exit 0 ;;
    -*) die "未知参数：${argument}（只接受 --confirm 和 --resume）" ;;
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

printf '================ MyPCBench 正式 collection ================\n'
printf '  COLLECTION_ID : %s\n' "$COLLECTION_ID"
printf '  采集配置      : %s\n' "${COLLECT_CONFIG#"${REPO_ROOT}/"}"
if (( TASK_COUNT > 0 )); then
  printf '  每个 agent    : %s tasks × %s repeats = %s episodes\n' \
    "$TASK_COUNT" "$REPEATS" "$((TASK_COUNT * REPEATS))"
else
  printf '  每个 agent    : ? tasks（读不到 %s）× %s repeats\n' "$TASKS_FILE" "$REPEATS"
fi
printf '  MAX_STEPS     : %s\n' "$MAX_STEPS"
printf '  TASK_TIMEOUT  : %ss\n' "$TASK_TIMEOUT"
printf '  Python        : %s\n' "$PYTHON_BIN"
printf '  输出          : artifacts/raw_rollouts/mypcbench/%s/<agent>/\n' "$COLLECTION_ID"
if (( ${#COLLECT_ENV_OVERRIDES[@]} > 0 )); then
  if [[ "${ALLOW_CONFIG_OVERRIDE:-0}" == "1" ]]; then
    printf '  ★ 环境变量覆盖 : %s\n' "${COLLECT_ENV_OVERRIDES[*]}"
  else
    printf '  ⚠ 以下环境变量与 config 冲突，已按 config 跑（collect_trajectories.sh 也会忽略它们）：%s\n' \
      "${COLLECT_ENV_OVERRIDES[*]}"
    printf '    要让它们生效请一并设 ALLOW_CONFIG_OVERRIDE=1；正式跑数请改 config。\n'
  fi
fi
printf '\n  %-20s %-8s %-10s %-9s %s\n' agent NUM_VMS 每步 预估 已有结果
total_hours=0
stale_total=0
for agent_id in "${AGENTS[@]}"; do
  num_vms="$(agent_num_vms "$agent_id")"
  step_s="$(agent_step_seconds "$agent_id")" \
    || die "${agent_id} 没有 eta_step_seconds；请在 configs/agents/${agent_id}.yaml 里补上"
  hours="$(awk -v e="$((TASK_COUNT * REPEATS))" -v s="$step_s" -v v="$num_vms" \
    -v n="$AVG_STEPS_PER_EPISODE" 'BEGIN{printf "%.1f", e*n*s/v/3600}')"
  total_hours="$(awk -v a="$total_hours" -v b="$hours" 'BEGIN{printf "%.1f", a+b}')"
  stale="$(existing_episodes "$agent_id")"
  stale_total=$((stale_total + stale))
  if (( stale > 0 )); then
    note="$( (( RESUME )) && printf '%s 个，跳过不重跑' "$stale" || printf '%s 个，将被删除重跑' "$stale" )"
  else
    note="-"
  fi
  printf '  %-20s %-8s %-10s ~%-8s %s\n' "$agent_id" "$num_vms" "${step_s}s" "${hours}h" "$note"
done
printf '  %-20s %-8s %-10s ~%-8s\n' '（合计）' '' '' "${total_hours}h"
cat <<EOF

  预估按每 episode 平均 ${AVG_STEPS_PER_EPISODE} 步估算，仅供参考。
  模型由 collect_trajectories.sh 自动启动 / 等 ready / 跑完自动停止，多个 agent 串行。
  会使用 .env 里的真实 OPENAI_API_KEY 给 VM 内 NPC 自动回复。
EOF
if (( stale_total > 0 )); then
  if (( RESUME )); then
    printf '\n  --resume：保留已完成的 %s 个 episode，只补跑缺的。\n' "$stale_total"
  else
    printf '\n  ⚠ 将删除 %s 个已有 episode 后重跑（REPEATS=%s，重跑即覆盖）。\n' \
      "$stale_total" "$REPEATS"
    printf '    想保留并只补跑缺的，改用 --resume。\n'
  fi
fi
printf '===========================================================\n'

grep -q '^formal_collection_authorized: true' "$MODELS_LOCK" || die \
  "configs/models.lock.yaml 里 formal_collection_authorized 还是 false。
     确认 revision、runtime image 和 endpoint gate 都核对过之后，手动改成 true 再跑。"

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
  die ".env 里 OPENAI_API_KEY 是空的。
     完整 184 任务采集需要真实可用的 key —— MyPCBench 会把它注入 VM 内的
     BuzzChat/WorkBuzz，供 NPC 自动回复；涉及聊天的任务没有它就做不了。
     填好之后重跑本命令。"
fi

if (( CONFIRMED == 0 )); then
  printf '\n这是预览，没有启动任何东西。确认无误后加 --confirm：\n'
  printf '  bash scripts/collection/collect_all.sh --confirm %s\n' "${AGENTS[*]}"
  exit 0
fi

if [[ -z "${TMUX:-}" && "${DERAIL_TMUX:-1}" != "0" ]]; then
  session="derail-collect-${COLLECTION_ID}"
  session="$(tr -c '[:alnum:]_-' '-' <<< "$session" | sed 's/-$//')"
  tmux has-session -t "=${session}" 2>/dev/null && die \
    "tmux session 已存在：${session}（跑完的旧 session 用 tmux kill-session -t =${session} 删掉）"
  local_flags=(--confirm)
  (( RESUME )) && local_flags+=(--resume)

  mkdir -p "$COLLECTION_LOG_DIR"
  : > "$COLLECTION_LOG"
  detached_env=(
    DERAIL_TMUX=0
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
  printf -v progress_text '%q ' \
    bash "${SCRIPT_DIR}/watch_progress.sh" "$COLLECTION_ID"
  tmux new-window -d -t "$session" -n progress -c "$REPO_ROOT" "$progress_text"
  tmux new-window -d -t "$session" -n monitor -c "$REPO_ROOT" "watch -n 2 nvidia-smi"
  tmux select-window -t "${session}:progress"
  printf '\n已启动（PID %s），与终端完全脱离，Ctrl-C 打不到它。\n\n' "$collection_pid"
  printf '  看进度条: tmux attach -t %s          （默认就停在 progress 窗口）\n' "$session"
  printf '  看日志  : tmux select-window -t %s:collection\n' "$session"
  printf '  日志文件: %s\n' "$COLLECTION_LOG"
  printf '  不进 tmux: bash scripts/collection/watch_progress.sh %s\n' "$COLLECTION_ID"
  printf '  要中止  : pkill -TERM -f "run_parallel_tasks.py.*%s"\n' "$COLLECTION_ID"
  printf '            （正常跑完会自动清理，不需要手动停）\n'
  exit 0
fi

export DERAIL_TMUX=0
export FORMAL_COLLECTION=1
export DERAIL_OPENAI_API_APPROVED=1
export DERAIL_OPENAI_API_PURPOSE=mypcbench_npc_replies

mkdir -p "$RUN_ROOT"
date +%s > "${RUN_ROOT}/.progress_started_at"

for agent_id in "${AGENTS[@]}"; do
  agent_vms="$(agent_num_vms "$agent_id")"
  export_agent_env "$agent_id"
  if (( RESUME == 0 )) && [[ -d "${RUN_ROOT}/${agent_id}" ]]; then
    stale="$(existing_episodes "$agent_id")"
    printf '[DERAIL] 清空 %s 的旧结果（%s 个 episode）以便重跑\n' "$agent_id" "$stale"
    rm -rf "${RUN_ROOT:?}/${agent_id:?}"
  fi
  printf '\n[DERAIL] ===== %s (NUM_VMS=%s) =====\n' "$agent_id" "$agent_vms"
  bash "${SCRIPT_DIR}/collect_trajectories.sh" "$agent_id"

  valid=0; empty=0
  while IFS= read -r d; do
    if compgen -G "${d}/step_*.png" >/dev/null; then valid=$((valid+1)); else empty=$((empty+1)); fi
  done < <(find "${RUN_ROOT}/${agent_id}" -name result.txt -printf '%h\n' 2>/dev/null)
  printf '[DERAIL] %s 完成校验：有效 %s，空壳 %s\n' "$agent_id" "$valid" "$empty"
  if (( empty > valid )); then
    die "${agent_id} 的空壳(${empty}) 多于有效(${valid})；endpoint 很可能中途挂了。
     中止队列，不再跑后面的 agent。检查 ${COLLECTION_LOG} 和 artifacts/serving_logs/。"
  fi
done

printf '\n[DERAIL] 全部完成：artifacts/raw_rollouts/mypcbench/%s\n' "$COLLECTION_ID"
