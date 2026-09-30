#!/usr/bin/env bash
# MyPCBench 正式 collection 的预设入口 —— 平时跑采集用这个。
#
# 与 01_collect_trajectories.sh 的分工：本脚本不含任何实验语义，只是把那一长串
# 环境变量（REPEATS、Qwen3.5 的 context 三件套、两个授权 flag）按 agent
# 填好，然后对每个 agent 依次调用 01_collect_trajectories.sh。真正的采集逻辑、
# 冻结 commit、manifest、endpoint 生命周期全部在那一个脚本里。它也不覆盖任何
# 已经显式导出的变量。要手工控制单次运行的参数，就直接跑 01_collect_trajectories.sh。
#
#   bash scripts/01_collect_all.sh                        # 预览默认队列（五个模型）
#   bash scripts/01_collect_all.sh qwen3_5_35b_a3b        # 预览单个
#   bash scripts/01_collect_all.sh --confirm qwen3_5_35b_a3b
#   bash scripts/01_collect_all.sh --confirm              # 五个模型串行跑完
#
# 为什么要 --confirm：正式采集会占满 8 张卡跑几十小时，并且会用 .env 里的真实
# OPENAI_API_KEY 给 VM 内 BuzzChat/WorkBuzz 的 NPC 自动回复。01 要求这个授权
# 必须逐次显式给出、不能写进 .env，所以这里用一个命令行 flag 承担那次确认。
#
# 模型由 01 自动启动、等 ready、跑完自动停止，不需要先手动跑 00。
# 同一时刻只能 serve 一个模型（8 张 A6000 装不下两个），所以多个 agent 会串行。

set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
MODELS_LOCK="${REPO_ROOT}/configs/models.lock.yaml"
ROOT_DOTENV="${REPO_ROOT}/.env"

# 默认队列：先跑快的，早点拿到一个模型的完整数据去验证下游链路。
DEFAULT_AGENTS=(
  qwen3_5_35b_a3b
)

# conda env 名为 derail；用绝对路径而不依赖 PATH，因为登录 shell 常残留 uv venv。
export PYTHON_BIN="${PYTHON_BIN:-python3}"

# 命名规范：<用途>_<模型>，不带日期。
# 01 已经把 agent 作为子目录，所以 COLLECTION_ID 只放用途，最终落盘路径读起来
# 就是 <用途>/<模型>：
#     artifacts/raw_rollouts/mypcbench/v1/qwen3_5_35b_a3b/repeat_1/...
# 不带日期是刻意的 —— 同样的设置重跑就该覆盖同一个目录（见下方 RESUME 逻辑），
# 而不是每天堆一个新目录出来。
# 用途取值：v1(正式) | smoke | probe | gate | retest
export COLLECTION_ID="${COLLECTION_ID:-v1}"

die() {
  printf '错误：%s\n' "$*" >&2
  exit 1
}

# shellcheck source=lib/collection_config.sh
source "${SCRIPT_DIR}/lib/collection_config.sh"

# ==================== 实验协议（docs/A6000_SERVER_HANDOFF.md §12）====================
# 会改变轨迹形状的参数（步数预算、重复轮数）只有一个真源：下面这份 config。
# 01_collect_trajectories.sh 用同一个默认路径读同一份文件，而且 config 的优先级
# 高于环境变量（要覆盖必须显式 ALLOW_CONFIG_OVERRIDE=1）。
#
# 所以这里**只读不导出**。以前这两行是 `export MAX_STEPS="${MAX_STEPS:-100}"`，
# 01 会把它当成"与 config 冲突的环境变量"记进已忽略清单，于是本脚本横幅上印的
# 数字和实际跑的数字可以静默地对不上 —— 2026-08-09 把 config 改成 150 之后，
# 横幅仍会理直气壮地印 100。
#
# 采集预算（论文 150 步 / 3600 秒 / 3 次）不要和 takeover 的 100 步接管预算混用。
COLLECT_CONFIG="${COLLECT_CONFIG:-${REPO_ROOT}/configs/collection/mypcbench_runtime.yaml}"
COLLECT_ENV_OVERRIDES=()   # 给人看：哪些环境变量和 config 不一致
COLLECT_ENV_FORWARD=()     # 给机器看：后台子进程要继承哪些显式覆盖

# 读 config 里的一个标量键。解析规则与 01 的 load_collection_config 保持一致
# （`key: value`，`#` 起注释，可带引号），键不存在就报错而不是回退到默认值 ——
# 静默回退正是这份 config 要消灭的东西。
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

# 把 config 的值落到脚本变量上，落下的**永远是实际生效的那个值** —— 横幅印什么
# 就跑什么，这是这次重构的全部目的。所以这里必须复刻 01 的优先级判定：显式设置
# 的同名环境变量只有在 ALLOW_CONFIG_OVERRIDE=1 时才赢，否则被 01 忽略，config 赢。
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

# 任务数实读 task 文件，不写死 184。third_party/ 不进 Git，所以文件缺失时退回
# 到"未知"而非让预览失败 —— 真正的存在性检查在 01 里，那里失败才有意义。
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

# 每 episode 的平均步数，仅用于时长预估。这是 v1 的实测均值，不是预算 ——
# 它跟着数据走，不跟着 MAX_STEPS 走。
AVG_STEPS_PER_EPISODE=40

# 每个 agent 的并发度。这里只用于预览表 —— 真正传给 runner 的值由 01 自己调
# 同一个函数算，本脚本不再 export NUM_VMS。
#
# 以前 export 是个真 bug：01 的采集 config 优先级高于环境变量，NUM_VMS=1 与
# config 的 num_vms: 4 冲突后被静默忽略，opencua_72b（TP8）会以 4 个 VM 起跑。
agent_num_vms() {
  agent_vm_count "$REPO_ROOT" "$1"
}

# 单步耗时，只用于时长预估。真源是 configs/agents/<agent_id>.yaml 的
# eta_step_seconds —— 以前是这里一张只覆盖 5 个开源 agent 的 case 表，另外 5 个
# 托管 API 的 agent 会被下面的 die 当成"未知 agent_id"打死，只能绕开本脚本单跑。
agent_step_seconds() {
  agent_eta_step_seconds "$REPO_ROOT" "$1"
}

# Qwen3.5 官方 wrapper 默认请求 32768 tokens/100-turn history。来源与证据见
# models.lock.yaml 的 models.qwen3_5_35b_a3b.probe_overrides。只有这个 agent
# 需要这三个变量。
#
# MYPCBENCH_QWEN_MAX_TOKENS=4096 是补全预算，不是上下文预算 —— Qwen3.5
# server context 是 49152，因此这个补全预算仍然保留。Qwen3.8 takeover
# 另用 98304，由 00_tmux_serve_open_source.sh 与 models.lock.yaml 记录。
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
# 采集的完整 stdout/stderr 落盘。以前只在 tmux 回滚里，窗口一关就没了。
COLLECTION_LOG_DIR="${REPO_ROOT}/artifacts/collection_logs"
COLLECTION_LOG="${COLLECTION_LOG_DIR}/${COLLECTION_ID}.log"

# 目录可能还不存在（全新的 COLLECTION_ID）。find 会以非零退出，在
# set -e + pipefail 下会直接把脚本打死，所以这里必须自己兜住。
existing_episodes() {
  local dir="${RUN_ROOT}/$1"
  [[ -d "$dir" ]] || { printf '0\n'; return 0; }
  local n
  n="$(find "$dir" -name result.txt 2>/dev/null | wc -l || true)"
  printf '%s\n' "${n:-0}"
}

# ==================== 预览 ====================
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
    printf '  ⚠ 以下环境变量与 config 冲突，已按 config 跑（01 也会忽略它们）：%s\n' \
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
  模型由 01 自动启动 / 等 ready / 跑完自动停止，多个 agent 串行。
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

# ==================== 闸门（提前检查，避免加载完模型才失败）====================
grep -q '^formal_collection_authorized: true' "$MODELS_LOCK" || die \
  "configs/models.lock.yaml 里 formal_collection_authorized 还是 false。
     确认 revision、runtime image 和 endpoint gate 都核对过之后，手动改成 true 再跑。"

# 01 在加载完模型之后才会检查这个；这里提前拦，省掉几分钟白等。
# 只判断非空，不读取也不打印值。
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
  printf '  bash scripts/01_collect_all.sh --confirm %s\n' "${AGENTS[*]}"
  exit 0
fi

# ==================== 真跑 ====================
# 每个 agent 的 NUM_VMS 不同，所以逐个调用 01；用 DERAIL_TMUX=0 让 01 在当前
# 进程里前台执行，否则它会各自建 session 导致五个模型同时启动、互相抢卡。
if [[ -z "${TMUX:-}" && "${DERAIL_TMUX:-1}" != "0" ]]; then
  session="derail-collect-${COLLECTION_ID}"
  session="$(tr -c '[:alnum:]_-' '-' <<< "$session" | sed 's/-$//')"
  # `=` 前缀 = 精确匹配。tmux 的 target-session 在找不到同名 session 时会退回
  # 前缀匹配，于是 COLLECTION_ID=v1 会被 derail-collect-v1_bashfix_probe20 挡住，
  # 而按提示去 kill 又会连那个不相干的 session 一起删掉。
  tmux has-session -t "=${session}" 2>/dev/null && die \
    "tmux session 已存在：${session}（跑完的旧 session 用 tmux kill-session -t =${session} 删掉）"
  local_flags=(--confirm)
  (( RESUME )) && local_flags+=(--resume)

  # 采集用 setsid 起在独立会话里，不挂在任何终端上，stdin 接 /dev/null。
  #
  # 为什么这么做：2026-08-04 的第一次正式采集，有人在 collection 窗口里想切
  # tmux 窗口、打错了几行、然后按 Ctrl-C 清行 —— 那一下 SIGINT 打到前台进程组，
  # 停掉了 4 个 endpoint。而 runner 的 VM 子进程用 start_new_session=True 跑在
  # 各自的进程组里，没收到信号，继续空转一小时，产出 174 个零截图但
  # result.txt=1.0 的空壳。22 小时的活白干。
  #
  # setsid 之后采集没有控制终端，tmux 窗口里的任何按键都送不到它。窗口改成
  # tail -f 日志，是纯只读的。要停采集只能显式 kill，不会再误杀。
  mkdir -p "$COLLECTION_LOG_DIR"
  : > "$COLLECTION_LOG"
  # 后台子进程不继承 REPEATS/MAX_STEPS —— 它会自己读同一份 COLLECT_CONFIG。
  # 只有调用者显式给出的覆盖（连同解锁它们所需的 ALLOW_CONFIG_OVERRIDE）才转发，
  # 否则就是把 config 的值抄成环境变量再传一遍，凭空造出第二个真源。
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

  # 三个窗口全部只读：日志、进度条、显卡。误按任何键都不会影响采集。
  printf -v log_text '%q ' tail -n +1 -f "$COLLECTION_LOG"
  tmux new-session -d -s "$session" -n collection -c "$REPO_ROOT" "$log_text"
  printf -v progress_text '%q ' \
    bash "${SCRIPT_DIR}/01_watch_progress.sh" "$COLLECTION_ID"
  tmux new-window -d -t "$session" -n progress -c "$REPO_ROOT" "$progress_text"
  tmux new-window -d -t "$session" -n monitor -c "$REPO_ROOT" "watch -n 2 nvidia-smi"
  tmux select-window -t "${session}:progress"
  printf '\n已启动（PID %s），与终端完全脱离，Ctrl-C 打不到它。\n\n' "$collection_pid"
  printf '  看进度条: tmux attach -t %s          （默认就停在 progress 窗口）\n' "$session"
  printf '  看日志  : tmux select-window -t %s:collection\n' "$session"
  printf '  日志文件: %s\n' "$COLLECTION_LOG"
  printf '  不进 tmux: bash scripts/01_watch_progress.sh %s\n' "$COLLECTION_ID"
  # 正常跑完不需要任何手动操作：run_parallel_tasks.py 的 atexit 回收器会杀掉
  # 自己那几个 QEMU 并清掉临时磁盘，01 的 trap 会停掉 endpoint。
  #
  # 只有中途要中止时才需要这条，而且必须发给 run_parallel_tasks.py 本身，
  # 不能 kill 整个进程组。VM 工作进程用 start_new_session=True 跑在各自的
  # 进程组里，组信号打不到它们；上游是靠父进程的 SIGINT/SIGTERM 回收器
  # （按 /tmp/mypcbench-<name>.pid 精确杀 QEMU）来收尸的。2026-08-04 用
  # kill -- -<pgid> 停过一次，bash 层先死、回收器没跑完，结果 4 个 QEMU 被
  # systemd 收养、又空转 1.5 小时写了 150 个空壳。
  printf '  要中止  : pkill -TERM -f "run_parallel_tasks.py.*%s"\n' "$COLLECTION_ID"
  printf '            （正常跑完会自动清理，不需要手动停）\n'
  exit 0
fi

export DERAIL_TMUX=0
export FORMAL_COLLECTION=1
# 这两个授权只对本次调用生效，不写进 .env。
export DERAIL_OPENAI_API_APPROVED=1
export DERAIL_OPENAI_API_PURPOSE=mypcbench_npc_replies

mkdir -p "$RUN_ROOT"
date +%s > "${RUN_ROOT}/.progress_started_at"

for agent_id in "${AGENTS[@]}"; do
  # 只用于下面那行提示；不 export —— 01 会自己从同一份 agent config 解析。
  agent_vms="$(agent_num_vms "$agent_id")"
  export_agent_env "$agent_id"
  # REPEATS=1 时「重跑」就等于「重做同一批任务」，官方 runner 会按已有 result.txt
  # 跳过，导致新旧配置的结果混在一个目录里。默认先清空该 agent 的结果目录，
  # 让每次 rerun 都是干净的一遍；--resume 保留旧行为（中断续跑）。
  if (( RESUME == 0 )) && [[ -d "${RUN_ROOT}/${agent_id}" ]]; then
    stale="$(existing_episodes "$agent_id")"
    printf '[DERAIL] 清空 %s 的旧结果（%s 个 episode）以便重跑\n' "$agent_id" "$stale"
    rm -rf "${RUN_ROOT:?}/${agent_id:?}"
  fi
  printf '\n[DERAIL] ===== %s (NUM_VMS=%s) =====\n' "$agent_id" "$agent_vms"
  bash "${SCRIPT_DIR}/01_collect_trajectories.sh" "$agent_id"

  # 跑完立刻校验：endpoint 中途挂掉时，runner 仍会给每个任务写 result.txt=1.0，
  # 但目录里零截图。不在这里拦住，就会像 2026-08-04 那次一样，184/184 看着满格、
  # 实际只有 10 个有数据，而且后面的 agent 会接着白跑。
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
