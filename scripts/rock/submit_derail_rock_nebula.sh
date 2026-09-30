#!/usr/bin/env bash
# =============================================================================
# DERAIL MyPCBench 采集 — Nebula + ROCK 提交入口（Mac 唯一入口）。
#
# 拓扑（照 MCUA closed-model legacy 形态）：
#   L1 推理   routify（外部 API，沙箱内经 OPENAI_BASE_URL 直连）
#   L2 编排   本脚本提交的 Nebula job：只跑 derail_rock_driver.py（无 vLLM），
#             通过 rl-rock SDK 管 ROCK 沙箱生命周期 + 结果对账
#   L3 环境   ROCK 沙箱（ROCK_CLUSTER / ROCK_SANDBOX_IMAGE, 8C/16G/100G），沙箱内
#             MyPCBench harness 自起 QEMU/KVM guest 跑 01_collect_trajectories.sh
#
# 凭证通道（与 MCUA 一致）：
#   - OSS 凭证：0600 --property_file（不进 argv / 不进入 shell transcript）
#   - routify key + 真实 OSS AK：0600 snippet 上传 oss://.../DERAIL/tmp/，
#     entry 从 bucket FUSE mount source 后自删（不进代码包）
#   - DERAIL 仓库只读克隆约束：本脚本只做本地文件操作 + nebulactl 提交，
#     绝不执行任何 git push。
#
# DRY_RUN=1 是默认值：只打印 nebulactl 命令，不提交。
#
# 用法：
#   bash scripts/rock/submit_derail_rock_nebula.sh                 # 干跑
#   DRY_RUN=0 AGENT_ID=dummy SHARD_FILE=configs/mypcbench_task_shards/smoke_one.json \
#     bash scripts/rock/submit_derail_rock_nebula.sh               # dummy smoke
#   DRY_RUN=0 AGENT_ID=gpt_5_5 SHARD_FILE=configs/mypcbench_task_shards/shard_1_of_4.json \
#     bash scripts/rock/submit_derail_rock_nebula.sh               # 正式分片（需当次确认）
# =============================================================================
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/../.." && pwd)"

# A formal Nebula bundle must correspond to a committed, reviewable source
# revision. Untracked local data (for example takeovewr_annotation/) is not
# packaged and does not affect this check.
ALLOW_DIRTY_SUBMIT="${ALLOW_DIRTY_SUBMIT:-0}"
if [[ "$ALLOW_DIRTY_SUBMIT" != "1" ]] \
  && { ! git -C "$REPO" diff --quiet --ignore-submodules -- \
    || ! git -C "$REPO" diff --cached --quiet --ignore-submodules --; }; then
  echo "ERROR: tracked changes are not committed; commit them before submitting, or set ALLOW_DIRTY_SUBMIT=1 for an explicitly non-reproducible run" >&2
  exit 2
fi

# Both submit entrypoints stage a fixed-name run config that must be present
# while nebulactl snapshots the worktree. Serialize local packaging so two
# fleets cannot overwrite or remove each other's config and secret reference.
SUBMIT_LOCK_DIR="$REPO/.submit_derail_nebula.lock"
_release_submit_lock() {
  rm -f "${OSS_CONFIG_FILE:+$REPO/$OSS_CONFIG_FILE}"
  rm -f "$SUBMIT_LOCK_DIR/pid"
  rmdir "$SUBMIT_LOCK_DIR" 2>/dev/null || true
}
_submit_lock_acquired=0
for _attempt in $(seq 1 300); do
  if mkdir "$SUBMIT_LOCK_DIR" 2>/dev/null; then
    printf '%s\n' "$$" > "$SUBMIT_LOCK_DIR/pid"
    _submit_lock_acquired=1
    break
  fi
  _owner="$(cat "$SUBMIT_LOCK_DIR/pid" 2>/dev/null || true)"
  if [[ "$_owner" =~ ^[0-9]+$ ]] && ! kill -0 "$_owner" 2>/dev/null; then
    rm -f "$SUBMIT_LOCK_DIR/pid"
    rmdir "$SUBMIT_LOCK_DIR" 2>/dev/null || true
    continue
  fi
  sleep 2
done
if [[ "$_submit_lock_acquired" != "1" ]]; then
  echo "ERROR: timed out waiting for local Nebula submit lock: $SUBMIT_LOCK_DIR" >&2
  exit 2
fi
trap _release_submit_lock EXIT

# --- 采集参数 -----------------------------------------------------------------
AGENT_ID="${AGENT_ID:-gpt_5_5}"
SHARD_FILE="${SHARD_FILE:-configs/mypcbench_task_shards/smoke_one.json}"
# COLLECTION_ID 留空则由 driver 生成（utc 时间戳 + agent + shard 标签）。
COLLECTION_ID="${COLLECTION_ID:-}"
# 断点续采：复用原 COLLECTION_ID + RESUME_SEED=1，driver 起跑前先从 OSS 回灌
# 部分结果，harness 只重跑无 result.txt 的任务（见 derail_rock_driver.py）。
RESUME_SEED="${RESUME_SEED:-0}"
# 重跑强制清场：rerun 的目标是已完成（带 result.txt）的任务，续采守卫会
# 静默跳过它们（2026-08-18 F4 opus 4.8 rerun 86 任务全跳过事故）。=1 时
# driver 在 seed 回灌后删除 SHARD_FILE 清单内全部任务目录，逼 harness 重跑。
RERUN_PURGE="${RERUN_PURGE:-0}"
# routify 网关要求 provider 前缀模型名；直连 OpenAI 官方时改为 gpt-5.5。
GPT55_MODEL="${GPT55_MODEL:-openai.gpt-5.5}"
# 其余原生 baseline 的模型名（实测可用名见 docs/ROUTIFY_API_MATRIX.md）。
GPT56_LUNA_MODEL="${GPT56_LUNA_MODEL:-gpt-5.6-luna}"
CLAUDE_OPUS_4_8_MODEL="${CLAUDE_OPUS_4_8_MODEL:-claude-opus-4-8}"
CLAUDE_SONNET_5_MODEL="${CLAUDE_SONNET_5_MODEL:-claude-sonnet-5}"
# kimi_k3 走 routify chat completions（同 OPENAI key）+ DERAIL 冻结 schema
# scaffold。行为参数全部读 configs/agents/kimi_k3.yaml，没有环境变量旁路。
KIMI_K3_MODEL="${KIMI_K3_MODEL:-kimi-k3}"
# Bedrock 拒绝 computer-use 与 prompt-caching 组合的 anthropic-beta 头，
# 而原生 computer 工具是 CUA baseline 的根本 —— 所以默认关掉 caching。
CLAUDE_PROMPT_CACHING_BETA="${CLAUDE_PROMPT_CACHING_BETA:-0}"
# 网关 RPM=60（按 key，跨模型共享）；429 走独立指数退避预算，不占用常规错误重试。
OPENAI_RATE_LIMIT_RETRIES="${OPENAI_RATE_LIMIT_RETRIES:-8}"
ANTHROPIC_RATE_LIMIT_RETRIES="${ANTHROPIC_RATE_LIMIT_RETRIES:-8}"
MYPCBENCH_OPENAI_REASONING_EFFORT="${MYPCBENCH_OPENAI_REASONING_EFFORT:-high}"
MYPCBENCH_OPENAI_EMPTY_OUTPUT_RETRIES="${MYPCBENCH_OPENAI_EMPTY_OUTPUT_RETRIES:-2}"
MYPCBENCH_SCREENSHOT_RETRIES="${MYPCBENCH_SCREENSHOT_RETRIES:-3}"
# 以下三个协议参数留空 = 取 configs/collection/mypcbench_runtime.yaml（论文统一
# 预算：3 次 / 150 步 / 3600 秒），由 driver 与沙箱内 01 脚本读取。显式给值只在
# 01 那边设了 ALLOW_CONFIG_OVERRIDE=1 时才会盖过 config。
REPEATS="${REPEATS:-}"
# 沙箱按单 VM 规格开；这是 01 的逃生口，压过 agent config 推导出的并发度。
NUM_VMS_OVERRIDE="${NUM_VMS_OVERRIDE:-${NUM_VMS:-1}}"
# 旧数据（F1-F4）均为 100 步口径，跨批次对比须交代。
MAX_STEPS="${MAX_STEPS:-}"
# bash 记账模式（tool_agent.py）：internal=bash 轮在步内循环不占步数（默认，
# 由 _MAX_BASH_ROUNDS_PER_STEP 兜底）；steps=统一记账，bash 轮每轮计为一个
# TOOL_CALL 步，与上游 openai_cuabash 语义对齐（2026-08-24 smoke 评估用）。
DERAIL_BASH_ACCOUNTING="${DERAIL_BASH_ACCOUNTING:-internal}"
# 单任务墙钟预算。v1 曾默认 7200s（2026-08-11 截断专项：sonnet 396s/step 在 3600s
# 内只够 ~9 步），F4 等旧批次是 7200 口径。
TASK_TIMEOUT="${TASK_TIMEOUT:-}"
# 任务来源（configs/collection/sources.yaml），留空取采集 config 的 task_source。
TASK_SOURCE="${TASK_SOURCE:-}"
ALLOW_CONFIG_OVERRIDE="${ALLOW_CONFIG_OVERRIDE:-0}"
TIMEOUT_PER_VM="${TIMEOUT_PER_VM:-259200}"
# 正式 184 采集门：shard 路径本就不触发 IS_FULL_TASK_SET，此 flag 只作为
# manifest 里的显式语义标记 + 提交侧纪律（默认 0，正式采集当次确认后设 1）。
FORMAL_COLLECTION="${FORMAL_COLLECTION:-0}"
# 提交动作本身即当次 API 授权（计划 Phase 3/4）；经 .rock_run.env 传给 driver，
# 由 driver 注入沙箱内 01 脚本的环境门。
DERAIL_OPENAI_API_APPROVED="${DERAIL_OPENAI_API_APPROVED:-1}"
DERAIL_OPENAI_API_PURPOSE="${DERAIL_OPENAI_API_PURPOSE:-mypcbench_collection_agent}"
DERAIL_WORKLOAD="${DERAIL_WORKLOAD:-collection}"
PHASE5_HAZARD_SMOKE="${PHASE5_HAZARD_SMOKE:-0}"
PHASE5_SHARD_COUNT="${PHASE5_SHARD_COUNT:-1}"
PHASE5_SHARD_INDEX="${PHASE5_SHARD_INDEX:-0}"
PHASE5_SMOKE_MODE="${PHASE5_SMOKE_MODE:-single_e2e}"
LOCK_QCOW2_SHA256="${LOCK_QCOW2_SHA256:-${PHASE5_QCOW2_SHA256:-}}"
ROCK_STARTUP_TIMEOUT="${ROCK_STARTUP_TIMEOUT:-600}"
ROCK_START_RETRIES="${ROCK_START_RETRIES:-8}"
DERAIL_ROCK_TOPOLOGY="${DERAIL_ROCK_TOPOLOGY:-in-sandbox}"
TAKEOVER_SOURCE_AGENT="${TAKEOVER_SOURCE_AGENT:-$AGENT_ID}"
TAKEOVER_TARGET_AGENT="${TAKEOVER_TARGET_AGENT:-$AGENT_ID}"
TAKEOVER_ANNOTATOR="${TAKEOVER_ANNOTATOR:-}"
TAKEOVER_BUILD_DIR="${TAKEOVER_BUILD_DIR:-}"
TAKEOVER_DEPTH="${TAKEOVER_DEPTH:-0}"
TAKEOVER_CONDITION="${TAKEOVER_CONDITION:-unaware}"
TAKEOVER_SHARD_COUNT="${TAKEOVER_SHARD_COUNT:-1}"
TAKEOVER_SHARD_OFFSET="${TAKEOVER_SHARD_OFFSET:-0}"
TAKEOVER_TOKENIZE_BASE_URL="${TAKEOVER_TOKENIZE_BASE_URL:-}"
TAKEOVER_TOKENIZE_MODE="${TAKEOVER_TOKENIZE_MODE:-vllm}"
TAKEOVER_CONTEXT_CAP="${TAKEOVER_CONTEXT_CAP:-0}"

# --- Nebula job 配置（照 MCUA 已验证默认值） ----------------------------------
# 集群侧标识不入库：提交前必须 export QUEUE / NEBULA_PROJECT。
QUEUE="${QUEUE:-<nebula-queue>}"
NEBULA_PROJECT="${NEBULA_PROJECT:-<nebula-project>}"
PLATFORM_ALGO_NAME="${PLATFORM_ALGO_NAME:-pytorch2100}"
WORKER_COUNT="${WORKER_COUNT:-1}"
# cluster.json 单位：cpu 百分之一核、gpu 百分比、memory MB。
# driver 是纯 asyncio 编排（无 vLLM/无推理，pip 装 rl-rock 为峰值负载），稳态
# <2C/<4G —— 原 32C/128G 照抄 MCUA，但 MCUA job 内要跑 vLLM 才需要那个规格；
# 8C/16G 已留 4x 余量。GPU：0 不在平台合法集合 {10,20,25,33,50,100,...}（MCUA
# 实测 task da6dde78），100=最小整卡；GPU_UNITS=10（0.1 卡）可试但需调度实测。
CPU="${CPU:-800}"
GPU_UNITS="${GPU_UNITS:-100}"
MEM="${MEM:-16384}"

# --- ROCK 沙箱配置 ------------------------------------------------------------
# ROCK 端点与个人 user_id 不入库：提交前必须 export ROCK_BASE_URL / ROCK_USER_ID。
ROCK_BASE_URL="${ROCK_BASE_URL:-<rock-endpoint>}"
ROCK_SANDBOX_IMAGE="${ROCK_SANDBOX_IMAGE:?set ROCK_SANDBOX_IMAGE}"
# 须是有 API egress、/dev/kvm 且能拉取 ROCK_SANDBOX_IMAGE 的集群。
ROCK_CLUSTER="${ROCK_CLUSTER:?set ROCK_CLUSTER}"
ROCK_DISK="${ROCK_DISK:-100g}"
ROCK_MEMORY="${ROCK_MEMORY:-16g}"
ROCK_CPUS="${ROCK_CPUS:-8}"
ROCK_USER_ID="${ROCK_USER_ID:-<rock-user-id>}"
ROCK_EXPERIMENT_ID="${ROCK_EXPERIMENT_ID:-derail-mypcbench}"
# 单片 46 任务中位 ~23h；arun 上限默认 30h，沙箱 TTL 48h 兜底自愈。
ROCK_RUN_TIMEOUT="${ROCK_RUN_TIMEOUT:-108000}"
ROCK_AUTO_CLEAR_SECONDS="${ROCK_AUTO_CLEAR_SECONDS:-172800}"

# --- OSS 路径 -----------------------------------------------------------------
# bucket 与个人前缀不入库：提交前必须 export OSS_BUCKET / OSS_PREFIX。
OSS_ENDPOINT="${OSS_ENDPOINT:?set OSS_ENDPOINT}"
OSS_BUCKET="${OSS_BUCKET:-<oss-bucket>}"
OSS_PREFIX="${OSS_PREFIX:-<oss-prefix>}"
OSS_ASSETS_URI="${OSS_ASSETS_URI:-oss://${OSS_BUCKET}/${OSS_PREFIX}/DERAIL/assets/mypcbench-vm}"
OSS_HARNESS_URI="${OSS_HARNESS_URI:-oss://${OSS_BUCKET}/${OSS_PREFIX}/DERAIL/assets/mypcbench-harness/mypcbench-caf9c754-takeover-native-v5.tar.gz}"
OSS_RESULTS_ROOT="${OSS_RESULTS_ROOT:-oss://${OSS_BUCKET}/${OSS_PREFIX}/DERAIL/results/raw/mypcbench}"
OSS_OPENCUA_SNAPSHOT_URI="${OSS_OPENCUA_SNAPSHOT_URI:-oss://${OSS_BUCKET}/${OSS_PREFIX}/DERAIL/assets/opencua-osworld/opencua-osworld-091f5ef.tar.gz}"
OSS_EVOCUA_SNAPSHOT_URI="${OSS_EVOCUA_SNAPSHOT_URI:-oss://${OSS_BUCKET}/${OSS_PREFIX}/DERAIL/assets/evocua/evocua-4a0ad5f.tar.gz}"
TAKEOVER_INPUT_ROOT="${TAKEOVER_INPUT_ROOT:-/data/oss_bucket_0/${OSS_PREFIX}/DERAIL/takeover_inputs/failure_prefix_v1_full_v3}"
PROXY_RESULTS_OSS_MOUNT="${PROXY_RESULTS_OSS_MOUNT:-/data/oss_bucket_0/${OSS_PREFIX}/DERAIL/results/raw/mypcbench}"
TAKEOVER_TRAJECTORY_ID_FILTER="${TAKEOVER_TRAJECTORY_ID_FILTER:-}"
TAKEOVER_TRAJECTORY_ID_FILE="${TAKEOVER_TRAJECTORY_ID_FILE:-}"

DRY_RUN="${DRY_RUN:-1}"

case "$DERAIL_WORKLOAD" in collection|takeover|phase5) ;; *)
  echo "ERROR: DERAIL_WORKLOAD must be collection|takeover|phase5" >&2; exit 2 ;;
esac
if [[ "$DERAIL_WORKLOAD" == "takeover" || "$DERAIL_WORKLOAD" == "phase5" ]]; then
  [[ "$DERAIL_ROCK_TOPOLOGY" == "proxy" ]] || {
    echo "ERROR: $DERAIL_WORKLOAD requires DERAIL_ROCK_TOPOLOGY=proxy" >&2; exit 2; }
fi
if [[ "$DERAIL_WORKLOAD" == "takeover" ]]; then
  for _req in TAKEOVER_SOURCE_AGENT TAKEOVER_TARGET_AGENT TAKEOVER_ANNOTATOR TAKEOVER_BUILD_DIR; do
    [[ -n "${!_req}" ]] || { echo "ERROR: $_req is required for takeover" >&2; exit 2; }
  done
fi
if [[ "$DERAIL_WORKLOAD" == "phase5" && "$DRY_RUN" == "0" ]]; then
  [[ "$LOCK_QCOW2_SHA256" =~ ^[0-9a-f]{64}$ ]] || {
    echo "ERROR: Phase 5 requires the accepted qcow2 SHA via PHASE5_QCOW2_SHA256" >&2
    exit 2
  }
fi

# 真跑（DRY_RUN=0）前强制要求内部基建参数已用环境变量接管占位符。
if [[ "$DRY_RUN" == "0" ]]; then
  for _req in ROCK_BASE_URL ROCK_USER_ID OSS_BUCKET OSS_PREFIX; do
    _val="${!_req}"
    if [[ -z "$_val" || "$_val" == "<"*">" ]]; then
      echo "ERROR: $_req 未设置（仍是占位符）；先 set -a; . <env文件>; set +a 再提交" >&2
      exit 2
    fi
  done
fi

# ROCK_API_KEY：driver 建沙箱的鉴权头（XRL-Authorization）。与 OSS/routify 凭证
# 同走 0600 mount-secret，不进代码包。从提交 shell 环境读取。
ROCK_API_KEY="${ROCK_API_KEY:-}"
if [[ -z "$ROCK_API_KEY" && "$DRY_RUN" == "0" ]]; then
  echo "ERROR: ROCK_API_KEY 未设置（driver 建沙箱会 401）；先 set -a; . <env文件>; set +a" >&2
  exit 2
fi

# --- 预检 ---------------------------------------------------------------------
[[ -f "$REPO/$SHARD_FILE" ]] || { echo "ERROR: 找不到 shard 文件：$REPO/$SHARD_FILE" >&2; exit 2; }
case "$AGENT_ID" in
  dummy)
    ;;
  gpt_5_5|gpt_5_6_luna|kimi_k3|kimi_k3_cuabash)
    # routify key 只走 0600 mount-secret 通道；从提交 shell 环境读取。
    if [[ -z "${OPENAI_API_KEY:-}" || -z "${OPENAI_BASE_URL:-}" ]]; then
      if [[ "$DRY_RUN" == "0" ]]; then
        echo "ERROR: AGENT_ID=$AGENT_ID 需要 OPENAI_API_KEY + OPENAI_BASE_URL（routify）" >&2
        echo "       在提交 shell 里 export 后再跑；它们经 0600 mount-secret 注入沙箱，不进代码包。" >&2
        exit 2
      fi
      echo "[submit-derail-rock] WARNING: OPENAI_API_KEY/OPENAI_BASE_URL 未设置；正式提交（DRY_RUN=0）会 fail fast。" >&2
    fi
    ;;
  claude_opus_4_8|claude_sonnet_5)
    # Claude 走 Anthropic 原生协议（routify /protocol/anthropic），SDK 自己读
    # ANTHROPIC_BASE_URL；同一把 routify key 当 ANTHROPIC_API_KEY 用。
    if [[ -z "${ANTHROPIC_API_KEY:-}" || -z "${ANTHROPIC_BASE_URL:-}" ]]; then
      if [[ "$DRY_RUN" == "0" ]]; then
        echo "ERROR: AGENT_ID=$AGENT_ID 需要 ANTHROPIC_API_KEY + ANTHROPIC_BASE_URL" >&2
        echo "       内部网关形式：ANTHROPIC_BASE_URL=<gateway>/protocol/anthropic" >&2
        exit 2
      fi
      echo "[submit-derail-rock] WARNING: ANTHROPIC_API_KEY/ANTHROPIC_BASE_URL 未设置；正式提交会 fail fast。" >&2
    fi
    ;;
  *)
    echo "ERROR: ROCK 流水线目前支持 AGENT_ID=dummy|gpt_5_5|gpt_5_6_luna|kimi_k3|kimi_k3_cuabash|claude_opus_4_8|claude_sonnet_5（其余 agent 未适配）" >&2
    exit 2
    ;;
esac

# --- 提交前 sweep 僵尸沙箱（429 教训：被杀的 job 不会走 finally stop） ---------
SWEEP_STALE_SANDBOXES="${SWEEP_STALE_SANDBOXES:-1}"
if [[ "$SWEEP_STALE_SANDBOXES" == "1" && "$DRY_RUN" == "0" ]]; then
  ROCKCLI_BIN="$(command -v rockcli 2>/dev/null || find "$HOME/.nvm/versions/node" -name rockcli -type l 2>/dev/null | head -1)"
  if [[ -n "$ROCKCLI_BIN" ]]; then
    # 实测 rockcli 无条件输出 ANSI 色码（管道/NO_COLOR/TERM=dumb 均关不掉），ID 列被
    # bold 包裹 —— 不 strip 则下游 ^..$ 锚定匹配全落空；实验沙箱状态是全大写
    # RUNNING/PENDING（原作 Pending 永远匹配不到，pending 僵尸必漏扫）。
    stale=$("$ROCKCLI_BIN" expr "$ROCK_EXPERIMENT_ID" sandboxes 2>/dev/null \
              | sed "s/$(printf '\033')\[[0-9;]*m//g" \
              | awk '/RUNNING|PENDING/ {print $1}' | grep -E '^[0-9a-f]{32}$' || true)
    for sb in $stale; do
      echo "[submit-derail-rock] sweeping stale sandbox $sb"
      "$ROCKCLI_BIN" sandbox "$sb" stop >/dev/null 2>&1 || true
    done
    [[ -z "$stale" ]] && echo "[submit-derail-rock] no stale sandboxes in $ROCK_EXPERIMENT_ID"
  else
    echo "[submit-derail-rock] WARNING: rockcli not found; skipping stale-sandbox sweep" >&2
  fi
fi

# --- 代码包 exclude ------------------------------------------------------------
# third_party 由 driver 在沙箱内从 OSS harness tar 恢复（含 .git，运行时零 github）；
# artifacts/draft/runs 是实验产物；.env 永不进代码包（routify key 走 mount-secret）。
IGNORE_LIST="${IGNORE_LIST:-third_party/*,takeovewr_annotation/*,artifacts/*,data/*,results/*,draft/*,runs/*,.git/*,.venv*,.env,docs/*,*.log,*.pyc,__pycache__/*,.props_*,.cluster_*}"

# --- OSS 凭证（0600 property_file，不进 argv） ---------------------------------
OSS_ID="${OSS_ACCESS_ID:-$(sed -n 's/^accessKeyID=//p' "${HOME}/.ossutilconfig" 2>/dev/null || true)}"
OSS_KEY="${OSS_ACCESS_KEY:-$(sed -n 's/^accessKeySecret=//p' "${HOME}/.ossutilconfig" 2>/dev/null || true)}"
[[ -n "${OSS_ID}" && -n "${OSS_KEY}" ]] \
  || { echo "ERROR: OSS creds missing (env OSS_ACCESS_ID/KEY or ~/.ossutilconfig)" >&2; exit 2; }

CLUSTER_FILE=".cluster_derail_rock.$$.json"
RUN_ENV_FILE=".rock_run.env"
PROPS_FILE=".props_derail_rock.$$"
OSS_CONFIG_FILE=".ossutil_derail_rock.$$"
PHASE5_PACKAGE_DIR=".phase5_package.$$"
# mount-staged secret 必须活得比本脚本久（entry 几分钟后才 source）；
# 只有未提交（DRY_RUN / 提交失败）时才在这里回收 OSS 上的 secret。
SUBMITTED=0
trap 'rm -f "$REPO/$CLUSTER_FILE" "$REPO/$RUN_ENV_FILE" "$REPO/$PROPS_FILE"; \
  rm -rf "$REPO/$PHASE5_PACKAGE_DIR"; \
  if [[ "${SUBMITTED:-0}" != "1" && "${DRY_RUN:-1}" != "1" ]]; then \
    "${OSSUTIL_BIN:-/usr/local/bin/ossutil}" -c "$REPO/$OSS_CONFIG_FILE" \
      rm -f "oss://${OSS_BUCKET}/${DERAIL_SECRET_OSS:-__none__}" \
      >/dev/null 2>&1 || true; \
  fi; _release_submit_lock' EXIT

cat > "$REPO/$CLUSTER_FILE" <<JSON
{"worker": {"cpu": ${CPU}, "gpu": ${GPU_UNITS}, "memory": ${MEM}}}
JSON
umask 077
printf 'oss_access_id=%s\noss_access_key=%s\noss_bucket=%s\noss_endpoint=%s\n' \
  "${OSS_ID}" "${OSS_KEY}" "${OSS_BUCKET}" "${OSS_ENDPOINT}" > "$REPO/$PROPS_FILE"
printf '[Credentials]\nlanguage=EN\nendpoint=%s\naccessKeyID=%s\naccessKeySecret=%s\n' \
  "${OSS_ENDPOINT}" "${OSS_ID}" "${OSS_KEY}" > "$REPO/$OSS_CONFIG_FILE"
umask 022

# --- 0600 secret 上 OSS mount 通道（OSS AK + routify key） ---------------------
DERAIL_SECRET_NAME=".derail_secrets.$$.sh"
DERAIL_SECRET_OSS="${OSS_PREFIX}/DERAIL/tmp/${DERAIL_SECRET_NAME}"
command -v ossutil >/dev/null 2>&1 || OSSUTIL_CANDIDATE="/usr/local/bin/ossutil"
OSSUTIL_BIN="$(command -v ossutil 2>/dev/null || echo "${OSSUTIL_CANDIDATE:-}")"
if [[ "$DRY_RUN" == "1" ]]; then
  echo "[submit-derail-rock] DRY_RUN=1 — skip mount-secret upload"
elif [[ -x "${OSSUTIL_BIN:-/nonexistent}" ]]; then
  _tmp_secret="$(mktemp)"; chmod 600 "$_tmp_secret"
  printf 'export OSS_ACCESS_ID=%q\nexport OSS_ACCESS_KEY=%q\nexport OSS_ENDPOINT=%q\n' \
    "${OSS_ID}" "${OSS_KEY}" "oss-${OSS_ENDPOINT%%.*}.aliyuncs.com" > "$_tmp_secret"
  # routify 凭证同车（entry source 后自删；沙箱内 agent loop 直接调 routify）。
  [[ -n "${OPENAI_API_KEY:-}" ]] && printf 'export OPENAI_API_KEY=%q\n' "$OPENAI_API_KEY" >> "$_tmp_secret"
  [[ -n "${OPENAI_BASE_URL:-}" ]] && printf 'export OPENAI_BASE_URL=%q\n' "$OPENAI_BASE_URL" >> "$_tmp_secret"
  # Anthropic 通道（claude_* agent）：同一把 routify key，SDK 从 env 读 base_url。
  [[ -n "${ANTHROPIC_API_KEY:-}" ]] && printf 'export ANTHROPIC_API_KEY=%q\n' "$ANTHROPIC_API_KEY" >> "$_tmp_secret"
  [[ -n "${ANTHROPIC_BASE_URL:-}" ]] && printf 'export ANTHROPIC_BASE_URL=%q\n' "$ANTHROPIC_BASE_URL" >> "$_tmp_secret"
  # ROCK_API_KEY 同车：driver 建沙箱的鉴权头，平台级 OSS AK 与它无关。
  [[ -n "$ROCK_API_KEY" ]] && printf 'export ROCK_API_KEY=%q\n' "$ROCK_API_KEY" >> "$_tmp_secret"
  if "${OSSUTIL_BIN}" -c "$REPO/$OSS_CONFIG_FILE" \
      cp -f "$_tmp_secret" "oss://${OSS_BUCKET}/${DERAIL_SECRET_OSS}" >/dev/null 2>&1; then
    echo "[submit-derail-rock] staged secrets on mount: oss://${OSS_BUCKET}/${DERAIL_SECRET_OSS}"
  else
    echo "ERROR: failed to stage secrets on mount; aborting before submit" >&2
    exit 1
  fi
  rm -f "$_tmp_secret"
else
  echo "ERROR: ossutil not found locally; cannot stage secrets" >&2
  exit 2
fi
echo "[submit-derail-rock] cluster.json -> $REPO/$CLUSTER_FILE : $(cat "$REPO/$CLUSTER_FILE")"

# --- .rock_run.env（随代码包走，entry source；绝不含任何 key） ------------------
cat > "$REPO/$RUN_ENV_FILE" <<ENVF
export AGENT_ID='${AGENT_ID}'
export SHARD_FILE='${SHARD_FILE}'
export COLLECTION_ID='${COLLECTION_ID}'
export RESUME_SEED='${RESUME_SEED}'
export RERUN_PURGE='${RERUN_PURGE}'
export GPT55_MODEL='${GPT55_MODEL}'
export GPT56_LUNA_MODEL='${GPT56_LUNA_MODEL}'
export CLAUDE_OPUS_4_8_MODEL='${CLAUDE_OPUS_4_8_MODEL}'
export CLAUDE_SONNET_5_MODEL='${CLAUDE_SONNET_5_MODEL}'
export KIMI_K3_MODEL='${KIMI_K3_MODEL}'
export CLAUDE_PROMPT_CACHING_BETA='${CLAUDE_PROMPT_CACHING_BETA}'
export OPENAI_RATE_LIMIT_RETRIES='${OPENAI_RATE_LIMIT_RETRIES}'
export ANTHROPIC_RATE_LIMIT_RETRIES='${ANTHROPIC_RATE_LIMIT_RETRIES}'
export MYPCBENCH_OPENAI_REASONING_EFFORT='${MYPCBENCH_OPENAI_REASONING_EFFORT}'
export MYPCBENCH_OPENAI_EMPTY_OUTPUT_RETRIES='${MYPCBENCH_OPENAI_EMPTY_OUTPUT_RETRIES}'
export MYPCBENCH_SCREENSHOT_RETRIES='${MYPCBENCH_SCREENSHOT_RETRIES}'
export REPEATS='${REPEATS}'
export NUM_VMS_OVERRIDE='${NUM_VMS_OVERRIDE}'
export MAX_STEPS='${MAX_STEPS}'
export DERAIL_BASH_ACCOUNTING='${DERAIL_BASH_ACCOUNTING}'
export TASK_TIMEOUT='${TASK_TIMEOUT}'
export TASK_SOURCE='${TASK_SOURCE}'
export ALLOW_CONFIG_OVERRIDE='${ALLOW_CONFIG_OVERRIDE}'
export TIMEOUT_PER_VM='${TIMEOUT_PER_VM}'
export FORMAL_COLLECTION='${FORMAL_COLLECTION}'
export DERAIL_OPENAI_API_APPROVED='${DERAIL_OPENAI_API_APPROVED}'
export DERAIL_OPENAI_API_PURPOSE='${DERAIL_OPENAI_API_PURPOSE}'
export DERAIL_WORKLOAD='${DERAIL_WORKLOAD}'
export PHASE5_HAZARD_SMOKE='${PHASE5_HAZARD_SMOKE}'
export PHASE5_SHARD_COUNT='${PHASE5_SHARD_COUNT}'
export PHASE5_SHARD_INDEX='${PHASE5_SHARD_INDEX}'
export PHASE5_SMOKE_MODE='${PHASE5_SMOKE_MODE}'
export PHASE5_PACKAGE_DIR='${PHASE5_PACKAGE_DIR}'
export LOCK_QCOW2_SHA256='${LOCK_QCOW2_SHA256}'
export ROCK_STARTUP_TIMEOUT='${ROCK_STARTUP_TIMEOUT}'
export ROCK_START_RETRIES='${ROCK_START_RETRIES}'
export DERAIL_ROCK_TOPOLOGY='${DERAIL_ROCK_TOPOLOGY}'
export TAKEOVER_INPUT_ROOT='${TAKEOVER_INPUT_ROOT}'
export TAKEOVER_SOURCE_AGENT='${TAKEOVER_SOURCE_AGENT}'
export TAKEOVER_TARGET_AGENT='${TAKEOVER_TARGET_AGENT}'
export TAKEOVER_ANNOTATOR='${TAKEOVER_ANNOTATOR}'
export TAKEOVER_BUILD_DIR='${TAKEOVER_BUILD_DIR}'
export TAKEOVER_DEPTH='${TAKEOVER_DEPTH}'
export TAKEOVER_CONDITION='${TAKEOVER_CONDITION}'
export TAKEOVER_SHARD_COUNT='${TAKEOVER_SHARD_COUNT}'
export TAKEOVER_SHARD_OFFSET='${TAKEOVER_SHARD_OFFSET}'
export TAKEOVER_TOKENIZE_BASE_URL='${TAKEOVER_TOKENIZE_BASE_URL}'
export TAKEOVER_TOKENIZE_MODE='${TAKEOVER_TOKENIZE_MODE}'
export TAKEOVER_CONTEXT_CAP='${TAKEOVER_CONTEXT_CAP}'
export PROXY_RESULTS_OSS_MOUNT='${PROXY_RESULTS_OSS_MOUNT}'
export TAKEOVER_TRAJECTORY_ID_FILTER='${TAKEOVER_TRAJECTORY_ID_FILTER}'
export TAKEOVER_TRAJECTORY_ID_FILE='${TAKEOVER_TRAJECTORY_ID_FILE}'
export OSS_OPENCUA_SNAPSHOT_URI='${OSS_OPENCUA_SNAPSHOT_URI}'
export OSS_EVOCUA_SNAPSHOT_URI='${OSS_EVOCUA_SNAPSHOT_URI}'
export ROCK_BASE_URL='${ROCK_BASE_URL}'
export ROCK_SANDBOX_IMAGE='${ROCK_SANDBOX_IMAGE}'
export ROCK_CLUSTER='${ROCK_CLUSTER}'
export ROCK_DISK='${ROCK_DISK}'
export ROCK_MEMORY='${ROCK_MEMORY}'
export ROCK_CPUS='${ROCK_CPUS}'
export ROCK_USER_ID='${ROCK_USER_ID}'
export ROCK_EXPERIMENT_ID='${ROCK_EXPERIMENT_ID}'
export ROCK_RUN_TIMEOUT='${ROCK_RUN_TIMEOUT}'
export ROCK_AUTO_CLEAR_SECONDS='${ROCK_AUTO_CLEAR_SECONDS}'
export OSS_ASSETS_URI='${OSS_ASSETS_URI}'
export OSS_HARNESS_URI='${OSS_HARNESS_URI}'
export OSS_RESULTS_ROOT='${OSS_RESULTS_ROOT}'
export DERAIL_SECRET_MOUNT='/data/oss_bucket_0/${DERAIL_SECRET_OSS}'
ENVF
chmod 600 "$REPO/$RUN_ENV_FILE"

if [[ "$DERAIL_WORKLOAD" == "phase5" ]]; then
  mkdir -p "$REPO/$PHASE5_PACKAGE_DIR/artifacts/phase5" \
    "$REPO/$PHASE5_PACKAGE_DIR/data/synthesis/generation" \
    "$REPO/$PHASE5_PACKAGE_DIR/data/synthesis/task_ir_v1"
  cp "$REPO/artifacts/phase5/run-manifest-option-a-20260912.json" \
    "$REPO/artifacts/phase5/hazards-terminal-option-a-20260912.jsonl" \
    "$REPO/$PHASE5_PACKAGE_DIR/artifacts/phase5/"
  cp "$REPO/data/synthesis/generation/final_v1.tar.gz" \
    "$REPO/$PHASE5_PACKAGE_DIR/data/synthesis/generation/"
  cp "$REPO/data/synthesis/task_ir_v1/accepted.tar.gz" \
    "$REPO/$PHASE5_PACKAGE_DIR/data/synthesis/task_ir_v1/"
  echo "[submit-derail-rock] staged frozen Phase 5 input package"
fi
echo "[submit-derail-rock] run-config -> $REPO/$RUN_ENV_FILE (AGENT_ID=$AGENT_ID SHARD=$SHARD_FILE FORMAL=$FORMAL_COLLECTION)"

ENTRY="scripts/rock/entry_derail_rock_nebula.py"
_shard_tag="$(basename "$SHARD_FILE" .json | tail -c 24)"
JOB_NAME="derail-rock-mypcbench-${AGENT_ID}-${_shard_tag}"
JOB_NAME_SUFFIX="${JOB_NAME_SUFFIX:-}"
[[ -n "$JOB_NAME_SUFFIX" ]] && JOB_NAME="${JOB_NAME}-${JOB_NAME_SUFFIX}"

# nebulactl 在 ~/bin/nebulactl（pip 包，非交互 shell 的 PATH 常缺 ~/bin）；
# 找不到时回退 nebula-cli（run mdl 语法兼容），最后回退 NVM 里的 nebula-cli。
NEBULACTL_BIN="$(command -v nebulactl 2>/dev/null \
  || { [[ -x "$HOME/bin/nebulactl" ]] && echo "$HOME/bin/nebulactl"; } \
  || command -v nebula-cli 2>/dev/null \
  || find "$HOME/.nvm/versions/node" -name nebula-cli -type l 2>/dev/null | head -1)"
[[ -x "${NEBULACTL_BIN:-/nonexistent}" ]] \
  || { echo "ERROR: nebulactl/nebula-cli not found（提交入口缺失）" >&2; exit 2; }

cmd=( "$NEBULACTL_BIN" run mdl )
# nebulactl 的尺寸检查在 --ignore 过滤前算全目录（artifacts 2.6G 必撞 1G 线）；
# 实际上传仍按 ignore 名单排除，故超线时用 NEBULACTL_FORCE=1 加 --force。
if [[ "${NEBULACTL_FORCE:-0}" == "1" ]]; then
  cmd+=( --force )
fi
cmd+=(
  --algo_name="${PLATFORM_ALGO_NAME}"
  --property_file="${PROPS_FILE}"
  --queue="${QUEUE}"
  --nebula_project="${NEBULA_PROJECT}"
  --job_name="${JOB_NAME}"
  --worker_count="${WORKER_COUNT}"
  --entry="${ENTRY}"
  --user_params=""
  --file.cluster_file="${CLUSTER_FILE}"
  --ignore="${IGNORE_LIST}"
)

echo
echo "# ---- ${JOB_NAME} ---- (credentials via --property_file + mount secret, not argv)"
printf '%q ' "${cmd[@]}"
echo
if [[ "$DRY_RUN" != "1" ]]; then
  if "${cmd[@]}"; then
    SUBMITTED=1
    echo "[submit-derail-rock] submitted. 监控: /oss-job-monitor；沙箱问题: /rock-debug"
  else
    echo "ERROR: nebulactl submission failed" >&2
    exit 1
  fi
else
  echo
  echo "[submit-derail-rock] DRY_RUN=1 — command printed only. Re-run with DRY_RUN=0 to submit."
fi
