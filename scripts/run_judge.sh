#!/usr/bin/env bash
# run_judge.sh — DERAIL MyPCBench 统一 judge 入口（唯一合法的打分启动方式）
#
# 背景（2026-08-08 事故）：直接裸跑上游 judge_results.py 时，后台进程未继承
# MYPCBENCH_RUBRIC_JUDGE_MODEL，judge 选择逻辑静默回退到论文默认 Gemini 路径，
# 缺 GEMINI_API_KEY 导致 shard_3/4 共 92 任务全量 judge_error。
# 本脚本固化判官配置并做 fail-fast 预检，杜绝复发。
#
# 用法：
#   bash scripts/run_judge.sh <result_dir> [<result_dir> ...] [--force]
# 例：
#   bash scripts/run_judge.sh artifacts/raw_rollouts/mypcbench/20260806T164231Z-gpt_5_5-shard_3_of_4
# 长时 judge（>1h）请在 tmux 会话内运行（仓库约定：所有长任务 tmux 托管）：
#   tmux new -s judge -d 'bash scripts/run_judge.sh <result_dir>'

set -euo pipefail

# ---- 判官配置 ----
# judge 模型唯一真源是 configs/judges/default.yaml（论文：GPT-5.6-terra），由
# judge_model_registry.py 解析；DERAIL_RUN_JUDGE_MODEL 只能等于它（试验需 ALLOW_CONFIG_OVERRIDE=1）。
# 历史：2026-08-10 曾改用 claude-opus-4-8，09 月 takeover 又混用过 7 个模型（见 docs/PAPER_AUDIT_20260930.md H10）。
PINNED_JUDGE_MODEL="${DERAIL_RUN_JUDGE_MODEL:-}"
PINNED_MAX_IMAGES="${DERAIL_RUN_JUDGE_MAX_IMAGES:-}"
# 注意 gpt 系后端仅 50 图硬上限（registry 的 max_images）。
# judge 子进程超时：上游默认 300s，但 100 图全轨迹负载下单约 270s，大面积超时（2026-08-12 片1 31/46）。
# 长轨迹重打时外部传 JUDGE_TIMEOUT=2400；默认取 registry 的 timeout_seconds。
JUDGE_TIMEOUT="${JUDGE_TIMEOUT:-}"
# rubric 并发度：保持上游默认 4。2026-08-15 曾短暂提至 8，实测打满 routify 配额，
# 全 rubric 秒级 Connection error 批量写假 0 分（evocua vm1 污染 16 任务），已回退；
# 需要提速时外部显式传同名 env，并先小规模验证不触发限流。
JUDGE_CONCURRENCY="${MYPCBENCH_OSWORLD_JUDGE_CONCURRENCY:-}"
# 凭证文件路径不入库：默认读 ~/.derail_creds.env，可用 DERAIL_CRED_ENV 覆盖。
CRED_ENV="${DERAIL_CRED_ENV:-$HOME/.derail_creds.env}"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
JUDGE_PY="$REPO_ROOT/third_party/MyPCBench/agent-harness/judge_results.py"
HARD_TIMEOUT_PY="$REPO_ROOT/scripts/31_run_with_timeout.py"
REGISTRY_PY="$REPO_ROOT/scripts/judge_model_registry.py"
JUDGE_WRAPPER="$REPO_ROOT/scripts/30_full_traj_judge.py"
BUNDLE_PY="$REPO_ROOT/scripts/14_takeover_bundle_prefix.py"
JUDGE_REGISTRY="${DERAIL_JUDGE_REGISTRY:-$REPO_ROOT/configs/judges/routify_model_registry.json}"

# ---- 凭证加载 ----
# 优先 CRED_ENV（DERAIL_CRED_ENV 或 ~/.derail_creds.env），没有就回落到仓库根的 .env。
# .env 是冒号格式且值带引号（KEY: "value"），必须 tr -d 掉引号，按 = 切会把整行带出来。
if [[ -f "$CRED_ENV" ]]; then
  set -a; . "$CRED_ENV"; set +a
elif [[ -f "$REPO_ROOT/.env" ]]; then
  echo "[run_judge] $CRED_ENV 不存在，凭证回落到 $REPO_ROOT/.env"
  if [[ -z "${OPENAI_API_KEY:-}" ]]; then
    OPENAI_API_KEY="$(sed -n 's/^OPENAI_API_KEY[=:][[:space:]]*//p' "$REPO_ROOT/.env" \
      | tr -d '"'"'"' ' | head -1)"
    export OPENAI_API_KEY
  fi
else
  echo "FATAL: 凭证既不在 $CRED_ENV 也不在 $REPO_ROOT/.env（可用 DERAIL_CRED_ENV 覆盖）" >&2
  exit 1
fi

# ---- 参数解析 ----
FORCE_FLAG=""
RESULT_DIRS=()
for arg in "$@"; do
  case "$arg" in
    --force) FORCE_FLAG="--force" ;;
    *)       RESULT_DIRS+=("$arg") ;;
  esac
done

# ---- fail-fast 预检：任何一项不满足直接退出，绝不允许静默回退判官 ----
fail() { echo "FATAL: $1" >&2; exit 1; }
[[ ${#RESULT_DIRS[@]} -gt 0 ]] || fail "缺少 <result_dir> 参数"
[[ -f "$REGISTRY_PY" ]] || fail "Judge model registry resolver 不存在: $REGISTRY_PY"
# 单一 judge + 被测对象不得兼任 judge（takeover_manifest.json 里的 source/target agent）。
registry_args=()
[[ -n "$PINNED_JUDGE_MODEL" ]] && registry_args+=("$PINNED_JUDGE_MODEL")
for d in "${RESULT_DIRS[@]}"; do registry_args+=(--result-dir "$d"); done
PINNED_JUDGE_MODEL="$(python3 "$REGISTRY_PY" "${registry_args[@]}" --print-model)" \
  || fail "judge 配置校验失败（单一 judge / 同源检查）"
[[ "$PINNED_JUDGE_MODEL" =~ ^[A-Za-z0-9._/-]+$ ]] || fail "judge model 含非法字符"
registry_values="$(python3 "$REGISTRY_PY" "$PINNED_JUDGE_MODEL" --registry "$JUDGE_REGISTRY")" \
  || fail "Judge model registry rejected model: $PINNED_JUDGE_MODEL"
IFS=$'\t' read -r JUDGE_PROTOCOL MODEL_MAX_IMAGES JUDGE_ADMISSION \
  MODEL_CONCURRENCY MODEL_TIMEOUT <<< "$registry_values"
PINNED_MAX_IMAGES="${PINNED_MAX_IMAGES:-$MODEL_MAX_IMAGES}"
JUDGE_TIMEOUT="${JUDGE_TIMEOUT:-$MODEL_TIMEOUT}"
JUDGE_CONCURRENCY="${JUDGE_CONCURRENCY:-$MODEL_CONCURRENCY}"
[[ "$PINNED_MAX_IMAGES" =~ ^[1-9][0-9]*$ ]] || fail "judge max images 必须是正整数"
(( ${#PINNED_MAX_IMAGES} <= ${#MODEL_MAX_IMAGES} && PINNED_MAX_IMAGES <= MODEL_MAX_IMAGES )) \
  || fail "$PINNED_JUDGE_MODEL max images must be <= $MODEL_MAX_IMAGES"
[[ "$JUDGE_TIMEOUT" =~ ^[1-9][0-9]*$ ]] || fail "judge timeout 必须是正整数"
[[ "$JUDGE_CONCURRENCY" =~ ^[1-9][0-9]*$ ]] || fail "judge concurrency 必须是正整数"
export MYPCBENCH_OSWORLD_JUDGE_CONCURRENCY="$JUDGE_CONCURRENCY"
[[ -n "${OPENAI_API_KEY:-}" ]]    || fail "OPENAI_API_KEY 为空（judge 无法调用 routify）"
if [[ -z "${OPENAI_BASE_URL:-}" ]]; then
  # 走 OpenAI 官方端点。必须 unset：openai SDK 把空串当成真的 base_url
  # （base_url='' -> APIConnectionError），只有变量不存在才回落到官方地址。
  unset OPENAI_BASE_URL
  echo "[run_judge] OPENAI_BASE_URL 未设，judge 走 OpenAI 官方端点"
fi
# DERAIL 包装：附录 D 的 prompt 原样，user 消息多一个最终状态 s_T 数据段（bundle 里有才加）。
export MYPCBENCH_RUBRIC_JUDGE_COMMAND="python3 $JUDGE_WRAPPER"
[[ -f "$JUDGE_PY" ]]              || fail "上游 judge 脚本不存在: $JUDGE_PY"
[[ -f "$HARD_TIMEOUT_PY" ]]       || fail "Judge hard-timeout wrapper 不存在: $HARD_TIMEOUT_PY"
[[ -f "$JUDGE_WRAPPER" ]]         || fail "Judge wrapper 不存在: $JUDGE_WRAPPER"
[[ -f "$BUNDLE_PY" ]]             || fail "bundle 构建脚本不存在: $BUNDLE_PY"

export MYPCBENCH_RUBRIC_JUDGE_MODEL="$PINNED_JUDGE_MODEL"
export MYPCBENCH_OSWORLD_JUDGE_MAX_IMAGES="$PINNED_MAX_IMAGES"
export MYPCBENCH_OSWORLD_JUDGE_REASONING_EFFORT="${MYPCBENCH_OSWORLD_JUDGE_REASONING_EFFORT:-low}"
export MYPCBENCH_OSWORLD_JUDGE_REQUEST_TIMEOUT="${MYPCBENCH_OSWORLD_JUDGE_REQUEST_TIMEOUT:-300}"
export MYPCBENCH_OSWORLD_JUDGE_PROCESS_TIMEOUT="${MYPCBENCH_OSWORLD_JUDGE_PROCESS_TIMEOUT:-900}"
HARD_JUDGE_TIMEOUT="${DERAIL_JUDGE_HARD_TIMEOUT:-$((JUDGE_TIMEOUT + MYPCBENCH_OSWORLD_JUDGE_REQUEST_TIMEOUT + 60))}"
[[ "$HARD_JUDGE_TIMEOUT" =~ ^[1-9][0-9]*$ ]] || fail "Judge hard timeout 必须是正整数"

echo "[run_judge] judge=$PINNED_JUDGE_MODEL (routify) protocol=$JUDGE_PROTOCOL admission=$JUDGE_ADMISSION max_images=$PINNED_MAX_IMAGES reasoning_effort=$MYPCBENCH_OSWORLD_JUDGE_REASONING_EFFORT request_timeout=${MYPCBENCH_OSWORLD_JUDGE_REQUEST_TIMEOUT}s process_timeout=${MYPCBENCH_OSWORLD_JUDGE_PROCESS_TIMEOUT}s task_timeout=${JUDGE_TIMEOUT}s hard_timeout=${HARD_JUDGE_TIMEOUT}s concurrency=$MYPCBENCH_OSWORLD_JUDGE_CONCURRENCY dirs=${#RESULT_DIRS[@]} force=${FORCE_FLAG:-no}"
for d in "${RESULT_DIRS[@]}"; do
  [[ -d "$d" ]] || fail "result_dir 不存在: $d"
  # 判官看到的证据在 bundle 一侧补齐：接管前缀、纯工具轮的调用与输出、最终状态 s_T（幂等）。
  python3 "$BUNDLE_PY" "$d" || fail "bundle 证据补齐失败: $d"
  echo "===== JUDGE $d start $(date '+%F %T')"
  python3 "$HARD_TIMEOUT_PY" \
    --timeout "$HARD_JUDGE_TIMEOUT" --grace 30 \
    python3 "$JUDGE_PY" --result_dir "$d" --timeout "$JUDGE_TIMEOUT" \
    --fail-on-errors $FORCE_FLAG
  echo "===== JUDGE $d done  $(date '+%F %T')"
done
