#!/usr/bin/env bash
# Usage: tmux new -s judge -d 'bash scripts/judge/run_judge.sh <result_dir> [<result_dir> ...] [--force]'

set -euo pipefail

PINNED_JUDGE_MODEL="${RECOVERY_RUN_JUDGE_MODEL:-}"
PINNED_MAX_IMAGES="${RECOVERY_RUN_JUDGE_MAX_IMAGES:-}"
JUDGE_TIMEOUT="${JUDGE_TIMEOUT:-}"
JUDGE_CONCURRENCY="${MYPCBENCH_OSWORLD_JUDGE_CONCURRENCY:-}"
CRED_ENV="${RECOVERY_CRED_ENV:-$HOME/.recovery_creds.env}"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
JUDGE_PY="$REPO_ROOT/third_party/MyPCBench/agent-harness/judge_results.py"
HARD_TIMEOUT_PY="$REPO_ROOT/scripts/judge/run_with_timeout.py"
REGISTRY_PY="$REPO_ROOT/scripts/judge/judge_model_registry.py"
JUDGE_WRAPPER="$REPO_ROOT/scripts/judge/full_traj_judge.py"
BUNDLE_PY="$REPO_ROOT/scripts/takeover/bundle_prefix.py"
JUDGE_REGISTRY="${RECOVERY_JUDGE_REGISTRY:-$REPO_ROOT/configs/judges/model_registry.json}"

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
  echo "FATAL: 凭证既不在 $CRED_ENV 也不在 $REPO_ROOT/.env（可用 RECOVERY_CRED_ENV 覆盖）" >&2
  exit 1
fi

FORCE_FLAG=""
RESULT_DIRS=()
for arg in "$@"; do
  case "$arg" in
    --force) FORCE_FLAG="--force" ;;
    *)       RESULT_DIRS+=("$arg") ;;
  esac
done

fail() { echo "FATAL: $1" >&2; exit 1; }
[[ ${#RESULT_DIRS[@]} -gt 0 ]] || fail "缺少 <result_dir> 参数"
[[ -f "$REGISTRY_PY" ]] || fail "Judge model registry resolver 不存在: $REGISTRY_PY"
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
[[ -n "${OPENAI_API_KEY:-}" ]]    || fail "OPENAI_API_KEY 为空（judge 无法调用网关）"
if [[ -z "${OPENAI_BASE_URL:-}" ]]; then
  unset OPENAI_BASE_URL
  echo "[run_judge] OPENAI_BASE_URL 未设，judge 走 OpenAI 官方端点"
fi
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
HARD_JUDGE_TIMEOUT="${RECOVERY_JUDGE_HARD_TIMEOUT:-$((JUDGE_TIMEOUT + MYPCBENCH_OSWORLD_JUDGE_REQUEST_TIMEOUT + 60))}"
[[ "$HARD_JUDGE_TIMEOUT" =~ ^[1-9][0-9]*$ ]] || fail "Judge hard timeout 必须是正整数"

echo "[run_judge] judge=$PINNED_JUDGE_MODEL (gateway) protocol=$JUDGE_PROTOCOL admission=$JUDGE_ADMISSION max_images=$PINNED_MAX_IMAGES reasoning_effort=$MYPCBENCH_OSWORLD_JUDGE_REASONING_EFFORT request_timeout=${MYPCBENCH_OSWORLD_JUDGE_REQUEST_TIMEOUT}s process_timeout=${MYPCBENCH_OSWORLD_JUDGE_PROCESS_TIMEOUT}s task_timeout=${JUDGE_TIMEOUT}s hard_timeout=${HARD_JUDGE_TIMEOUT}s concurrency=$MYPCBENCH_OSWORLD_JUDGE_CONCURRENCY dirs=${#RESULT_DIRS[@]} force=${FORCE_FLAG:-no}"
for d in "${RESULT_DIRS[@]}"; do
  [[ -d "$d" ]] || fail "result_dir 不存在: $d"
  python3 "$BUNDLE_PY" "$d" || fail "bundle 证据补齐失败: $d"
  echo "===== JUDGE $d start $(date '+%F %T')"
  python3 "$HARD_TIMEOUT_PY" \
    --timeout "$HARD_JUDGE_TIMEOUT" --grace 30 \
    python3 "$JUDGE_PY" --result_dir "$d" --timeout "$JUDGE_TIMEOUT" \
    --fail-on-errors $FORCE_FLAG
  echo "===== JUDGE $d done  $(date '+%F %T')"
done
