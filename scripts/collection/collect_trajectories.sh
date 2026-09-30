#!/usr/bin/env bash

set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

AGENTS=(
  "evocua_32b"
)

if (( $# > 0 )); then
  AGENTS=("$@")
fi


COLLECT_CONFIG="${COLLECT_CONFIG:-${REPO_ROOT}/configs/collection/mypcbench_runtime.yaml}"
COLLECT_OVERRIDDEN=()
COLLECT_IGNORED=()
PORT_BASE="${PORT_BASE:-25000}"
NUM_VMS_OVERRIDE="${NUM_VMS_OVERRIDE:-}"
DRY_RUN="${DRY_RUN:-0}"
FORMAL_COLLECTION="${FORMAL_COLLECTION:-0}"
PYTHON_BIN="${PYTHON_BIN:-python3}"

RECOVERY_AUTO_SERVE="${RECOVERY_AUTO_SERVE:-1}"
SERVE_REPLICAS="${SERVE_REPLICAS:-}"
SERVE_READY_TIMEOUT="${SERVE_READY_TIMEOUT:-3600}"
SERVE_POLL_SECONDS="${SERVE_POLL_SECONDS:-10}"
SERVE_STARTUP_GRACE="${SERVE_STARTUP_GRACE:-180}"
SERVE_SCRIPT="${SCRIPT_DIR}/serve_open_source.sh"
SERVING_PORT_BASE="${SERVING_PORT_BASE:-8000}"
SERVING_LOG_ROOT="${SERVING_LOG_ROOT:-${REPO_ROOT}/artifacts/serving_logs}"

# shellcheck source=../lib/third_party.sh
source "${SCRIPT_DIR}/../lib/third_party.sh"
third_party_paths "$REPO_ROOT"

TASKS_FILE="${TASKS_FILE:-}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${REPO_ROOT}/artifacts/raw_rollouts/mypcbench}"
COLLECTION_ID="${COLLECTION_ID:-$(date -u +%Y%m%dT%H%M%SZ)}"
ENV_YAML="${ENV_YAML:-${REPO_ROOT}/env.yaml}"
ROOT_DOTENV="${ROOT_DOTENV:-${REPO_ROOT}/.env}"
MODELS_LOCK="${MODELS_LOCK:-${REPO_ROOT}/configs/models.lock.yaml}"
RECOVERY_TMUX="${RECOVERY_TMUX:-1}"
RECOVERY_OPENAI_API_APPROVED="${RECOVERY_OPENAI_API_APPROVED:-0}"
RECOVERY_OPENAI_API_PURPOSE="${RECOVERY_OPENAI_API_PURPOSE:-}"
RECOVERY_ANTHROPIC_API_APPROVED="${RECOVERY_ANTHROPIC_API_APPROVED:-0}"
RECOVERY_ANTHROPIC_API_PURPOSE="${RECOVERY_ANTHROPIC_API_PURPOSE:-}"

die() {
  printf '错误：%s\n' "$*" >&2
  exit 1
}

info() {
  printf '[RECOVERY collection] %s\n' "$*"
}

is_positive_integer() {
  [[ "$1" =~ ^[1-9][0-9]*$ ]]
}

# shellcheck source=../lib/collection_config.sh
source "${SCRIPT_DIR}/../lib/collection_config.sh"

resolve_agent_num_vms() {
  if [[ -n "$NUM_VMS_OVERRIDE" ]]; then
    printf '%s\n' "$NUM_VMS_OVERRIDE"
    return 0
  fi
  agent_vm_count "$REPO_ROOT" "$1"
}

resolve_agent_max_steps() {
  agent_max_steps "$REPO_ROOT" "$1" "$MAX_STEPS"
}

resolve_agent_task_timeout() {
  local value
  value="$(config_scalar "$(agent_config_path "$REPO_ROOT" "$1")" task_timeout || true)"
  [[ "$value" =~ ^[1-9][0-9]*$ ]] || value="$TASK_TIMEOUT"
  printf '%s\n' "$value"
}

start_in_tmux_if_needed() {
  [[ "$RECOVERY_TMUX" == "0" || -n "${TMUX:-}" ]] && return 0
  command -v tmux >/dev/null 2>&1 || die \
    "RECOVERY_TMUX=1 但找不到 tmux；安装 tmux 或显式设置 RECOVERY_TMUX=0"

  local session_name="${RECOVERY_TMUX_SESSION:-recovery-collect-${COLLECTION_ID}}"
  session_name="$(tr -c '[:alnum:]_-' '-' <<< "$session_name" | sed 's/-$//')"
  tmux has-session -t "=${session_name}" 2>/dev/null && die \
    "tmux session 已存在：${session_name}"

  local -a env_unset=()
  local -a env_assign=(RECOVERY_TMUX=0)
  local -a stale_from_tmux=()
  local tmux_global_names=""
  tmux_global_names="$(tmux show-environment -g 2>/dev/null | awk -F= 'NF > 1 { print $1 }' || true)"
  local variable
  for variable in \
    REPEATS REPEAT_START_INDEX NUM_VMS_OVERRIDE MAX_STEPS RECOVERY_BASH_ACCOUNTING BACKEND TIMEOUT_PER_VM TASK_TIMEOUT PORT_BASE DRY_RUN \
    CONTEXT_IMAGES TASK_SOURCE MYPCBENCH_QWEN_IMAGE_MAX \
    FORMAL_COLLECTION PYTHON_BIN MYPCBENCH_COMMIT MYPCBENCH_ROOT \
    EVOCUA_COMMIT EVOCUA_ROOT OPENCUA_OSWORLD_COMMIT OPENCUA_OSWORLD_ROOT \
    TASKS_FILE OUTPUT_ROOT COLLECTION_ID ROOT_DOTENV ENV_YAML MODELS_LOCK \
    MYPCBENCH_QCOW2 MYPCBENCH_OVMF_CODE MYPCBENCH_OVMF_VARS \
    MYPCBENCH_QEMU_BINARY \
    RECOVERY_AUTO_SERVE SERVE_REPLICAS SERVE_READY_TIMEOUT SERVE_POLL_SECONDS \
    SERVE_STARTUP_GRACE SERVING_PORT_BASE SERVING_LOG_ROOT SERVING_CPUSET HF_CACHE_ROOT \
    ALLOW_BUSY_GPU CUDA_VISIBLE_DEVICES RECOVERY_GPU_COUNT \
    ALLOW_NO_KVM MYPCBENCH_DIAG_DIR MYPCBENCH_QWEN_MAX_TOKENS \
    MYPCBENCH_QWEN_HISTORY_N MYPCBENCH_QWEN_CONTEXT_POLICY \
    OPENAI_BASE_URL QWEN35_BASE_URLS QWEN35_BASE_URL \
    RERAIL_CHECKPOINT RERAIL_BASE_URLS RERAIL_BASE_URL \
    EVOCUA_BASE_URLS EVOCUA_BASE_URL \
    OPENCUA_BASE_URLS OPENCUA_BASE_URL \
    ANTHROPIC_API_KEY ANTHROPIC_BASE_URL CLAUDE_PROMPT_CACHING_BETA \
    CLAUDE_OPUS_4_8_MODEL GPT55_MODEL \
    OPENAI_ZDR_STATELESS OPENAI_ZDR_KEEP_IMAGES \
    OPENAI_RATE_LIMIT_RETRIES ANTHROPIC_RATE_LIMIT_RETRIES \
    RECOVERY_ANTHROPIC_API_APPROVED RECOVERY_ANTHROPIC_API_PURPOSE \
    RECOVERY_OPENAI_API_APPROVED RECOVERY_OPENAI_API_PURPOSE \
    COLLECT_CONFIG ALLOW_CONFIG_OVERRIDE; do
    if [[ -n "${!variable+x}" ]]; then
      env_assign+=("${variable}=${!variable}")
    else
      env_unset+=(-u "$variable")
      if grep -qx -- "$variable" <<< "$tmux_global_names"; then
        stale_from_tmux+=("$variable")
      fi
    fi
  done
  if (( ${#stale_from_tmux[@]} > 0 )); then
    info "已丢弃 tmux server global environment 里的残留变量（本次命令未设置）：${stale_from_tmux[*]}"
    info "  它们来自这个 tmux server 上更早的 session；要彻底清掉：tmux set-environment -gu <名字>"
  fi
  local -a command=(env "${env_unset[@]}" "${env_assign[@]}" bash "$0" "$@")

  local command_text
  printf -v command_text '%q ' "${command[@]}"
  command_text+=$'; status=$?; printf "\\n[RECOVERY collection] exit=%s\\n" "$status"; exec bash'
  tmux new-session -d -s "$session_name" -n collection -c "$REPO_ROOT" "$command_text"
  tmux new-window -d -t "$session_name" -n monitor -c "$REPO_ROOT" \
    "watch -n 2 nvidia-smi"
  info "已在 tmux 启动；没有在当前 terminal 中后台裸跑"
  info "查看 collection: tmux attach -t ${session_name}"
  info "查看 GPU: tmux select-window -t ${session_name}:monitor"
  exit 0
}

sha256_file() {
  local path="$1"
  if command -v sha256sum >/dev/null 2>&1; then
    sha256sum "$path" | awk '{print $1}'
  elif command -v shasum >/dev/null 2>&1; then
    shasum -a 256 "$path" | awk '{print $1}'
  else
    die "找不到 sha256sum 或 shasum，无法记录 prompt/image hash"
  fi
}

load_yaml_env() {
  local yaml_path="$1"
  local parsed_env
  local env_line
  local env_key
  [[ -f "$yaml_path" ]] || return 0

  parsed_env="$(
    "$PYTHON_BIN" - "$yaml_path" <<'PY'
import json
import re
import sys

path = sys.argv[1]
key_pattern = re.compile(r"^[A-Z][A-Z0-9_]*$")

with open(path, encoding="utf-8") as handle:
    for line_number, raw_line in enumerate(handle, start=1):
        stripped = raw_line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if ":" not in raw_line:
            raise SystemExit(f"{path}:{line_number}: 应为 KEY: VALUE")
        key, raw_value = raw_line.split(":", 1)
        key = key.strip()
        value_text = raw_value.strip()
        if not key_pattern.fullmatch(key):
            raise SystemExit(f"{path}:{line_number}: 非法环境变量名 {key!r}")
        if not value_text or value_text in {"null", "~"}:
            continue
        if value_text.startswith('"'):
            try:
                value = json.loads(value_text)
            except json.JSONDecodeError as exc:
                raise SystemExit(f"{path}:{line_number}: 双引号字符串无效: {exc}")
        elif value_text.startswith("'"):
            if len(value_text) < 2 or not value_text.endswith("'"):
                raise SystemExit(f"{path}:{line_number}: 单引号字符串没有闭合")
            value = value_text[1:-1].replace("''", "'")
        else:
            value = value_text
        if not isinstance(value, str):
            raise SystemExit(f"{path}:{line_number}: VALUE 必须是字符串")
        if "\n" in value or "\x00" in value:
            raise SystemExit(f"{path}:{line_number}: VALUE 只能占一行")
        if value:
            print(f"{key}={value}")
PY
  )"

  while IFS= read -r env_line; do
    [[ -n "$env_line" ]] || continue
    env_key="${env_line%%=*}"
    export "$env_line"
    info "已从 env.yaml 加载 ${env_key}"
  done <<< "$parsed_env"
}

load_collection_config() {
  local config_path="$1"
  local parsed line key value
  [[ -f "$config_path" ]] || die "找不到采集配置：${config_path}（用 COLLECT_CONFIG= 覆盖）"

  parsed="$(
    "$PYTHON_BIN" - "$config_path" "$REPO_ROOT" <<'PY'
import re
import sys
from pathlib import Path

path, repo_root = Path(sys.argv[1]), Path(sys.argv[2])
VAR_MAP = {
    "repeats": "REPEATS",
    "max_steps": "MAX_STEPS",
    "backend": "BACKEND",
    "timeout_per_vm": "TIMEOUT_PER_VM",
    "task_timeout": "TASK_TIMEOUT",
    "context_images": "CONTEXT_IMAGES",
    "task_source": "TASK_SOURCE",
    "environment": None,
    "collection_id": None,
    "status": None,
}
value_pattern = re.compile(r"^[A-Za-z0-9_.\-]+$")


def parse_flat(p):
    out = {}
    for lineno, raw in enumerate(p.read_text(encoding="utf-8").splitlines(), start=1):
        stripped = raw.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if ":" not in stripped:
            raise SystemExit(f"{p}:{lineno}: 应为 key: value")
        k, text = stripped.split(":", 1)
        out[k.strip()] = text.split("#", 1)[0].strip().strip('"').strip("'")
    return out


cfg = parse_flat(path)
for key, text in cfg.items():
    if key not in VAR_MAP:
        raise SystemExit(f"{path}: 未知配置键 {key!r}（白名单见 load_collection_config）")
    if VAR_MAP[key] is None or not text:
        continue
    if not value_pattern.fullmatch(text):
        raise SystemExit(f"{path}: 键 {key} 的值 {text!r} 含非法字符")
    print(f"{VAR_MAP[key]}={text}")

env_id = cfg.get("environment", "")
if env_id:
    env_path = repo_root / "configs" / "environments" / f"{env_id}.yaml"
    if not env_path.is_file():
        raise SystemExit(f"{path}: environment={env_id} 对应的 {env_path} 不存在")
    env_cfg = parse_flat(env_path)
    for field, var in (("screen_width", "SCREEN_WIDTH"), ("screen_height", "SCREEN_HEIGHT")):
        text = env_cfg.get(field, "")
        if not text or not value_pattern.fullmatch(text):
            raise SystemExit(f"{env_path}: 缺少可用的 {field}")
        print(f"{var}={text}")
    print(f"ENVIRONMENT_CONFIG={env_path}")
PY
  )" || die "解析采集配置失败：${config_path}"

  while IFS= read -r line; do
    [[ -n "$line" ]] || continue
    key="${line%%=*}"
    value="${line#*=}"
    if [[ -n "${!key:-}" && "${!key}" != "$value" ]]; then
      if [[ "${ALLOW_CONFIG_OVERRIDE:-0}" == "1" ]]; then
        COLLECT_OVERRIDDEN+=("${key}=${!key}（config 为 ${value}）")
        continue
      fi
      COLLECT_IGNORED+=("${key}=${!key}")
    fi
    printf -v "$key" '%s' "$value"
  done <<< "$parsed"
}

load_dotenv() {
  local dotenv_path="$1"
  local parsed_env
  local env_line
  local env_key
  [[ -f "$dotenv_path" ]] || return 0

  parsed_env="$(
    "$PYTHON_BIN" - "$dotenv_path" <<'PY'
import ast
import re
import sys

path = sys.argv[1]
key_pattern = re.compile(r"^[A-Z][A-Z0-9_]*$")
with open(path, encoding="utf-8") as handle:
    for line_number, raw_line in enumerate(handle, start=1):
        stripped = raw_line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if stripped.startswith("export "):
            stripped = stripped[7:].lstrip()
        if "=" in stripped:
            key, value_text = stripped.split("=", 1)
        elif ":" in stripped:
            key, value_text = stripped.split(":", 1)
        else:
            raise SystemExit(f"{path}:{line_number}: 应为 KEY=VALUE 或 KEY: VALUE")
        key = key.strip()
        value_text = value_text.strip()
        if not key_pattern.fullmatch(key):
            raise SystemExit(f"{path}:{line_number}: 非法环境变量名 {key!r}")
        if value_text.startswith(("\"", "'")):
            try:
                value = ast.literal_eval(value_text)
            except (SyntaxError, ValueError) as exc:
                raise SystemExit(f"{path}:{line_number}: quoted value 无效: {exc}")
        else:
            value = value_text
        if not isinstance(value, str) or "\n" in value or "\x00" in value:
            raise SystemExit(f"{path}:{line_number}: VALUE 必须是单行字符串")
        if value:
            print(f"{key}={value}")
PY
  )"

  while IFS= read -r env_line; do
    [[ -n "$env_line" ]] || continue
    env_key="${env_line%%=*}"
    export "$env_line"
    info "已从 $(basename "$dotenv_path") 加载 ${env_key}"
  done <<< "$parsed_env"
}

resolve_agent() {
  local agent_id="$1"
  RESOLVED_AGENT_TYPE=""
  RESOLVED_MODEL=""
  RESOLVED_REQUIRED_ENV=""
  RESOLVED_BASE_URLS_ENV=""
  RESOLVED_BASE_URL_ENV=""
  case "$agent_id" in
    gpt_5_5)
      RESOLVED_AGENT_TYPE="openai_cuabash"
      RESOLVED_MODEL="${GPT55_MODEL:-gpt-5.5}"
      RESOLVED_REQUIRED_ENV="OPENAI_API_KEY"
      ;;
    kimi_k3)
      RESOLVED_AGENT_TYPE="recovery_kimi_k3"
      RESOLVED_MODEL="${KIMI_K3_MODEL:-kimi-k3}"
      RESOLVED_REQUIRED_ENV="OPENAI_API_KEY"
      ;;
    kimi_k3_cuabash)
      RESOLVED_AGENT_TYPE="recovery_kimi_k3_cuabash"
      RESOLVED_MODEL="${KIMI_K3_MODEL:-kimi-k3}"
      RESOLVED_REQUIRED_ENV="OPENAI_API_KEY"
      ;;
    qwen3_5_35b_a3b)
      RESOLVED_AGENT_TYPE="qwen_cuabash"
      RESOLVED_MODEL="${QWEN35_MODEL:-Qwen/Qwen3.5-35B-A3B}"
      RESOLVED_REQUIRED_ENV="LOCAL_ENDPOINT"
      RESOLVED_BASE_URLS_ENV="QWEN35_BASE_URLS"
      RESOLVED_BASE_URL_ENV="QWEN35_BASE_URL"
      ;;
    rerail_35b_a3b)
      [[ -n "${RERAIL_CHECKPOINT:-}" ]] || die "rerail_35b_a3b 需要 RERAIL_CHECKPOINT"
      RESOLVED_AGENT_TYPE="qwen_cuabash"
      RESOLVED_MODEL="${RERAIL_CHECKPOINT}"
      RESOLVED_REQUIRED_ENV="LOCAL_ENDPOINT"
      RESOLVED_BASE_URLS_ENV="RERAIL_BASE_URLS"
      RESOLVED_BASE_URL_ENV="RERAIL_BASE_URL"
      ;;
    claude_opus_4_8)
      [[ -n "${CLAUDE_OPUS_4_8_MODEL:-}" ]] || die \
        "claude_opus_4_8 尚未冻结真实 API model ID；请先设置 CLAUDE_OPUS_4_8_MODEL"
      RESOLVED_AGENT_TYPE="claude_cuabash"
      RESOLVED_MODEL="${CLAUDE_OPUS_4_8_MODEL}"
      RESOLVED_REQUIRED_ENV="ANTHROPIC_API_KEY"
      ;;
    evocua_32b)
      RESOLVED_AGENT_TYPE="recovery_evocua"
      RESOLVED_MODEL="${EVOCUA_MODEL:-EvoCUA}"
      RESOLVED_REQUIRED_ENV="LOCAL_ENDPOINT"
      RESOLVED_BASE_URLS_ENV="EVOCUA_BASE_URLS"
      RESOLVED_BASE_URL_ENV="EVOCUA_BASE_URL"
      ;;
    opencua_72b)
      RESOLVED_AGENT_TYPE="recovery_opencua"
      RESOLVED_MODEL="${OPENCUA_MODEL:-opencua-72b}"
      RESOLVED_REQUIRED_ENV="LOCAL_ENDPOINT"
      RESOLVED_BASE_URLS_ENV="OPENCUA_BASE_URLS"
      RESOLVED_BASE_URL_ENV="OPENCUA_BASE_URL"
      ;;
    dummy)
      RESOLVED_AGENT_TYPE="dummy"
      RESOLVED_MODEL="dummy"
      RESOLVED_REQUIRED_ENV=""
      RESOLVED_BASE_URLS_ENV=""
      RESOLVED_BASE_URL_ENV=""
      ;;
    *)
      die "未知 agent_id：${agent_id}；请使用 configs/agents/ 中定义的名称"
      ;;
  esac
}

default_serve_replicas() {
  resolve_agent_num_vms "$1"
}

serving_container_count() {
  local names
  names="$(docker ps --filter label=recovery.project=RECOVERY \
    --filter "label=recovery.agent_id=$1" --format '{{.Names}}' 2>/dev/null || true)"
  if [[ -z "$names" ]]; then
    printf '0\n'
  else
    printf '%s\n' "$names" | grep -c .
  fi
}

serving_base_urls() {
  local replicas="$1"
  local urls=""
  local index
  for ((index = 0; index < replicas; index++)); do
    [[ -z "$urls" ]] || urls+=","
    urls+="http://127.0.0.1:$((SERVING_PORT_BASE + index))/v1"
  done
  printf '%s\n' "$urls"
}

assert_endpoint_is_ours() {
  local agent_id="$1" url="$2"
  [[ -n "$AUTO_SERVED_AGENT" ]] || return 0
  local port="${url##*:}"
  port="${port%%/*}"
  [[ "$port" =~ ^[0-9]+$ ]] || return 0
  local published
  published="$(docker ps --filter label=recovery.project=RECOVERY \
    --filter "label=recovery.agent_id=${agent_id}" --format '{{.Ports}}' 2>/dev/null || true)"
  grep -q "127\.0\.0\.1:${port}->" <<< "$published" || die \
    "端口 ${port} 上有服务在应答，但它不是 ${agent_id} 的 container（本次发布的端口：${published//$'\n'/ }）；换一段端口重跑：SERVING_PORT_BASE=8100"
}

wait_for_endpoints() {
  local agent_id="$1"
  local urls="$2"
  local deadline=$((SECONDS + SERVE_READY_TIMEOUT))
  local startup_grace=$((SECONDS + SERVE_STARTUP_GRACE))
  local seen_container=0
  local container_count
  local -a url_list=()
  IFS=',' read -r -a url_list <<< "$urls"
  local url
  local http_code
  for url in "${url_list[@]}"; do
    info "等待 endpoint ready：${url}（最多 ${SERVE_READY_TIMEOUT}s）"
    while true; do
      http_code="$(curl -s -o /dev/null -w '%{http_code}' --max-time 5 \
        "${url}/models" || true)"
      if [[ "$http_code" == "200" ]]; then
        assert_endpoint_is_ours "$agent_id" "$url"
        info "endpoint ready：${url}"
        break
      fi
      container_count="$(serving_container_count "$agent_id")"
      (( container_count > 0 )) && seen_container=1
      if (( seen_container == 1 && container_count == 0 )); then
        die "${agent_id} 的 serving container 已退出；日志见 ${SERVING_LOG_ROOT} 下最新的 endpoint_*.log"
      fi
      if (( seen_container == 0 && SECONDS > startup_grace )); then
        die "${agent_id} 的 serving container 在 ${SERVE_STARTUP_GRACE}s 内没有出现；检查 tmux session recovery-serve-* 和 ${SERVING_LOG_ROOT}"
      fi
      (( SECONDS < deadline )) || die \
        "${agent_id} 的 endpoint 在 ${SERVE_READY_TIMEOUT}s 内没有 ready：${url}"
      sleep "$SERVE_POLL_SECONDS"
    done
  done
}

AUTO_SERVED_AGENT=""

auto_serve_stop() {
  [[ -n "$AUTO_SERVED_AGENT" ]] || return 0
  local agent_id="$AUTO_SERVED_AGENT"
  AUTO_SERVED_AGENT=""
  info "停止自动启动的 endpoint：${agent_id}"
  bash "$SERVE_SCRIPT" stop "$agent_id" || \
    printf '[RECOVERY collection] 警告：停止 %s 失败，请手动检查 docker ps\n' "$agent_id" >&2
  sleep "${SERVE_SETTLE_SECONDS:-15}"
}

ENDPOINT_WATCH_INTERVAL="${ENDPOINT_WATCH_INTERVAL:-60}"
ENDPOINT_WATCH_FAILURES="${ENDPOINT_WATCH_FAILURES:-3}"
ENDPOINT_WATCHDOG_PID=""
ENDPOINT_WATCHDOG_TRIP=""

endpoint_watchdog_start() {
  local urls="$1"
  local runner_pattern="$2"
  ENDPOINT_WATCHDOG_PID=""
  ENDPOINT_WATCHDOG_TRIP=""
  [[ -n "$urls" ]] || return 0
  (( ENDPOINT_WATCH_INTERVAL > 0 )) || return 0

  ENDPOINT_WATCHDOG_TRIP="$(mktemp -t recovery-endpoint-trip.XXXXXX)"
  rm -f "$ENDPOINT_WATCHDOG_TRIP"
  local trip_file="$ENDPOINT_WATCHDOG_TRIP"
  local interval="$ENDPOINT_WATCH_INTERVAL"
  local max_failures="$ENDPOINT_WATCH_FAILURES"

  (
    declare -a url_list=()
    IFS=',' read -r -a url_list <<< "$urls"
    declare -A failures=()
    declare url http_code
    for url in "${url_list[@]}"; do failures["$url"]=0; done
    while true; do
      sleep "$interval"
      for url in "${url_list[@]}"; do
        http_code="$(curl -s -o /dev/null -w '%{http_code}' --max-time 10 \
          "${url}/models" || true)"
        if [[ "$http_code" == "200" ]]; then
          failures["$url"]=0
          continue
        fi
        failures["$url"]=$(( failures["$url"] + 1 ))
        if (( failures["$url"] >= max_failures )); then
          printf '%s\n' "$url" > "$trip_file"
          printf '[RECOVERY collection] endpoint %s 连续 %d 次探测失败，正在中止 runner\n' \
            "$url" "${failures["$url"]}" >&2
          pkill -TERM -f "run_parallel_tasks.py.*${runner_pattern}" || true
          exit 0
        fi
      done
    done
  ) &
  ENDPOINT_WATCHDOG_PID="$!"
}

endpoint_watchdog_stop() {
  [[ -n "$ENDPOINT_WATCHDOG_PID" ]] || return 0
  local pid="$ENDPOINT_WATCHDOG_PID"
  ENDPOINT_WATCHDOG_PID=""
  kill "$pid" 2>/dev/null || true
  wait "$pid" 2>/dev/null || true
}

endpoint_watchdog_tripped() {
  [[ -n "$ENDPOINT_WATCHDOG_TRIP" && -s "$ENDPOINT_WATCHDOG_TRIP" ]]
}

collection_cleanup() {
  endpoint_watchdog_stop
  if [[ -n "$ENDPOINT_WATCHDOG_TRIP" ]]; then
    rm -f "$ENDPOINT_WATCHDOG_TRIP"
  fi
  auto_serve_stop
}

trap collection_cleanup EXIT INT TERM

auto_serve_start() {
  local agent_id="$1"
  local urls_variable="$2"
  local replicas
  local running

  running="$(serving_container_count "$agent_id")"
  if (( running > 0 )); then
    replicas="$running"
    info "${agent_id} 已有 ${replicas} 个 serving container 在跑；复用，结束时不会停掉它"
  else
    replicas="${SERVE_REPLICAS:-$(default_serve_replicas "$agent_id")}"
    local ceiling
    ceiling="$(default_serve_replicas "$agent_id")"
    if (( replicas > ceiling )); then
      info "${agent_id} 在本机的 endpoint 上限是 ${ceiling}（按可用 GPU 推导）；忽略 SERVE_REPLICAS=${replicas}"
      replicas="$ceiling"
    fi
    info "自动启动 ${agent_id} 的 vLLM endpoint（${replicas} 个 replica）"
    bash "$SERVE_SCRIPT" start "$agent_id" "$replicas"
    AUTO_SERVED_AGENT="$agent_id"
  fi

  local urls
  urls="$(serving_base_urls "$replicas")"
  if [[ -n "$urls_variable" ]]; then
    if [[ -n "${!urls_variable:-}" && "${!urls_variable}" != "$urls" ]]; then
      info "覆盖 ${urls_variable}：${!urls_variable} → ${urls}（RECOVERY_AUTO_SERVE=0 可保留 .env 的值）"
    fi
    export "${urls_variable}=${urls}"
  fi
  wait_for_endpoints "$agent_id" "$urls"
}

start_in_tmux_if_needed "$@"

load_collection_config "$COLLECT_CONFIG"

is_positive_integer "$REPEATS" || die "REPEATS 必须是正整数"
is_positive_integer "$MAX_STEPS" || die "MAX_STEPS 必须是正整数"
is_positive_integer "$TIMEOUT_PER_VM" || die "TIMEOUT_PER_VM 必须是正整数"
is_positive_integer "$TASK_TIMEOUT" || die "TASK_TIMEOUT 必须是正整数"
is_positive_integer "$CONTEXT_IMAGES" || die "CONTEXT_IMAGES 必须是正整数"
(( TASK_TIMEOUT < TIMEOUT_PER_VM )) || die "TASK_TIMEOUT 必须小于 TIMEOUT_PER_VM，否则单任务上限形同虚设"
(( ${#AGENTS[@]} > 0 )) || die "AGENTS 不能为空"
[[ "$DRY_RUN" == "0" || "$DRY_RUN" == "1" ]] || die "DRY_RUN 只能是 0 或 1"
[[ "$FORMAL_COLLECTION" == "0" || "$FORMAL_COLLECTION" == "1" ]] || \
  die "FORMAL_COLLECTION 只能是 0 或 1"
[[ "$RECOVERY_TMUX" == "0" || "$RECOVERY_TMUX" == "1" ]] || \
  die "RECOVERY_TMUX 只能是 0 或 1"
[[ "$RECOVERY_OPENAI_API_APPROVED" == "0" || "$RECOVERY_OPENAI_API_APPROVED" == "1" ]] || \
  die "RECOVERY_OPENAI_API_APPROVED 只能是 0 或 1"
[[ "$BACKEND" == "qemu" || "$BACKEND" == "docker" ]] || \
  die "BACKEND 只能是 qemu 或 docker"
[[ "$RECOVERY_AUTO_SERVE" == "0" || "$RECOVERY_AUTO_SERVE" == "1" ]] || \
  die "RECOVERY_AUTO_SERVE 只能是 0 或 1"
is_positive_integer "$SERVE_READY_TIMEOUT" || die "SERVE_READY_TIMEOUT 必须是正整数"
is_positive_integer "$SERVE_POLL_SECONDS" || die "SERVE_POLL_SECONDS 必须是正整数"
is_positive_integer "$SERVE_STARTUP_GRACE" || die "SERVE_STARTUP_GRACE 必须是正整数"
if [[ -n "$SERVE_REPLICAS" ]]; then
  is_positive_integer "$SERVE_REPLICAS" || die "SERVE_REPLICAS 必须是正整数"
fi
if [[ -n "$NUM_VMS_OVERRIDE" ]]; then
  is_positive_integer "$NUM_VMS_OVERRIDE" || die "NUM_VMS_OVERRIDE 必须是正整数"
fi
if [[ "$RECOVERY_AUTO_SERVE" == "1" && "$DRY_RUN" == "0" ]]; then
  [[ -x "$SERVE_SCRIPT" || -f "$SERVE_SCRIPT" ]] || die \
    "RECOVERY_AUTO_SERVE=1 但找不到 serving 脚本：${SERVE_SCRIPT}"
  command -v curl >/dev/null 2>&1 || die \
    "RECOVERY_AUTO_SERVE=1 需要 curl 探测 endpoint ready；请安装 curl 或设置 RECOVERY_AUTO_SERVE=0"
  command -v docker >/dev/null 2>&1 || die \
    "RECOVERY_AUTO_SERVE=1 需要 docker；请安装 docker 或设置 RECOVERY_AUTO_SERVE=0"
fi
command -v "$PYTHON_BIN" >/dev/null 2>&1 || die "找不到 Python：${PYTHON_BIN}"
command -v git >/dev/null 2>&1 || die "找不到 git"
[[ -f "$MODELS_LOCK" ]] || die "找不到模型锁文件：${MODELS_LOCK}"
export PYTHONPATH="${REPO_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"

for image_var in OPENAI_ZDR_KEEP_IMAGES MYPCBENCH_QWEN_IMAGE_MAX; do
  if [[ -n "${!image_var:-}" && "${!image_var}" != "$CONTEXT_IMAGES" ]]; then
    if [[ "${ALLOW_CONFIG_OVERRIDE:-0}" == "1" ]]; then
      COLLECT_OVERRIDDEN+=("${image_var}=${!image_var}（config context_images 为 ${CONTEXT_IMAGES}）")
      continue
    fi
    COLLECT_IGNORED+=("${image_var}=${!image_var}")
  fi
  export "${image_var}=${CONTEXT_IMAGES}"
done

export RECOVERY_ENVIRONMENT_CONFIG="$ENVIRONMENT_CONFIG"
export PYTHONPATH="${REPO_ROOT}/src/recovery/rollout/site_hook:${PYTHONPATH}"

if git -C "$REPO_ROOT" rev-parse --is-inside-work-tree >/dev/null 2>&1; then
  for secret_file in .env env.yaml; do
    if git -C "$REPO_ROOT" ls-files --error-unmatch "$secret_file" >/dev/null 2>&1; then
      die "${secret_file} 已进入 Git index；请先将它从 index 移除，禁止上传敏感信息"
    fi
    git -C "$REPO_ROOT" check-ignore --quiet "$secret_file" || \
      die "${secret_file} 没有被 .gitignore 保护"
  done
fi

setup_mypcbench
actual_commit="$(git -C "$MYPCBENCH_ROOT" rev-parse HEAD)"

RUNNER="${MYPCBENCH_ROOT}/agent-harness/run_parallel_tasks.py"
PROMPT_SOURCE="${MYPCBENCH_ROOT}/agent-harness/agents/prompts.py"
[[ -f "$RUNNER" ]] || die "找不到官方 runner：${RUNNER}"
[[ -f "$PROMPT_SOURCE" ]] || die "找不到官方 prompt source：${PROMPT_SOURCE}"

SOURCE_TASKS_FILE="${TASKS_FILE:-$("$PYTHON_BIN" -m recovery.rollout.tasks --source "$TASK_SOURCE")}" \
  || die "解析 task_source=${TASK_SOURCE} 失败"
[[ -f "$SOURCE_TASKS_FILE" ]] || die "找不到 ${TASK_SOURCE} 的任务文件：${SOURCE_TASKS_FILE}"
if [[ "$TASK_SOURCE" == "mypcbench" ]]; then
  TASKS_FILE="$SOURCE_TASKS_FILE"
else
  if [[ "$DRY_RUN" == "1" ]]; then
    converted_tasks="$(mktemp -d)/${TASK_SOURCE}.json"
  else
    converted_tasks="${OUTPUT_ROOT}/${COLLECTION_ID}/_task_source/${TASK_SOURCE}.json"
  fi
  task_source_summary="$("$PYTHON_BIN" -m recovery.rollout.tasks --source "$TASK_SOURCE" \
    --tasks-file "$SOURCE_TASKS_FILE" --out "$converted_tasks")" \
    || die "转换 ${TASK_SOURCE} 任务失败"
  info "task_source=${TASK_SOURCE}：${task_source_summary}"
  TASKS_FILE="$converted_tasks"
fi
[[ -f "$TASKS_FILE" ]] || die "找不到 task 文件：${TASKS_FILE}"

full_tasks_path="$($PYTHON_BIN - "${MYPCBENCH_ROOT}/tasks/final/all_tasks_with_grading.json" <<'PY'
from pathlib import Path
import sys
print(Path(sys.argv[1]).resolve())
PY
)"
selected_tasks_path="$($PYTHON_BIN - "$SOURCE_TASKS_FILE" <<'PY'
from pathlib import Path
import sys
print(Path(sys.argv[1]).resolve())
PY
)"
IS_FULL_TASK_SET=0
if [[ "$selected_tasks_path" == "$full_tasks_path" ]]; then
  IS_FULL_TASK_SET=1
fi
LOCK_FORMAL_AUTHORIZED="$($PYTHON_BIN - "$MODELS_LOCK" <<'PY'
from pathlib import Path
import sys
from recovery.mypcbench.launch_contract import formal_collection_authorized
print("1" if formal_collection_authorized(Path(sys.argv[1]).read_text(encoding="utf-8")) else "0")
PY
)"
export FORMAL_COLLECTION LOCK_FORMAL_AUTHORIZED IS_FULL_TASK_SET
export TASK_TIMEOUT TIMEOUT_PER_VM TASKS_FILE CONTEXT_IMAGES TASK_SOURCE SOURCE_TASKS_FILE
if [[ "$DRY_RUN" == "0" && "$IS_FULL_TASK_SET" == "1" ]]; then
  [[ "$FORMAL_COLLECTION" == "1" ]] || die \
    "检测到完整正式 task 文件；必须在得到用户确认后显式设置 FORMAL_COLLECTION=1"
  [[ "$LOCK_FORMAL_AUTHORIZED" == "1" ]] || die \
    "models.lock.yaml 仍为 formal_collection_authorized: false；禁止启动 184×3"
fi

if [[ -f "${MYPCBENCH_ROOT}/.env" ]]; then
  load_dotenv "${MYPCBENCH_ROOT}/.env"
fi

load_yaml_env "$ENV_YAML"
load_dotenv "$ROOT_DOTENV"

if [[ "$DRY_RUN" != "1" && -n "${OPENAI_API_KEY:-}" ]]; then
  [[ "$RECOVERY_OPENAI_API_APPROVED" == "1" ]] || die \
    "检测到 OPENAI_API_KEY，但本次运行未获 API 授权；请先说明用途并在获批后仅对本次命令设置 RECOVERY_OPENAI_API_APPROVED=1"
  case "$RECOVERY_OPENAI_API_PURPOSE" in
    mypcbench_npc_replies)
      info "本次已获批的 OpenAI API 用途：MyPCBench NPC replies；不会用于 judge"
      ;;
    mypcbench_collection_agent)
      info "本次已获批的 OpenAI API 用途：MyPCBench collection agent 推理；不会用于 judge"
      ;;
    *)
      die "collection 的 API 用途必须显式为 RECOVERY_OPENAI_API_PURPOSE=mypcbench_npc_replies 或 mypcbench_collection_agent"
      ;;
  esac
fi
if [[ "$DRY_RUN" != "1" && "$IS_FULL_TASK_SET" == "1" ]]; then
  [[ -n "${OPENAI_API_KEY:-}" ]] || die \
    "完整 MyPCBench collection 需要 .env 中的 OPENAI_API_KEY 供 NPC replies 使用"
fi

if [[ "$DRY_RUN" != "1" && -n "${ANTHROPIC_API_KEY:-}" ]]; then
  [[ "$RECOVERY_ANTHROPIC_API_APPROVED" == "1" ]] || die \
    "检测到 ANTHROPIC_API_KEY，但本次运行未获 API 授权；请先说明用途并在获批后仅对本次命令设置 RECOVERY_ANTHROPIC_API_APPROVED=1"
  case "$RECOVERY_ANTHROPIC_API_PURPOSE" in
    mypcbench_collection_agent)
      info "本次已获批的 Anthropic API 用途：MyPCBench collection agent 推理；不会用于 judge"
      ;;
    *)
      die "collection 的 Anthropic API 用途必须显式为 RECOVERY_ANTHROPIC_API_PURPOSE=mypcbench_collection_agent"
      ;;
  esac
fi
GENERIC_OPENAI_BASE_URL="${OPENAI_BASE_URL:-}"

for agent_id in "${AGENTS[@]}"; do
  resolve_agent "$agent_id"
  case "$agent_id" in
    evocua_32b) setup_evocua ;;
    opencua_72b) setup_opencua_osworld ;;
  esac
done

export RECOVERY_REPO_ROOT="$REPO_ROOT"
export RECOVERY_EVOCUA_ROOT="$EVOCUA_ROOT"
export RECOVERY_OPENCUA_OSWORLD_ROOT="$OPENCUA_OSWORLD_ROOT"
export RECOVERY_AGENT_MAX_STEPS="$MAX_STEPS"

QWEN35_ENDPOINT_CONTRACTS_JSON="[]"
if [[ "$DRY_RUN" == "0" ]]; then
  for agent_id in "${AGENTS[@]}"; do
    [[ "$(agent_family "$REPO_ROOT" "$agent_id" || true)" == qwen35 ]] || continue
    [[ -n "${MYPCBENCH_QWEN_MAX_TOKENS:-}" ]] || die \
      "Qwen3.5 真实运行必须显式设置 MYPCBENCH_QWEN_MAX_TOKENS；probe 的 4096 不会自动转为正式配置"
    [[ -n "${MYPCBENCH_QWEN_HISTORY_N:-}" ]] || die \
      "Qwen3.5 真实运行必须显式设置 MYPCBENCH_QWEN_HISTORY_N；probe 的历史策略不会自动转为正式配置"
    [[ "${MYPCBENCH_QWEN_CONTEXT_POLICY:-}" == "tokenize_oldest_first_v1" ]] || die \
      "Qwen3.5 真实运行必须显式设置 MYPCBENCH_QWEN_CONTEXT_POLICY=tokenize_oldest_first_v1"
  done
fi
export QWEN35_ENDPOINT_CONTRACTS_JSON

qwen35_endpoint_preflight() {
  local endpoint_urls="$1"
  if ! QWEN35_ENDPOINT_CONTRACTS_JSON="$($PYTHON_BIN - \
    "$endpoint_urls" "$RESOLVED_MODEL" "$MYPCBENCH_QWEN_MAX_TOKENS" \
    "$MYPCBENCH_QWEN_HISTORY_N" <<'PY'
import json
import os
import sys
from recovery.mypcbench.launch_contract import LaunchContractError, fetch_vllm_contract

urls = [url.strip() for url in sys.argv[1].split(",") if url.strip()]
if not urls:
    raise SystemExit("Qwen3.5 endpoint list is empty")
try:
    contracts = [
        fetch_vllm_contract(
            url,
            sys.argv[2],
            sys.argv[3],
            sys.argv[4],
            api_key=os.environ.get("QWEN35_API_KEY"),
        )
        for url in urls
    ]
except LaunchContractError as exc:
    raise SystemExit(f"Qwen3.5 endpoint contract failed: {exc}") from exc
print(json.dumps(contracts, separators=(",", ":")))
PY
  )"; then
    die "Qwen3.5 endpoint/context preflight 未通过"
  fi
  export QWEN35_ENDPOINT_CONTRACTS_JSON
  info "Qwen3.5 endpoint/context preflight 通过：${QWEN35_ENDPOINT_CONTRACTS_JSON}"
}

record_qwen35_contracts() {
  local manifest_path="$1"
  [[ -f "$manifest_path" ]] || return 0
  "$PYTHON_BIN" - "$manifest_path" <<'PY'
import json
import os
import pathlib
import sys

path = pathlib.Path(sys.argv[1])
manifest = json.loads(path.read_text(encoding="utf-8"))
manifest.setdefault("launch_contract", {})["qwen35_endpoint_contracts"] = json.loads(
    os.environ.get("QWEN35_ENDPOINT_CONTRACTS_JSON", "[]")
)
path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
PY
}

if [[ "$DRY_RUN" != "1" ]]; then
  if [[ "$BACKEND" == "qemu" ]]; then
    [[ "$(uname -s)" == "Linux" ]] || die "QEMU 正式运行需要 Linux host"
    if [[ ! -e /dev/kvm && "${ALLOW_NO_KVM:-0}" != "1" ]]; then
      die "没有 /dev/kvm；如确实接受纯软件模拟，请显式设置 ALLOW_NO_KVM=1"
    fi
  else
    command -v docker >/dev/null 2>&1 || die "BACKEND=docker 但找不到 docker"
  fi
fi

prompt_sha256="$(sha256_file "$PROMPT_SOURCE")"
info "MyPCBench commit: ${actual_commit}"
info "prompt SHA-256: ${prompt_sha256}"
info "tasks: ${TASKS_FILE}"
agent_vms_summary=""
agent_steps_summary=""
agent_timeout_summary=""
for banner_agent in "${AGENTS[@]}"; do
  [[ -z "$agent_vms_summary" ]] || agent_vms_summary+=" "
  agent_vms_summary+="${banner_agent}=$(resolve_agent_num_vms "$banner_agent")"
  [[ -z "$agent_steps_summary" ]] || agent_steps_summary+=" "
  agent_steps_summary+="${banner_agent}=$(resolve_agent_max_steps "$banner_agent")"
  [[ -z "$agent_timeout_summary" ]] || agent_timeout_summary+=" "
  agent_timeout_summary+="${banner_agent}=$(resolve_agent_task_timeout "$banner_agent")"
done
info "agents: ${AGENTS[*]}；repeats: ${REPEATS}；VMs: ${agent_vms_summary}"
info "采集配置: $(realpath --relative-to="$REPO_ROOT" "$COLLECT_CONFIG")"
info "  max_steps/agent: ${agent_steps_summary}（采集 config 默认 ${MAX_STEPS}）"
info "  task_timeout/agent: ${agent_timeout_summary}（采集 config 默认 ${TASK_TIMEOUT}s）"
info "  timeout_per_vm=${TIMEOUT_PER_VM}s backend=${BACKEND} screen=${SCREEN_WIDTH}x${SCREEN_HEIGHT} context_images=${CONTEXT_IMAGES}"
info "  task_source=${TASK_SOURCE} environment_hooks=$(realpath --relative-to="$REPO_ROOT" "$RECOVERY_ENVIRONMENT_CONFIG")"
if (( ${#COLLECT_OVERRIDDEN[@]} > 0 )); then
  info "  ★ ALLOW_CONFIG_OVERRIDE=1，以下用环境变量而非 config：${COLLECT_OVERRIDDEN[*]}"
fi
if (( ${#COLLECT_IGNORED[@]} > 0 )); then
  info "  已忽略与 config 冲突的环境变量：${COLLECT_IGNORED[*]}（要生效请设 ALLOW_CONFIG_OVERRIDE=1）"
fi

QCOW2_ARGS=()
image_sha256="not-applicable"
if [[ "$BACKEND" == "qemu" ]]; then
  qcow2_path="${MYPCBENCH_QCOW2:-${MYPCBENCH_ROOT}/mypcbench-vm/mypcbench.qcow2}"
  if [[ "$DRY_RUN" != "1" && ! -f "$qcow2_path" ]]; then
    info "首次下载 MyPCBench QEMU image；文件较大，请耐心等待"
    bash "${MYPCBENCH_ROOT}/scripts/get-eval-image.sh" \
      --out "${MYPCBENCH_ROOT}/mypcbench-vm"
  fi
  if [[ "$DRY_RUN" != "1" ]]; then
    [[ -f "$qcow2_path" ]] || die "找不到 QEMU image：${qcow2_path}"
    qcow2_path="$(cd "$(dirname "$qcow2_path")" && pwd)/$(basename "$qcow2_path")"
    image_sha256="$(sha256_file "$qcow2_path")"
  fi
  QCOW2_ARGS=(--qcow2-path "$qcow2_path")
fi

run_root="${OUTPUT_ROOT}/${COLLECTION_ID}"
if [[ "$DRY_RUN" != "1" ]]; then
  mkdir -p "$run_root"
  NUM_VMS_PER_AGENT_JSON="$(
    for manifest_agent in "${AGENTS[@]}"; do
      printf '%s\t%s\n' "$manifest_agent" "$(resolve_agent_num_vms "$manifest_agent")"
    done | "$PYTHON_BIN" -c '
import json
import sys

print(json.dumps({
    agent: int(vms)
    for agent, vms in (line.rstrip("\n").split("\t") for line in sys.stdin if line.strip())
}))
'
  )"
  TASK_TIMEOUT_PER_AGENT_JSON="$(
    for manifest_agent in "${AGENTS[@]}"; do
      printf '%s\t%s\n' "$manifest_agent" "$(resolve_agent_task_timeout "$manifest_agent")"
    done | "$PYTHON_BIN" -c 'import json, sys; print(json.dumps({a: int(v) for a, v in (l.rstrip("\n").split("\t") for l in sys.stdin if l.strip())}))'
  )"
  MAX_STEPS_PER_AGENT_JSON="$(
    for manifest_agent in "${AGENTS[@]}"; do
      printf '%s\t%s\n' "$manifest_agent" "$(resolve_agent_max_steps "$manifest_agent")"
    done | "$PYTHON_BIN" -c '
import json
import sys

print(json.dumps({
    agent: int(steps)
    for agent, steps in (line.rstrip("\n").split("\t") for line in sys.stdin if line.strip())
}))
'
  )"
  export NUM_VMS_PER_AGENT_JSON MAX_STEPS_PER_AGENT_JSON TASK_TIMEOUT_PER_AGENT_JSON
  "$PYTHON_BIN" - "$run_root/collection_manifest.json" \
    "$actual_commit" "$prompt_sha256" "$image_sha256" "$REPEATS" \
    "$MAX_STEPS" "$REPO_ROOT" "$MYPCBENCH_PATCH" "$EVOCUA_ROOT" \
    "$OPENCUA_OSWORLD_ROOT" "$EVOCUA_COMMIT" "$OPENCUA_OSWORLD_COMMIT" \
    "${AGENTS[@]}" <<'PY'
import hashlib
import importlib.metadata
import json
import os
import pathlib
import platform
import sys

path = pathlib.Path(sys.argv[1])
repo_root = pathlib.Path(sys.argv[7])
mypcbench_patch = pathlib.Path(sys.argv[8])
evocua_root = pathlib.Path(sys.argv[9])
opencua_root = pathlib.Path(sys.argv[10])

source_files = [
    mypcbench_patch,
    repo_root / "third_party/MyPCBench/agent-harness/agents/qwen_cua.py",
    repo_root / "third_party/MyPCBench/agent-harness/agents/vendored_paper_results/qwen35vl_agent.py",
    *sorted((repo_root / "src/recovery/mypcbench").glob("*.py")),
    *sorted((repo_root / "prompts/agents").glob("*.txt")),
    *sorted((repo_root / "configs/agents").glob("*.yaml")),
    *sorted((repo_root / "src/recovery/rollout").rglob("*.py")),
    pathlib.Path(os.environ["RECOVERY_ENVIRONMENT_CONFIG"]),
    repo_root / "infra/snapshot/changelog_replay.py",
    repo_root / "infra/volatile_columns.json",
    repo_root / "configs/collection/sources.yaml",
]
for external_file in (
    evocua_root / "mm_agents/evocua/evocua_agent.py",
    evocua_root / "mm_agents/evocua/prompts.py",
    evocua_root / "mm_agents/evocua/utils.py",
    opencua_root / "mm_agents/opencua/opencua_agent.py",
    opencua_root / "mm_agents/opencua/prompts.py",
    opencua_root / "mm_agents/opencua/utils.py",
):
    if external_file.is_file():
        source_files.append(external_file)

source_hashes = {
    str(source.relative_to(repo_root) if source.is_relative_to(repo_root) else source):
        hashlib.sha256(source.read_bytes()).hexdigest()
    for source in source_files
}

from recovery.mypcbench.agent_config import AGENT_ID_BY_TYPE, load_agent_config

_factory_agents = set(AGENT_ID_BY_TYPE.values())
agent_live_config = {
    agent_id: dict(load_agent_config(agent_id).live)
    for agent_id in sys.argv[13:]
    if agent_id in _factory_agents
}
tasks_file = pathlib.Path(os.environ["TASKS_FILE"]).resolve()
tasks_total = len(json.loads(tasks_file.read_text(encoding="utf-8")))

manifest = {
    "source_benchmark": "mypcbench",
    "mypcbench_commit": sys.argv[2],
    "prompt_sha256": sys.argv[3],
    "image_sha256": sys.argv[4],
    "tasks_file": str(tasks_file),
    "tasks_total": tasks_total,
    "task_source": os.environ["TASK_SOURCE"],
    "task_source_file": str(pathlib.Path(os.environ["SOURCE_TASKS_FILE"]).resolve()),
    "repeats": int(sys.argv[5]),
    "num_vms_per_agent": json.loads(os.environ.get("NUM_VMS_PER_AGENT_JSON", "{}")),
    "max_steps_per_agent": json.loads(os.environ.get("MAX_STEPS_PER_AGENT_JSON", "{}")),
    "max_steps": int(sys.argv[6]),
    "task_timeout_seconds": int(os.environ.get("TASK_TIMEOUT", "0")) or None,
    "task_timeout_per_agent": json.loads(os.environ.get("TASK_TIMEOUT_PER_AGENT_JSON", "{}")),
    "context_images": int(os.environ["CONTEXT_IMAGES"]),
    "context_image_env": {
        name: os.environ.get(name)
        for name in ("OPENAI_ZDR_KEEP_IMAGES", "MYPCBENCH_QWEN_IMAGE_MAX")
    },
    "environment_config": os.environ["RECOVERY_ENVIRONMENT_CONFIG"],
    "timeout_per_vm_seconds": int(os.environ.get("TIMEOUT_PER_VM", "0")) or None,
    "evocua_agent_commit": sys.argv[11],
    "opencua_osworld_agent_commit": sys.argv[12],
    "agents": sys.argv[13:],
    "launch_contract": {
        "full_task_set": os.environ.get("IS_FULL_TASK_SET") == "1",
        "formal_collection_requested": os.environ.get("FORMAL_COLLECTION") == "1",
        "formal_collection_authorized": os.environ.get("LOCK_FORMAL_AUTHORIZED") == "1",
        "qwen35_endpoint_contracts": json.loads(
            os.environ.get("QWEN35_ENDPOINT_CONTRACTS_JSON", "[]")
        ),
        "qwen35_context_policy": os.environ.get("MYPCBENCH_QWEN_CONTEXT_POLICY"),
    },
    "adapter_policy": {
        "runner_owns_max_steps": True,
        "official_upstream_guard_offset": 1,
        "invalid_native_tool_call": "one_schema_repair_then_FAIL",
    },
    "harness_runtime": {
        "python_executable": sys.executable,
        "python_version": platform.python_version(),
        "packages": {
            package: importlib.metadata.version(package)
            for package in ("backoff", "openai", "Pillow", "requests")
        },
    },
    "adapter_source_sha256": source_hashes,
    "agent_live_config": agent_live_config,
}
path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
PY
fi

for agent_id in "${AGENTS[@]}"; do
  resolve_agent "$agent_id"
  agent_vms="$(resolve_agent_num_vms "$agent_id")"
  agent_max_steps_value="$(resolve_agent_max_steps "$agent_id")"
  agent_task_timeout_value="$(resolve_agent_task_timeout "$agent_id")"
  (( agent_task_timeout_value < TIMEOUT_PER_VM )) || die \
    "${agent_id} 的 task_timeout(${agent_task_timeout_value}) 必须小于 TIMEOUT_PER_VM"
  export RECOVERY_AGENT_MAX_STEPS="$agent_max_steps_value"
  if [[ "$agent_max_steps_value" != "$MAX_STEPS" ]]; then
    info "${agent_id}：max_steps 取 agent config 声明的 ${agent_max_steps_value}（采集 config 默认 ${MAX_STEPS}）"
  fi
  if [[ -n "$NUM_VMS_OVERRIDE" ]]; then
    info "${agent_id}：★ VM 数被 NUM_VMS_OVERRIDE 压到 ${agent_vms}（不覆盖时是 $(agent_vm_count "$REPO_ROOT" "$agent_id")）"
  else
    info "${agent_id}：VM 数 ${agent_vms}（本地 serving 按可用 GPU / tensor_parallel_size 推导，托管 API 取 yaml 的 num_vms）"
  fi
  VLLM_ARGS=()
  endpoint_urls=""
  if [[ "$RESOLVED_REQUIRED_ENV" == "LOCAL_ENDPOINT" ]]; then
    if [[ "$RECOVERY_AUTO_SERVE" == "1" && "$DRY_RUN" == "0" ]]; then
      auto_serve_start "$agent_id" "$RESOLVED_BASE_URLS_ENV"
    fi
    endpoint_urls="${!RESOLVED_BASE_URLS_ENV:-}"
    if [[ -z "$endpoint_urls" ]]; then
      endpoint_urls="${!RESOLVED_BASE_URL_ENV:-$GENERIC_OPENAI_BASE_URL}"
    fi
    [[ -n "$endpoint_urls" ]] || die \
      "${agent_id} 需要 ${RESOLVED_BASE_URLS_ENV}、${RESOLVED_BASE_URL_ENV} 或 OPENAI_BASE_URL"
    VLLM_ARGS=(--vllm-base-urls "$endpoint_urls")
    endpoint_count="$(awk -F',' '{print NF}' <<< "$endpoint_urls")"
    if (( agent_vms > endpoint_count )); then
      info "注意：${agent_id} 的 ${agent_vms} 个 VM 将共享 ${endpoint_count} 个推理 endpoint"
    fi
    if [[ "$(agent_family "$REPO_ROOT" "$agent_id" || true)" == qwen35 && "$DRY_RUN" == "0" ]]; then
      qwen35_endpoint_preflight "$endpoint_urls"
      record_qwen35_contracts "${run_root}/collection_manifest.json"
    fi
  elif [[ -n "$RESOLVED_REQUIRED_ENV" && -z "${!RESOLVED_REQUIRED_ENV:-}" ]]; then
    die "${agent_id} 需要环境变量 ${RESOLVED_REQUIRED_ENV}"
  fi

  export RECOVERY_AGENT_ID="$agent_id"

  repeat_start_index="${REPEAT_START_INDEX:-1}"
  is_positive_integer "$repeat_start_index" || die "REPEAT_START_INDEX 必须是正整数"
  repeat_end_index=$((repeat_start_index + REPEATS - 1))
  for repeat_index in $(seq "$repeat_start_index" "$repeat_end_index"); do
    result_dir="${run_root}/${agent_id}/repeat_${repeat_index}"
    container_base="recovery-${COLLECTION_ID}-${agent_id}-r${repeat_index}"
    command=(
      "$PYTHON_BIN" "$RUNNER"
      --backend "$BACKEND"
      --tasks-file "$TASKS_FILE"
      --num-vms "$agent_vms"
      --agent-type "$RESOLVED_AGENT_TYPE"
      --model "$RESOLVED_MODEL"
      --max-steps "$agent_max_steps_value"
      --timeout-per-vm "$TIMEOUT_PER_VM"
      --task-timeout "$agent_task_timeout_value"
      --port-base "$PORT_BASE"
      --container-base "$container_base"
      --result-dir "$result_dir"
      --screen-width "$SCREEN_WIDTH"
      --screen-height "$SCREEN_HEIGHT"
      "${VLLM_ARGS[@]}"
      "${QCOW2_ARGS[@]}"
    )

    info "${agent_id}，repeat ${repeat_index}/${REPEATS}"
    if [[ "$DRY_RUN" == "1" ]]; then
      printf '  '
      printf '%q ' "${command[@]}"
      printf '\n'
    else
      endpoint_watchdog_start "${endpoint_urls:-}" "$container_base"
      runner_status=0
      "${command[@]}" || runner_status=$?
      endpoint_watchdog_stop
      if endpoint_watchdog_tripped; then
        dead_endpoint="$(cat "$ENDPOINT_WATCHDOG_TRIP")"
        rm -f "$ENDPOINT_WATCHDOG_TRIP"
        die "${agent_id} repeat ${repeat_index}：endpoint ${dead_endpoint} 在采集途中消失，
     runner 已被中止。${result_dir} 下这一刻之后的 episode 都是空壳（PREDICT_CRASH +
     result.txt=1.0），必须删掉重跑，不要拿去判分。
     先确认没有别的 collection 在抢同一批显卡，再看 ${SERVING_LOG_ROOT} 下最新的 endpoint_*.log。"
      fi
      if [[ -n "$ENDPOINT_WATCHDOG_TRIP" ]]; then
        rm -f "$ENDPOINT_WATCHDOG_TRIP"
      fi
      (( runner_status == 0 )) || die \
        "${agent_id} repeat ${repeat_index}：runner 以退出码 ${runner_status} 结束"
    fi
  done

  auto_serve_stop

  if [[ "$DRY_RUN" != "1" ]]; then
    "$PYTHON_BIN" - "$run_root" "$agent_id" <<'PY'
import json
import pathlib
import sys

run_root = pathlib.Path(sys.argv[1])
agent_id = sys.argv[2]
shared = run_root / "collection_manifest.json"
manifest = json.loads(shared.read_text(encoding="utf-8"))

manifest["agents"] = [agent_id]
for field in ("num_vms_per_agent", "max_steps_per_agent", "task_timeout_per_agent", "agent_live_config"):
    table = manifest.get(field)
    if isinstance(table, dict):
        manifest[field] = {agent_id: table[agent_id]} if agent_id in table else {}

target = run_root / f"collection_manifest.{agent_id}.json"
target.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
print(f"[RECOVERY collection] 已留存 {target.name}")
PY
  fi
done

if [[ "$DRY_RUN" == "1" ]]; then
  info "dry run 完成，没有启动 VM 或调用模型"
else
  info "collection 完成：${run_root}"
fi
