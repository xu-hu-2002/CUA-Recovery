#!/usr/bin/env bash
# Usage: tmux new -s recovery-judge 'bash scripts/judge/judge_rubrics.sh [COLLECTION_ID [AGENT_ID]]'

set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

PYTHON_BIN="${PYTHON_BIN:-python3}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${REPO_ROOT}/artifacts/raw_rollouts/mypcbench}"
JUDGE_SCRIPT="${JUDGE_SCRIPT:-${REPO_ROOT}/third_party/MyPCBench/agent-harness/judge_results.py}"
JUDGE_LOG_ROOT="${JUDGE_LOG_ROOT:-${REPO_ROOT}/artifacts/judge_logs}"
ROOT_DOTENV="${ROOT_DOTENV:-${REPO_ROOT}/.env}"
JUDGE_CONFIG="${JUDGE_CONFIG:-${REPO_ROOT}/configs/judges/mypcbench_rubric.yaml}"
REGISTRY_PY="${SCRIPT_DIR}/judge_model_registry.py"
BUNDLE_PY="${SCRIPT_DIR}/../takeover/bundle_prefix.py"
JUDGE_WRAPPER="${SCRIPT_DIR}/full_traj_judge.py"
FORCE="${FORCE:-0}"

die() {
  printf '错误：%s\n' "$*" >&2
  exit 1
}

info() {
  printf '[RECOVERY judge] %s\n' "$*"
}

load_dotenv() {
  local dotenv_path="$1"
  local parsed_env env_line env_key
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
    case "$env_key" in
      OPENAI_API_KEY|OPENAI_BASE_URL|GEMINI_API_KEY|GOOGLE_API_KEY) ;;
      *) continue ;;
    esac
    export "$env_line"
    info "已从 $(basename "$dotenv_path") 加载 ${env_key}"
  done <<< "$parsed_env"
}

load_judge_config() {
  local config_path="$1"
  local parsed line key value
  [[ -f "$config_path" ]] || die "找不到判官配置：${config_path}（用 JUDGE_CONFIG= 覆盖）"

  parsed="$(
    "$PYTHON_BIN" - "$config_path" <<'PY'
import re
import sys

path = sys.argv[1]
ENV_MAP = {
    "flavor": "MYPCBENCH_JUDGE_FLAVOR",
    "provider": "MYPCBENCH_JUDGE_PROVIDER",
    "max_images": "MYPCBENCH_OSWORLD_JUDGE_MAX_IMAGES",
    "max_image_mb": "MYPCBENCH_OSWORLD_JUDGE_MAX_IMAGE_MB",
    "image_format": "MYPCBENCH_OSWORLD_JUDGE_IMAGE_FORMAT",
    "image_quality": "MYPCBENCH_OSWORLD_JUDGE_IMAGE_QUALITY",
    "reasoning_effort": "MYPCBENCH_OSWORLD_JUDGE_REASONING_EFFORT",
    "max_completion_tokens": "MYPCBENCH_OSWORLD_JUDGE_MAX_COMPLETION_TOKENS",
    "max_retries": "MYPCBENCH_OSWORLD_JUDGE_MAX_RETRIES",
    "concurrency": "MYPCBENCH_OSWORLD_JUDGE_CONCURRENCY",
    "timeout_seconds": "RECOVERY_JUDGE_TIMEOUT",
    "api_key_env": "RECOVERY_JUDGE_API_KEY_ENV",
    "force_official_endpoint": "RECOVERY_JUDGE_FORCE_OFFICIAL_ENDPOINT",
    "judge_id": None,
    "status": None,
}
value_pattern = re.compile(r'^[A-Za-z0-9_.\-]+$')

with open(path, encoding="utf-8") as handle:
    for lineno, raw in enumerate(handle, start=1):
        stripped = raw.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if ":" not in stripped:
            raise SystemExit(f"{path}:{lineno}: 应为 key: value")
        key, text = stripped.split(":", 1)
        key = key.strip()
        text = text.split("#", 1)[0].strip().strip('"').strip("'")
        if key not in ENV_MAP:
            raise SystemExit(f"{path}:{lineno}: 未知配置键 {key!r}（白名单见 load_judge_config）")
        if ENV_MAP[key] is None or not text:
            continue
        if not value_pattern.fullmatch(text):
            raise SystemExit(f"{path}:{lineno}: 值 {text!r} 含非法字符")
        print(f"{ENV_MAP[key]}={text}")
PY
  )" || die "解析判官配置失败：${config_path}"

  while IFS= read -r line; do
    [[ -n "$line" ]] || continue
    key="${line%%=*}"
    value="${line#*=}"
    if [[ -n "${!key:-}" && "${!key}" != "$value" ]]; then
      if [[ "${ALLOW_CONFIG_OVERRIDE:-0}" == "1" ]]; then
        JUDGE_OVERRIDDEN+=("${key}=${!key}（config 为 ${value}）")
        continue
      fi
      JUDGE_IGNORED+=("${key}=${!key}")
    fi
    export "${key}=${value}"
  done <<< "$parsed"
}

collection_id="${1:-}"
requested_agent="${2:-}"

[[ -x "$PYTHON_BIN" ]] || die "找不到 python：${PYTHON_BIN}（用 PYTHON_BIN= 覆盖）"
[[ -f "$JUDGE_SCRIPT" ]] || die "找不到 judge 脚本：${JUDGE_SCRIPT}"

JUDGE_OVERRIDDEN=()
JUDGE_IGNORED=()
load_judge_config "$JUDGE_CONFIG"
JUDGE_MODEL="$("$PYTHON_BIN" "$REGISTRY_PY" ${JUDGE_MODEL:+"$JUDGE_MODEL"} --print-model)" \
  || die "judge 模型校验失败（configs/judges/default.yaml 是唯一真源）"
JUDGE_TIMEOUT="${JUDGE_TIMEOUT:-${RECOVERY_JUDGE_TIMEOUT:-1000}}"

if [[ -z "$collection_id" ]]; then
  collection_id="$(ls -t "$OUTPUT_ROOT" 2>/dev/null | head -1)"
  [[ -n "$collection_id" ]] || die "找不到任何 collection：${OUTPUT_ROOT}"
fi
run_root="${OUTPUT_ROOT}/${collection_id}"
[[ -d "$run_root" ]] || die "collection 不存在：${run_root}"

load_dotenv "$ROOT_DOTENV"

export MYPCBENCH_RUBRIC_JUDGE_MODEL="$JUDGE_MODEL"
case "$JUDGE_MODEL" in
  gemini*)
    [[ -n "${GEMINI_API_KEY:-}${GOOGLE_API_KEY:-}" ]] \
      || die "JUDGE_MODEL=${JUDGE_MODEL} 需要 GEMINI_API_KEY（或 GOOGLE_API_KEY）"
    ;;
  *)
    [[ -n "${OPENAI_API_KEY:-}" ]] \
      || die "JUDGE_MODEL=${JUDGE_MODEL} 需要 OPENAI_API_KEY"
    if [[ "${RECOVERY_JUDGE_FORCE_OFFICIAL_ENDPOINT:-true}" == "true" \
       && -n "${OPENAI_BASE_URL:-}" && "${KEEP_OPENAI_BASE_URL:-0}" != "1" ]]; then
      info "忽略 .env 的 OPENAI_BASE_URL（判分走官方 endpoint；要保留就设 KEEP_OPENAI_BASE_URL=1）"
      unset OPENAI_BASE_URL
    fi
    "$PYTHON_BIN" -c 'import openai' 2>/dev/null \
      || die "${PYTHON_BIN} 里没有 openai 包"
    ;;
esac

shopt -s nullglob
agent_dirs=("$run_root"/*/)
shopt -u nullglob
(( ${#agent_dirs[@]} > 0 )) || die "${run_root} 下没有 agent 目录"

for agent_dir in "${agent_dirs[@]}"; do
  agent_id="$(basename "$agent_dir")"
  [[ "$agent_id" == .* ]] && continue
  [[ -n "$requested_agent" && "$agent_id" != "$requested_agent" ]] && continue
  "$PYTHON_BIN" "$REGISTRY_PY" "$JUDGE_MODEL" --agent "$agent_id" --print-model >/dev/null \
    || die "judge ${JUDGE_MODEL} 与被测 agent ${agent_id} 同源"
done

mapfile -t targets < <(
  for agent_dir in "${agent_dirs[@]}"; do
    agent_id="$(basename "$agent_dir")"
    [[ "$agent_id" == .* ]] && continue
    [[ -n "$requested_agent" && "$agent_id" != "$requested_agent" ]] && continue
    find "$agent_dir" -name rubric_bundle.json -printf '%h\n' 2>/dev/null
  done | xargs -r -n1 dirname | sort -u
)

(( ${#targets[@]} > 0 )) || die "没找到任何 rubric_bundle.json —— 这个 collection 还没采集完？"

mkdir -p "$JUDGE_LOG_ROOT"
judge_log="${JUDGE_LOG_ROOT}/${collection_id}.log"

date +%s > "${run_root}/.judge_started_at" 2>/dev/null || true

total_bundles="$(find "$run_root" -name rubric_bundle.json 2>/dev/null | wc -l)"
already="$(find "$run_root" -name rubric_judge_result.json 2>/dev/null | wc -l)"

info "collection : ${collection_id}"
info "judge model: ${JUDGE_MODEL}（单 task 超时 ${JUDGE_TIMEOUT}s）"
info "判官配置   : $(realpath --relative-to="$REPO_ROOT" "$JUDGE_CONFIG")"
info "  flavor=${MYPCBENCH_JUDGE_FLAVOR:-<上游默认>} max_images=${MYPCBENCH_OSWORLD_JUDGE_MAX_IMAGES:-<上游默认>} concurrency=${MYPCBENCH_OSWORLD_JUDGE_CONCURRENCY:-<上游默认>}"
info "  reasoning_effort=${MYPCBENCH_OSWORLD_JUDGE_REASONING_EFFORT:-<上游默认>} max_completion_tokens=${MYPCBENCH_OSWORLD_JUDGE_MAX_COMPLETION_TOKENS:-<上游默认>} max_retries=${MYPCBENCH_OSWORLD_JUDGE_MAX_RETRIES:-<上游默认>}"
info "  image=${MYPCBENCH_OSWORLD_JUDGE_IMAGE_FORMAT:-<上游默认>}/q${MYPCBENCH_OSWORLD_JUDGE_IMAGE_QUALITY:-<上游默认>} max_image_mb=${MYPCBENCH_OSWORLD_JUDGE_MAX_IMAGE_MB:-<上游默认>}"
if (( ${#JUDGE_OVERRIDDEN[@]} > 0 )); then
  info "  ★ ALLOW_CONFIG_OVERRIDE=1，以下用环境变量而非 config：${JUDGE_OVERRIDDEN[*]}"
fi
if (( ${#JUDGE_IGNORED[@]} > 0 )); then
  info "  已忽略与 config 冲突的环境变量：${JUDGE_IGNORED[*]}（要生效请设 ALLOW_CONFIG_OVERRIDE=1）"
fi
info "待判目录   : ${#targets[@]} 个"
info "task 总数  : ${total_bundles}（已判 ${already}，$( ((FORCE)) && echo '本次 FORCE=1 会全部重判' || echo '已判的会跳过，不重复计费')）"
info "日志       : ${judge_log}"

export MYPCBENCH_RUBRIC_JUDGE_COMMAND="$(printf '%q %q' "$PYTHON_BIN" "$JUDGE_WRAPPER")"

force_flag=()
(( FORCE )) && force_flag=(--force)

rc_total=0
{
  printf '[RECOVERY judge] 开始：%s  model=%s  timeout=%ss\n' \
    "$(date '+%F %T')" "$JUDGE_MODEL" "$JUDGE_TIMEOUT"
  for vm_dir in "${targets[@]}"; do
    printf '\n[RECOVERY judge] ===== %s =====\n' "${vm_dir#${run_root}/}"
    "$PYTHON_BIN" "$BUNDLE_PY" "$vm_dir" || { rc_total=$?; continue; }
    "$PYTHON_BIN" -u "$JUDGE_SCRIPT" \
      --result_dir "$vm_dir" \
      --timeout "$JUDGE_TIMEOUT" \
      "${force_flag[@]+"${force_flag[@]}"}" || rc_total=$?
  done
  printf '\n[RECOVERY judge] 结束：%s  退出码 %s\n' "$(date '+%F %T')" "$rc_total"
} 2>&1 | tee -a "$judge_log"

(( rc_total == 0 )) || die "有判分目录以退出码 ${rc_total} 结束，看 ${judge_log}"

info "全部完成。"
