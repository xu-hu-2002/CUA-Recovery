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
JUDGE_CONFIG="${JUDGE_CONFIG:-${REPO_ROOT}/configs/judges/default.yaml}"
REGISTRY_PY="${SCRIPT_DIR}/judge_model_registry.py"
BUNDLE_PY="${SCRIPT_DIR}/../takeover/bundle_prefix.py"
JUDGE_WRAPPER="${SCRIPT_DIR}/full_traj_judge.py"
FORCE="${FORCE:-0}"

die() {
  printf 'error: %s\n' "$*" >&2
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
            raise SystemExit(f"{path}:{line_number}: expected KEY=VALUE or KEY: VALUE")
        key = key.strip()
        value_text = value_text.strip()
        if not key_pattern.fullmatch(key):
            raise SystemExit(f"{path}:{line_number}: invalid environment variable name {key!r}")
        if value_text.startswith(("\"", "'")):
            try:
                value = ast.literal_eval(value_text)
            except (SyntaxError, ValueError) as exc:
                raise SystemExit(f"{path}:{line_number}: invalid quoted value: {exc}")
        else:
            value = value_text
        if not isinstance(value, str) or "\n" in value or "\x00" in value:
            raise SystemExit(f"{path}:{line_number}: VALUE must be a single-line string")
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
    info "Loaded ${env_key} from $(basename "$dotenv_path")"
  done <<< "$parsed_env"
}

load_judge_config() {
  local config_path="$1"
  local parsed line key value
  [[ -f "$config_path" ]] || die "Judge config not found: ${config_path} (override with JUDGE_CONFIG=)"

  parsed="$(
    "$PYTHON_BIN" - "$config_path" <<'PY'
import re
import sys

import yaml

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
}
value_pattern = re.compile(r'^[A-Za-z0-9_.\-]+$')

with open(path, encoding="utf-8") as handle:
    section = (yaml.safe_load(handle) or {}).get("rubric")
if not isinstance(section, dict):
    raise SystemExit(f"{path}: missing the rubric section")
for key, value in section.items():
    if key not in ENV_MAP:
        raise SystemExit(f"{path}: unknown rubric key {key!r} (see the allowlist in load_judge_config)")
    if value is None or value == "":
        continue
    text = str(value).lower() if isinstance(value, bool) else str(value)
    if not value_pattern.fullmatch(text):
        raise SystemExit(f"{path}: rubric.{key} value {text!r} contains invalid characters")
    print(f"{ENV_MAP[key]}={text}")
PY
  )" || die "Failed to parse judge config: ${config_path}"

  while IFS= read -r line; do
    [[ -n "$line" ]] || continue
    key="${line%%=*}"
    value="${line#*=}"
    if [[ -n "${!key:-}" && "${!key}" != "$value" ]]; then
      if [[ "${ALLOW_CONFIG_OVERRIDE:-0}" == "1" ]]; then
        JUDGE_OVERRIDDEN+=("${key}=${!key} (config: ${value})")
        continue
      fi
      JUDGE_IGNORED+=("${key}=${!key}")
    fi
    export "${key}=${value}"
  done <<< "$parsed"
}

collection_id="${1:-}"
requested_agent="${2:-}"

[[ -x "$PYTHON_BIN" ]] || die "python not found: ${PYTHON_BIN} (override with PYTHON_BIN=)"
[[ -f "$JUDGE_SCRIPT" ]] || die "Judge script not found: ${JUDGE_SCRIPT}"

JUDGE_OVERRIDDEN=()
JUDGE_IGNORED=()
load_judge_config "$JUDGE_CONFIG"
JUDGE_MODEL="$("$PYTHON_BIN" "$REGISTRY_PY" ${JUDGE_MODEL:+"$JUDGE_MODEL"} --print-model)" \
  || die "Judge model check failed (configs/judges/default.yaml is the single source of truth)"
JUDGE_TIMEOUT="${JUDGE_TIMEOUT:-${RECOVERY_JUDGE_TIMEOUT:-1000}}"

if [[ -z "$collection_id" ]]; then
  collection_id="$(ls -t "$OUTPUT_ROOT" 2>/dev/null | head -1)"
  [[ -n "$collection_id" ]] || die "No collection found: ${OUTPUT_ROOT}"
fi
run_root="${OUTPUT_ROOT}/${collection_id}"
[[ -d "$run_root" ]] || die "Collection does not exist: ${run_root}"

load_dotenv "$ROOT_DOTENV"

export MYPCBENCH_RUBRIC_JUDGE_MODEL="$JUDGE_MODEL"
case "$JUDGE_MODEL" in
  gemini*)
    [[ -n "${GEMINI_API_KEY:-}${GOOGLE_API_KEY:-}" ]] \
      || die "JUDGE_MODEL=${JUDGE_MODEL} needs GEMINI_API_KEY (or GOOGLE_API_KEY)"
    ;;
  *)
    [[ -n "${OPENAI_API_KEY:-}" ]] \
      || die "JUDGE_MODEL=${JUDGE_MODEL} needs OPENAI_API_KEY"
    if [[ "${RECOVERY_JUDGE_FORCE_OFFICIAL_ENDPOINT:-true}" == "true" \
       && -n "${OPENAI_BASE_URL:-}" && "${KEEP_OPENAI_BASE_URL:-0}" != "1" ]]; then
      info "Ignoring OPENAI_BASE_URL from .env (judging uses the official endpoint; set KEEP_OPENAI_BASE_URL=1 to keep it)"
      unset OPENAI_BASE_URL
    fi
    "$PYTHON_BIN" -c 'import openai' 2>/dev/null \
      || die "${PYTHON_BIN} has no openai package"
    ;;
esac

shopt -s nullglob
agent_dirs=("$run_root"/*/)
shopt -u nullglob
(( ${#agent_dirs[@]} > 0 )) || die "No agent directories under ${run_root}"

for agent_dir in "${agent_dirs[@]}"; do
  agent_id="$(basename "$agent_dir")"
  [[ "$agent_id" == .* ]] && continue
  [[ -n "$requested_agent" && "$agent_id" != "$requested_agent" ]] && continue
  "$PYTHON_BIN" "$REGISTRY_PY" "$JUDGE_MODEL" --agent "$agent_id" --print-model >/dev/null \
    || die "judge ${JUDGE_MODEL} shares a provider with evaluated agent ${agent_id}"
done

mapfile -t targets < <(
  for agent_dir in "${agent_dirs[@]}"; do
    agent_id="$(basename "$agent_dir")"
    [[ "$agent_id" == .* ]] && continue
    [[ -n "$requested_agent" && "$agent_id" != "$requested_agent" ]] && continue
    find "$agent_dir" -name rubric_bundle.json -printf '%h\n' 2>/dev/null
  done | xargs -r -n1 dirname | sort -u
)

(( ${#targets[@]} > 0 )) || die "No rubric_bundle.json found; is this collection still running?"

mkdir -p "$JUDGE_LOG_ROOT"
judge_log="${JUDGE_LOG_ROOT}/${collection_id}.log"

date +%s > "${run_root}/.judge_started_at" 2>/dev/null || true

total_bundles="$(find "$run_root" -name rubric_bundle.json 2>/dev/null | wc -l)"
already="$(find "$run_root" -name rubric_judge_result.json 2>/dev/null | wc -l)"

info "collection : ${collection_id}"
info "judge model: ${JUDGE_MODEL} (per-task timeout ${JUDGE_TIMEOUT}s)"
info "judge config: $(realpath --relative-to="$REPO_ROOT" "$JUDGE_CONFIG")"
info "  flavor=${MYPCBENCH_JUDGE_FLAVOR:-<upstream default>} max_images=${MYPCBENCH_OSWORLD_JUDGE_MAX_IMAGES:-<upstream default>} concurrency=${MYPCBENCH_OSWORLD_JUDGE_CONCURRENCY:-<upstream default>}"
info "  reasoning_effort=${MYPCBENCH_OSWORLD_JUDGE_REASONING_EFFORT:-<upstream default>} max_completion_tokens=${MYPCBENCH_OSWORLD_JUDGE_MAX_COMPLETION_TOKENS:-<upstream default>} max_retries=${MYPCBENCH_OSWORLD_JUDGE_MAX_RETRIES:-<upstream default>}"
info "  image=${MYPCBENCH_OSWORLD_JUDGE_IMAGE_FORMAT:-<upstream default>}/q${MYPCBENCH_OSWORLD_JUDGE_IMAGE_QUALITY:-<upstream default>} max_image_mb=${MYPCBENCH_OSWORLD_JUDGE_MAX_IMAGE_MB:-<upstream default>}"
if (( ${#JUDGE_OVERRIDDEN[@]} > 0 )); then
  info "  ★ ALLOW_CONFIG_OVERRIDE=1, using environment instead of config for: ${JUDGE_OVERRIDDEN[*]}"
fi
if (( ${#JUDGE_IGNORED[@]} > 0 )); then
  info "  Ignored environment variables that conflict with config: ${JUDGE_IGNORED[*]} (set ALLOW_CONFIG_OVERRIDE=1 to apply them)"
fi
info "targets     : ${#targets[@]}"
info "tasks       : ${total_bundles} (already judged ${already}, $( ((FORCE)) && echo 'FORCE=1 will rejudge all' || echo 'judged ones are skipped, no double billing'))"
info "log         : ${judge_log}"

export MYPCBENCH_RUBRIC_JUDGE_COMMAND="$(printf '%q %q' "$PYTHON_BIN" "$JUDGE_WRAPPER")"

force_flag=()
(( FORCE )) && force_flag=(--force)

rc_total=0
{
  printf '[RECOVERY judge] start: %s  model=%s  timeout=%ss\n' \
    "$(date '+%F %T')" "$JUDGE_MODEL" "$JUDGE_TIMEOUT"
  for vm_dir in "${targets[@]}"; do
    printf '\n[RECOVERY judge] ===== %s =====\n' "${vm_dir#${run_root}/}"
    "$PYTHON_BIN" "$BUNDLE_PY" "$vm_dir" || { rc_total=$?; continue; }
    "$PYTHON_BIN" -u "$JUDGE_SCRIPT" \
      --result_dir "$vm_dir" \
      --timeout "$JUDGE_TIMEOUT" \
      "${force_flag[@]+"${force_flag[@]}"}" || rc_total=$?
  done
  printf '\n[RECOVERY judge] end: %s  exit code %s\n' "$(date '+%F %T')" "$rc_total"
} 2>&1 | tee -a "$judge_log"

(( rc_total == 0 )) || die "Some judge targets ended with exit code ${rc_total}; see ${judge_log}"

info "All done."
