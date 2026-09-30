#!/usr/bin/env bash
set -euo pipefail

SOURCE_AGENT="${SOURCE_AGENT:-}"
TARGET_AGENT="${TARGET_AGENT:-}"
ANNOTATOR_ID="${ANNOTATOR_ID:-}"
BUILD_DIR="${BUILD_DIR:-}"
HUMAN_LABELS_DIR="${HUMAN_LABELS_DIR:-artifacts/derail_builds/human_labels}"
TRAJECTORY_ID_FILTER="${TRAJECTORY_ID_FILTER:-}"
TRAJECTORY_ID_FILE="${TRAJECTORY_ID_FILE:-}"
TRAJECTORY_ID_EXCLUDES="${TRAJECTORY_ID_EXCLUDES:-}"
QCOW2="${QCOW2:-third_party/MyPCBench/mypcbench-vm/mypcbench.qcow2}"
QCOW2_SHA256="${QCOW2_SHA256:-}"
QCOW2_HASH_PREVERIFIED="${QCOW2_HASH_PREVERIFIED:-0}"
RUN_TAG="${RUN_TAG:-takeover_v1}"
TAKEOVER_CONFIG="${TAKEOVER_CONFIG:-configs/takeover/takeover.yaml}"
CONDITIONS="${CONDITIONS:-}"
DEPTHS="${DEPTHS:-}"
REPEATS="${REPEATS:-}"
MAX_STEPS="${MAX_STEPS:-}"
TIMEOUT="${TIMEOUT:-}"
PREFIX_SOURCE="${PREFIX_SOURCE:-}"
REPAIRED_PREFIX_DIR="${REPAIRED_PREFIX_DIR:-}"
ON_MISSING_REPAIRED="${ON_MISSING_REPAIRED:-}"
NUM_WORKERS="${NUM_WORKERS:-1}"
SHARD_COUNT="${SHARD_COUNT:-$NUM_WORKERS}"
SHARD_OFFSET="${SHARD_OFFSET:-0}"
WORKER_INDICES="${WORKER_INDICES:-}"
PORT_BASE="${PORT_BASE:-24000}"
TARGET_BASE_URLS="${TARGET_BASE_URLS:-}"
TOKENIZE_BASE_URL="${TOKENIZE_BASE_URL:-}"
TOKENIZE_MODE="${TOKENIZE_MODE:-vllm}"
EXPERIMENT_CONTEXT_CAP="${EXPERIMENT_CONTEXT_CAP:-0}"
RUN_JUDGE="${RUN_JUDGE:-1}"
FORCE_JUDGE="${FORCE_JUDGE:-0}"
DRY_RUN="${DRY_RUN:-0}"
RETRY_FAILED_ONLY="${RETRY_FAILED_ONLY:-0}"
JOB_MAX_ATTEMPTS="${JOB_MAX_ATTEMPTS:-3}"
JOB_RETRY_DELAY="${JOB_RETRY_DELAY:-10}"

need_value() {
  [[ $# -ge 2 && -n "$2" ]] || { echo "FATAL: $1 需要一个值" >&2; exit 2; }
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --source-agent) need_value "$@"; SOURCE_AGENT="$2"; shift 2 ;;
    --takeover-agent|--target-agent) need_value "$@"; TARGET_AGENT="$2"; shift 2 ;;
    --depth|--depths) need_value "$@"; DEPTHS="$2"; shift 2 ;;
    --condition|--conditions|--prompt-condition) need_value "$@"; CONDITIONS="$2"; shift 2 ;;
    --annotator-id) need_value "$@"; ANNOTATOR_ID="$2"; shift 2 ;;
    --run-tag) need_value "$@"; RUN_TAG="$2"; shift 2 ;;
    --repeats) need_value "$@"; REPEATS="$2"; shift 2 ;;
    --takeover-config) need_value "$@"; TAKEOVER_CONFIG="$2"; shift 2 ;;
    --dry-run) DRY_RUN=1; shift ;;
    --no-judge) RUN_JUDGE=0; shift ;;
    --run-judge) RUN_JUDGE=1; shift ;;
    --retry-failed-only) RETRY_FAILED_ONLY=1; shift ;;
    *) echo "FATAL: unknown argument: $1" >&2; exit 2 ;;
  esac
done

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
resolve_path() {
  if [[ "$1" = /* ]]; then printf '%s\n' "$1"; else printf '%s/%s\n' "$REPO_ROOT" "$1"; fi
}
[[ -n "$SOURCE_AGENT" && -n "$TARGET_AGENT" ]] || { echo "FATAL: --source-agent and --target-agent are required" >&2; exit 1; }
[[ -n "$BUILD_DIR" ]] || { echo "FATAL: BUILD_DIR is required" >&2; exit 1; }
BUILD_DIR="$(resolve_path "$BUILD_DIR")"
HUMAN_LABELS_DIR="$(resolve_path "$HUMAN_LABELS_DIR")"
QCOW2="$(resolve_path "$QCOW2")"
TAKEOVER_CONFIG="$(resolve_path "$TAKEOVER_CONFIG")"
eval "$(python3 - "$TAKEOVER_CONFIG" <<'PY'
import shlex
import sys

import yaml

config = yaml.safe_load(open(sys.argv[1], encoding="utf-8"))
for name, value in (
    ("REPEATS", config["repeats"]),
    ("MAX_STEPS", config["max_steps"]),
    ("TIMEOUT", config["timeout_seconds"]),
    ("DEPTHS", " ".join(str(depth) for depth in config["depths"])),
    ("CONDITIONS", " ".join(config["conditions"])),
):
    print(f'{name}="${{{name}:-}}"; [[ -n "${name}" ]] || {name}={shlex.quote(str(value))}')
PY
)"
OUTPUT_ROOT="${OUTPUT_ROOT:-$REPO_ROOT/artifacts/model_outputs/takeover/$RUN_TAG/${SOURCE_AGENT}_to_${TARGET_AGENT}}"
SELECTION_MANIFEST="$OUTPUT_ROOT/selection_manifest.json"
SELECTION_LIST="$OUTPUT_ROOT/selection.tsv"
HISTORY_PREFLIGHT_REPORT="$OUTPUT_ROOT/native_history_preflight.json"
PREFLIGHT_EXCLUSIONS="$OUTPUT_ROOT/token_overflow_exclusions.tsv"
TOKEN_COUNT_CACHE="${TOKEN_COUNT_CACHE:-$OUTPUT_ROOT/token_count_cache.json}"

fail() { echo "FATAL: $1" >&2; exit 1; }
[[ -d "$BUILD_DIR/canonical" ]] || fail "missing canonical build: $BUILD_DIR/canonical"
[[ -d "$HUMAN_LABELS_DIR" ]] || fail "missing human labels: $HUMAN_LABELS_DIR"
if [[ "$DRY_RUN" != "1" ]]; then
  [[ -f "$QCOW2" ]] || fail "missing qcow2: $QCOW2"
fi
if [[ -z "$TARGET_BASE_URLS" ]]; then
  case "$TARGET_AGENT" in
    qwen3_8_27b) TARGET_BASE_URLS="${QWEN38_BASE_URLS:-${QWEN38_BASE_URL:-${OPENAI_BASE_URL:-}}}" ;;
    qwen3_6_27b) TARGET_BASE_URLS="${QWEN36_BASE_URLS:-${QWEN36_BASE_URL:-${OPENAI_BASE_URL:-}}}" ;;
    evocua_32b) TARGET_BASE_URLS="${EVOCUA_BASE_URLS:-${EVOCUA_BASE_URL:-${OPENAI_BASE_URL:-}}}" ;;
    opencua_72b) TARGET_BASE_URLS="${OPENCUA_BASE_URLS:-${OPENCUA_BASE_URL:-${OPENAI_BASE_URL:-}}}" ;;
    holo_3_1_35b_a3b) TARGET_BASE_URLS="${HOLO31_BASE_URLS:-${HOLO31_BASE_URL:-${OPENAI_BASE_URL:-}}}" ;;
    *) TARGET_BASE_URLS="${OPENAI_BASE_URL:-}" ;;
  esac
fi
[[ "$NUM_WORKERS" =~ ^[1-9][0-9]*$ ]] || fail "NUM_WORKERS must be a positive integer"
[[ "$SHARD_COUNT" =~ ^[1-9][0-9]*$ ]] || fail "SHARD_COUNT must be a positive integer"
[[ "$SHARD_OFFSET" =~ ^[0-9]+$ ]] || fail "SHARD_OFFSET must be a non-negative integer"
(( SHARD_OFFSET + NUM_WORKERS <= SHARD_COUNT )) || fail \
  "SHARD_OFFSET + NUM_WORKERS must not exceed SHARD_COUNT"
[[ "$REPEATS" =~ ^[1-9][0-9]*$ ]] || fail "REPEATS must be a positive integer"
[[ "$MAX_STEPS" =~ ^[1-9][0-9]*$ ]] || fail "MAX_STEPS must be a positive integer"
[[ "$TIMEOUT" =~ ^[1-9][0-9]*$ ]] || fail "TIMEOUT must be a positive integer"
[[ "$JOB_MAX_ATTEMPTS" =~ ^[1-9][0-9]*$ ]] || fail "JOB_MAX_ATTEMPTS must be a positive integer"
[[ "$JOB_RETRY_DELAY" =~ ^[0-9]+$ ]] || fail "JOB_RETRY_DELAY must be a non-negative integer"
[[ "$RETRY_FAILED_ONLY" == "0" || "$RETRY_FAILED_ONLY" == "1" ]] || \
  fail "RETRY_FAILED_ONLY must be 0 or 1"
[[ "$PORT_BASE" =~ ^[0-9]+$ ]] || fail "PORT_BASE must be an integer"
case "$TOKENIZE_MODE" in vllm|chat_usage|anthropic_usage|none) ;; *) fail "TOKENIZE_MODE must be vllm|chat_usage|anthropic_usage|none" ;; esac
[[ "$EXPERIMENT_CONTEXT_CAP" =~ ^[0-9]+$ ]] || fail "EXPERIMENT_CONTEXT_CAP must be an integer"
if [[ "$TOKENIZE_MODE" != "vllm" && "$TOKENIZE_MODE" != "none" && "$EXPERIMENT_CONTEXT_CAP" == "0" ]]; then
  fail "usage token counting requires a positive EXPERIMENT_CONTEXT_CAP"
fi
IFS=',' read -r -a ENDPOINT_VALUES <<< "$TARGET_BASE_URLS"
if [[ "$DRY_RUN" != "1" ]]; then
  [[ -n "$TARGET_BASE_URLS" ]] || fail "no target endpoint; set TARGET_BASE_URLS or OPENAI_BASE_URL"
  (( NUM_WORKERS <= ${#ENDPOINT_VALUES[@]} )) || fail \
    "NUM_WORKERS=$NUM_WORKERS exceeds the ${#ENDPOINT_VALUES[@]} configured target endpoints"
fi

if [[ -n "$WORKER_INDICES" ]]; then
  read -r -a ACTIVE_WORKER_INDICES <<< "$WORKER_INDICES"
else
  ACTIVE_WORKER_INDICES=()
  for ((worker_index = 0; worker_index < NUM_WORKERS; worker_index++)); do
    ACTIVE_WORKER_INDICES+=("$worker_index")
  done
fi
for worker_index in "${ACTIVE_WORKER_INDICES[@]}"; do
  [[ "$worker_index" =~ ^[0-9]+$ ]] || fail "WORKER_INDICES must contain integers"
  (( worker_index < NUM_WORKERS )) || fail \
    "worker index $worker_index is outside the NUM_WORKERS=$NUM_WORKERS shard range"
done
if [[ -n "$WORKER_INDICES" && "$RUN_JUDGE" == "1" ]]; then
  fail "selective WORKER_INDICES recovery requires RUN_JUDGE=0; judge only after all shards finish"
fi
if [[ -n "$TRAJECTORY_ID_EXCLUDES" && "$RUN_JUDGE" == "1" ]]; then
  fail "TRAJECTORY_ID_EXCLUDES recovery requires RUN_JUDGE=0"
fi
EXCLUDED_TRAJECTORY_VALUES=()
if [[ -n "$TRAJECTORY_ID_EXCLUDES" ]]; then
  read -r -a EXCLUDED_TRAJECTORY_VALUES <<< "$TRAJECTORY_ID_EXCLUDES"
fi

if [[ "$DRY_RUN" != "1" ]]; then
  if [[ "$QCOW2_HASH_PREVERIFIED" == "1" ]]; then
    [[ "$QCOW2_SHA256" =~ ^[0-9a-f]{64}$ ]] || \
      fail "QCOW2_HASH_PREVERIFIED=1 requires a 64-character QCOW2_SHA256"
  else
    ACTUAL_QCOW2_SHA256="$(sha256sum "$QCOW2" | awk '{print $1}')"
    if [[ -n "$QCOW2_SHA256" && "$QCOW2_SHA256" != "$ACTUAL_QCOW2_SHA256" ]]; then
      fail "provided QCOW2_SHA256 does not match $QCOW2"
    fi
    QCOW2_SHA256="$ACTUAL_QCOW2_SHA256"
  fi
  printf 'Verified qcow2 sha256=%s path=%s\n' "$QCOW2_SHA256" "$QCOW2"
fi

if [[ "$CONDITIONS" == "all" ]]; then CONDITIONS="unaware notified diagnosed"; fi
if [[ "$DEPTHS" == "all" ]]; then DEPTHS="0 5 10 15 20 25"; fi
CONDITIONS="${CONDITIONS//\// }"
CONDITIONS="${CONDITIONS//,/ }"
DEPTHS="${DEPTHS//\// }"
DEPTHS="${DEPTHS//,/ }"
read -r -a CONDITION_VALUES <<< "$CONDITIONS"
read -r -a DEPTH_VALUES <<< "$DEPTHS"
[[ ${#CONDITION_VALUES[@]} -gt 0 ]] || fail "CONDITIONS is empty"
[[ ${#DEPTH_VALUES[@]} -gt 0 ]] || fail "DEPTHS is empty"
for index in "${!CONDITION_VALUES[@]}"; do
  condition="${CONDITION_VALUES[$index]}"
  case "$condition" in
    unaware|notified|diagnosed) ;;
    diagnosis|diagonosis) CONDITION_VALUES[$index]="diagnosed" ;;
    *) fail "unknown condition: $condition" ;;
  esac
done
for depth in "${DEPTH_VALUES[@]}"; do
  case "$depth" in
    0|5|10|15|20|25) ;;
    *) fail "depth must be one of: 0 5 10 15 20 25 (received $depth)" ;;
  esac
done

mkdir -p "$OUTPUT_ROOT"
selection=(
  python3 "$REPO_ROOT/scripts/benchmark/select_takeover_failures.py"
  --build-dir "$BUILD_DIR"
  --human-labels-dir "$HUMAN_LABELS_DIR"
  --source-agent "$SOURCE_AGENT"
  --depths "${DEPTH_VALUES[@]}"
  --manifest "$SELECTION_MANIFEST"
  --list-file "$SELECTION_LIST"
)
[[ -n "$ANNOTATOR_ID" ]] && selection+=(--annotator-id "$ANNOTATOR_ID")
[[ -n "$TRAJECTORY_ID_FILTER" ]] && selection+=(--trajectory-id "$TRAJECTORY_ID_FILTER")
if [[ -n "$TRAJECTORY_ID_FILE" ]]; then
  selection+=(--trajectory-id-file "$(resolve_path "$TRAJECTORY_ID_FILE")")
fi
"${selection[@]}"

preflight=(
  python3 "$REPO_ROOT/scripts/takeover/preflight_history.py"
  --selection-list "$SELECTION_LIST"
  --target-agent "$TARGET_AGENT"
  --depths "${DEPTH_VALUES[@]}"
  --conditions "${CONDITION_VALUES[@]}"
  --shard-count "$SHARD_COUNT"
  --shard-offset "$SHARD_OFFSET"
  --shard-workers "$NUM_WORKERS"
  --overflow-policy exclude
  --takeover-config "$TAKEOVER_CONFIG"
  --report "$HISTORY_PREFLIGHT_REPORT"
)
PREFIX_OPTIONS=()
[[ -n "$PREFIX_SOURCE" ]] && PREFIX_OPTIONS+=(--prefix-source "$PREFIX_SOURCE")
[[ -n "$REPAIRED_PREFIX_DIR" ]] && PREFIX_OPTIONS+=(--repaired-prefix-dir "$(resolve_path "$REPAIRED_PREFIX_DIR")")
[[ -n "$ON_MISSING_REPAIRED" ]] && PREFIX_OPTIONS+=(--on-missing-repaired "$ON_MISSING_REPAIRED")
preflight+=("${PREFIX_OPTIONS[@]}")
if [[ -n "$TRAJECTORY_ID_EXCLUDES" ]]; then
  for trajectory_id in "${EXCLUDED_TRAJECTORY_VALUES[@]}"; do
    preflight+=(--exclude-trajectory-id "$trajectory_id")
  done
fi
if [[ "$DRY_RUN" != "1" && "$TOKENIZE_MODE" != "none" ]]; then
  preflight+=(--tokenize-base-url "${TOKENIZE_BASE_URL:-${ENDPOINT_VALUES[0]}}")
  preflight+=(--tokenize-mode "$TOKENIZE_MODE")
  if [[ "$EXPERIMENT_CONTEXT_CAP" != "0" ]]; then
    preflight+=(--context-cap "$EXPERIMENT_CONTEXT_CAP")
  fi
  if [[ "$TOKENIZE_MODE" != "vllm" ]]; then
    preflight+=(--token-cache "$TOKEN_COUNT_CACHE")
  fi
fi
"${preflight[@]}"
python3 - "$HISTORY_PREFLIGHT_REPORT" "$PREFLIGHT_EXCLUSIONS" <<'PY'
import json
import sys
from pathlib import Path

report_path, output_path = map(Path, sys.argv[1:])
report = json.loads(report_path.read_text(encoding="utf-8"))
rows = {
    (str(item["trajectory_id"]), str(item["depth"]), str(item["condition"]), "token_overflow")
    for item in report["token_preflight"]["overflows"]
}
rows |= {
    (str(item["trajectory_id"]), str(item["depth"]), str(item["condition"]), str(item["reason"]))
    for item in report.get("prefix_exclusions", [])
}
payload = "".join("\t".join(row) + "\n" for row in sorted(rows))
output_path.write_text(payload, encoding="utf-8")
PY
if [[ "$DRY_RUN" != "1" ]]; then
  rm -f "$OUTPUT_ROOT/.takeover_paused_at"
  date +%s > "$OUTPUT_ROOT/.takeover_active_run_started_at"
fi

if [[ "$DRY_RUN" != "1" ]]; then
  if [[ ! -f "$OUTPUT_ROOT/.takeover_started_at" ]]; then
    date +%s > "$OUTPUT_ROOT/.takeover_started_at"
  fi
fi

JUDGE_CELLS=()
for ((repeat = 1; repeat <= REPEATS; repeat++)); do
for depth in "${DEPTH_VALUES[@]}"; do
  for condition in "${CONDITION_VALUES[@]}"; do
    if awk -F '\t' -v depth="$depth" '
      {
        n = split($8, values, ",")
        for (i = 1; i <= n; i++) if (values[i] == depth) { found = 1; exit }
      }
      END { exit(found ? 0 : 1) }
    ' "$SELECTION_LIST"; then
      JUDGE_CELLS+=("$repeat $depth $condition")
    fi
  done
done
done

run_worker() {
  local worker_index="$1"
  local shard_index=$((SHARD_OFFSET + worker_index))
  local endpoint="${ENDPOINT_VALUES[$worker_index]:-}"
  local job_index=0
  local repeat depth condition result_dir scheduled attempt status job_failures=0
  local exclusion_reason
  for ((repeat = 1; repeat <= REPEATS; repeat++)); do
  for depth in "${DEPTH_VALUES[@]}"; do
    for condition in "${CONDITION_VALUES[@]}"; do
      result_dir="$OUTPUT_ROOT/repeat_${repeat}/depth_${depth}/$condition"
      scheduled=0
    while IFS=$'\t' read -r trajectory_id canonical_trajectory normalization_report task_json annotation_json root_index last_action_index available_depths; do
      [[ -n "$trajectory_id" ]] || continue
      if [[ -n "$TRAJECTORY_ID_EXCLUDES" ]]; then
        for excluded_trajectory_id in "${EXCLUDED_TRAJECTORY_VALUES[@]}"; do
          if [[ "$trajectory_id" == "$excluded_trajectory_id" ]]; then
            continue 2
          fi
        done
      fi
      case ",$available_depths," in
        *",$depth,"*) ;;
        *) continue ;;
      esac
      scheduled=$((scheduled + 1))
      if (( job_index % SHARD_COUNT != shard_index )); then
        job_index=$((job_index + 1))
        continue
      fi
      exclusion_reason="$(awk -F '\t' -v trajectory="$trajectory_id" -v depth="$depth" \
          -v condition="$condition" \
          '$1 == trajectory && $2 == depth && $3 == condition { print $4; exit }' \
          "$PREFLIGHT_EXCLUSIONS")"
      if [[ -n "$exclusion_reason" ]]; then
        printf 'PROTOCOL_EXCLUSION: worker=%s job=%s repeat=%s depth=%s condition=%s reason=%s report=%s\n' \
          "$worker_index" "$trajectory_id" "$repeat" "$depth" "$condition" \
          "$exclusion_reason" "$HISTORY_PREFLIGHT_REPORT" >&2
        job_index=$((job_index + 1))
        continue
      fi
      command=(
        python3 "$REPO_ROOT/scripts/takeover/run_rollout.py"
        --source-agent "$SOURCE_AGENT"
        --target-agent "$TARGET_AGENT"
        --condition "$condition"
        --depth "$depth"
        --canonical-trajectory "$canonical_trajectory"
        --normalization-report "$normalization_report"
        --task-json "$task_json"
        --annotation-json "$annotation_json"
        --qcow2 "$QCOW2"
        --result-dir "$result_dir"
        --takeover-config "$TAKEOVER_CONFIG"
        --repeat "$repeat"
        --max-steps "$MAX_STEPS"
        --timeout "$TIMEOUT"
        --worker-index "$worker_index"
        --port-base "$PORT_BASE"
        --skip-completed
      )
      if [[ -n "$QCOW2_SHA256" ]]; then
        command+=(--qcow2-sha256 "$QCOW2_SHA256")
      fi
      if [[ "$RETRY_FAILED_ONLY" == "1" ]]; then
        command+=(--retry-failed-only)
      fi
      command+=("${PREFIX_OPTIONS[@]}")
      if [[ "$DRY_RUN" != "1" ]]; then
        command+=(--qcow2-hash-preverified)
      fi
      if [[ "$DRY_RUN" == "1" ]]; then
        printf '[dry-run] worker=%s shard=%s/%s endpoint=%s repeat=%s depth=%s %s %s -> %s:' \
          "$worker_index" "$shard_index" "$SHARD_COUNT" "${endpoint:-unset}" \
          "$repeat" "$depth" "$condition" "$SOURCE_AGENT" "$TARGET_AGENT"
        printf ' %q' "${command[@]}"
        printf '\n'
      else
        status=0
        for ((attempt = 1; attempt <= JOB_MAX_ATTEMPTS; attempt++)); do
          if OPENAI_BASE_URL="$endpoint" "${command[@]}"; then
            status=0
            break
          else
            status=$?
          fi
          if (( status == 3 )); then
            printf 'PROTOCOL_EXCLUSION: worker=%s job=%s repeat=%s depth=%s condition=%s see=%s\n' \
              "$worker_index" "$trajectory_id" "$repeat" "$depth" "$condition" \
              "$result_dir" >&2
            status=0
            break
          fi
          if (( attempt < JOB_MAX_ATTEMPTS )); then
            printf 'WARN: worker=%s job=%s depth=%s condition=%s exited=%s; retrying fresh VM (%s/%s) in %ss\n' \
              "$worker_index" "$trajectory_id" "$depth" "$condition" "$status" \
              "$((attempt + 1))" "$JOB_MAX_ATTEMPTS" "$JOB_RETRY_DELAY" >&2
            sleep "$JOB_RETRY_DELAY"
          fi
        done
        if (( status != 0 )); then
          printf 'ERROR: worker=%s job=%s depth=%s condition=%s failed after %s attempts (last exit=%s); continuing shard\n' \
            "$worker_index" "$trajectory_id" "$depth" "$condition" \
            "$JOB_MAX_ATTEMPTS" "$status" >&2
          job_failures=$((job_failures + 1))
        fi
      fi
      job_index=$((job_index + 1))
    done < "$SELECTION_LIST"
    done
  done
  done
  if (( job_failures > 0 )); then
    printf 'ERROR: worker=%s completed shard with %s exhausted job(s)\n' \
      "$worker_index" "$job_failures" >&2
    return 1
  fi
}

if [[ "$DRY_RUN" == "1" ]]; then
  for worker_index in "${ACTIVE_WORKER_INDICES[@]}"; do
    run_worker "$worker_index"
  done
else
  worker_pids=()
  for worker_index in "${ACTIVE_WORKER_INDICES[@]}"; do
    worker_log="$OUTPUT_ROOT/worker_${worker_index}.log"
    if [[ -n "$WORKER_INDICES" ]]; then
      worker_log="$OUTPUT_ROOT/worker_${worker_index}_resume_$(date +%Y%m%dT%H%M%S).log"
    fi
    run_worker "$worker_index" > "$worker_log" 2>&1 &
    worker_pids+=("$!")
    printf 'Started takeover worker %s shard=%s/%s pid=%s endpoint=%s log=%s\n' \
      "$worker_index" "$((SHARD_OFFSET + worker_index))" "$SHARD_COUNT" "$!" \
      "${ENDPOINT_VALUES[$worker_index]}" "$worker_log"
  done
  worker_failed=0
  for worker_pid in "${worker_pids[@]}"; do
    if ! wait "$worker_pid"; then worker_failed=1; fi
  done
  (( worker_failed == 0 )) || fail "one or more takeover workers failed; inspect $OUTPUT_ROOT/worker_*.log"
fi

if [[ "$RUN_JUDGE" == "1" ]]; then
  staging_dirs=()
  for cell in "${JUDGE_CELLS[@]}"; do
    read -r repeat depth condition <<< "$cell"
    staging="$OUTPUT_ROOT/repeat_${repeat}/_judge_d${depth}_${condition}_ok"
    if [[ "$DRY_RUN" == "1" ]] || python3 "$REPO_ROOT/scripts/judge/takeover_judge_selection.py" \
        "$OUTPUT_ROOT/repeat_${repeat}" "$depth" "$condition" "$staging" > /dev/null; then
      staging_dirs+=("$staging")
    else
      printf 'WARN: no judgeable episodes in repeat=%s depth=%s condition=%s\n' \
        "$repeat" "$depth" "$condition" >&2
    fi
  done
  judge=(bash "$REPO_ROOT/scripts/judge/run_judge.sh" "${staging_dirs[@]}")
  [[ "$FORCE_JUDGE" == "1" ]] && judge+=(--force)
  summarize=(python3 "$REPO_ROOT/scripts/judge/summarize_takeover.py"
    --output-root "$OUTPUT_ROOT" --depths "${DEPTH_VALUES[@]}" --conditions "${CONDITION_VALUES[@]}")
  if [[ "$DRY_RUN" == "1" ]]; then
    printf '[dry-run]'
    printf ' %q' "${judge[@]}"
    printf '\n[dry-run]'
    printf ' %q' "${summarize[@]}"
    printf '\n'
  else
    (( ${#staging_dirs[@]} > 0 )) || fail "no judgeable takeover episodes under $OUTPUT_ROOT"
    date +%s > "$OUTPUT_ROOT/.judge_started_at"
    "${judge[@]}"
    "${summarize[@]}"
  fi
fi
