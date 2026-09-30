#!/usr/bin/env bash
set -euo pipefail
SOURCE_AGENT="${SOURCE_AGENT:-}"
TAKEOVER_AGENT="${TAKEOVER_AGENT:-${TARGET_AGENT:-}}"
DEPTH="${DEPTH:-}"
CONDITION="${CONDITION:-}"
RUN_TAG="${RUN_TAG:-takeover_v1}"
JUDGE_MODEL="${JUDGE_MODEL:-}"
MAX_IMAGES="${MAX_IMAGES:-}"
REASONING_EFFORT="${REASONING_EFFORT:-medium}"
MAX_COMPLETION_TOKENS="${MAX_COMPLETION_TOKENS:-16384}"
CONCURRENCY="${CONCURRENCY:-}"
CSV_LABEL="${CSV_LABEL:-}"
REPEAT="${REPEAT:-}"
CSV_OUT_DIR="${CSV_OUT_DIR:-}"
JUDGE_ARCHIVE_ROOT="${JUDGE_ARCHIVE_ROOT:-}"
SHIP_TO_OSS="${SHIP_TO_OSS:-0}"
FORCE=0
PREPARE_ONLY=0
ERROR_AWARENESS=0
ERROR_AWARENESS_ONLY=0
EAR_SKIP_AGGREGATE=0
EAR_TASK_IDS_FILE=""

usage() {
  cat <<'EOF'
Usage:
  bash scripts/judge/run_takeover_judge.sh \
    --source-agent evocua_32b \
    --takeover-agent qwen3_8_27b \
    --depth 0/5/10/15/20/25 \
    --prompt-condition unaware/notified/diagnosed [options]

Required selectors:
  --source-agent ID       Agent that produced the failure prefix.
  --takeover-agent ID     Agent that continued from the prefix.
  --depth N[/N...]        Takeover depth(s): 0, 5, 10, 15, 20, 25, or all.
  --condition NAME[/...]  unaware, notified, diagnosed, or all.
  --prompt-condition NAME Alias for --condition.

Options:
  --run-tag TAG           Output run tag (default: takeover_v1).
  --judge-model MODEL     Must equal configs/judges/default.yaml model (default: not passed);
                          image limits come from configs/judges/routify_model_registry.json.
  --max-images N          Positive integer <= the registered model image limit.
  --csv-label LABEL       CSV filename prefix; known agent IDs get paper-facing defaults.
  --csv-out-dir DIR       CSV destination (default: artifacts/takeover).
  --archive-root DIR      Persist scores/records/ledger under the independent judge tree.
  --ship                  Upload after successful archiving; requires --archive-root.
  --force                 Rejudge every completed episode in the selected cell.
  --prepare-only          Build/validate staging without API calls or uploads, even with --ship.
  --error-awareness       Also judge Error-Awareness: the first three post-takeover
                          thoughts get a binary verdict, EAR is aggregated by error type x
                          depth over every --depth at once, and each episode's verdict lands
                          in the per-task rubric CSV as the error_aware column.  Runs before
                          the rubric judge so that column is populated in the same pass.
    --error-awareness-only  Judge Error-Awareness and stop; no rubric calls are billed.
    --ear-task-ids-file PATH       Select an explicit newline-delimited EAR task partition.
    --ear-skip-aggregate     Write a partition only; aggregate after all partitions finish.
  -h, --help              Show this help.

The same values can be supplied through SOURCE_AGENT, TAKEOVER_AGENT (or TARGET_AGENT),
DEPTH, CONDITION, RUN_TAG, REPEAT, JUDGE_MODEL, MAX_IMAGES, CSV_LABEL, CSV_OUT_DIR,
JUDGE_ARCHIVE_ROOT, and SHIP_TO_OSS (default: 0; set 1 to opt in).
OSSUTIL overrides the uploader executable (default: ossutil).
Uploads go under $JUDGE_OSS_ROOT (required with --ship),
followed by <model>/<source>/<target>/<condition>/d<N>/.
Only task directories with result.txt=1.0, rubric_bundle.json and no protocol_exclusion.json
are staged.  Without REPEAT, a run with repeat_<k>/ directories is judged repeat by repeat and
summarized once (scripts/judge/summarize_takeover.py).
Multiple depths and conditions run as their Cartesian product, one cell at a time.
EOF
}

need_value() {
  [[ $# -ge 2 && -n "$2" ]] || { echo "FATAL: $1 requires a value" >&2; exit 2; }
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --source-agent) need_value "$@"; SOURCE_AGENT="$2"; shift 2 ;;
    --takeover-agent|--target-agent) need_value "$@"; TAKEOVER_AGENT="$2"; shift 2 ;;
    --depth) need_value "$@"; DEPTH="$2"; shift 2 ;;
    --condition|--prompt-condition) need_value "$@"; CONDITION="$2"; shift 2 ;;
    --run-tag) need_value "$@"; RUN_TAG="$2"; shift 2 ;;
    --judge-model) need_value "$@"; JUDGE_MODEL="$2"; shift 2 ;;
    --max-images) need_value "$@"; MAX_IMAGES="$2"; shift 2 ;;
    --csv-label) need_value "$@"; CSV_LABEL="$2"; shift 2 ;;
    --csv-out-dir) need_value "$@"; CSV_OUT_DIR="$2"; shift 2 ;;
    --archive-root) need_value "$@"; JUDGE_ARCHIVE_ROOT="$2"; shift 2 ;;
    --ship) SHIP_TO_OSS=1; shift ;;
    --force) FORCE=1; shift ;;
    --prepare-only) PREPARE_ONLY=1; shift ;;
    --error-awareness) ERROR_AWARENESS=1; shift ;;
    --error-awareness-only) ERROR_AWARENESS=1; ERROR_AWARENESS_ONLY=1; shift ;;
    --ear-task-ids-file) need_value "$@"; EAR_TASK_IDS_FILE="$2"; shift 2 ;;
    --ear-skip-aggregate) EAR_SKIP_AGGREGATE=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "FATAL: unknown argument: $1" >&2; usage >&2; exit 2 ;;
  esac
done

fail() { echo "FATAL: $1" >&2; exit 1; }
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
REGISTRY_PY="$REPO_ROOT/scripts/judge/judge_model_registry.py"
safe_id='^[A-Za-z0-9_.-]+$'
[[ -n "$SOURCE_AGENT" && "$SOURCE_AGENT" =~ $safe_id ]] || fail "set a valid --source-agent"
[[ -n "$TAKEOVER_AGENT" && "$TAKEOVER_AGENT" =~ $safe_id ]] || fail "set a valid --takeover-agent"
[[ "$SOURCE_AGENT" != "." && "$SOURCE_AGENT" != ".." ]] || fail "set a valid --source-agent"
[[ "$TAKEOVER_AGENT" != "." && "$TAKEOVER_AGENT" != ".." ]] || fail "set a valid --takeover-agent"
[[ "$RUN_TAG" =~ $safe_id ]] || fail "--run-tag contains invalid characters"
[[ -f "$REGISTRY_PY" ]] || fail "Judge model registry resolver does not exist: $REGISTRY_PY"
EFFECTIVE_JUDGE_MODEL="$(python3 "$REGISTRY_PY" ${JUDGE_MODEL:+"$JUDGE_MODEL"} --print-model)" \
  || fail "judge model does not match configs/judges/default.yaml: ${JUDGE_MODEL:-<unset>}"
registry_values="$(python3 "$REGISTRY_PY" "$EFFECTIVE_JUDGE_MODEL")" \
  || fail "Judge model registry rejected model: $EFFECTIVE_JUDGE_MODEL"
IFS=$'\t' read -r JUDGE_PROTOCOL MODEL_MAX_IMAGES JUDGE_ADMISSION \
  MODEL_CONCURRENCY MODEL_TIMEOUT <<< "$registry_values"
MAX_IMAGES="${MAX_IMAGES:-$MODEL_MAX_IMAGES}"
CONCURRENCY="${CONCURRENCY:-$MODEL_CONCURRENCY}"
[[ "$MAX_IMAGES" =~ ^[1-9][0-9]*$ ]] || fail "--max-images must be a positive integer"
(( ${#MAX_IMAGES} <= ${#MODEL_MAX_IMAGES} && MAX_IMAGES <= MODEL_MAX_IMAGES )) \
  || fail "$EFFECTIVE_JUDGE_MODEL --max-images must be <= $MODEL_MAX_IMAGES"
[[ "$MAX_COMPLETION_TOKENS" =~ ^[1-9][0-9]*$ ]] || fail "MAX_COMPLETION_TOKENS must be positive"
[[ "$CONCURRENCY" =~ ^[1-9][0-9]*$ ]] || fail "CONCURRENCY must be positive"
[[ "$SHIP_TO_OSS" == "0" || "$SHIP_TO_OSS" == "1" ]] || fail "SHIP_TO_OSS must be 0 or 1"
[[ "$SHIP_TO_OSS" != "1" || -n "$JUDGE_ARCHIVE_ROOT" ]] || fail "--ship requires --archive-root (or JUDGE_ARCHIVE_ROOT)"
[[ -z "$EAR_TASK_IDS_FILE" || "$ERROR_AWARENESS" == "1" ]] || fail "--ear-task-ids-file requires --error-awareness"

if [[ "$DEPTH" == "all" ]]; then DEPTH="0/5/10/15/20/25"; fi
if [[ "$CONDITION" == "all" ]]; then CONDITION="unaware/notified/diagnosed"; fi
IFS='/,' read -r -a DEPTH_VALUES <<< "$DEPTH"
IFS='/,' read -r -a CONDITION_VALUES_RAW <<< "$CONDITION"
(( ${#DEPTH_VALUES[@]} > 0 )) || fail "set --depth"
(( ${#CONDITION_VALUES_RAW[@]} > 0 )) || fail "set --condition"

for index in "${!DEPTH_VALUES[@]}"; do
  value="${DEPTH_VALUES[$index]//[[:space:]]/}"
  [[ "$value" =~ ^[0-9]+$ ]] || fail "--depth values must be non-negative integers separated by /"
  DEPTH_VALUES[$index]="$value"
done
CONDITION_VALUES=()
for value in "${CONDITION_VALUES_RAW[@]}"; do
  value="${value//[[:space:]]/}"
  case "$value" in
    unaware|notified|diagnosed) ;;
    diagnosis|diagonosis) value="diagnosed" ;;
    *) fail "--condition values must be unaware, notified, or diagnosed, separated by /" ;;
  esac
  CONDITION_VALUES+=("$value")
done

SCRIPT_PATH="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/$(basename "${BASH_SOURCE[0]}")"
JUDGE_MODEL_ARGS=()
[[ -n "$JUDGE_MODEL" ]] && JUDGE_MODEL_ARGS=(--judge-model "$JUDGE_MODEL")

RUN_ROOT="${OUTPUT_ROOT:-$REPO_ROOT/artifacts/model_outputs/takeover/$RUN_TAG/${SOURCE_AGENT}_to_${TAKEOVER_AGENT}}"
if [[ "$RUN_ROOT" != /* ]]; then RUN_ROOT="$REPO_ROOT/$RUN_ROOT"; fi
REPEAT_DIRS=()
if [[ -z "$REPEAT" && "$ERROR_AWARENESS_ONLY" != "1" ]]; then
  for repeat_dir in "$RUN_ROOT"/repeat_*/; do
    [[ -d "$repeat_dir" ]] && REPEAT_DIRS+=("${repeat_dir%/}")
  done
fi
if (( ${#REPEAT_DIRS[@]} > 0 )); then
  forward=(--source-agent "$SOURCE_AGENT" --takeover-agent "$TAKEOVER_AGENT"
    --depth "$(IFS=/; echo "${DEPTH_VALUES[*]}")" --run-tag "$RUN_TAG" "${JUDGE_MODEL_ARGS[@]}")
  [[ -n "$CSV_OUT_DIR" ]] && forward+=(--csv-out-dir "$CSV_OUT_DIR")
  [[ "$FORCE" == "1" ]] && forward+=(--force)
  [[ "$PREPARE_ONLY" == "1" ]] && forward+=(--prepare-only)
  if [[ "$ERROR_AWARENESS" == "1" ]]; then
    for condition_value in "${CONDITION_VALUES[@]}"; do
      ear_all=(bash "$SCRIPT_PATH" "${forward[@]}" --condition "$condition_value"
        --error-awareness-only)
      [[ -n "$CSV_LABEL" ]] && ear_all+=(--csv-label "$CSV_LABEL")
      OUTPUT_ROOT="$RUN_ROOT" "${ear_all[@]}"
    done
  fi
  for repeat_dir in "${REPEAT_DIRS[@]}"; do
    repeat_value="${repeat_dir##*/repeat_}"
    child=(bash "$SCRIPT_PATH" "${forward[@]}"
      --condition "$(IFS=/; echo "${CONDITION_VALUES[*]}")" --max-images "$MAX_IMAGES")
    [[ -n "$CSV_LABEL" ]] && child+=(--csv-label "${CSV_LABEL}_r${repeat_value}")
    [[ -n "$JUDGE_ARCHIVE_ROOT" ]] && child+=(--archive-root "$JUDGE_ARCHIVE_ROOT")
    [[ "$SHIP_TO_OSS" == "1" ]] && child+=(--ship)
    REPEAT="$repeat_value" OUTPUT_ROOT="$repeat_dir" "${child[@]}"
  done
  if [[ "$PREPARE_ONLY" != "1" ]]; then
    python3 "$REPO_ROOT/scripts/judge/summarize_takeover.py" --output-root "$RUN_ROOT" \
      --depths "${DEPTH_VALUES[@]}" --conditions "${CONDITION_VALUES[@]}"
  fi
  echo "[takeover judge] repeats complete: ${#REPEAT_DIRS[@]}"
  exit 0
fi

if [[ "$ERROR_AWARENESS_ONLY" != "1" ]] \
   && (( ${#DEPTH_VALUES[@]} > 1 || ${#CONDITION_VALUES[@]} > 1 )); then
  if [[ "$ERROR_AWARENESS" == "1" ]]; then
    ear_parent=(bash "$SCRIPT_PATH" --source-agent "$SOURCE_AGENT"
      --takeover-agent "$TAKEOVER_AGENT" --depth "$DEPTH" --condition "${CONDITION_VALUES[0]}"
      --run-tag "$RUN_TAG" "${JUDGE_MODEL_ARGS[@]}" --error-awareness-only)
    [[ -n "$CSV_LABEL" ]] && ear_parent+=(--csv-label "$CSV_LABEL")
    [[ -n "$CSV_OUT_DIR" ]] && ear_parent+=(--csv-out-dir "$CSV_OUT_DIR")
    [[ "$FORCE" == "1" ]] && ear_parent+=(--force)
    [[ "$PREPARE_ONLY" == "1" ]] && ear_parent+=(--prepare-only)
    "${ear_parent[@]}"
    ERROR_AWARENESS=0
  fi
  printf '[takeover judge] batch: %s depth(s) x %s condition(s) = %s cell(s)\n' \
    "${#DEPTH_VALUES[@]}" "${#CONDITION_VALUES[@]}" \
    "$(( ${#DEPTH_VALUES[@]} * ${#CONDITION_VALUES[@]} ))"
  for depth_value in "${DEPTH_VALUES[@]}"; do
    for condition_value in "${CONDITION_VALUES[@]}"; do
      child=(bash "$SCRIPT_PATH"
        --source-agent "$SOURCE_AGENT"
        --takeover-agent "$TAKEOVER_AGENT"
        --depth "$depth_value"
        --condition "$condition_value"
        --run-tag "$RUN_TAG"
        "${JUDGE_MODEL_ARGS[@]}"
        --max-images "$MAX_IMAGES")
      [[ -n "$CSV_LABEL" ]] && child+=(--csv-label "$CSV_LABEL")
      [[ -n "$CSV_OUT_DIR" ]] && child+=(--csv-out-dir "$CSV_OUT_DIR")
      [[ -n "$JUDGE_ARCHIVE_ROOT" ]] && child+=(--archive-root "$JUDGE_ARCHIVE_ROOT")
      [[ "$SHIP_TO_OSS" == "1" ]] && child+=(--ship)
      [[ "$FORCE" == "1" ]] && child+=(--force)
      [[ "$PREPARE_ONLY" == "1" ]] && child+=(--prepare-only)
      "${child[@]}"
    done
  done
  echo "[takeover judge] batch complete"
  exit 0
fi

DEPTH="${DEPTH_VALUES[0]}"
CONDITION="${CONDITION_VALUES[0]}"

CSV_OUT_DIR="${CSV_OUT_DIR:-$REPO_ROOT/artifacts/takeover}"
if [[ "$CSV_OUT_DIR" != /* ]]; then CSV_OUT_DIR="$REPO_ROOT/$CSV_OUT_DIR"; fi

display_agent_id() {
  case "$1" in
    evocua_32b) echo "evocua32b" ;;
    qwen3_8_27b) echo "qwen3.8_27B" ;;
    qwen3_6_27b) echo "qwen3.6_27B" ;;
    qwen3_5_35b_a3b) echo "qwen3.5_35B_A3B" ;;
    holo_3_1_35b_a3b) echo "holo3.1_35B_A3B" ;;
    opencua_72b) echo "opencua72b" ;;
    kimi_k3) echo "kimi_k3" ;;
    gpt_5_5) echo "gpt-5.5" ;;
    *) echo "$1" ;;
  esac
}
if [[ -z "$CSV_LABEL" ]]; then
  CSV_LABEL="$(display_agent_id "$SOURCE_AGENT")_to_$(display_agent_id "$TAKEOVER_AGENT")${REPEAT:+_r$REPEAT}"
fi
[[ "$CSV_LABEL" =~ ^[A-Za-z0-9_.-]+$ ]] || fail "--csv-label contains invalid characters"

OUTPUT_ROOT="${OUTPUT_ROOT:-$REPO_ROOT/artifacts/model_outputs/takeover/$RUN_TAG/${SOURCE_AGENT}_to_${TAKEOVER_AGENT}${REPEAT:+/repeat_$REPEAT}}"
if [[ "$OUTPUT_ROOT" != /* ]]; then OUTPUT_ROOT="$REPO_ROOT/$OUTPUT_ROOT"; fi
if [[ "$ERROR_AWARENESS" == "1" ]]; then
  (( ${#CONDITION_VALUES[@]} == 1 )) || fail "--error-awareness takes exactly one --condition"
  EAR_SCRIPT="$REPO_ROOT/scripts/judge/error_awareness.py"
  [[ -f "$EAR_SCRIPT" ]] || fail "EAR judge does not exist: $EAR_SCRIPT"
  ear=(python3 "$EAR_SCRIPT"
    --run-dir "$OUTPUT_ROOT"
    --depth "$(IFS=/; echo "${DEPTH_VALUES[*]}")"
    --condition "$CONDITION"
    "${JUDGE_MODEL_ARGS[@]}"
    --label "$CSV_LABEL"
    --out-dir "$CSV_OUT_DIR"
    --concurrency "$CONCURRENCY")
  [[ "$FORCE" == "1" ]] && ear+=(--force)
  [[ -n "$EAR_TASK_IDS_FILE" ]] && ear+=(--task-ids-file "$EAR_TASK_IDS_FILE")
  [[ "$EAR_SKIP_AGGREGATE" == "1" ]] && ear+=(--skip-aggregate)
  if [[ "$PREPARE_ONLY" == "1" ]]; then
    "${ear[@]}" --prepare-only
    if [[ "$ERROR_AWARENESS_ONLY" == "1" ]]; then exit 0; fi
    ERROR_AWARENESS=0
  fi
  if [[ "$PREPARE_ONLY" != "1" ]]; then
  if [[ -n "${DERAIL_CRED_ENV:-}" && -f "$DERAIL_CRED_ENV" ]]; then
    set -a; . "$DERAIL_CRED_ENV"; set +a
  fi
  if [[ -z "${OPENAI_API_KEY:-}" ]]; then
    [[ -f "$REPO_ROOT/.env" ]] || fail "$REPO_ROOT/.env 不存在，且环境里没有 OPENAI_API_KEY"
    OPENAI_API_KEY="$(sed -n 's/^OPENAI_API_KEY[=:][[:space:]]*//p' "$REPO_ROOT/.env" \
      | tr -d '"'"'"' ' | head -1)"
    export OPENAI_API_KEY
  fi
  [[ -n "${OPENAI_API_KEY:-}" ]] || fail "OPENAI_API_KEY 为空（.env 里没读到）"
  if [[ -z "${OPENAI_BASE_URL:-}" ]]; then unset OPENAI_BASE_URL; fi
  export MYPCBENCH_OSWORLD_JUDGE_REASONING_EFFORT="$REASONING_EFFORT"
  export MYPCBENCH_OSWORLD_JUDGE_MAX_COMPLETION_TOKENS="$MAX_COMPLETION_TOKENS"
  export DERAIL_OPENAI_API_APPROVED=1
  export DERAIL_OPENAI_API_PURPOSE=takeover_error_awareness
  printf '[takeover judge] error-awareness: model=%s reasoning=%s depths=%s condition=%s\n' \
    "$EFFECTIVE_JUDGE_MODEL" "$REASONING_EFFORT" "${DEPTH_VALUES[*]}" "$CONDITION"
  if [[ "$ERROR_AWARENESS" == "1" ]]; then "${ear[@]}"; fi
  ear_status=$?
  (( ear_status == 0 )) || exit "$ear_status"
  fi
  if [[ "$ERROR_AWARENESS_ONLY" == "1" ]]; then exit 0; fi
fi

CELL="$OUTPUT_ROOT/depth_${DEPTH}/$CONDITION"
STAGING="$OUTPUT_ROOT/_judge_d${DEPTH}_${CONDITION}_ok"
[[ -d "$CELL" ]] || fail "takeover cell does not exist: $CELL"

python3 "$REPO_ROOT/scripts/takeover/bundle_prefix.py" "$CELL" \
  || fail "could not prepend the replayed prefix to the rubric bundles under $CELL"

read -r COMPLETED EXCLUDED JUDGED EXISTING_MODELS < <(
  python3 "$REPO_ROOT/scripts/judge/takeover_judge_selection.py" \
    "$OUTPUT_ROOT" "$DEPTH" "$CONDITION" "$STAGING"
)

PENDING=$((COMPLETED - JUDGED))
printf '[takeover judge] cell=%s\n' "$CELL"
printf '[takeover judge] source=%s takeover=%s depth=%s condition=%s\n' \
  "$SOURCE_AGENT" "$TAKEOVER_AGENT" "$DEPTH" "$CONDITION"
printf '[takeover judge] runner-clean=%s excluded=%s judged=%s pending=%s staging=%s\n' \
  "$COMPLETED" "$EXCLUDED" "$JUDGED" "$PENDING" "$STAGING"
printf '[takeover judge] model=%s max_images=%s reasoning=%s max_completion_tokens=%s concurrency=%s\n' \
  "$EFFECTIVE_JUDGE_MODEL" "$MAX_IMAGES" "$REASONING_EFFORT" "$MAX_COMPLETION_TOKENS" "$CONCURRENCY"
printf '[takeover judge] protocol=%s admission=%s registry_timeout=%ss\n' \
  "$JUDGE_PROTOCOL" "$JUDGE_ADMISSION" "$MODEL_TIMEOUT"
printf '[takeover judge] csv=%s/%s_d%s_%s.csv\n' \
  "$CSV_OUT_DIR" "$CSV_LABEL" "$DEPTH" "$CONDITION"

if [[ "$EXISTING_MODELS" != "-" && ",$EXISTING_MODELS," != *",$EFFECTIVE_JUDGE_MODEL,"* ]]; then
  [[ "$FORCE" == "1" ]] || fail \
    "cell already contains judge model(s) [$EXISTING_MODELS], not $EFFECTIVE_JUDGE_MODEL; use the matching model or --force to rejudge the whole cell"
fi
if [[ "$EXISTING_MODELS" == *,* && "$FORCE" != "1" ]]; then
  fail "cell already mixes judge models [$EXISTING_MODELS]; use --force to normalize it"
fi

if [[ "$PREPARE_ONLY" == "1" ]]; then
  echo "[takeover judge] prepare-only: no API calls made"
  exit 0
fi

export MYPCBENCH_OSWORLD_JUDGE_REASONING_EFFORT="$REASONING_EFFORT"
export MYPCBENCH_OSWORLD_JUDGE_MAX_COMPLETION_TOKENS="$MAX_COMPLETION_TOKENS"
export MYPCBENCH_OSWORLD_JUDGE_CONCURRENCY="$CONCURRENCY"
export DERAIL_OPENAI_API_APPROVED=1
export DERAIL_OPENAI_API_PURPOSE=trajectory_rubric_judge
judge=(bash "$REPO_ROOT/scripts/judge/run_judge.sh" "$STAGING")
[[ "$FORCE" == "1" ]] && judge+=(--force)
DERAIL_RUN_JUDGE_MODEL="$JUDGE_MODEL" \
DERAIL_RUN_JUDGE_MAX_IMAGES="$MAX_IMAGES" \
  "${judge[@]}"

echo "[takeover judge] complete: $CELL"
CSV_SCRIPT="$REPO_ROOT/scripts/judge/rubric_csv.py"
[[ -f "$CSV_SCRIPT" ]] || fail "CSV builder does not exist: $CSV_SCRIPT"
python3 "$CSV_SCRIPT" \
  --run-dir "$OUTPUT_ROOT" \
  --depth "$DEPTH" \
  --condition "$CONDITION" \
  --label "$CSV_LABEL" \
  --out-dir "$CSV_OUT_DIR"
printf '[takeover judge] CSV complete: %s/%s_d%s_%s.csv\n' \
  "$CSV_OUT_DIR" "$CSV_LABEL" "$DEPTH" "$CONDITION"
if [[ -n "$JUDGE_ARCHIVE_ROOT" ]]; then
  python3 "$REPO_ROOT/scripts/judge/archive.py" \
    --cell "$CELL" \
    --scores "$STAGING/scores.json" \
    --csv "$CSV_OUT_DIR/${CSV_LABEL}_d${DEPTH}_${CONDITION}.csv" \
    --archive-root "$JUDGE_ARCHIVE_ROOT" \
    --judge-model "$EFFECTIVE_JUDGE_MODEL" \
    --source-agent "$SOURCE_AGENT" \
    --target-agent "$TAKEOVER_AGENT" \
    --condition "$CONDITION" \
    --depth "$DEPTH" \
    --exclusions "$OUTPUT_ROOT/judge_exclusions.json"
  if [[ "$SHIP_TO_OSS" == "1" ]]; then
    ARCHIVE_DEPTH="$(python3 -c 'import sys; print(int(sys.argv[1]))' "$DEPTH")"
    ARCHIVE_SUFFIX="$EFFECTIVE_JUDGE_MODEL/$SOURCE_AGENT/$TAKEOVER_AGENT/$CONDITION/d$ARCHIVE_DEPTH"
    python3 "$REPO_ROOT/scripts/judge/ship_archive.py" \
      --source "${JUDGE_ARCHIVE_ROOT%/}/$ARCHIVE_SUFFIX" \
      --destination "${JUDGE_OSS_ROOT:?set JUDGE_OSS_ROOT to ship judge archives}${ARCHIVE_SUFFIX}/" \
      --ossutil "${OSSUTIL:-ossutil}"
  fi
fi
