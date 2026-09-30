#!/usr/bin/env bash
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/../.." && pwd)"

ALLOW_DIRTY_SUBMIT="${ALLOW_DIRTY_SUBMIT:-0}"
if [[ "$ALLOW_DIRTY_SUBMIT" != "1" ]] \
  && { ! git -C "$REPO" diff --quiet --ignore-submodules -- \
    || ! git -C "$REPO" diff --cached --quiet --ignore-submodules --; }; then
  echo "ERROR: tracked changes are not committed; commit them before submitting, or set ALLOW_DIRTY_SUBMIT=1 for an explicitly non-reproducible run" >&2
  exit 2
fi

SUBMIT_LOCK_DIR="$REPO/.submit_derail_nebula.lock"
_release_submit_lock() {
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

MCUA_ENV_FILE="${MCUA_ENV_FILE:-$HOME/Desktop/Long_horizon-MCUA/.env}"
if [[ -z "${ROCK_API_KEY:-}" && -f "$MCUA_ENV_FILE" ]]; then
  set -a; . "$MCUA_ENV_FILE" 2>/dev/null || true; set +a
  echo "[submit-opencua-smoke] loaded ROCK creds from $MCUA_ENV_FILE"
fi

SMOKE_PHASE="${SMOKE_PHASE:-probe}"
case "$SMOKE_PHASE" in probe|full) ;; *)
  echo "ERROR: SMOKE_PHASE 只能是 probe|full" >&2; exit 2 ;;
esac
OPENCUA_MODEL="${OPENCUA_MODEL:-opencua-72b}"
SERVE_PORT="${SERVE_PORT:-8000}"
SERVE_READY_TIMEOUT="${SERVE_READY_TIMEOUT:-5400}"
SANDBOX_REACH_PROBE="${SANDBOX_REACH_PROBE:-0}"

AGENT_ID="${AGENT_ID:-opencua_72b}"
SHARD_FILE="${SHARD_FILE:-configs/mypcbench_task_shards/smoke_one.json}"
COLLECTION_ID="${COLLECTION_ID:-smoke_opencua72b_nebula_$(date -u +%Y%m%dT%H%M%SZ)}"
REPEATS="${REPEATS:-}"
NUM_VMS_OVERRIDE="1"
MAX_STEPS="${MAX_STEPS:-8}"
TASK_TIMEOUT="${TASK_TIMEOUT:-}"
TIMEOUT_PER_VM="${TIMEOUT_PER_VM:-259200}"

QUEUE="${QUEUE:-<nebula-queue>}"
NEBULA_PROJECT="${NEBULA_PROJECT:-<nebula-project>}"
PLATFORM_ALGO_NAME="${PLATFORM_ALGO_NAME:-pytorch2100}"
WORKER_COUNT="${WORKER_COUNT:-1}"
DERAIL_WORKLOAD="${DERAIL_WORKLOAD:-collection}"
TAKEOVER_SOURCE_AGENT="${TAKEOVER_SOURCE_AGENT:-opencua_72b}"
TAKEOVER_TARGET_AGENT="${TAKEOVER_TARGET_AGENT:-opencua_72b}"
TAKEOVER_ANNOTATOR="${TAKEOVER_ANNOTATOR:-dingyi}"
TAKEOVER_BUILD_DIR="${TAKEOVER_BUILD_DIR:-artifacts/derail_builds/opencua72b_dingyi_takeover}"
TAKEOVER_DEPTH="${TAKEOVER_DEPTH:-0}"
TAKEOVER_CONDITION="${TAKEOVER_CONDITION:-unaware}"
TAKEOVER_SHARD_COUNT="${TAKEOVER_SHARD_COUNT:-1}"
TAKEOVER_SHARD_OFFSET="${TAKEOVER_SHARD_OFFSET:-0}"
TAKEOVER_TRAJECTORY_ID_FILTER="${TAKEOVER_TRAJECTORY_ID_FILTER:-}"
TAKEOVER_TRAJECTORY_ID_FILE="${TAKEOVER_TRAJECTORY_ID_FILE:-}"
OPENCUA_MAX_TOKENS_OVERRIDE="${OPENCUA_MAX_TOKENS_OVERRIDE:-}"
CPU="${CPU:-3200}"
GPU_UNITS="${GPU_UNITS:-800}"
MEM="${MEM:-131072}"

ROCK_BASE_URL="${ROCK_BASE_URL:-<rock-endpoint>}"
ROCK_SANDBOX_IMAGE="${ROCK_SANDBOX_IMAGE:?set ROCK_SANDBOX_IMAGE}"
ROCK_CLUSTER="${ROCK_CLUSTER:?set ROCK_CLUSTER}"
ROCK_DISK="${ROCK_DISK:-100g}"
ROCK_MEMORY="${ROCK_MEMORY:-16g}"
ROCK_CPUS="${ROCK_CPUS:-8}"
ROCK_USER_ID="${ROCK_USER_ID:-<rock-user-id>}"
ROCK_EXPERIMENT_ID="${ROCK_EXPERIMENT_ID:-derail-mypcbench}"
ROCK_RUN_TIMEOUT="${ROCK_RUN_TIMEOUT:-108000}"
ROCK_AUTO_CLEAR_SECONDS="${ROCK_AUTO_CLEAR_SECONDS:-172800}"

OSS_ENDPOINT="${OSS_ENDPOINT:?set OSS_ENDPOINT}"
OSS_BUCKET="${OSS_BUCKET:-<oss-bucket>}"
OSS_PREFIX="${OSS_PREFIX:-<oss-prefix>}"
OSS_ASSETS_URI="${OSS_ASSETS_URI:-oss://${OSS_BUCKET}/${OSS_PREFIX}/DERAIL/assets/mypcbench-vm}"
OSS_HARNESS_URI="${OSS_HARNESS_URI:-oss://${OSS_BUCKET}/${OSS_PREFIX}/DERAIL/assets/mypcbench-harness/mypcbench-caf9c754-takeover-native-v5.tar.gz}"
OSS_RESULTS_ROOT="${OSS_RESULTS_ROOT:-oss://${OSS_BUCKET}/${OSS_PREFIX}/DERAIL/results/raw/mypcbench}"
OSS_SMOKE_EVIDENCE_DIR="${OSS_SMOKE_EVIDENCE_DIR:-/data/oss_bucket_0/${OSS_PREFIX}/DERAIL/results/smoke/opencua72b}"
OPENCUA_WEIGHTS_OSS_DIR="${OPENCUA_WEIGHTS_OSS_DIR:-/data/oss_bucket_0/${OSS_PREFIX}/MCUA/OpenCUA-72B}"
LOCAL_MODEL_CACHE_DIR="${LOCAL_MODEL_CACHE_DIR:-}"
MODEL_WARM_WORKERS="${MODEL_WARM_WORKERS:-4}"
SHIP_INCREMENT_SECONDS="${SHIP_INCREMENT_SECONDS:-1800}"
TAKEOVER_TOKENIZE_BASE_URL="${TAKEOVER_TOKENIZE_BASE_URL:-}"
TAKEOVER_TOKENIZE_MODE="${TAKEOVER_TOKENIZE_MODE:-vllm}"
TAKEOVER_CONTEXT_CAP="${TAKEOVER_CONTEXT_CAP:-0}"
MYPCBENCH_QWEN_MAX_TOKENS="${MYPCBENCH_QWEN_MAX_TOKENS:-}"
MYPCBENCH_QWEN_HISTORY_N="${MYPCBENCH_QWEN_HISTORY_N:-}"
MYPCBENCH_QWEN_CONTEXT_POLICY="${MYPCBENCH_QWEN_CONTEXT_POLICY:-}"
VLLM_EXTRA_ARGS="${VLLM_EXTRA_ARGS:-}"
if [[ "$DERAIL_WORKLOAD" == "takeover" ]]; then
  _default_proxy_results="/data/oss_bucket_0/${OSS_PREFIX}/DERAIL/results/takeover/failure_prefix_v1/${TAKEOVER_SOURCE_AGENT}/${TAKEOVER_TARGET_AGENT}/${TAKEOVER_CONDITION}/d${TAKEOVER_DEPTH}"
else
  _default_proxy_results="/data/oss_bucket_0/${OSS_PREFIX}/DERAIL/results/raw/mypcbench"
fi
PROXY_RESULTS_OSS_MOUNT="${PROXY_RESULTS_OSS_MOUNT:-${_default_proxy_results}}"
OSS_OPENCUA_SNAPSHOT_URI="${OSS_OPENCUA_SNAPSHOT_URI:-oss://${OSS_BUCKET}/${OSS_PREFIX}/DERAIL/assets/opencua-osworld/opencua-osworld-091f5ef.tar.gz}"
OSS_EVOCUA_SNAPSHOT_URI="${OSS_EVOCUA_SNAPSHOT_URI:-oss://${OSS_BUCKET}/${OSS_PREFIX}/DERAIL/assets/evocua/evocua-4a0ad5f.tar.gz}"
TAKEOVER_INPUT_ROOT="${TAKEOVER_INPUT_ROOT:-/data/oss_bucket_0/${OSS_PREFIX}/DERAIL/takeover_inputs/failure_prefix_v1_full_v3}"

DRY_RUN="${DRY_RUN:-1}"
ALLOW_PLACEHOLDER="${ALLOW_PLACEHOLDER:-0}"

if [[ "$ALLOW_PLACEHOLDER" != "1" ]]; then
  for _req in QUEUE NEBULA_PROJECT ROCK_BASE_URL ROCK_USER_ID OSS_BUCKET OSS_PREFIX; do
    _val="${!_req}"
    if [[ -z "$_val" || "$_val" == "<"*">" ]]; then
      echo "ERROR: $_req 未设置（仍是占位符）；先 set -a; . \$DERAIL_CRED_ENV; set +a 再跑（DRY_RUN 也需真实值；临时绕过设 ALLOW_PLACEHOLDER=1）" >&2
      exit 2
    fi
  done
fi

if [[ "$DRY_RUN" == "0" ]]; then
  if [[ "${SANDBOX_REACH_PROBE}" == "1" || "$SMOKE_PHASE" == "full" ]]; then
    [[ -n "${ROCK_API_KEY:-}" ]] || {
      echo "ERROR: ROCK_API_KEY 未设置（建沙箱会 401）；或设 SANDBOX_REACH_PROBE=0 且 SMOKE_PHASE=probe" >&2
      exit 2; }
  fi
fi

[[ -f "$REPO/$SHARD_FILE" ]] || { echo "ERROR: 找不到 shard 文件：$REPO/$SHARD_FILE" >&2; exit 2; }

SWEEP_STALE_SANDBOXES="${SWEEP_STALE_SANDBOXES:-1}"
if [[ "$SWEEP_STALE_SANDBOXES" == "1" && "$DRY_RUN" == "0" \
      && ( "${SANDBOX_REACH_PROBE}" == "1" || "$SMOKE_PHASE" == "full" ) ]]; then
  ROCKCLI_BIN="$(command -v rockcli 2>/dev/null || find "$HOME/.nvm/versions/node" -name rockcli -type l 2>/dev/null | head -1)"
  if [[ -n "$ROCKCLI_BIN" ]]; then
    stale=$("$ROCKCLI_BIN" expr "$ROCK_EXPERIMENT_ID" sandboxes 2>/dev/null \
              | sed "s/$(printf '\033')\[[0-9;]*m//g" \
              | awk '/RUNNING|PENDING/ {print $1}' | grep -E '^[0-9a-f]{32}$' || true)
    for sb in $stale; do
      echo "[submit-opencua-smoke] sweeping stale sandbox $sb"
      "$ROCKCLI_BIN" sandbox "$sb" stop >/dev/null 2>&1 || true
    done
    [[ -z "$stale" ]] && echo "[submit-opencua-smoke] no stale sandboxes in $ROCK_EXPERIMENT_ID"
  else
    echo "[submit-opencua-smoke] WARNING: rockcli not found; skipping stale-sandbox sweep" >&2
  fi
fi

OSS_ID="${OSS_ACCESS_ID:-$(sed -n 's/^accessKeyID=//p' "${HOME}/.ossutilconfig" 2>/dev/null || true)}"
OSS_KEY="${OSS_ACCESS_KEY:-$(sed -n 's/^accessKeySecret=//p' "${HOME}/.ossutilconfig" 2>/dev/null || true)}"
[[ -n "${OSS_ID}" && -n "${OSS_KEY}" ]] \
  || { echo "ERROR: OSS creds missing (env OSS_ACCESS_ID/KEY or ~/.ossutilconfig)" >&2; exit 2; }

CLUSTER_FILE=".cluster_opencua_smoke.$$.json"
RUN_ENV_FILE=".opencua_run.env"
PROPS_FILE=".props_opencua_smoke.$$"
DERAIL_SECRET_NAME=".derail_opencua_secrets.$$.sh"
DERAIL_SECRET_OSS="${OSS_PREFIX}/DERAIL/tmp/${DERAIL_SECRET_NAME}"
SUBMITTED=0
STAGING_DIR=""
trap 'rm -f "$REPO/$CLUSTER_FILE" "$REPO/$RUN_ENV_FILE" "$REPO/$PROPS_FILE"; \
  if [[ -n "${STAGING_DIR:-}" && -d "$STAGING_DIR" ]]; then rm -rf "$STAGING_DIR"; fi; \
  if [[ "${SUBMITTED:-0}" != "1" ]]; then \
    "${OSSUTIL_BIN:-/usr/local/bin/ossutil}" -e "${OSS_ENDPOINT}" \
      -i "${OSS_ID:-}" -k "${OSS_KEY:-}" rm -f "oss://${OSS_BUCKET}/${DERAIL_SECRET_OSS:-__none__}" \
      >/dev/null 2>&1 || true; \
  fi; _release_submit_lock' EXIT

cat > "$REPO/$CLUSTER_FILE" <<JSON
{"worker": {"cpu": ${CPU}, "gpu": ${GPU_UNITS}, "memory": ${MEM}}}
JSON
umask 077
printf 'oss_access_id=%s\noss_access_key=%s\noss_bucket=%s\noss_endpoint=%s\n' \
  "${OSS_ID}" "${OSS_KEY}" "${OSS_BUCKET}" "${OSS_ENDPOINT}" > "$REPO/$PROPS_FILE"
umask 022

OSSUTIL_BIN="$(command -v ossutil 2>/dev/null || echo "/usr/local/bin/ossutil")"
if [[ -x "${OSSUTIL_BIN:-/nonexistent}" ]]; then
  _tmp_secret="$(mktemp)"; chmod 600 "$_tmp_secret"
  printf 'export OSS_ACCESS_ID=%q\nexport OSS_ACCESS_KEY=%q\nexport OSS_ENDPOINT=%q\n' \
    "${OSS_ID}" "${OSS_KEY}" "oss-${OSS_ENDPOINT%%.*}.aliyuncs.com" > "$_tmp_secret"
  [[ -n "${ROCK_API_KEY:-}" ]] && printf 'export ROCK_API_KEY=%q\n' "$ROCK_API_KEY" >> "$_tmp_secret"
  if "${OSSUTIL_BIN}" -e "${OSS_ENDPOINT}" -i "${OSS_ID}" -k "${OSS_KEY}" \
      cp -f "$_tmp_secret" "oss://${OSS_BUCKET}/${DERAIL_SECRET_OSS}" >/dev/null 2>&1; then
    echo "[submit-opencua-smoke] staged secrets on mount: oss://${OSS_BUCKET}/${DERAIL_SECRET_OSS}"
  elif [[ "$DRY_RUN" == "1" ]]; then
    echo "[submit-opencua-smoke] WARNING: DRY_RUN 下 secret staging 失败（多半是占位符 OSS_PREFIX），跳过；真提交前必须修" >&2
  else
    echo "ERROR: failed to stage secrets on mount; aborting before submit" >&2
    exit 1
  fi
  rm -f "$_tmp_secret"
else
  echo "ERROR: ossutil not found locally; cannot stage secrets" >&2
  exit 2
fi
echo "[submit-opencua-smoke] cluster.json -> $REPO/$CLUSTER_FILE : $(cat "$REPO/$CLUSTER_FILE")"

cat > "$REPO/$RUN_ENV_FILE" <<ENVF
export SMOKE_PHASE='${SMOKE_PHASE}'
export OPENCUA_MODEL='${OPENCUA_MODEL}'
export SERVE_PORT='${SERVE_PORT}'
export SERVE_READY_TIMEOUT='${SERVE_READY_TIMEOUT}'
export OPENCUA_WEIGHTS_OSS_DIR='${OPENCUA_WEIGHTS_OSS_DIR}'
export LOCAL_MODEL_CACHE_DIR='${LOCAL_MODEL_CACHE_DIR}'
export MODEL_WARM_WORKERS='${MODEL_WARM_WORKERS}'
export SHIP_INCREMENT_SECONDS='${SHIP_INCREMENT_SECONDS}'
export SANDBOX_REACH_PROBE='${SANDBOX_REACH_PROBE}'
export OSS_SMOKE_EVIDENCE_DIR='${OSS_SMOKE_EVIDENCE_DIR}'
export AGENT_ID='${AGENT_ID}'
export SHARD_FILE='${SHARD_FILE}'
export COLLECTION_ID='${COLLECTION_ID}'
export REPEATS='${REPEATS}'
export REPEAT_START_INDEX='${REPEAT_START_INDEX:-1}'
export NUM_VMS_OVERRIDE='${NUM_VMS_OVERRIDE}'
export MAX_STEPS='${MAX_STEPS}'
export TASK_TIMEOUT='${TASK_TIMEOUT}'
export TIMEOUT_PER_VM='${TIMEOUT_PER_VM}'
export FORMAL_COLLECTION='${FORMAL_COLLECTION:-0}'
export ALLOW_CONFIG_OVERRIDE='${ALLOW_CONFIG_OVERRIDE:-0}'
export RESUME_SEED='${RESUME_SEED:-0}'
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
export OSS_OPENCUA_SNAPSHOT_URI='${OSS_OPENCUA_SNAPSHOT_URI}'
export OSS_EVOCUA_SNAPSHOT_URI='${OSS_EVOCUA_SNAPSHOT_URI}'
export DERAIL_ROCK_TOPOLOGY='proxy'
export DERAIL_WORKLOAD='${DERAIL_WORKLOAD}'
export TAKEOVER_INPUT_ROOT='${TAKEOVER_INPUT_ROOT}'
export TAKEOVER_SOURCE_AGENT='${TAKEOVER_SOURCE_AGENT}'
export TAKEOVER_TARGET_AGENT='${TAKEOVER_TARGET_AGENT}'
export TAKEOVER_ANNOTATOR='${TAKEOVER_ANNOTATOR}'
export TAKEOVER_BUILD_DIR='${TAKEOVER_BUILD_DIR}'
export TAKEOVER_DEPTH='${TAKEOVER_DEPTH}'
export TAKEOVER_CONDITION='${TAKEOVER_CONDITION}'
export TAKEOVER_SHARD_COUNT='${TAKEOVER_SHARD_COUNT}'
export TAKEOVER_SHARD_OFFSET='${TAKEOVER_SHARD_OFFSET}'
export TAKEOVER_TRAJECTORY_ID_FILTER='${TAKEOVER_TRAJECTORY_ID_FILTER}'
export TAKEOVER_TRAJECTORY_ID_FILE='${TAKEOVER_TRAJECTORY_ID_FILE}'
export OPENCUA_MAX_TOKENS_OVERRIDE='${OPENCUA_MAX_TOKENS_OVERRIDE}'
export TAKEOVER_TOKENIZE_BASE_URL='${TAKEOVER_TOKENIZE_BASE_URL}'
export TAKEOVER_TOKENIZE_MODE='${TAKEOVER_TOKENIZE_MODE}'
export TAKEOVER_CONTEXT_CAP='${TAKEOVER_CONTEXT_CAP}'
export MYPCBENCH_QWEN_MAX_TOKENS='${MYPCBENCH_QWEN_MAX_TOKENS}'
export MYPCBENCH_QWEN_HISTORY_N='${MYPCBENCH_QWEN_HISTORY_N}'
export MYPCBENCH_QWEN_CONTEXT_POLICY='${MYPCBENCH_QWEN_CONTEXT_POLICY}'
export VLLM_EXTRA_ARGS='${VLLM_EXTRA_ARGS}'
export PROXY_RESULTS_OSS_MOUNT='${PROXY_RESULTS_OSS_MOUNT}'
export DERAIL_SECRET_MOUNT='/data/oss_bucket_0/${DERAIL_SECRET_OSS}'
ENVF
chmod 600 "$REPO/$RUN_ENV_FILE"
echo "[submit-opencua-smoke] run-config -> $REPO/$RUN_ENV_FILE (phase=$SMOKE_PHASE shard=$SHARD_FILE)"

STAGING_DIR="$(mktemp -d "${TMPDIR:-/tmp}/derail-nebula-submit.XXXXXX")"
git -C "$REPO" archive HEAD | tar -x -C "$STAGING_DIR"
cp "$REPO/$CLUSTER_FILE" "$REPO/$RUN_ENV_FILE" "$REPO/$PROPS_FILE" "$STAGING_DIR/"
chmod 600 "$STAGING_DIR/$PROPS_FILE"
echo "[submit-opencua-smoke] clean staging package: $(du -sh "$STAGING_DIR" | awk '{print $1}')"

ENTRY="scripts/rock/entry_derail_opencua_nebula.py"
JOB_NAME="derail-${AGENT_ID//_/-}-${SMOKE_PHASE}-$(date -u +%Y%m%d%H%M)"
if [[ -n "${JOB_NAME_SUFFIX:-}" ]]; then
  JOB_NAME="${JOB_NAME}-${JOB_NAME_SUFFIX}"
fi
IGNORE_LIST="${IGNORE_LIST:-third_party/*,takeovewr_annotation/*,artifacts/raw_rollouts/*,artifacts/derail_builds/*,artifacts/model_outputs/*,artifacts/takeover/bundles/*,artifacts/phase5/*,data/synthesis/*,draft/*,runs/*,.git/*,.venv*,.env,docs/*,*.log,*.pyc,__pycache__/*,.props_*,.cluster_*}"

NEBULACTL_BIN="$(command -v nebulactl 2>/dev/null \
  || { [[ -x "$HOME/bin/nebulactl" ]] && echo "$HOME/bin/nebulactl"; } \
  || command -v nebula-cli 2>/dev/null \
  || find "$HOME/.nvm/versions/node" -name nebula-cli -type l 2>/dev/null | head -1)"
[[ -x "${NEBULACTL_BIN:-/nonexistent}" ]] \
  || { echo "ERROR: nebulactl/nebula-cli not found（提交入口缺失）" >&2; exit 2; }

cmd=( "$NEBULACTL_BIN" run mdl )
if [[ "${NEBULACTL_FORCE:-0}" == "1" ]]; then
  cmd+=( --force )
fi
cmd+=(
  --algo_name="${PLATFORM_ALGO_NAME}"
  --property_file="${STAGING_DIR}/${PROPS_FILE}"
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
  if (cd "$STAGING_DIR" && "${cmd[@]}"); then
    SUBMITTED=1
    echo "[submit-opencua-smoke] submitted. 监控: /oss-job-monitor"
  else
    echo "ERROR: nebulactl submission failed" >&2
    exit 1
  fi
else
  echo
  echo "[submit-opencua-smoke] DRY_RUN=1 — command printed only. Re-run with DRY_RUN=0 to submit."
fi
