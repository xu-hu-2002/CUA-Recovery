#!/usr/bin/env bash

set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
HF_CACHE_ROOT="${HF_CACHE_ROOT:-${HF_HOME:-$HOME/.cache/huggingface}}"
SERVING_LOG_ROOT="${SERVING_LOG_ROOT:-${REPO_ROOT}/artifacts/serving_logs}"
SERVING_PORT_BASE="${SERVING_PORT_BASE:-8000}"
SERVING_CPUSET="${SERVING_CPUSET:-0-15,32-47}"
DRY_RUN="${DRY_RUN:-0}"

VLLM_019_IMAGE="vllm/vllm-openai@sha256:7a0f0fdd2771464b6976625c2b2d5dd46f566aa00fbc53eceab86ef50883da90"
VLLM_012_IMAGE="vllm/vllm-openai@sha256:f2309d913a07da49ea20b2a694703f4cfcb5ad8e7437ec0f26145479ac01e002"
EVOCUA_IMAGE="recovery/evocua-vllm:0.11.0-transformers4.57.3"
MODELS_LOCK="${REPO_ROOT}/configs/models.lock.yaml"
EVOCUA_DOCKERFILE="${REPO_ROOT}/serving/evocua-vllm.Dockerfile"
EVOCUA_IMAGE_ID="${EVOCUA_IMAGE_ID:-}"

usage() {
  cat <<'EOF'
Usage:
  bash scripts/collection/serve_open_source.sh start <agent_id> [replicas]
  bash scripts/collection/serve_open_source.sh stop <agent_id>
  bash scripts/collection/serve_open_source.sh status [agent_id]
  bash scripts/collection/serve_open_source.sh attach <agent_id>

agent_id:
  qwen3_5_35b_a3b | rerail_35b_a3b | evocua_32b | opencua_72b

Replicas are derived, not declared: usable GPUs / tensor_parallel_size, where the
TP size comes from configs/agents/<agent_id>.yaml and the GPU count from
nvidia-smi (override with RECOVERY_GPU_COUNT / CUDA_VISIBLE_DEVICES). On eight GPUs
that is 4 endpoints for a TP2 model and 1 for TP8. The derived value is also the
ceiling — asking for more is an error. RECOVERY_MAX_ENDPOINTS caps it (default 4,
matching the SERVING_PORT_BASE..+3 port lane).

This script never calls an endpoint or OpenAI API. After the server logs say ready,
run endpoint/single-task probes separately and only with the required approval.
EOF
}

die() {
  printf 'error: %s\n' "$*" >&2
  exit 1
}

info() {
  printf '[RECOVERY serving] %s\n' "$*"
}

# shellcheck source=../lib/collection_config.sh
source "${SCRIPT_DIR}/../lib/collection_config.sh"

max_replicas() {
  agent_vm_count "$REPO_ROOT" "$1"
}

cpuset_slice() {
  local spec="$1" index="$2" total="$3"
  awk -v spec="$spec" -v idx="$index" -v total="$total" '
    BEGIN {
      if (total <= 0) exit 1
      n = split(spec, parts, ",")
      out = ""
      for (i = 1; i <= n; i++) {
        if (parts[i] == "") continue
        if (parts[i] ~ /-/) {
          split(parts[i], bound, "-")
          lo = bound[1] + 0; hi = bound[2] + 0
        } else {
          lo = parts[i] + 0; hi = lo
        }
        size = hi - lo + 1
        per = int(size / total)
        if (per < 1) per = 1
        first = lo + idx * per
        if (first > hi) first = hi - per + 1
        if (first < lo) first = lo
        last = first + per - 1
        if (last > hi) last = hi
        chunk = (first == last) ? first : first "-" last
        out = out (out == "" ? "" : ",") chunk
        emitted = 1
      }
      if (!emitted) exit 1
      print out
    }
  '
}

agent_gpu_group() {
  local agent_id="$1" index="$2" tp
  tp="$(agent_tensor_parallel_size "$REPO_ROOT" "$agent_id")" \
    || die "configs/agents/${agent_id}.yaml is missing tensor_parallel_size"
  awk -v devices="$(usable_gpu_indices)" -v tp="$tp" -v idx="$index" '
    BEGIN {
      n = split(devices, all, ",")
      out = ""
      for (i = idx * tp; i < (idx + 1) * tp && i < n; i++)
        out = out (out == "" ? "" : ",") all[i + 1]
      print out
    }
  '
}

short_name() {
  case "$1" in
    qwen3_5_35b_a3b) printf 'qwen35\n' ;;
    rerail_35b_a3b) printf 'rerail\n' ;;
    evocua_32b) printf 'evocua\n' ;;
    opencua_72b) printf 'opencua72b\n' ;;
    *) die "Unknown agent_id: $1" ;;
  esac
}

session_name() {
  printf 'recovery-serve-%s\n' "$(short_name "$1")"
}

assert_tools() {
  command -v docker >/dev/null 2>&1 || die "docker not found"
  command -v tmux >/dev/null 2>&1 || die "tmux not found"
  command -v nvidia-smi >/dev/null 2>&1 || die "nvidia-smi not found"
  [[ -d "$HF_CACHE_ROOT" ]] || die "Hugging Face cache does not exist: ${HF_CACHE_ROOT}"
}

assert_evocua_image() {
  docker image inspect "$EVOCUA_IMAGE" >/dev/null 2>&1 || die \
    "EvoCUA runtime image not found locally: ${EVOCUA_IMAGE}
     It is not on any registry; build it first: bash serving/build_evocua_image.sh"

  local want_dockerfile actual_dockerfile
  want_dockerfile="$(config_scalar "$MODELS_LOCK" dockerfile_sha256 || true)"
  [[ -n "$want_dockerfile" ]] || die "models.lock.yaml has no dockerfile_sha256"
  [[ -f "$EVOCUA_DOCKERFILE" ]] || die "${EVOCUA_DOCKERFILE} not found"
  actual_dockerfile="$(sha256sum "$EVOCUA_DOCKERFILE" | awk '{print $1}')"
  [[ "$actual_dockerfile" == "$want_dockerfile" ]] || die \
    "EvoCUA Dockerfile does not match the lock: expected ${want_dockerfile}, got ${actual_dockerfile}"

  local base_digest base_layers image_layers
  base_digest="$(config_scalar "$MODELS_LOCK" base_amd64_digest || true)"
  [[ -n "$base_digest" ]] || die "models.lock.yaml has no base_amd64_digest"
  base_layers="$(docker image inspect "vllm/vllm-openai@${base_digest}" \
    --format '{{range .RootFS.Layers}}{{.}}
{{end}}' 2>/dev/null || true)"
  if [[ -z "$base_layers" ]]; then
    info "Warning: frozen base image vllm/vllm-openai@${base_digest} not present locally; skipping base check"
  else
    image_layers="$(docker image inspect "$EVOCUA_IMAGE" \
      --format '{{range .RootFS.Layers}}{{.}}
{{end}}')"
    [[ "${image_layers}"$'\n' == "${base_layers}"$'\n'* ]] || die \
      "${EVOCUA_IMAGE} was not built from frozen base ${base_digest}; rebuild: bash serving/build_evocua_image.sh"
  fi

  if [[ -n "$EVOCUA_IMAGE_ID" ]]; then
    local actual_id
    actual_id="$(docker image inspect "$EVOCUA_IMAGE" --format '{{.Id}}')"
    [[ "$actual_id" == "$EVOCUA_IMAGE_ID" ]] || die \
      "EvoCUA image ID does not match the explicit pin: expected ${EVOCUA_IMAGE_ID}, got ${actual_id}"
  fi
}

assert_runtime_image() {
  local agent_id="$1"
  local image
  case "$agent_id" in
    qwen3_5_35b_a3b | rerail_35b_a3b) image="$VLLM_019_IMAGE" ;;
    opencua_72b) image="$VLLM_012_IMAGE" ;;
    evocua_32b)
      assert_evocua_image
      return 0
      ;;
  esac
  docker image inspect "$image" >/dev/null 2>&1 || die "Frozen runtime image does not exist: ${image}"
}

agent_checkpoint() {
  local value name
  value="$(config_scalar_required "$(agent_config_path "$REPO_ROOT" "$1")" checkpoint)" || exit 1
  if [[ "$value" =~ ^\$\{([A-Za-z_][A-Za-z0-9_]*)\}$ ]]; then
    name="${BASH_REMATCH[1]}"
    value="${!name:-}"
    [[ -n "$value" ]] || die "checkpoint of ${1} needs environment variable ${name}"
  fi
  printf '%s\n' "$value"
}

snapshot_path() {
  local config checkpoint revision
  config="$(agent_config_path "$REPO_ROOT" "$1")"
  checkpoint="$(agent_checkpoint "$1")" || exit 1
  if [[ "$checkpoint" == /* ]]; then
    printf '%s\n' "$checkpoint"
    return
  fi
  revision="$(config_scalar_required "$config" revision)" || exit 1
  printf '%s/models--%s/snapshots/%s\n' \
    "$HF_CACHE_ROOT/hub" "${checkpoint//\//--}" "$revision"
}

append_model_args() {
  local agent_id="$1" config tp checkpoint revision
  config="$(agent_config_path "$REPO_ROOT" "$agent_id")"
  tp="$(config_scalar_required "$config" tensor_parallel_size)" || exit 1
  checkpoint="$(agent_checkpoint "$agent_id")" || exit 1
  revision="$(config_scalar "$config" revision || true)"
  DOCKER_VOLUME_ARGS=()
  [[ "$checkpoint" == /* ]] && DOCKER_VOLUME_ARGS=(--volume "${checkpoint}:${checkpoint}:ro")
  case "$agent_id" in
    qwen3_5_35b_a3b | rerail_35b_a3b)
      MODEL_ARGS=(
        "$checkpoint"
        ${revision:+--revision "$revision"}
        --served-model-name "$checkpoint"
        --tensor-parallel-size "$tp" --dtype bfloat16
        --gpu-memory-utilization 0.92 --max-model-len 49152 --max-num-seqs 1
        --disable-custom-all-reduce
        --enable-auto-tool-choice --tool-call-parser hermes
        --host 0.0.0.0 --port 8000 --trust-remote-code
      )
      RUNTIME_IMAGE="$VLLM_019_IMAGE"
      ;;
    evocua_32b)
      MODEL_ARGS=(
        --model "$checkpoint"
        --revision "$revision"
        --served-model-name EvoCUA --tensor-parallel-size "$tp" --dtype bfloat16
        --max-model-len 49152 --gpu-memory-utilization 0.95 --max-num-seqs 1
        --disable-custom-all-reduce --host 0.0.0.0 --port 8000
      )
      RUNTIME_IMAGE="$EVOCUA_IMAGE"
      ;;
    opencua_72b)
      MODEL_ARGS=(
        "$checkpoint"
        --revision "$revision"
        --trust-remote-code --tensor-parallel-size "$tp" --dtype bfloat16
        --gpu-memory-utilization 0.95
        --served-model-name opencua-72b --host 0.0.0.0 --port 8000
      )
      RUNTIME_IMAGE="$VLLM_012_IMAGE"
      ;;
  esac
}

print_base_urls() {
  local agent_id="$1"
  local replicas="$2"
  local variable
  case "$agent_id" in
    qwen3_5_35b_a3b) variable=QWEN35_BASE_URLS ;;
    rerail_35b_a3b) variable=RERAIL_BASE_URLS ;;
    evocua_32b) variable=EVOCUA_BASE_URLS ;;
    opencua_72b) variable=OPENCUA_BASE_URLS ;;
  esac
  local urls=""
  local index
  for ((index = 0; index < replicas; index++)); do
    [[ -z "$urls" ]] || urls+=","
    urls+="http://127.0.0.1:$((SERVING_PORT_BASE + index))/v1"
  done
  info "Endpoint lines for local .env (no secrets):"
  printf '%s="%s"\n' "$variable" "$urls"
}

start_model() {
  local agent_id="$1"
  local replicas="${2:-}"
  local short
  local session
  short="$(short_name "$agent_id")"
  session="$(session_name "$agent_id")"
  local ceiling
  ceiling="$(max_replicas "$agent_id")" || die "Cannot derive endpoint count for ${agent_id}"
  [[ -z "$replicas" ]] && replicas="$ceiling"
  [[ "$replicas" =~ ^[1-9][0-9]*$ ]] || die "replicas must be a positive integer"
  if (( replicas > ceiling )); then
    die "${agent_id} can run at most ${ceiling} endpoints on this host
     (usable GPUs $(usable_gpu_count) / tensor_parallel_size $(agent_tensor_parallel_size "$REPO_ROOT" "$agent_id"),
     capped by RECOVERY_MAX_ENDPOINTS=${RECOVERY_MAX_ENDPOINTS:-4}); requested ${replicas} exceeds it."
  fi
  info "${agent_id}: usable GPUs $(usable_gpu_count), TP $(agent_tensor_parallel_size "$REPO_ROOT" "$agent_id"), starting ${replicas} endpoints"

  assert_tools
  assert_runtime_image "$agent_id"
  local snapshot
  snapshot="$(snapshot_path "$agent_id")"
  [[ -d "$snapshot" ]] || die "Frozen model snapshot does not exist: ${snapshot}"
  tmux has-session -t "=${session}" 2>/dev/null && die "tmux session already exists: ${session}"
  if [[ "${ALLOW_BUSY_GPU:-0}" != "1" ]] && \
    nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits | grep -q '[0-9]'; then
    die "GPU compute process detected; refusing to take the GPU. Once ownership is confirmed, set ALLOW_BUSY_GPU=1"
  fi

  append_model_args "$agent_id"
  local started_at
  started_at="$(date -u +%Y%m%dT%H%M%SZ)"
  local log_dir="${SERVING_LOG_ROOT}/${started_at}_${short}"
  if [[ "$DRY_RUN" != "1" ]]; then
    mkdir -p "$log_dir"
    chmod 700 "$log_dir"
  fi

  local index
  local probe_port
  for ((index = 0; index < replicas; index++)); do
    probe_port=$((SERVING_PORT_BASE + index))
    if ss -ltn "sport = :${probe_port}" 2>/dev/null | grep -q ":${probe_port}"; then
      die "Port ${probe_port} is in use (${agent_id} needs ${SERVING_PORT_BASE}..$((SERVING_PORT_BASE + replicas - 1))); use another range: SERVING_PORT_BASE=8100"
    fi
  done

  for ((index = 0; index < replicas; index++)); do
    local port=$((SERVING_PORT_BASE + index))
    local container="recovery-serve-${short}-r${index}"
    local gpu_request
    local cpu_set
    gpu_request="\"device=$(agent_gpu_group "$agent_id" "$index")\""
    cpu_set="$(cpuset_slice "$SERVING_CPUSET" "$index" "$replicas")" \
      || die "SERVING_CPUSET=${SERVING_CPUSET} cannot be split into ${replicas} parts"
    local -a docker_command=(
      docker run --rm --name "$container"
      --label recovery.project=RECOVERY --label "recovery.agent_id=${agent_id}"
      --gpus "$gpu_request" --ipc host --cpuset-cpus "$cpu_set"
      --publish "127.0.0.1:${port}:8000"
      --volume "${HF_CACHE_ROOT}:/root/.cache/huggingface" "${DOCKER_VOLUME_ARGS[@]}"
      --env PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
      "$RUNTIME_IMAGE" "${MODEL_ARGS[@]}"
    )
    local docker_text
    local log_path="${log_dir}/endpoint_${index}.log"
    printf -v docker_text '%q ' "${docker_command[@]}"
    local window_text="set -o pipefail; ${docker_text}2>&1 | tee $(printf '%q' "$log_path"); status=\${PIPESTATUS[0]}; printf '\\n[RECOVERY serving] container exit=%s\\n' \"\$status\"; exec bash"
    if [[ "$DRY_RUN" == "1" ]]; then
      printf 'tmux window r%d: %s\n' "$index" "$docker_text"
    elif (( index == 0 )); then
      tmux new-session -d -s "$session" -n "r${index}" -c "$REPO_ROOT" \
        "bash -lc $(printf '%q' "$window_text")"
    else
      tmux new-window -d -t "$session" -n "r${index}" -c "$REPO_ROOT" \
        "bash -lc $(printf '%q' "$window_text")"
    fi
  done
  if [[ "$DRY_RUN" != "1" ]]; then
    tmux new-window -d -t "$session" -n monitor -c "$REPO_ROOT" "watch -n 2 nvidia-smi"
    tmux select-window -t "${session}:r0"
    info "Started tmux session ${session}; logs: ${log_dir}"
    info "View: tmux attach -t ${session}"
  else
    info "dry-run: no tmux session, container or endpoint created, and no API request"
  fi
  print_base_urls "$agent_id" "$replicas"
  info "This script did not call /v1/models or the OpenAI API; wait for the log to show server ready"
}

stop_model() {
  local agent_id="$1"
  assert_tools
  local session
  session="$(session_name "$agent_id")"
  local -a containers=()
  mapfile -t containers < <(
    docker ps --filter label=recovery.project=RECOVERY \
      --filter "label=recovery.agent_id=${agent_id}" --format '{{.ID}}'
  )
  if (( ${#containers[@]} )); then
    docker stop --time 30 "${containers[@]}"
  fi
  if tmux has-session -t "=${session}" 2>/dev/null; then
    tmux kill-session -t "=${session}"
  fi
  info "Stopped exactly matching container/session of ${agent_id}"
}

status_model() {
  local agent_id="${1:-}"
  assert_tools
  if [[ -n "$agent_id" ]]; then
    local session
    session="$(session_name "$agent_id")"
    tmux has-session -t "=${session}" 2>/dev/null \
      && info "tmux: ${session} running" || info "tmux: ${session} absent"
    docker ps --filter "label=recovery.agent_id=${agent_id}" \
      --format 'table {{.Names}}\t{{.Status}}\t{{.Ports}}'
  else
    tmux list-sessions 2>/dev/null | grep '^recovery-serve-' || true
    docker ps --filter label=recovery.project=RECOVERY \
      --format 'table {{.Names}}\t{{.Status}}\t{{.Ports}}'
  fi
  nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu \
    --format=csv,noheader
}

attach_model() {
  local agent_id="$1"
  command -v tmux >/dev/null 2>&1 || die "tmux not found"
  local session
  session="$(session_name "$agent_id")"
  tmux has-session -t "=${session}" 2>/dev/null || die "tmux session does not exist: ${session}"
  exec tmux attach -t "=${session}"
}

command_name="${1:-help}"
case "$command_name" in
  start)
    [[ $# -ge 2 && $# -le 3 ]] || { usage; exit 2; }
    start_model "$2" "${3:-}"
    ;;
  stop)
    [[ $# -eq 2 ]] || { usage; exit 2; }
    stop_model "$2"
    ;;
  status)
    [[ $# -le 2 ]] || { usage; exit 2; }
    status_model "${2:-}"
    ;;
  attach)
    [[ $# -eq 2 ]] || { usage; exit 2; }
    attach_model "$2"
    ;;
  help|-h|--help)
    usage
    ;;
  *)
    usage
    exit 2
    ;;
esac
