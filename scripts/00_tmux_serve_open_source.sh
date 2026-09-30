#!/usr/bin/env bash
# 手动管理六个开源 baseline 的冻结 vLLM endpoint。只启动/停止本地服务；
# 不发 API 请求，也不自动启动 MyPCBench collection。

set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
HF_CACHE_ROOT="${HF_CACHE_ROOT:-${HF_HOME:-$HOME/.cache/huggingface}}"
SERVING_LOG_ROOT="${SERVING_LOG_ROOT:-${REPO_ROOT}/artifacts/serving_logs}"
SERVING_PORT_BASE="${SERVING_PORT_BASE:-8000}"
SERVING_CPUSET="${SERVING_CPUSET:-0-15,32-47}"
DRY_RUN="${DRY_RUN:-0}"

VLLM_019_IMAGE="vllm/vllm-openai@sha256:7a0f0fdd2771464b6976625c2b2d5dd46f566aa00fbc53eceab86ef50883da90"
VLLM_012_IMAGE="vllm/vllm-openai@sha256:f2309d913a07da49ea20b2a694703f4cfcb5ad8e7437ec0f26145479ac01e002"
EVOCUA_IMAGE="derail/evocua-vllm:0.11.0-transformers4.57.3"
MODELS_LOCK="${REPO_ROOT}/configs/models.lock.yaml"
EVOCUA_DOCKERFILE="${REPO_ROOT}/serving/evocua-vllm.Dockerfile"
# 可选的逐字节 pin：只有当你想断言本机镜像与某一次具体构建完全一致时才设置
# （值是那次 `docker image inspect --format '{{.Id}}'` 的输出）。默认为空 —— 见
# assert_evocua_image 的注释，本地构建的 image ID 不是跨机器可复现的量。
EVOCUA_IMAGE_ID="${EVOCUA_IMAGE_ID:-}"

usage() {
  cat <<'EOF'
Usage:
  bash scripts/00_tmux_serve_open_source.sh start <agent_id> [replicas]
  bash scripts/00_tmux_serve_open_source.sh stop <agent_id>
  bash scripts/00_tmux_serve_open_source.sh status [agent_id]
  bash scripts/00_tmux_serve_open_source.sh attach <agent_id>

agent_id:
  qwen3_5_35b_a3b | evocua_32b | qwen3_6_27b | qwen3_8_27b |
  holo_3_1_35b_a3b | opencua_72b

Replicas are derived, not declared: usable GPUs / tensor_parallel_size, where the
TP size comes from configs/agents/<agent_id>.yaml and the GPU count from
nvidia-smi (override with DERAIL_GPU_COUNT / CUDA_VISIBLE_DEVICES). On eight GPUs
that is 4 endpoints for a TP2 model and 1 for TP8. The derived value is also the
ceiling — asking for more is an error. DERAIL_MAX_ENDPOINTS caps it (default 4,
matching the SERVING_PORT_BASE..+3 port lane).

This script never calls an endpoint or OpenAI API. After the server logs say ready,
run endpoint/single-task probes separately and only with the required approval.
EOF
}

die() {
  printf '错误：%s\n' "$*" >&2
  exit 1
}

info() {
  printf '[DERAIL serving] %s\n' "$*"
}

# shellcheck source=lib/collection_config.sh
source "${SCRIPT_DIR}/lib/collection_config.sh"

# 该 agent 能起几个 endpoint = 可用 GPU 数 / yaml 声明的 tensor_parallel_size。
# 与 01 的 VM 并发度是同一个数字（runner 按 VM round-robin 分配 endpoint）。
max_replicas() {
  agent_vm_count "$REPO_ROOT" "$1"
}

# 把 SERVING_CPUSET（形如 "0-15,32-47"）按 endpoint 序号切成 total 份。
# 以前是四组写死的 cpuset，replica 数一旦不是 4 就对不上。
#
# 关键是**逐段**切而不是先展开成一个大列表：0-15 是物理核、32-47 是它们的 SMT
# 兄弟（32 是 0 的兄弟）。整体切的话 r0 会拿到 0-7、r2 拿到 32-39，两个 replica
# 落在同一批物理核上互相抢。逐段各取 1/total 再拼起来，就还原成原来那种
# "0-3,32-35" 的配对。
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

# 一个 endpoint 占几张卡，以及本轮实际可用的卡号，供切分 GPU 组用。
agent_gpu_group() {
  local agent_id="$1" index="$2" tp
  tp="$(agent_tensor_parallel_size "$REPO_ROOT" "$agent_id")" \
    || die "configs/agents/${agent_id}.yaml 缺少 tensor_parallel_size"
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
    evocua_32b) printf 'evocua\n' ;;
    qwen3_6_27b) printf 'qwen36\n' ;;
    qwen3_8_27b) printf 'qwen38\n' ;;
    holo_3_1_35b_a3b) printf 'holo31\n' ;;
    opencua_72b) printf 'opencua72b\n' ;;
    *) die "未知 agent_id：$1" ;;
  esac
}

session_name() {
  printf 'derail-serve-%s\n' "$(short_name "$1")"
}

assert_tools() {
  command -v docker >/dev/null 2>&1 || die "找不到 docker"
  command -v tmux >/dev/null 2>&1 || die "找不到 tmux"
  command -v nvidia-smi >/dev/null 2>&1 || die "找不到 nvidia-smi"
  [[ -d "$HF_CACHE_ROOT" ]] || die "Hugging Face cache 不存在：${HF_CACHE_ROOT}"
}

# EvoCUA 的 runtime 不在任何 registry 上，必须本地构建（serving/evocua-vllm.Dockerfile
# 在冻结的 vLLM 0.11.0 base 上装 transformers==4.57.3）。
#
# 这里不能拿 image ID 当身份：`docker build` 出来的 ID 是本机构建产物的内容哈希，
# 受 buildkit 版本、pip 解析到的 wheel、时间戳等影响，另一台机器按同一个 Dockerfile
# 重建必然得到不同的 ID。之前把某一次构建的 ID 写死在脚本里，等于任何 clone 下来
# 重建的人都会被 die 掉。
#
# 换成校验真正可复现的两样东西：
#   1. 配方本身 —— Dockerfile 的 sha256 要等于 models.lock.yaml 记的值；
#   2. 构建基座 —— 镜像的 layer 前缀必须就是冻结 base image 的全部 layer，
#      也就是它确实是从锁定的那个 digest FROM 出来的。
# 想额外断言与某次具体构建逐字节一致，设 EVOCUA_IMAGE_ID 即可，默认不设。
assert_evocua_image() {
  docker image inspect "$EVOCUA_IMAGE" >/dev/null 2>&1 || die \
    "本地没有 EvoCUA runtime image：${EVOCUA_IMAGE}
     它不在任何 registry 上，请先构建：bash serving/build_evocua_image.sh"

  local want_dockerfile actual_dockerfile
  want_dockerfile="$(config_scalar "$MODELS_LOCK" dockerfile_sha256 || true)"
  [[ -n "$want_dockerfile" ]] || die "models.lock.yaml 里没有 dockerfile_sha256"
  [[ -f "$EVOCUA_DOCKERFILE" ]] || die "找不到 ${EVOCUA_DOCKERFILE}"
  actual_dockerfile="$(sha256sum "$EVOCUA_DOCKERFILE" | awk '{print $1}')"
  [[ "$actual_dockerfile" == "$want_dockerfile" ]] || die \
    "EvoCUA Dockerfile 与 lock 不一致：期望 ${want_dockerfile}，实际 ${actual_dockerfile}"

  local base_digest base_layers image_layers
  base_digest="$(config_scalar "$MODELS_LOCK" base_amd64_digest || true)"
  [[ -n "$base_digest" ]] || die "models.lock.yaml 里没有 base_amd64_digest"
  base_layers="$(docker image inspect "vllm/vllm-openai@${base_digest}" \
    --format '{{range .RootFS.Layers}}{{.}}
{{end}}' 2>/dev/null || true)"
  if [[ -z "$base_layers" ]]; then
    info "警告：本机没有冻结 base image vllm/vllm-openai@${base_digest}，跳过 base 校验"
  else
    image_layers="$(docker image inspect "$EVOCUA_IMAGE" \
      --format '{{range .RootFS.Layers}}{{.}}
{{end}}')"
    [[ "${image_layers}"$'\n' == "${base_layers}"$'\n'* ]] || die \
      "${EVOCUA_IMAGE} 不是从冻结 base ${base_digest} 构建的；请重新构建：bash serving/build_evocua_image.sh"
  fi

  if [[ -n "$EVOCUA_IMAGE_ID" ]]; then
    local actual_id
    actual_id="$(docker image inspect "$EVOCUA_IMAGE" --format '{{.Id}}')"
    [[ "$actual_id" == "$EVOCUA_IMAGE_ID" ]] || die \
      "EvoCUA image ID 与显式 pin 不符：期望 ${EVOCUA_IMAGE_ID}，实际 ${actual_id}"
  fi
}

assert_runtime_image() {
  local agent_id="$1"
  local image
  case "$agent_id" in
    qwen3_5_35b_a3b|qwen3_6_27b|qwen3_8_27b|holo_3_1_35b_a3b) image="$VLLM_019_IMAGE" ;;
    opencua_72b) image="$VLLM_012_IMAGE" ;;
    evocua_32b)
      assert_evocua_image
      return 0
      ;;
  esac
  docker image inspect "$image" >/dev/null 2>&1 || die "冻结 runtime image 不存在：${image}"
}

# 冻结快照在 HF cache 里的路径。checkpoint 与 revision 只写在
# configs/agents/<agent_id>.yaml，这里按 HF 的目录命名规则拼出来 —— 以前是五份
# 抄写，改一个 revision 要记得同时改 yaml、这里和 append_model_args 三处。
snapshot_path() {
  local config checkpoint revision
  config="$(agent_config_path "$REPO_ROOT" "$1")"
  checkpoint="$(config_scalar_required "$config" checkpoint)" || exit 1
  revision="$(config_scalar_required "$config" revision)" || exit 1
  printf '%s/models--%s/snapshots/%s\n' \
    "$HF_CACHE_ROOT/hub" "${checkpoint//\//--}" "$revision"
}

# A6000 没有 NVLink，vLLM 的自定义 all-reduce 内核在 CUDA graph 捕获时会以
# "custom_all_reduce.cuh:455 'invalid argument'" 崩掉。EvoCUA 早先就撞过这个
# （见 models.lock.yaml 的 evocua_32b.probe_serving_overrides），用
# --disable-custom-all-reduce 解决；另外三个模型当时改用了 --enforce-eager，
# 那同样能绕过崩溃，但代价是关掉 CUDA graph。
#
# 2026-08-04 在本机实测 Qwen3.5-35B-A3B（GPU 2,3，其余参数完全一致）：
#   --enforce-eager              11.2 tok/s
#   --disable-custom-all-reduce  98.7 tok/s   → 8.8 倍
# MoE 模型每个 token 要调度大量小的 expert kernel，eager 模式下的 kernel launch
# 开销被成倍放大，所以惩罚远大于稠密模型。KV cache 只从 241,824 掉到 239,712
# tokens，显存不是问题。
#
# 代价：贪心解码不再与 eager 模式逐字一致（实测前 705 字符相同后开始同义改写），
# 属于浮点归约顺序差异。同一套配置内部仍然稳定。
# qwen3_5 / qwen3_6 / evocua / holo 仍使用 49152。Qwen3.8 takeover 的
# native prefix 明显更长：d0/unaware 已观测最长输入是 58,494 tokens，
# 再加 2,048-token completion 后 49,152 与 65,536 都缺少足够的 long-horizon
# 增长空间。Qwen3.8 因此单独用 98,304；模型原生上限是 262,144，
# 本机 A6000 TP2 的 vLLM 启动日志实测 KV cache 可容纳 126,224 tokens。
#
# 起因：全部 scaffold 的上下文策略对齐到上游的 history_n=100 / image_max=20 之后，
# 单步 prompt 实测约 27k tokens（qwen3_5 的 agent_metadata:
# original_prompt_tokens=27429），12288 装不下。
#   - 走上游 scaffold 的 qwen3_5 在 12288 下不会报错，但它的 tokenize_oldest_first_v1
#     会把 27429 一路压到 8159，实际窗口只剩名义值的三成 —— 名义 100/20、实跑却
#     不是，横比时那是个说不清的混淆项。
#   - 走 DERAIL scaffold 的 qwen3_6 / holo_3_1 没有那层裁剪，12288 下是每步 400。
#
# 提到 49152 之后两边都拿得到完整窗口。代价是 KV cache 显存；三个模型都是 TP2 +
# --max-num-seqs 1，evocua_32b 的实测说 native 262144 需要 32 GiB KV/GPU，按比例
# 49152 约 6 GiB，留得出。起服务后请看一眼日志里的 KV cache 行再跑正式采集。
append_model_args() {
  local agent_id="$1" config tp checkpoint revision
  # 权重身份与 TP 尺寸都读 configs/agents/<agent_id>.yaml，不在这里重抄一份。
  config="$(agent_config_path "$REPO_ROOT" "$agent_id")"
  tp="$(config_scalar_required "$config" tensor_parallel_size)" || exit 1
  checkpoint="$(config_scalar_required "$config" checkpoint)" || exit 1
  revision="$(config_scalar_required "$config" revision)" || exit 1
  case "$agent_id" in
    qwen3_5_35b_a3b)
      MODEL_ARGS=(
        "$checkpoint"
        --revision "$revision"
        --served-model-name "$checkpoint"
        --tensor-parallel-size "$tp" --dtype bfloat16
        --gpu-memory-utilization 0.92 --max-model-len 49152 --max-num-seqs 1
        --disable-custom-all-reduce
        --enable-auto-tool-choice --tool-call-parser hermes
        --host 0.0.0.0 --port 8000 --trust-remote-code
      )
      RUNTIME_IMAGE="$VLLM_019_IMAGE"
      ;;
    qwen3_6_27b)
      MODEL_ARGS=(
        "$checkpoint"
        --revision "$revision"
        --served-model-name "$checkpoint"
        --tensor-parallel-size "$tp" --dtype bfloat16
        --max-model-len 49152 --max-num-seqs 1 --gpu-memory-utilization 0.90
        --disable-custom-all-reduce
        --reasoning-parser qwen3 --enable-auto-tool-choice
        --tool-call-parser qwen3_coder --host 0.0.0.0 --port 8000
      )
      RUNTIME_IMAGE="$VLLM_019_IMAGE"
      ;;
    qwen3_8_27b)
      MODEL_ARGS=(
        "$checkpoint"
        --revision "$revision"
        --served-model-name "$checkpoint"
        --tensor-parallel-size "$tp" --dtype bfloat16
        --max-model-len 98304 --max-num-seqs 1 --gpu-memory-utilization 0.90
        --disable-custom-all-reduce
        --reasoning-parser qwen3 --enable-auto-tool-choice
        --tool-call-parser qwen3_coder --host 0.0.0.0 --port 8000
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
    holo_3_1_35b_a3b)
      MODEL_ARGS=(
        "$checkpoint"
        --revision "$revision"
        --served-model-name "$checkpoint"
        --tensor-parallel-size "$tp" --dtype bfloat16
        --max-model-len 49152 --max-num-seqs 1 --gpu-memory-utilization 0.90
        --disable-custom-all-reduce
        --reasoning-parser qwen3 --enable-auto-tool-choice
        --tool-call-parser qwen3_coder --host 0.0.0.0 --port 8000
        --trust-remote-code
      )
      RUNTIME_IMAGE="$VLLM_019_IMAGE"
      ;;
    opencua_72b)
      # 这里**没有** --max-model-len，和官方 TP8 命令一致，是有意的：OpenCUA-72B 的
      # config.json 写着 max_position_embeddings=32768，vLLM 就按这个推导窗口。
      # （同一份 config 里的 text_config.pretraining_sequence_length=131072 不是
      # vLLM 读的量，别拿它估显存。）2026-08-13 试图给 TP4 配 --max-model-len 98304
      # 时 vLLM 直接 ValidationError：超过 derived max_model_len 只能靠
      # VLLM_ALLOW_LONG_MAX_MODEL_LEN=1 强开，而 RoPE 外推会出 nan —— 不要开。
      # 结论是 TP4 根本不需要压窗口：32768 token × 0.3125 MiB/token
      # （2×80 层×8 KV head×128 dim×2 B）只要 10 GiB KV，四张卡装得下，
      # 窗口与 TP8 逐字相同。
      #
      # --gpu-memory-utilization 0.95 是 TP4 才加的（官方 TP8 用默认 0.9）：四张卡
      # 总预算 192 GiB，0.9 只剩 172.8 GiB，扣掉 137 GiB 权重后留给激活+KV 的
      # 35.8 GiB 偏紧；0.95 给到 45.4 GiB。这个旋钮只影响显存分配，不改变输出。
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
    qwen3_6_27b) variable=QWEN36_BASE_URLS ;;
    qwen3_8_27b) variable=QWEN38_BASE_URLS ;;
    evocua_32b) variable=EVOCUA_BASE_URLS ;;
    holo_3_1_35b_a3b) variable=HOLO31_BASE_URLS ;;
    opencua_72b) variable=OPENCUA_BASE_URLS ;;
  esac
  local urls=""
  local index
  for ((index = 0; index < replicas; index++)); do
    [[ -z "$urls" ]] || urls+=","
    urls+="http://127.0.0.1:$((SERVING_PORT_BASE + index))/v1"
  done
  info "写入本地 .env 的 endpoint 行（不含密钥）："
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
  ceiling="$(max_replicas "$agent_id")" || die "无法推导 ${agent_id} 的 endpoint 数"
  [[ -z "$replicas" ]] && replicas="$ceiling"
  [[ "$replicas" =~ ^[1-9][0-9]*$ ]] || die "replicas 必须是正整数"
  if (( replicas > ceiling )); then
    die "${agent_id} 在本机最多起 ${ceiling} 个 endpoint
     （可用 GPU $(usable_gpu_count) 张 / tensor_parallel_size $(agent_tensor_parallel_size "$REPO_ROOT" "$agent_id")，
     再受 DERAIL_MAX_ENDPOINTS=${DERAIL_MAX_ENDPOINTS:-4} 限制）；请求的 ${replicas} 超了。"
  fi
  info "${agent_id}：可用 GPU $(usable_gpu_count) 张，TP $(agent_tensor_parallel_size "$REPO_ROOT" "$agent_id")，起 ${replicas} 个 endpoint"

  assert_tools
  assert_runtime_image "$agent_id"
  local snapshot
  snapshot="$(snapshot_path "$agent_id")"
  [[ -d "$snapshot" ]] || die "冻结 model snapshot 不存在：${snapshot}"
  # `=` 前缀 = 精确匹配。没有它 tmux 会退回前缀匹配，名字更长的别的 session
  # （agent id 互为前缀、或人手建的 derail-serve-xxx-debug）会假报"已存在"。
  tmux has-session -t "=${session}" 2>/dev/null && die "tmux session 已存在：${session}"
  if [[ "${ALLOW_BUSY_GPU:-0}" != "1" ]] && \
    nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits | grep -q '[0-9]'; then
    die "检测到 GPU compute process；拒绝抢卡。确认归属后可显式设置 ALLOW_BUSY_GPU=1"
  fi

  append_model_args "$agent_id"
  local started_at
  started_at="$(date -u +%Y%m%dT%H%M%SZ)"
  local log_dir="${SERVING_LOG_ROOT}/${started_at}_${short}"
  if [[ "$DRY_RUN" != "1" ]]; then
    mkdir -p "$log_dir"
    chmod 700 "$log_dir"
  fi

  # 端口预检。docker run 在 tmux window 里跑，绑不上端口只会在 endpoint_N.log
  # 里留一行 "address already in use" 然后那个 replica 就没了 —— 而 01 的
  # readiness 探针看到该端口上**别人**的服务照样返回 200，于是采集会拿别人的模型
  # 当成本 agent 跑完整整一批。2026-08-10：8000 上有个 Qwen3.6 的服务，holo 的
  # r0 因此没起来，四个 endpoint 里有一个指向 Qwen。宁可现在就报错。
  local index
  local probe_port
  for ((index = 0; index < replicas; index++)); do
    probe_port=$((SERVING_PORT_BASE + index))
    if ss -ltn "sport = :${probe_port}" 2>/dev/null | grep -q ":${probe_port}"; then
      die "端口 ${probe_port} 已被占用（${agent_id} 需要 ${SERVING_PORT_BASE}..$((SERVING_PORT_BASE + replicas - 1))）；换一段：SERVING_PORT_BASE=8100"
    fi
  done

  # GPU 组与 cpuset 都按 replica 序号切出来 —— TP8 拿到全部八张、TP2 拿到相邻两张，
  # 不再需要给 opencua_72b 开特例。
  for ((index = 0; index < replicas; index++)); do
    local port=$((SERVING_PORT_BASE + index))
    local container="derail-serve-${short}-r${index}"
    local gpu_request
    local cpu_set
    gpu_request="\"device=$(agent_gpu_group "$agent_id" "$index")\""
    cpu_set="$(cpuset_slice "$SERVING_CPUSET" "$index" "$replicas")" \
      || die "SERVING_CPUSET=${SERVING_CPUSET} 切不出 ${replicas} 份"
    local -a docker_command=(
      docker run --rm --name "$container"
      --label derail.project=DERAIL --label "derail.agent_id=${agent_id}"
      --gpus "$gpu_request" --ipc host --cpuset-cpus "$cpu_set"
      --publish "127.0.0.1:${port}:8000"
      --volume "${HF_CACHE_ROOT}:/root/.cache/huggingface"
      --env PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
      "$RUNTIME_IMAGE" "${MODEL_ARGS[@]}"
    )
    local docker_text
    local log_path="${log_dir}/endpoint_${index}.log"
    printf -v docker_text '%q ' "${docker_command[@]}"
    local window_text="set -o pipefail; ${docker_text}2>&1 | tee $(printf '%q' "$log_path"); status=\${PIPESTATUS[0]}; printf '\\n[DERAIL serving] container exit=%s\\n' \"\$status\"; exec bash"
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
    info "已启动 tmux session ${session}；日志：${log_dir}"
    info "查看：tmux attach -t ${session}"
  else
    info "dry-run：没有创建 tmux session、container 或 endpoint，也没有 API 请求"
  fi
  print_base_urls "$agent_id" "$replicas"
  info "脚本未调用 /v1/models，也未调用 OpenAI API；请等待日志显示 server ready"
}

stop_model() {
  local agent_id="$1"
  assert_tools
  local session
  session="$(session_name "$agent_id")"
  local -a containers=()
  mapfile -t containers < <(
    docker ps --filter label=derail.project=DERAIL \
      --filter "label=derail.agent_id=${agent_id}" --format '{{.ID}}'
  )
  if (( ${#containers[@]} )); then
    docker stop --time 30 "${containers[@]}"
  fi
  # 同样要精确匹配：kill-session 也走前缀匹配，停一个 agent 会顺手杀掉名字以它
  # 为前缀的别的 session —— 这个函数声称自己"只停精确匹配的东西"。
  if tmux has-session -t "=${session}" 2>/dev/null; then
    tmux kill-session -t "=${session}"
  fi
  info "已停止 ${agent_id} 的精确匹配 container/session"
}

status_model() {
  local agent_id="${1:-}"
  assert_tools
  if [[ -n "$agent_id" ]]; then
    local session
    session="$(session_name "$agent_id")"
    tmux has-session -t "=${session}" 2>/dev/null \
      && info "tmux: ${session} running" || info "tmux: ${session} absent"
    docker ps --filter "label=derail.agent_id=${agent_id}" \
      --format 'table {{.Names}}\t{{.Status}}\t{{.Ports}}'
  else
    tmux list-sessions 2>/dev/null | grep '^derail-serve-' || true
    docker ps --filter label=derail.project=DERAIL \
      --format 'table {{.Names}}\t{{.Status}}\t{{.Ports}}'
  fi
  nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu \
    --format=csv,noheader
}

attach_model() {
  local agent_id="$1"
  command -v tmux >/dev/null 2>&1 || die "找不到 tmux"
  local session
  session="$(session_name "$agent_id")"
  tmux has-session -t "=${session}" 2>/dev/null || die "tmux session 不存在：${session}"
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
