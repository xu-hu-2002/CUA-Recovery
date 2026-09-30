# 被 00/01 三个脚本 source 的共享 config 读取器。没有可执行逻辑，只有函数。
#
# 为什么要有这个文件：`opencua_72b 用 1 个 VM、其余用 4 个` 这条规则曾经被抄在
# 三个地方 —— 00_tmux_serve_open_source.sh 的 replicas 默认值、01_collect_all.sh
# 的 agent_num_vms()、01_collect_trajectories.sh 的 default_serve_replicas()。
# 三份抄写不但要一起改，而且其中一份还是错的：01_collect_all 通过 export NUM_VMS
# 传值，而 01 的 config 优先级高于环境变量，于是 opencua_72b 的 1 被静默忽略，
# 一个 TP8 模型会以 4 个 VM 起跑。
#
# 现在真源是 configs/agents/<agent_id>.yaml，三个脚本都从这里读。
#
# 用 awk 而不是 python：00 在装 conda env 之前就要能跑，不能依赖 PYTHON_BIN。

# 从 `key: value` 形式的扁平 config 里读一个标量。
# 解析规则与 01_collect_trajectories.sh 的 load_collection_config 保持一致：
# `#` 起注释、值可带引号、空值视为未设置。找不到返回 1 并且不输出。
config_scalar() {
  local path="$1" key="$2"
  [[ -f "$path" ]] || return 1
  awk -v want="$key" '
    { sub(/\r$/, "") }
    /^[[:space:]]*#/ { next }
    {
      pos = index($0, ":")
      if (pos == 0) next
      k = substr($0, 1, pos - 1)
      gsub(/^[[:space:]]+|[[:space:]]+$/, "", k)
      if (k != want) next
      v = substr($0, pos + 1)
      sub(/#.*$/, "", v)
      gsub(/^[[:space:]]+|[[:space:]]+$/, "", v)
      gsub(/^["\047]|["\047]$/, "", v)
      if (v == "") next
      print v
      found = 1
      exit
    }
    END { if (!found) exit 1 }
  ' "$path"
}

agent_config_path() {
  printf '%s/configs/agents/%s.yaml\n' "$1" "$2"
}

config_scalar_required() {
  local path="$1" key="$2" value
  value="$(config_scalar "$path" "$key" || true)"
  if [[ -z "$value" ]]; then
    printf '错误：%s 缺少 %s\n' "$path" "$key" >&2
    return 1
  fi
  printf '%s\n' "$value"
}

# 本机这轮采集能用的 GPU 张数。
#
# 顺序：显式声明 > CUDA_VISIBLE_DEVICES 圈定的子集 > nvidia-smi 实际枚举。没有
# nvidia-smi 且没有显式声明时返回 0，由调用方决定是报错还是跳过（托管 API 的
# agent 根本不需要卡）。DERAIL_GPU_COUNT 是唯一的逃生口：容器里看不到
# nvidia-smi、或者只想用八张里的四张时用它。
usable_gpu_count() {
  if [[ -n "${DERAIL_GPU_COUNT:-}" ]]; then
    printf '%s\n' "$DERAIL_GPU_COUNT"
    return 0
  fi
  if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
    awk -F, '{ n = 0; for (i = 1; i <= NF; i++) if ($i != "") n++; print n }' \
      <<< "$CUDA_VISIBLE_DEVICES"
    return 0
  fi
  if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi -L 2>/dev/null | grep -c '^GPU ' || printf '0\n'
    return 0
  fi
  printf '0\n'
}

# 本轮采集实际可用的 GPU 序号列表（逗号分隔），供 00 切分 endpoint 用。
# CUDA_VISIBLE_DEVICES 已经圈定子集时按它的顺序原样返回。
usable_gpu_indices() {
  if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
    printf '%s\n' "$CUDA_VISIBLE_DEVICES"
    return 0
  fi
  local count
  count="$(usable_gpu_count)"
  awk -v n="$count" 'BEGIN {
    out = ""
    for (i = 0; i < n; i++) out = out (i ? "," : "") i
    print out
  }'
}

# 该 agent 一个 endpoint 占几张卡。托管 API 的 agent 没有这一项，返回空并退出 1。
agent_tensor_parallel_size() {
  config_scalar "$(agent_config_path "$1" "$2")" tensor_parallel_size
}

# 该 agent 这轮能起几个 endpoint / 跑几个 VM。
#
# 本地 serving：可用 GPU 数 / tensor_parallel_size —— 八张卡 TP2 得四个，TP8 只
# 得一个。这个数以前是 yaml 里写死的 num_vms，换台卡数不同的机器就是错的。
# 托管 API：不占卡，上限是配额而非显存，只能由 yaml 的 num_vms 显式声明。
#
# 官方 runner 按 VM round-robin 分配 endpoint，所以每个 VM 配一个 replica 就够，
# 两个数字是同一个。
agent_vm_count() {
  local repo_root="$1" agent_id="$2" path tp gpus count
  path="$(agent_config_path "$repo_root" "$agent_id")"
  [[ -f "$path" ]] || { printf '错误：找不到 %s\n' "$path" >&2; return 1; }

  tp="$(config_scalar "$path" tensor_parallel_size || true)"
  if [[ -z "$tp" ]]; then
    config_scalar_required "$path" num_vms
    return
  fi
  [[ "$tp" =~ ^[1-9][0-9]*$ ]] || {
    printf '错误：%s 的 tensor_parallel_size=%s 不是正整数\n' "$path" "$tp" >&2
    return 1
  }

  gpus="$(usable_gpu_count)"
  if (( gpus < tp )); then
    printf '错误：%s 需要每个 endpoint %s 张卡，本机只有 %s 张可用。\n' \
      "$agent_id" "$tp" "$gpus" >&2
    printf '       容器里看不到 nvidia-smi 时用 DERAIL_GPU_COUNT 显式声明。\n' >&2
    return 1
  fi
  count=$((gpus / tp))
  # 端口与 cpuset 按 endpoint 顺序分配，多到一定程度只会互相抢 CPU；上限本身也
  # 可调，默认 4 与 SERVING_PORT_BASE..+3 这段端口对齐。
  local ceiling="${DERAIL_MAX_ENDPOINTS:-4}"
  (( count > ceiling )) && count="$ceiling"
  printf '%s\n' "$count"
}

# 某个 agent 的每局步数上限；没声明就退回采集 config 的默认值。
#
# 与 agent_vm_count 不同，这是**协议参数**——它改变轨迹长度、因而改变分数。之所以
# 仍住在 agent yaml：2026-08-11 起 evocua_32b 用 120 而其余用 150，这个数已经按
# agent 取值而不是按采集批次取值。跨 agent 比分时必须记得它不一致，manifest 的
# max_steps_per_agent 就是为了让这件事事后查得到。
agent_max_steps() {
  local repo_root="$1" agent_id="$2" fallback="$3" value
  value="$(config_scalar "$(agent_config_path "$repo_root" "$agent_id")" max_steps || true)"
  [[ "$value" =~ ^[1-9][0-9]*$ ]] || value="$fallback"
  printf '%s\n' "$value"
}

# 单步耗时的预估值，只用于 01_collect_all.sh 的时长预览，不影响采集行为。
# 以前是那个脚本里的一张 case 表，只覆盖 5 个开源 agent，其余 5 个 agent 会被当成
# "未知 agent_id" 直接打死 —— 于是托管 API 的 agent 只能绕过 01_collect_all 单跑。
agent_eta_step_seconds() {
  config_scalar_required "$(agent_config_path "$1" "$2")" eta_step_seconds
}
