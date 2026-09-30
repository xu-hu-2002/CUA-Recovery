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

agent_tensor_parallel_size() {
  config_scalar "$(agent_config_path "$1" "$2")" tensor_parallel_size
}

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
  local ceiling="${DERAIL_MAX_ENDPOINTS:-4}"
  (( count > ceiling )) && count="$ceiling"
  printf '%s\n' "$count"
}

agent_max_steps() {
  local repo_root="$1" agent_id="$2" fallback="$3" value
  value="$(config_scalar "$(agent_config_path "$repo_root" "$agent_id")" max_steps || true)"
  [[ "$value" =~ ^[1-9][0-9]*$ ]] || value="$fallback"
  printf '%s\n' "$value"
}

agent_eta_step_seconds() {
  config_scalar_required "$(agent_config_path "$1" "$2")" eta_step_seconds
}
