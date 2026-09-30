MYPCBENCH_REPOSITORY="${MYPCBENCH_REPOSITORY:-https://github.com/ljang0/MyPCBench.git}"
MYPCBENCH_COMMIT="${MYPCBENCH_COMMIT:-caf9c754ffe0b41c7e629e17fff19299774af1cb}"

EVOCUA_REPOSITORY="${EVOCUA_REPOSITORY:-https://github.com/meituan/EvoCUA.git}"
EVOCUA_COMMIT="${EVOCUA_COMMIT:-4a0ad5fd4eb1d5b65966e1c7cc3feaa3b534eadd}"
OPENCUA_OSWORLD_REPOSITORY="${OPENCUA_OSWORLD_REPOSITORY:-https://github.com/xlang-ai/OSWorld.git}"
OPENCUA_OSWORLD_COMMIT="${OPENCUA_OSWORLD_COMMIT:-091f5ef1d5544bc74953c77875d5feb5bed30108}"

third_party_paths() {
  local repo_root="$1"
  MYPCBENCH_ROOT="${MYPCBENCH_ROOT:-${repo_root}/third_party/MyPCBench}"
  EVOCUA_ROOT="${EVOCUA_ROOT:-${repo_root}/third_party/EvoCUA}"
  OPENCUA_OSWORLD_ROOT="${OPENCUA_OSWORLD_ROOT:-${repo_root}/third_party/OpenCUA-OSWorld}"
  MYPCBENCH_PATCH="${repo_root}/patches/mypcbench_caf9c754_derail_agents.patch"
  MYPCBENCH_MESSAGE_PATCH="${repo_root}/patches/mypcbench_caf9c754_message_trajectory_agents.patch"
  MYPCBENCH_NATIVE_RESPONSES_PATCH="${repo_root}/patches/mypcbench_caf9c754_native_responses_state.patch"
  MYPCBENCH_GPT55_RESILIENCE_PATCH="${repo_root}/patches/mypcbench_caf9c754_gpt55_recollection_resilience.patch"
  MYPCBENCH_JUDGE_PATCH="${repo_root}/patches/mypcbench_caf9c754_derail_judge.patch"
  MYPCBENCH_PROXY_PATCH="${repo_root}/patches/mypcbench_caf9c754_derail_proxy.patch"
  MYPCBENCH_MESSAGE_JUDGE_PATCH="${repo_root}/patches/mypcbench_caf9c754_message_first_judge.patch"
  MYPCBENCH_JUDGE_RESILIENCE_PATCH="${repo_root}/patches/mypcbench_caf9c754_judge_resilience_v2.patch"
  MYPCBENCH_JUDGE_REASONING_PATCH="${repo_root}/patches/mypcbench_caf9c754_judge_reasoning_effort_low.patch"
  MYPCBENCH_JUDGE_REQUEST_TIMEOUT_PATCH="${repo_root}/patches/mypcbench_caf9c754_judge_request_timeout.patch"
  MYPCBENCH_JUDGE_PROCESS_TIMEOUT_PATCH="${repo_root}/patches/mypcbench_caf9c754_judge_process_timeout.patch"
  MYPCBENCH_JUDGE_FAIL_ON_ERRORS_PATCH="${repo_root}/patches/mypcbench_caf9c754_judge_fail_on_errors.patch"
}

clone_frozen_repo() {
  local repository="$1"
  local expected_commit="$2"
  local target="$3"
  local label="$4"
  local actual_commit

  if [[ ! -d "${target}/.git" ]]; then
    info "clone ${label} 到 ${target}"
    mkdir -p "$(dirname "$target")"
    git clone "$repository" "$target"
    git -C "$target" checkout --detach "$expected_commit"
  fi
  actual_commit="$(git -C "$target" rev-parse HEAD)"
  [[ "$actual_commit" == "$expected_commit" ]] || die \
    "${label} commit 不匹配：期望 ${expected_commit}，实际 ${actual_commit}"
}

apply_mypcbench_patch() {
  local patch_path="$1"
  local label="$2"
  [[ -f "$patch_path" ]] || die "找不到 MyPCBench ${label} patch：${patch_path}"
  if git -C "$MYPCBENCH_ROOT" apply --unidiff-zero --reverse --check \
    "$patch_path" >/dev/null 2>&1; then
    info "MyPCBench ${label} hook 已存在"
  elif git -C "$MYPCBENCH_ROOT" apply --unidiff-zero --check \
    "$patch_path" >/dev/null 2>&1; then
    git -C "$MYPCBENCH_ROOT" apply --unidiff-zero "$patch_path"
    info "已应用 MyPCBench ${label} hook"
  else
    die "MyPCBench ${label} patch 无法干净应用；请检查 third_party/MyPCBench 的本地改动"
  fi
}

apply_mypcbench_patch_series() {
  local base_patch="$1"
  local base_label="$2"
  local tail_patch="$3"
  local tail_label="$4"
  if git -C "$MYPCBENCH_ROOT" apply --unidiff-zero --reverse --check \
    "$tail_patch" >/dev/null 2>&1; then
    info "MyPCBench ${base_label} + ${tail_label} hook 已存在"
    return
  fi
  apply_mypcbench_patch "$base_patch" "$base_label"
  apply_mypcbench_patch "$tail_patch" "$tail_label"
}

setup_mypcbench() {
  clone_frozen_repo \
    "$MYPCBENCH_REPOSITORY" "$MYPCBENCH_COMMIT" "$MYPCBENCH_ROOT" "MyPCBench"
  if git -C "$MYPCBENCH_ROOT" apply --unidiff-zero --reverse --check \
    "$MYPCBENCH_GPT55_RESILIENCE_PATCH" >/dev/null 2>&1; then
    info "MyPCBench DERAIL adapter + native Responses + GPT-5.5 resilience hooks 已存在"
    return
  fi
  if git -C "$MYPCBENCH_ROOT" apply --unidiff-zero --reverse --check \
    "$MYPCBENCH_NATIVE_RESPONSES_PATCH" >/dev/null 2>&1; then
    apply_mypcbench_patch \
      "$MYPCBENCH_GPT55_RESILIENCE_PATCH" "GPT-5.5 recollection resilience"
    return
  fi
  apply_mypcbench_patch_series \
    "$MYPCBENCH_PATCH" "DERAIL adapter" \
    "$MYPCBENCH_MESSAGE_PATCH" "tool-message trajectory"
  apply_mypcbench_patch \
    "$MYPCBENCH_NATIVE_RESPONSES_PATCH" "native Responses state"
  apply_mypcbench_patch \
    "$MYPCBENCH_GPT55_RESILIENCE_PATCH" "GPT-5.5 recollection resilience"
}

setup_mypcbench_judge() {
  if git -C "$MYPCBENCH_ROOT" apply --unidiff-zero --reverse --check \
    "$MYPCBENCH_JUDGE_FAIL_ON_ERRORS_PATCH" >/dev/null 2>&1; then
    info "MyPCBench DERAIL judge chain + fail-on-errors hook 已存在"
    return
  fi
  if git -C "$MYPCBENCH_ROOT" apply --unidiff-zero --reverse --check \
    "$MYPCBENCH_JUDGE_PROCESS_TIMEOUT_PATCH" >/dev/null 2>&1; then
    apply_mypcbench_patch \
      "$MYPCBENCH_JUDGE_FAIL_ON_ERRORS_PATCH" "judge fail on errors"
    return
  fi
  if ! git -C "$MYPCBENCH_ROOT" apply --unidiff-zero --reverse --check \
    "$MYPCBENCH_JUDGE_REASONING_PATCH" >/dev/null 2>&1; then
    if ! git -C "$MYPCBENCH_ROOT" apply --unidiff-zero --reverse --check \
      "$MYPCBENCH_JUDGE_RESILIENCE_PATCH" >/dev/null 2>&1; then
      apply_mypcbench_patch_series \
        "$MYPCBENCH_JUDGE_PATCH" "DERAIL judge" \
        "$MYPCBENCH_MESSAGE_JUDGE_PATCH" "message-first judge"
      apply_mypcbench_patch \
        "$MYPCBENCH_JUDGE_RESILIENCE_PATCH" "judge resilience v2"
    fi
    apply_mypcbench_patch \
      "$MYPCBENCH_JUDGE_REASONING_PATCH" "judge reasoning effort low"
  fi
  apply_mypcbench_patch \
    "$MYPCBENCH_JUDGE_REQUEST_TIMEOUT_PATCH" "judge request timeout"
  apply_mypcbench_patch \
    "$MYPCBENCH_JUDGE_PROCESS_TIMEOUT_PATCH" "judge process timeout"
  apply_mypcbench_patch \
    "$MYPCBENCH_JUDGE_FAIL_ON_ERRORS_PATCH" "judge fail on errors"
}

setup_mypcbench_proxy() {
  apply_mypcbench_patch "$MYPCBENCH_PROXY_PATCH" "DERAIL proxy"
}

setup_evocua() {
  clone_frozen_repo "$EVOCUA_REPOSITORY" "$EVOCUA_COMMIT" "$EVOCUA_ROOT" "EvoCUA"
}

setup_opencua_osworld() {
  clone_frozen_repo \
    "$OPENCUA_OSWORLD_REPOSITORY" "$OPENCUA_OSWORLD_COMMIT" \
    "$OPENCUA_OSWORLD_ROOT" "OpenCUA-OSWorld"
}
