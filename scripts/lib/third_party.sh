# 被 setup_third_party.sh 和 01_collect_trajectories.sh 共同 source 的
# third_party 获取逻辑。没有可执行语句，只有变量声明和函数。
#
# 为什么要有这个文件：`third_party/` 在 .gitignore 里，所以 clone 下来的仓库是空
# 的——这是刻意的，三个上游各自带 .git，直接收录只会记成 gitlink，别人 clone 到手
# 的还是空目录，而 MyPCBench 那 16.5G 的 qcow2 也不该进 git 历史。代价是"怎么把
# third_party 建起来"必须有个显式入口，否则新人只能靠跑一遍采集脚本来触发它。
#
# 抽出来的第二个原因和 collection_config.sh 一样：pin 住的 commit 曾经只写在
# 01_collect_trajectories.sh 里，任何第二个需要这些仓库的脚本都得把三组
# repository/commit/root 再抄一遍，而抄错 commit 是不会报错的——实验协议会静默
# 漂移。现在真源在这里。

# 冻结官方 runner 版本。若要升级，先审计 prompt/runner diff，再显式更新该 commit。
MYPCBENCH_REPOSITORY="${MYPCBENCH_REPOSITORY:-https://github.com/ljang0/MyPCBench.git}"
MYPCBENCH_COMMIT="${MYPCBENCH_COMMIT:-caf9c754ffe0b41c7e629e17fff19299774af1cb}"

# EvoCUA/OpenCUA 的 prompt、history 和 action parser 直接来自各自官方仓库。
# 固定 commit 是为了避免上游更新悄悄改变实验协议。
EVOCUA_REPOSITORY="${EVOCUA_REPOSITORY:-https://github.com/meituan/EvoCUA.git}"
EVOCUA_COMMIT="${EVOCUA_COMMIT:-4a0ad5fd4eb1d5b65966e1c7cc3feaa3b534eadd}"
OPENCUA_OSWORLD_REPOSITORY="${OPENCUA_OSWORLD_REPOSITORY:-https://github.com/xlang-ai/OSWorld.git}"
OPENCUA_OSWORLD_COMMIT="${OPENCUA_OSWORLD_COMMIT:-091f5ef1d5544bc74953c77875d5feb5bed30108}"

# 根据 repo 根目录推导三个 checkout 位置和 patch 路径。调用方先定好 REPO_ROOT
# 再调用；env 显式指定的位置优先，便于把 checkout 放到别的盘。
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

# 已有目录不会被覆盖或自动 checkout，只校验 commit 是否就是 pin 住的那个。
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

# 对冻结的 MyPCBench 只加很小的 factory hook。patch 可逆、可审计，adapter 实现
# 仍全部保留在 DERAIL 自己的 src/ 中。幂等：已打过的 patch 会被 reverse-check
# 识别并跳过，不会重复应用也不会报错。
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

# 后续 patch 会改动前一个 patch 新增的行，因此完整 patch 链打完后，前一个 patch
# 未必还能单独通过 --reverse --check。先检查链尾；链尾存在就说明它依赖的前序状态
# 也已经存在。否则再按顺序逐个幂等应用。
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

# agents patch 是采集必需的；judge patch 服务于 02_judge_rubrics.sh，两者改的文件
# 不重叠，一起打不会冲突。
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

# proxy patch 只服务 ROCK proxy 拓扑（agent loop 在 Nebula 侧）的 full 阶段：
# runner 跳过本地起 VM / 端口滑窗，env.py 的 reset 改走 driver 的 lifecycle
# 端点。本地直连采集（agents/judge 路径）不设 MYPCBENCH_REMOTE_ATTACH 时行为
# 完全不变，所以一起打也无副作用。注意：该 patch 的基线是"已含 DERAIL 本地
# 适配"的 harness——即版本化 OSS takeover harness tar 恢复出的那份；对
# pristine caf9c754 HEAD 打不上（适配改动已在基线里）。
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
