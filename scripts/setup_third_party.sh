#!/usr/bin/env bash
# 把 third_party/ 下的三个上游仓库建起来：clone 到 pin 住的 commit，并给
# MyPCBench 打上 DERAIL 的 adapter / judge patch。
#
# 为什么需要这个脚本：`third_party/` 被 .gitignore 屏蔽，clone 本仓库拿不到它们。
# 这是刻意的——三个上游各自带 .git，收录进来只会记成 gitlink（别人 clone 到手依旧
# 是空目录），而 MyPCBench 的 qcow2 镜像有 16.5G，不该进 git 历史。
# 获取逻辑本来只藏在 01_collect_trajectories.sh 里，等于"想拿到代码得先跑一遍采集"。
#
# 用法：
#   bash scripts/setup_third_party.sh              # MyPCBench（采集与判官都需要）
#   bash scripts/setup_third_party.sh --all        # 再加 EvoCUA 和 OpenCUA-OSWorld
#   bash scripts/setup_third_party.sh evocua       # 只补某一个
#
# 只依赖 git，不需要 conda env 或 GPU。可重复执行：已存在的 checkout 不会被覆盖，
# 只校验 commit；已打过的 patch 会被跳过。

set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

die() {
  printf '错误：%s\n' "$*" >&2
  exit 1
}

info() {
  printf '[DERAIL setup] %s\n' "$*"
}

# shellcheck source=lib/third_party.sh
source "${SCRIPT_DIR}/lib/third_party.sh"
third_party_paths "$REPO_ROOT"

TARGETS=()
for arg in "$@"; do
  case "$arg" in
    --all) TARGETS=(mypcbench evocua opencua) ;;
    mypcbench|evocua|opencua) TARGETS+=("$arg") ;;
    -h|--help)
      sed -n '2,20p' "${BASH_SOURCE[0]}"
      exit 0
      ;;
    *) die "未知参数：${arg}（可用：mypcbench / evocua / opencua / --all）" ;;
  esac
done
# 不给参数时只建 MyPCBench：它是唯一无条件需要的，另外两个只有跑对应 agent 才用得上。
(( ${#TARGETS[@]} )) || TARGETS=(mypcbench)

command -v git >/dev/null 2>&1 || die "找不到 git"

for target in "${TARGETS[@]}"; do
  case "$target" in
    mypcbench)
      setup_mypcbench
      setup_mypcbench_judge
      ;;
    evocua) setup_evocua ;;
    opencua) setup_opencua_osworld ;;
  esac
done

info "third_party 就绪：$(printf '%s ' "${TARGETS[@]}")"

# VM 镜像单独提示而不自动下载：16.5G，且只有真正要跑 rollout 的人才需要。
if [[ " ${TARGETS[*]} " == *" mypcbench "* && ! -f "${MYPCBENCH_ROOT}/mypcbench-vm/mypcbench.qcow2" ]]; then
  cat <<EOF

还差 VM 镜像（约 16.5G，只有要跑 rollout 才需要）：
  bash ${MYPCBENCH_ROOT}/scripts/get-eval-image.sh --out ${MYPCBENCH_ROOT}/mypcbench-vm

镜像跑起来需要 KVM。宿主机能不能直跑，实测别猜：
  [ -r /dev/kvm ] && [ -w /dev/kvm ] && echo "qemu backend 可用" || echo "改用 docker backend"
不可用时把 configs/collection/mypcbench_runtime.yaml 的 backend 设为 docker。
EOF
fi
