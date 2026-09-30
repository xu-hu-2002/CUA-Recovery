#!/usr/bin/env bash
# DERAIL ROCK 资产预置：在 ROCK 工具沙箱内下载/校验/解压 MyPCBench VM 镜像。
#
# 输入固定在 configs/models.lock.yaml：
#   - HF dataset ljang0/mypcbench-qemu-baseline @ aca6ec99 的 michael_scott.qcow2
#   - published_compressed_sha256 / local_uncompressed_sha256 用于完整性校验
# OVMF 固件取自沙箱系统包（osworld-rock:v7 的 /usr/share/OVMF/*_4M.fd），
# 与 lock 无关（lock 只锁 qcow2）；其 sha256 记录进资产 manifest 供审计。
#
# 用法（沙箱内）：
#   bash stage_vm_assets.sh            # 下载+校验+解压，落 /storage/derail-assets
#   成功后目录内有 STAGING_DONE 标记。
set -Eeuo pipefail
export PATH=/usr/local/bin:/usr/bin:/bin:/sbin
WORK="${DERAIL_ASSET_WORKDIR:-/storage/derail-assets}"
mkdir -p "$WORK"
cd "$WORK"

step() { echo "[stage $(date +%H:%M:%S)] $*"; }

HF_URL="https://huggingface.co/datasets/ljang0/mypcbench-qemu-baseline/resolve/aca6ec996c60f3567b6701028eb7d177297f405f/michael_scott.qcow2"
LOCK_COMPRESSED_SHA="61729f831588720b56a8ed46548fac003205cc53e414cec573ce269d1a52cef3"
LOCK_UNCOMPRESSED_SHA="29e6ef0230655501920ad7b08c38e16c0d39e94c5be8220de2cf723885759e90"

if [[ ! -f michael_scott.qcow2 ]]; then
  step "downloading michael_scott.qcow2 (pinned HF revision aca6ec99)"
  curl -fL --retry 3 --retry-delay 5 -C - -o michael_scott.qcow2 "$HF_URL"
fi
step "verifying compressed sha256"
echo "${LOCK_COMPRESSED_SHA}  michael_scott.qcow2" | sha256sum -c -

if [[ ! -f mypcbench.qcow2 ]]; then
  step "decompressing qcow2 (qemu-img convert, ~5G -> ~10G)"
  qemu-img convert -p -O qcow2 -o cluster_size=64K michael_scott.qcow2 mypcbench.qcow2.tmp
  mv mypcbench.qcow2.tmp mypcbench.qcow2
fi
step "uncompressed sha256 (lock expects ${LOCK_UNCOMPRESSED_SHA}):"
sha256sum mypcbench.qcow2 | tee mypcbench.qcow2.sha256

cp -f /usr/share/OVMF/OVMF_CODE_4M.fd OVMF_CODE.fd
cp -f /usr/share/OVMF/OVMF_VARS_4M.fd OVMF_VARS.fd
step "ovmf sha256:"
sha256sum OVMF_CODE.fd OVMF_VARS.fd | tee ovmf.sha256

step "STAGING_DONE"
touch "$WORK/STAGING_DONE"
