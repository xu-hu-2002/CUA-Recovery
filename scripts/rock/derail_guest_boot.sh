#!/usr/bin/env bash
# =============================================================================
# DERAIL MyPCBench guest boot — 跑在 ROCK 沙箱容器内（L3 环境层）。
#
# MCUA 拓扑的 MyPCBench 形态：agent loop 与模型采样在 Nebula（L2），guest
# 只留执行环境。本脚本用与 agent-harness/env.py _start_qemu 完全一致的参数
# 起 QEMU（CoW overlay + OVMF + user-net hostfwd），把 guest 控制面 :5000
# 映射到沙箱容器 :15000，供 derail_guest_gateway.py(:8080) 路由给平台 Proxy。
# 沙箱永不回连 Nebula —— 方向只有 Nebula → Proxy → 本 gateway → guest。
#
# 幂等：guest 已起且健康则直接 GUEST_ALREADY_UP；残留进程先清。
# 产物：GUEST_READY / GUEST_DIED / GUEST_READY_TIMEOUT（driver 侧判成功）。
# =============================================================================
set -Eeuo pipefail
export PATH="/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin${PATH:+:$PATH}"

VM_DIR="${VM_DIR:-/storage/mypcbench-vm}"
BASE_QCOW2="${GUEST_QCOW2:-${VM_DIR}/mypcbench.qcow2}"
NAME="${GUEST_NAME:-mypcbench-guest}"
API_FWD="${API_FWD_PORT:-15000}"      # 沙箱口 → guest 控制面 :5000
VNC_FWD="${VNC_FWD_PORT:-15901}"      # 沙箱口 → guest VNC :5901
PIDFILE="/tmp/${NAME}.pid"
OVERLAY="/tmp/${NAME}-overlay.qcow2"
VARS="/tmp/${NAME}-vars.qcow2"

already_healthy() {
  [[ -f "$PIDFILE" ]] && kill -0 "$(cat "$PIDFILE")" 2>/dev/null \
    && curl -fsS "http://127.0.0.1:${API_FWD}/health" >/dev/null 2>&1
}
if already_healthy; then
  echo "GUEST_ALREADY_UP (pid $(cat "$PIDFILE"))"
  exit 0
fi
[[ -f "$PIDFILE" ]] && { kill "$(cat "$PIDFILE")" 2>/dev/null || true; sleep 2; }

for f in "$BASE_QCOW2" "${VM_DIR}/OVMF_CODE.fd" "${VM_DIR}/OVMF_VARS.fd"; do
  [[ -f "$f" ]] || { echo "ERROR: missing $f (prep 未拉资产)" >&2; exit 1; }
done
command -v qemu-img >/dev/null 2>&1 || { echo "ERROR: qemu-img not found" >&2; exit 1; }
QEMU_BIN="$(command -v qemu-system-x86_64 || command -v qemu-kvm || echo /usr/libexec/qemu-kvm)"
[[ -x "$QEMU_BIN" ]] || { echo "ERROR: no QEMU binary in sandbox image" >&2; exit 1; }
[[ -e /dev/kvm ]] || { echo "ERROR: /dev/kvm missing (TCG 太慢，拒跑)" >&2; exit 1; }

# 与 env.py _start_qemu 同参：新 CoW overlay（=OSWorld revert_to_snapshot 语义）
qemu-img create -f qcow2 -b "$BASE_QCOW2" -F qcow2 "$OVERLAY"
qemu-img convert -O qcow2 -f raw "${VM_DIR}/OVMF_VARS.fd" "$VARS"

# hostfwd：控制面/VNC 用让开平台口的 1xxxx 段；app 口 3001-3018 与 guest 同号
# （env.py 的 _discover_ports_qemu 默认 MYPCBENCH_HOST_APP_PORT_<cp>=<cp>）。
HOSTFWD="hostfwd=tcp::${API_FWD}-:5000,hostfwd=tcp::${VNC_FWD}-:5901"
for cp in $(seq 3001 3018); do
  HOSTFWD="${HOSTFWD},hostfwd=tcp::${cp}-:${cp}"
done

nohup setsid "$QEMU_BIN" \
  -name "$NAME" \
  -machine q35,accel=kvm:tcg \
  -enable-kvm -cpu host \
  -m 8G -smp 4 \
  -drive "if=pflash,format=raw,readonly=on,file=${VM_DIR}/OVMF_CODE.fd" \
  -drive "if=pflash,format=qcow2,file=${VARS}" \
  -drive "file=${OVERLAY},format=qcow2,if=virtio,cache=unsafe,aio=threads" \
  -netdev "user,id=net0,${HOSTFWD}" \
  -device virtio-net-pci,netdev=net0 \
  -vga virtio -vnc :1 -display none \
  -serial "file:/tmp/${NAME}-serial.log" \
  -monitor none \
  </dev/null > "/tmp/${NAME}-stderr.log" 2>&1 &
echo $! > "$PIDFILE"
echo "[guest-boot] QEMU pid=$(cat "$PIDFILE") api_fwd=${API_FWD}"

# guest 桌面自举（GDM autologin + controller.service）KVM 下约 2-4min。
for _ in $(seq 1 120); do
  if curl -fsS "http://127.0.0.1:${API_FWD}/health" >/dev/null 2>&1; then
    echo "GUEST_READY"
    exit 0
  fi
  kill -0 "$(cat "$PIDFILE")" 2>/dev/null || {
    echo "GUEST_DIED" >&2
    tail -20 "/tmp/${NAME}-stderr.log" >&2 || true
    exit 1
  }
  sleep 5
done
echo "GUEST_READY_TIMEOUT" >&2
tail -20 "/tmp/${NAME}-serial.log" >&2 || true
exit 1
