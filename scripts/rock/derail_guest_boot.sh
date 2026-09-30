#!/usr/bin/env bash
set -Eeuo pipefail
export PATH="/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin${PATH:+:$PATH}"

VM_DIR="${VM_DIR:-/storage/mypcbench-vm}"
BASE_QCOW2="${GUEST_QCOW2:-${VM_DIR}/mypcbench.qcow2}"
NAME="${GUEST_NAME:-mypcbench-guest}"
API_FWD="${API_FWD_PORT:-15000}"
VNC_FWD="${VNC_FWD_PORT:-15901}"
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

qemu-img create -f qcow2 -b "$BASE_QCOW2" -F qcow2 "$OVERLAY"
qemu-img convert -O qcow2 -f raw "${VM_DIR}/OVMF_VARS.fd" "$VARS"

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
