#!/usr/bin/env bash
# Allocate a per-Nebula-job localhost port block for the ROCK proxy bridge.
# Nebula jobs can share a host network namespace, so fixed localhost ports are
# unsafe once several takeover shards start concurrently.

if [[ -z "${PROXY_PORT_BASE:-}" || -z "${PROXY_LIFECYCLE_PORT:-}" ]]; then
  _proxy_port_key="$(hostname 2>/dev/null || echo unknown):${COLLECTION_ID:-unknown}"
  _proxy_port_hash="$(printf '%s' "$_proxy_port_key" | cksum | awk '{print $1}')"
  # Each job owns 32 consecutive ports: bridge uses offsets 0 and 10..27;
  # lifecycle uses offset 28. Keep the range below the ephemeral port floor.
  _proxy_start=$((20000 + (_proxy_port_hash % 700) * 32))
  PROXY_PORT_BASE="$(python3 - "$_proxy_start" <<'PY'
import socket
import sys

start = int(sys.argv[1])
for block in range(700):
    base = 20000 + ((start - 20000 + block * 32) % (700 * 32))
    sockets = []
    try:
        for port in range(base, base + 32):
            sock = socket.socket()
            sock.bind(("127.0.0.1", port))
            sockets.append(sock)
    except OSError:
        continue
    finally:
        for sock in sockets:
            sock.close()
    print(base)
    break
else:
    raise SystemExit("no free 32-port proxy block")
PY
)"
  PROXY_LIFECYCLE_PORT=$((PROXY_PORT_BASE + 28))
  export PROXY_PORT_BASE PROXY_LIFECYCLE_PORT
  unset _proxy_port_key _proxy_port_hash _proxy_start
fi

echo "[derail-rock] proxy ports: base=${PROXY_PORT_BASE} lifecycle=${PROXY_LIFECYCLE_PORT}"
