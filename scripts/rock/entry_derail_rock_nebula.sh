#!/usr/bin/env bash
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/../.." && pwd)"
cd "$REPO"
export PYTHONPATH="$REPO${PYTHONPATH:+:$PYTHONPATH}"

if [[ -f "$REPO/.rock_run.env" ]]; then
  set -a; # shellcheck disable=SC1091
  source "$REPO/.rock_run.env"; set +a
  echo "[derail-rock] sourced run-config .rock_run.env"
else
  echo "[derail-rock] ERROR: .rock_run.env missing (submit script writes it)" >&2
  exit 2
fi

if [[ "${DERAIL_WORKLOAD:-}" == "phase5" ]]; then
  PACKAGE_ROOT="$REPO/${PHASE5_PACKAGE_DIR:?PHASE5_PACKAGE_DIR is required}"
  [[ -d "$PACKAGE_ROOT" ]] || {
    echo "[derail-rock] ERROR: Phase 5 package is missing: $PACKAGE_ROOT" >&2; exit 2; }
  mkdir -p "$REPO/artifacts/phase5" "$REPO/data/synthesis/generation" \
    "$REPO/data/synthesis/task_ir_v1"
  cp "$PACKAGE_ROOT/artifacts/phase5/"* "$REPO/artifacts/phase5/"
  tar xzf "$PACKAGE_ROOT/data/synthesis/generation/final_v1.tar.gz" \
    -C "$REPO/data/synthesis/generation/"
  tar xzf "$PACKAGE_ROOT/data/synthesis/task_ir_v1/accepted.tar.gz" \
    -C "$REPO/data/synthesis/task_ir_v1/"
  echo "[derail-rock] restored frozen Phase 5 inputs"
fi

# shellcheck disable=SC1091
source "$HERE/allocate_proxy_ports.sh"

if [[ -n "${DERAIL_SECRET_MOUNT:-}" && -f "${DERAIL_SECRET_MOUNT}" ]]; then
  set -a; # shellcheck disable=SC1090
  source "${DERAIL_SECRET_MOUNT}"; set +a
  rm -f "${DERAIL_SECRET_MOUNT}" || true
  echo "[derail-rock] secrets loaded from staged mount file (and removed)"
else
  echo "[derail-rock] ERROR: DERAIL_SECRET_MOUNT not found: ${DERAIL_SECRET_MOUNT:-<unset>}" >&2
  exit 2
fi
export OSS_ACCESS_ID="${OSS_ACCESS_ID:-${OSS_ID:-${OSS_ACCESS_KEY_ID:-}}}"
export OSS_ACCESS_KEY="${OSS_ACCESS_KEY:-${OSS_KEY:-${OSS_ACCESS_KEY_SECRET:-}}}"
[[ -n "$OSS_ACCESS_ID" && -n "$OSS_ACCESS_KEY" ]] || {
  echo "[derail-rock] ERROR: OSS creds empty after secret source" >&2; exit 2; }
if [[ "${AGENT_ID:-}" != "dummy" ]]; then
  [[ -n "${OPENAI_API_KEY:-}" && -n "${OPENAI_BASE_URL:-}" ]] || {
    echo "[derail-rock] ERROR: AGENT_ID=${AGENT_ID} 需要 secret 里的 OPENAI_API_KEY/OPENAI_BASE_URL" >&2
    exit 2; }
fi

PY="${DERAIL_ENTRY_PYTHON:-${PYTHONHOME:+$PYTHONHOME/bin/python}}"
PY="${PY:-python3}"
if ! command -v "$PY" >/dev/null 2>&1; then
  for _p in python python3.12 python3.11 python3.10; do
    if command -v "$_p" >/dev/null 2>&1; then
      echo "[derail-rock] '$PY' not on this image -> using $_p"
      PY="$_p"; break
    fi
  done
fi

ROCK_SDK_PROBE='from rock.sdk.sandbox.config import SandboxConfig; from rock.actions import CreateBashSessionRequest; from rock.sdk.sandbox.client import Sandbox'
rock_sdk_ok() { "$PY" -c "$ROCK_SDK_PROBE" 2>/dev/null; }

if ! rock_sdk_ok; then
  for attempt in 1 2 3; do
    echo "[derail-rock] ROCK SDK not importable -- install attempt ${attempt}/3 with $PY -m pip"
    extra=""
    [ "$attempt" -gt 1 ] && extra="--force-reinstall --no-cache-dir"
    _pip_index=""; [ -n "${PIP_INDEX_URL:-}" ] && _pip_index="-i $PIP_INDEX_URL"
    "$PY" -m pip install -q $extra "rl-rock" $_pip_index \
      || "$PY" -m pip install -q $extra "rock-rl" $_pip_index \
      || "$PY" -m pip install -q $extra "rl-rock" \
      || "$PY" -m pip install -q $extra "rock-rl" \
      || echo "[derail-rock] pip install attempt ${attempt} returned non-zero"
    if rock_sdk_ok; then
      echo "[derail-rock] ROCK SDK deep import OK after attempt ${attempt}"
      break
    fi
    [ "$attempt" -eq 3 ] && {
      echo "[derail-rock] ERROR: ROCK SDK still not importable after 3 attempts" >&2
      "$PY" -c "$ROCK_SDK_PROBE" 2>&1 | tail -5 >&2
      exit 1
    }
    sleep 20
  done
fi
echo "[derail-rock] ROCK SDK ready: $("$PY" -c 'import rock; print(getattr(rock,"__version__","?"))' 2>/dev/null || echo '?')"

ROCK_SITE_PACKAGES="$($PY -c 'import site; print(":".join(site.getsitepackages()))')"
export PYTHONPATH="${ROCK_SITE_PACKAGES}${PYTHONPATH:+:$PYTHONPATH}"
"$PY" -c "$ROCK_SDK_PROBE"

CORE_DEPS_PROBE='import yaml, jsonschema'
deps_ok() { "$PY" -c "$CORE_DEPS_PROBE" 2>/dev/null; }
if ! deps_ok; then
  for attempt in 1 2 3; do
    echo "[derail-rock] core deps (yaml/jsonschema) missing -- pip install -e . attempt ${attempt}/3"
    _pip_index=""; [ -n "${PIP_INDEX_URL:-}" ] && _pip_index="-i $PIP_INDEX_URL"
    "$PY" -m pip install -q -e "$REPO" $_pip_index \
      || "$PY" -m pip install -q -e "$REPO" \
      || "$PY" -m pip install -q PyYAML jsonschema \
      || echo "[derail-rock] deps install attempt ${attempt} returned non-zero"
    deps_ok && break
    sleep 15
  done
fi
deps_ok || { echo "[derail-rock] ERROR: core deps still missing after 3 attempts -- refusing fake start" >&2; exit 1; }
echo "[derail-rock] core deps ready (yaml/jsonschema)"

if [[ "${PHASE5_HAZARD_SMOKE:-0}" == "1" ]]; then
  echo "[derail-rock] launching Phase 5 representative hazard smoke"
  PYTHONUNBUFFERED=1 exec "$PY" -u "$REPO/scripts/phase5/run_hazard_smoke.py"
fi

ROCK_URL="${ROCK_BASE_URL:-http://<rock-endpoint>}"
if command -v curl >/dev/null 2>&1; then
  auth_args=()
  if [[ -n "${ROCK_API_KEY:-}" ]]; then
    auth_args=(-H "XRL-Authorization: Bearer ${ROCK_API_KEY}")
  fi
  code=$(curl -s -o /dev/null -w "%{http_code}" --connect-timeout 5 \
    "${auth_args[@]}" "${ROCK_URL}/apis/envs/sandbox/v1/get_status" 2>/dev/null || echo "000")
  echo "[derail-rock] ROCK probe: ${ROCK_URL} -> HTTP ${code} (000=unreachable)"
  if [[ "$code" == "401" || "$code" == "403" ]]; then
    echo "[derail-rock] ERROR: ROCK authentication rejected; refusing to start driver" >&2
    exit 1
  fi
fi

echo "[derail-rock] launching driver: AGENT_ID=${AGENT_ID} SHARD_FILE=${SHARD_FILE} FORMAL_COLLECTION=${FORMAL_COLLECTION:-0}"
PYTHONUNBUFFERED=1 exec "$PY" -u "$REPO/scripts/rock/derail_rock_driver.py"
