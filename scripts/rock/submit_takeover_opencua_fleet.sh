#!/usr/bin/env bash
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SUBMIT_SCRIPT="${SUBMIT_SCRIPT:-$HERE/submit_takeover_opencua.sh}"

DEPTH="${DEPTH:-0}"
CONDITION="${CONDITION:-unaware}"
SHARD_COUNT="${SHARD_COUNT:-16}"
WORKER_INDICES="${WORKER_INDICES:-}"
MAX_FLEET_WORKERS="${MAX_FLEET_WORKERS:-32}"

case "$DEPTH" in 0|5|10|15|20|25) ;; *) echo "invalid depth: $DEPTH" >&2; exit 2 ;; esac
case "$CONDITION" in unaware|notified) ;; *) echo "invalid condition: $CONDITION" >&2; exit 2 ;; esac
[[ "$SHARD_COUNT" =~ ^[1-9][0-9]*$ ]] || { echo "invalid SHARD_COUNT" >&2; exit 2; }
[[ "$MAX_FLEET_WORKERS" =~ ^[1-9][0-9]*$ ]] || { echo "invalid MAX_FLEET_WORKERS" >&2; exit 2; }
(( MAX_FLEET_WORKERS <= 32 )) || { echo "MAX_FLEET_WORKERS cannot exceed the 32-worker production cap" >&2; exit 2; }

if [[ -z "$WORKER_INDICES" ]]; then
  indices=()
  for ((index = 0; index < SHARD_COUNT; index++)); do
    indices+=("$index")
  done
else
  read -r -a indices <<< "${WORKER_INDICES//,/ }"
fi

(( ${#indices[@]} <= MAX_FLEET_WORKERS )) || {
  echo "selected ${#indices[@]} workers exceeds MAX_FLEET_WORKERS=$MAX_FLEET_WORKERS" >&2
  exit 2
}

validated_csv=""
for index in "${indices[@]}"; do
  [[ "$index" =~ ^[0-9]+$ ]] || { echo "invalid worker index: $index" >&2; exit 2; }
  (( index < SHARD_COUNT )) || { echo "worker index $index is outside shard count $SHARD_COUNT" >&2; exit 2; }
  case ",$validated_csv," in
    *",$index,"*) echo "duplicate worker index: $index" >&2; exit 2 ;;
  esac
  validated_csv="${validated_csv:+$validated_csv,}$index"
done

echo "[opencua-fleet] depth=$DEPTH condition=$CONDITION workers=${#indices[@]} shards=$SHARD_COUNT"
echo "[opencua-fleet] production cap=$MAX_FLEET_WORKERS; at least $((40 - MAX_FLEET_WORKERS)) of 40 ROCK slots remain"

# Only the first submission may perform the pre-flight stale-sandbox sweep. Every
# subsequent submission must leave already-started siblings untouched.
first=1
for index in "${indices[@]}"; do
  if (( first )); then
    shard_sweep="${SWEEP_STALE_SANDBOXES:-0}"
    first=0
  else
    shard_sweep=0
  fi
  echo "[opencua-fleet] submit shard=$index/$SHARD_COUNT sweep=$shard_sweep"
  DEPTH="$DEPTH" \
  CONDITION="$CONDITION" \
  SHARD_COUNT="$SHARD_COUNT" \
  SHARD_OFFSET="$index" \
  SWEEP_STALE_SANDBOXES="$shard_sweep" \
  bash "$SUBMIT_SCRIPT"
done
