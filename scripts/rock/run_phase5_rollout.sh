#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
MODEL="${PHASE5_MODEL:?PHASE5_MODEL is required}"
BATCH="${PHASE5_BATCH:?PHASE5_BATCH is required}"
SHARD_COUNT="${PHASE5_SHARD_COUNT:-1}"
SHARD_INDEX="${PHASE5_SHARD_INDEX:-0}"
SMOKE_MODE="${PHASE5_SMOKE_MODE:-single_e2e}"
WORKLOAD_DIR="${PHASE5_WORKLOAD_DIR:-/tmp/derail-phase5-workloads}"
WORKLOAD_PATH="${WORKLOAD_DIR}/${BATCH}-${MODEL}-s${SHARD_INDEX}of${SHARD_COUNT}.json"
SMOKE_CONFIG="${REPO_ROOT}/configs/phase5/smoke_v1.json"

mkdir -p "$WORKLOAD_DIR"
python3 "${REPO_ROOT}/scripts/phase5/build_rollout_workload.py" \
  --model "$MODEL" --shard-count "$SHARD_COUNT" --shard-index "$SHARD_INDEX" \
  --out "$WORKLOAD_PATH"

COMBINATIONS=()
while IFS= read -r combination_id; do
  COMBINATIONS+=("$combination_id")
done < <(python3 - "$SMOKE_CONFIG" "$SMOKE_MODE" <<'PY'
import json, sys
config = json.load(open(sys.argv[1], encoding="utf-8"))
for combination_id in config.get(sys.argv[2], []):
    print(combination_id)
PY
)

if [[ "$SMOKE_MODE" != "full" && "${#COMBINATIONS[@]}" -eq 0 ]]; then
  printf 'unknown or empty PHASE5_SMOKE_MODE=%s\n' "$SMOKE_MODE" >&2
  exit 2
fi

COMMAND=(python3 "${REPO_ROOT}/scripts/phase5/run_rollout_workload.py"
  --workload "$WORKLOAD_PATH" --batch "$BATCH")
for combination_id in "${COMBINATIONS[@]}"; do
  COMMAND+=(--combination-id "$combination_id")
done

if [[ "${DRY_RUN:-0}" == "1" ]]; then
  printf 'PHASE5_DRY_RUN model=%s batch=%s shard=%s/%s smoke=%s combinations=%s\n' \
    "$MODEL" "$BATCH" "$SHARD_INDEX" "$SHARD_COUNT" "$SMOKE_MODE" "${#COMBINATIONS[@]}"
  printf 'workload=%s manifest_sha256=%s\n' "$WORKLOAD_PATH" \
    "$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["source_manifest_sha256"])' "$WORKLOAD_PATH")"
  exit 0
fi

exec "${COMMAND[@]}"
