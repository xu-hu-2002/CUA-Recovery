#!/usr/bin/env bash
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

DEPTH="${DEPTH:-0}"
CONDITION="${CONDITION:-unaware}"
SHARD_COUNT="${SHARD_COUNT:-1}"
SHARD_OFFSET="${SHARD_OFFSET:-0}"
case "$DEPTH" in 0|5|10|15|20|25) ;; *) echo "invalid depth: $DEPTH" >&2; exit 2 ;; esac
case "$CONDITION" in unaware) ;; *) echo "invalid OpenCUA main condition: $CONDITION" >&2; exit 2 ;; esac
[[ "$SHARD_COUNT" =~ ^[1-9][0-9]*$ ]] || { echo "invalid SHARD_COUNT" >&2; exit 2; }
[[ "$SHARD_OFFSET" =~ ^[0-9]+$ ]] || { echo "invalid SHARD_OFFSET" >&2; exit 2; }
(( SHARD_OFFSET < SHARD_COUNT )) || { echo "SHARD_OFFSET must be below SHARD_COUNT" >&2; exit 2; }

export SMOKE_PHASE=full
export DERAIL_WORKLOAD=takeover
export TAKEOVER_SOURCE_AGENT=opencua_72b
export TAKEOVER_TARGET_AGENT=opencua_72b
export TAKEOVER_ANNOTATOR=annotator5
export TAKEOVER_BUILD_DIR=artifacts/derail_builds/opencua72b_annotator5_takeover
export TAKEOVER_DEPTH="$DEPTH"
export TAKEOVER_CONDITION="$CONDITION"
export TAKEOVER_SHARD_COUNT="$SHARD_COUNT"
export TAKEOVER_SHARD_OFFSET="$SHARD_OFFSET"
export COLLECTION_ID="${COLLECTION_ID:-takeover-opencua-d${DEPTH}-${CONDITION}-s${SHARD_OFFSET}of${SHARD_COUNT}}"
export PROXY_RESULTS_OSS_MOUNT="${PROXY_RESULTS_OSS_MOUNT:-/data/oss_bucket_0/${OSS_PREFIX:-<oss-prefix>}/DERAIL/results/takeover/failure_prefix_v1/opencua_72b/opencua_72b/${CONDITION}/d${DEPTH}}"

exec bash "$HERE/submit_derail_opencua_smoke.sh"
