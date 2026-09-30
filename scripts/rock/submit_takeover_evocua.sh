#!/usr/bin/env bash
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

DEPTH="${DEPTH:-0}"
CONDITION="${CONDITION:-notified}"
SHARD_COUNT="${SHARD_COUNT:-4}"
SHARD_OFFSET="${SHARD_OFFSET:-0}"

case "$CONDITION" in
  notified) case "$DEPTH" in 0|10|20) ;; *) echo "invalid EvoCUA notified depth: $DEPTH" >&2; exit 2 ;; esac ;;
  unaware) case "$DEPTH" in 0|5|10|15|20) ;; *) echo "invalid EvoCUA unaware depth: $DEPTH" >&2; exit 2 ;; esac ;;
  *) echo "invalid EvoCUA condition: $CONDITION" >&2; exit 2 ;;
esac
[[ "$SHARD_COUNT" =~ ^[1-9][0-9]*$ ]] || { echo "invalid SHARD_COUNT" >&2; exit 2; }
[[ "$SHARD_OFFSET" =~ ^[0-9]+$ ]] || { echo "invalid SHARD_OFFSET" >&2; exit 2; }
(( SHARD_OFFSET < SHARD_COUNT )) || { echo "SHARD_OFFSET must be below SHARD_COUNT" >&2; exit 2; }

export SMOKE_PHASE=full
export DERAIL_WORKLOAD=takeover
export AGENT_ID=evocua_32b
export OPENCUA_MODEL=EvoCUA
export OPENCUA_WEIGHTS_OSS_DIR="${OPENCUA_WEIGHTS_OSS_DIR:?set OPENCUA_WEIGHTS_OSS_DIR}"
export LOCAL_MODEL_CACHE_DIR="${LOCAL_MODEL_CACHE_DIR:-/tmp/derail-model-cache/EvoCUA-32B}"
export TAKEOVER_CONTEXT_CAP=49152
export VLLM_EXTRA_ARGS="${VLLM_EXTRA_ARGS:---max-model-len 49152 --disable-custom-all-reduce --enforce-eager}"
export TAKEOVER_TOKENIZE_MODE=vllm
export TAKEOVER_SOURCE_AGENT=evocua_32b
export TAKEOVER_TARGET_AGENT=evocua_32b
export TAKEOVER_ANNOTATOR=Jinxin
export TAKEOVER_BUILD_DIR=artifacts/derail_builds/evocua_32b_jinxin_takeover_full_v3
export TAKEOVER_DEPTH="$DEPTH"
export TAKEOVER_CONDITION="$CONDITION"
export TAKEOVER_SHARD_COUNT="$SHARD_COUNT"
export TAKEOVER_SHARD_OFFSET="$SHARD_OFFSET"
export JOB_NAME_SUFFIX="evocua-self-d${DEPTH}-${CONDITION}-s${SHARD_OFFSET}"
export COLLECTION_ID="${COLLECTION_ID:-takeover-evocua-self-d${DEPTH}-${CONDITION}-s${SHARD_OFFSET}of${SHARD_COUNT}}"
export PROXY_RESULTS_OSS_MOUNT="${PROXY_RESULTS_OSS_MOUNT:-/data/oss_bucket_0/${OSS_PREFIX:?set OSS_PREFIX}/DERAIL/results/takeover/failure_prefix_v1/evocua_32b/evocua_32b/${CONDITION}/d${DEPTH}}"

exec bash "$HERE/submit_derail_opencua_smoke.sh"
