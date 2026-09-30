#!/usr/bin/env bash
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEPTH="${DEPTH:-0}"
CONDITION="${CONDITION:-unaware}"
SHARD_COUNT="${SHARD_COUNT:-1}"
SHARD_OFFSET="${SHARD_OFFSET:-0}"

case "$DEPTH" in 0|5|10|15|20|25) ;; *) echo "invalid depth: $DEPTH" >&2; exit 2 ;; esac
case "$CONDITION" in unaware|notified) ;; *) echo "invalid condition: $CONDITION" >&2; exit 2 ;; esac
if [[ "$CONDITION" == notified && "$DEPTH" != 0 && "$DEPTH" != 10 && "$DEPTH" != 20 ]]; then
  echo "notified is frozen to depth 0/10/20" >&2; exit 2
fi
[[ "$SHARD_COUNT" =~ ^[1-9][0-9]*$ ]] || { echo "invalid SHARD_COUNT" >&2; exit 2; }
[[ "$SHARD_OFFSET" =~ ^[0-9]+$ ]] || { echo "invalid SHARD_OFFSET" >&2; exit 2; }
(( SHARD_OFFSET < SHARD_COUNT )) || { echo "SHARD_OFFSET must be below SHARD_COUNT" >&2; exit 2; }

export SMOKE_PHASE=full
export DERAIL_WORKLOAD=takeover
export AGENT_ID=qwen3_5_35b_a3b
export OPENCUA_MODEL=Qwen/Qwen3.5-35B-A3B
export OPENCUA_WEIGHTS_OSS_DIR="${OPENCUA_WEIGHTS_OSS_DIR:-/data/oss_bucket_0/dev/AI4S/model/Qwen/Qwen3.5-35B-A3B}"
export LOCAL_MODEL_CACHE_DIR="${LOCAL_MODEL_CACHE_DIR:-/tmp/derail-model-cache/qwen3.5-35b-a3b}"
export VLLM_EXTRA_ARGS="${VLLM_EXTRA_ARGS:---max-model-len 49152 --max-num-seqs 1 --disable-custom-all-reduce --enforce-eager --enable-auto-tool-choice --tool-call-parser hermes}"
export TAKEOVER_TOKENIZE_MODE=vllm
export TAKEOVER_CONTEXT_CAP=49152
export MYPCBENCH_QWEN_MAX_TOKENS=4096
export MYPCBENCH_QWEN_HISTORY_N=100
export MYPCBENCH_QWEN_CONTEXT_POLICY=tokenize_oldest_first_v1
export TAKEOVER_SOURCE_AGENT=qwen3_5_35b_a3b
export TAKEOVER_TARGET_AGENT=qwen3_5_35b_a3b
export TAKEOVER_ANNOTATOR=licheng
export TAKEOVER_BUILD_DIR=artifacts/derail_builds/qwen3_5_35b_a3b_licheng_takeover_full_v5
export TAKEOVER_DEPTH="$DEPTH"
export TAKEOVER_CONDITION="$CONDITION"
export TAKEOVER_SHARD_COUNT="$SHARD_COUNT"
export TAKEOVER_SHARD_OFFSET="$SHARD_OFFSET"
export COLLECTION_ID="${COLLECTION_ID:-takeover-qwen35-d${DEPTH}-${CONDITION}-s${SHARD_OFFSET}of${SHARD_COUNT}}"
export PROXY_RESULTS_OSS_MOUNT="${PROXY_RESULTS_OSS_MOUNT:-/data/oss_bucket_0/${OSS_PREFIX:?set OSS_PREFIX}/DERAIL/results/takeover/failure_prefix_v1/qwen3_5_35b_a3b/qwen3_5_35b_a3b/${CONDITION}/d${DEPTH}}"

exec bash "$HERE/submit_derail_opencua_smoke.sh"
