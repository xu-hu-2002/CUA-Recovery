#!/usr/bin/env bash
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

DEPTH="${DEPTH:-0}"
CONDITION="${CONDITION:-unaware}"
SHARD_COUNT="${SHARD_COUNT:-1}"
SHARD_OFFSET="${SHARD_OFFSET:-0}"
case "$DEPTH" in 0|5|10|15|20|25) ;; *) echo "invalid depth: $DEPTH" >&2; exit 2 ;; esac
case "$CONDITION" in unaware) ;; *) echo "invalid Kimi main condition: $CONDITION" >&2; exit 2 ;; esac
[[ "$SHARD_COUNT" =~ ^[1-9][0-9]*$ ]] || { echo "invalid SHARD_COUNT" >&2; exit 2; }
[[ "$SHARD_OFFSET" =~ ^[0-9]+$ ]] || { echo "invalid SHARD_OFFSET" >&2; exit 2; }
(( SHARD_OFFSET < SHARD_COUNT )) || { echo "SHARD_OFFSET must be below SHARD_COUNT" >&2; exit 2; }

export AGENT_ID=kimi_k3_cuabash
export DERAIL_WORKLOAD=takeover
export DERAIL_ROCK_TOPOLOGY=proxy
export TAKEOVER_SOURCE_AGENT=kimi_k3
export TAKEOVER_TARGET_AGENT=kimi_k3
export TAKEOVER_ANNOTATOR=kangshuo
export TAKEOVER_BUILD_DIR=artifacts/derail_builds/kimi_k3_kangshuo_takeover_full_v3
export TAKEOVER_DEPTH="$DEPTH"
export TAKEOVER_CONDITION="$CONDITION"
export TAKEOVER_SHARD_COUNT="$SHARD_COUNT"
export TAKEOVER_SHARD_OFFSET="$SHARD_OFFSET"
export TAKEOVER_TRAJECTORY_ID_FILTER="${TAKEOVER_TRAJECTORY_ID_FILTER:-}"
export TAKEOVER_TOKENIZE_BASE_URL="${KIMI_TOKENIZE_BASE_URL:-${OPENAI_BASE_URL:-}}"
export TAKEOVER_TOKENIZE_MODE=chat_usage
export TAKEOVER_CONTEXT_CAP="${KIMI_EXPERIMENT_CONTEXT_CAP:-262144}"
export NUM_VMS_OVERRIDE=1
export MAX_STEPS="${MAX_STEPS:-}"
export COLLECTION_ID="${COLLECTION_ID:-takeover-kimi-self-d${DEPTH}-${CONDITION}-s${SHARD_OFFSET}of${SHARD_COUNT}}"
export PROXY_RESULTS_OSS_MOUNT="${PROXY_RESULTS_OSS_MOUNT:-/data/oss_bucket_0/${OSS_PREFIX:-<oss-prefix>}/DERAIL/results/takeover/failure_prefix_v1/kimi_k3/kimi_k3/${CONDITION}/d${DEPTH}}"

export IGNORE_LIST="${IGNORE_LIST:-third_party/*,takeovewr_annotation/*,artifacts/raw_rollouts/*,artifacts/derail_builds/*,artifacts/model_outputs/*,artifacts/takeover/bundles/*,draft/*,runs/*,.git/*,.venv*,.env,docs/*,*.log,*.pyc,__pycache__/*,.props_*,.cluster_*}"

if [[ "${DRY_RUN:-1}" == "0" && -z "$TAKEOVER_TOKENIZE_BASE_URL" ]]; then
  echo "ERROR: OPENAI_BASE_URL is required for Kimi upstream token counting" >&2
  exit 2
fi

exec bash "$HERE/submit_derail_rock_nebula.sh"
