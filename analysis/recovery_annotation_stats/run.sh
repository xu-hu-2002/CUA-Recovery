#!/usr/bin/env bash
set -euo pipefail

RECOVERY_ROOT="${RECOVERY_ROOT:-${1:-}}"
: "${RECOVERY_ROOT:?export RECOVERY_ROOT=/path/to/RECOVERY (or pass it as \$1)}"
: "${RECOVERY_ANALYSIS_PYTHON:?export RECOVERY_ANALYSIS_PYTHON=/path/to/analysis/python}"

export RECOVERY_BUILDS_DIR="${RECOVERY_BUILDS_DIR:-$RECOVERY_ROOT/artifacts/recovery_builds}"
export RECOVERY_ANALYSIS_OUT="${RECOVERY_ANALYSIS_OUT:?export RECOVERY_ANALYSIS_OUT=/path/to/output}"

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export PYTHONPATH="$HERE/src${PYTHONPATH:+:$PYTHONPATH}"
exec "$RECOVERY_ANALYSIS_PYTHON" -m "recovery_annot_stats.${RECOVERY_ANALYSIS_ENTRY:-main}" "${@:2}"
