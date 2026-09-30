#!/usr/bin/env bash
# Wrapper that pins the analysis interpreter and the two required path variables.
# Nothing here is a personal path fallback: DERAIL_ROOT must already be exported,
# or be passed as the first argument.
set -euo pipefail

DERAIL_ROOT="${DERAIL_ROOT:-${1:-}}"
: "${DERAIL_ROOT:?export DERAIL_ROOT=/path/to/DERAIL (or pass it as \$1)}"
: "${DERAIL_ANALYSIS_PYTHON:?export DERAIL_ANALYSIS_PYTHON=/path/to/analysis/python}"

export DERAIL_BUILDS_DIR="${DERAIL_BUILDS_DIR:-$DERAIL_ROOT/artifacts/derail_builds}"
export DERAIL_ANALYSIS_OUT="${DERAIL_ANALYSIS_OUT:?export DERAIL_ANALYSIS_OUT=/path/to/output}"

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export PYTHONPATH="$HERE/src${PYTHONPATH:+:$PYTHONPATH}"
exec "$DERAIL_ANALYSIS_PYTHON" -m "derail_annot_stats.${DERAIL_ANALYSIS_ENTRY:-main}" "${@:2}"
