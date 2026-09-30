#!/usr/bin/env bash
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export SUBMIT_SCRIPT="$HERE/submit_takeover_evocua.sh"
export CONDITION="${CONDITION:-notified}"

exec bash "$HERE/submit_takeover_opencua_fleet.sh"
