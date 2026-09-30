#!/usr/bin/env bash
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export SUBMIT_SCRIPT="$HERE/submit_takeover_claude.sh"

exec bash "$HERE/submit_takeover_opencua_fleet.sh"
