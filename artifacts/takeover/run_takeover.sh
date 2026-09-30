#!/usr/bin/env bash
# Compatibility entrypoint; production packages execute the versioned launcher in scripts/.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
exec bash "$REPO_ROOT/scripts/rock/run_takeover.sh" "$@"
