#!/usr/bin/env bash
# Usage: RECOVERY_FIX_PREVIOUS=<run> RECOVERY_FIX_ROUND=N tmux new -d -s task_ir_fix 'bash scripts/synthesis/run_task_ir_v1_fix.sh'
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
CRED_ENV="${RECOVERY_CRED_ENV:-$HOME/.recovery_creds.env}"
PYTHON_BIN="${PYTHON_BIN:-python}"
: "${RECOVERY_VM_DB_DIR:?need RECOVERY_VM_DB_DIR}"
: "${RECOVERY_FIX_PREVIOUS:?need RECOVERY_FIX_PREVIOUS (previous run directory with rubric_check/)}"
: "${RECOVERY_FIX_ROUND:?need RECOVERY_FIX_ROUND (1-based round number)}"
OUTPUT_DIR="${RECOVERY_FIX_OUTPUT_DIR:-$(dirname "$RECOVERY_FIX_PREVIOUS")/fix_0${RECOVERY_FIX_ROUND}}"
WORKERS="${RECOVERY_EXTRACTION_WORKERS:-1}"

if [[ -f "$CRED_ENV" ]]; then
  set -a; . "$CRED_ENV"; set +a
elif [[ -f "$REPO_ROOT/.env" ]]; then
  eval "$("$PYTHON_BIN" - "$REPO_ROOT/.env" <<'PY'
import re, shlex, sys
wanted = {"OPENAI_API_KEY", "OPENAI_BASE_URL"}
for line in open(sys.argv[1], encoding="utf-8"):
    match = re.match(r'^\s*([A-Z_][A-Z0-9_]*)\s*:\s*"?([^"]*?)"?\s*$', line)
    if match and match.group(1) in wanted and match.group(2):
        print("export %s=%s" % (match.group(1), shlex.quote(match.group(2))))
PY
)"
fi
[[ -n "${OPENAI_API_KEY:-}" ]] || { echo "FATAL: OPENAI_API_KEY is empty" >&2; exit 1; }
case "${OPENAI_BASE_URL:-}" in
  http://*|https://*|"") ;;
  *) echo "WARNING: OPENAI_BASE_URL is not an http(s) URL (length ${#OPENAI_BASE_URL}); ignoring it" >&2; OPENAI_BASE_URL="" ;;
esac

export RECOVERY_OPENAI_API_APPROVED=1
export RECOVERY_OPENAI_API_PURPOSE=task_ir_extraction
mkdir -p "$OUTPUT_DIR"
echo "round=$RECOVERY_FIX_ROUND previous=$RECOVERY_FIX_PREVIOUS output=$OUTPUT_DIR key_len=${#OPENAI_API_KEY}"
cd "$REPO_ROOT"
"$PYTHON_BIN" scripts/synthesis/fix_task_ir_v1.py --previous-run "$RECOVERY_FIX_PREVIOUS" \
  --output-dir "$OUTPUT_DIR" --round "$RECOVERY_FIX_ROUND" --workers "$WORKERS" "$@" 2>&1 | tee "$OUTPUT_DIR/fix.log"
