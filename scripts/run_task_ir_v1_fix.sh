#!/usr/bin/env bash
# One approved fix round of Task-IR v1 extraction (execution doc v1.2 section 5.3).
#
# Same credential handling as run_task_ir_v1_extraction.sh (~/.derail_creds.env or the colon
# format .env; values exported, never printed).  Running this script is the explicit per-run
# approval.  Always run it under tmux:
#   DERAIL_FIX_PREVIOUS=<run dir with rubric_check/> DERAIL_FIX_ROUND=2 \
#     tmux new -d -s task_ir_fix2 'bash scripts/run_task_ir_v1_fix.sh'
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CRED_ENV="${DERAIL_CRED_ENV:-$HOME/.derail_creds.env}"
PYTHON_BIN="${PYTHON_BIN:-python}"
: "${DERAIL_VM_DB_DIR:?need DERAIL_VM_DB_DIR}"
: "${DERAIL_FIX_PREVIOUS:?need DERAIL_FIX_PREVIOUS (previous run directory with rubric_check/)}"
: "${DERAIL_FIX_ROUND:?need DERAIL_FIX_ROUND (1-based round number)}"
OUTPUT_DIR="${DERAIL_FIX_OUTPUT_DIR:-$(dirname "$DERAIL_FIX_PREVIOUS")/fix_0${DERAIL_FIX_ROUND}}"
WORKERS="${DERAIL_EXTRACTION_WORKERS:-1}"

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

export DERAIL_OPENAI_API_APPROVED=1
export DERAIL_OPENAI_API_PURPOSE=task_ir_extraction
mkdir -p "$OUTPUT_DIR"
echo "round=$DERAIL_FIX_ROUND previous=$DERAIL_FIX_PREVIOUS output=$OUTPUT_DIR key_len=${#OPENAI_API_KEY}"
cd "$REPO_ROOT"
"$PYTHON_BIN" scripts/fix_task_ir_v1.py --previous-run "$DERAIL_FIX_PREVIOUS" \
  --output-dir "$OUTPUT_DIR" --round "$DERAIL_FIX_ROUND" --workers "$WORKERS" "$@" 2>&1 | tee "$OUTPUT_DIR/fix.log"
