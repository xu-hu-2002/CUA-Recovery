#!/usr/bin/env bash
# Usage: tmux new -s task_ir_v1 -d 'bash scripts/synthesis/run_task_ir_v1_extraction.sh'
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
CRED_ENV="${RECOVERY_CRED_ENV:-$HOME/.recovery_creds.env}"
PYTHON_BIN="${PYTHON_BIN:-python}"
RUN_ID="${RECOVERY_EXTRACTION_RUN_ID:-dev_v1_run_$(date -u +%Y%m%dT%H%M%SZ)}"
OUTPUT_DIR="${RECOVERY_EXTRACTION_OUTPUT_DIR:-$REPO_ROOT/data/synthesis/task_ir_v1/$RUN_ID}"
WORKERS="${RECOVERY_EXTRACTION_WORKERS:-1}"
: "${RECOVERY_VM_DB_DIR:?need RECOVERY_VM_DB_DIR (directory of <app>.sqlite copies)}"

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
OPENAI_BASE_URL="${OPENAI_BASE_URL:-}"
echo "run_id=$RUN_ID output=$OUTPUT_DIR db_dir=$RECOVERY_VM_DB_DIR base_url_len=${#OPENAI_BASE_URL} key_len=${#OPENAI_API_KEY}"
cd "$REPO_ROOT"
SELECTION="--dev-sample"
for arg in "$@"; do [[ "$arg" == "--task-id" ]] && SELECTION=""; done
[[ "${RECOVERY_EXTRACTION_ALL_TASKS:-0}" == "1" ]] && SELECTION=""
"$PYTHON_BIN" scripts/synthesis/extract_task_ir_v1.py \
  --tasks third_party/MyPCBench/tasks/final/all_tasks_with_grading.json \
  --splits data/synthesis/source_splits.json \
  $SELECTION --call-model --workers "$WORKERS" \
  --output-dir "$OUTPUT_DIR" "$@" 2>&1 | tee "$OUTPUT_DIR/extract.log"
