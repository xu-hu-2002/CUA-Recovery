#!/usr/bin/env bash
# Usage: tmux new -d -s realize 'RECOVERY_GEN_BUNDLE=<bundle> bash scripts/synthesis/run_realization_v1.sh [--limit N] [--all]'
set -euo pipefail
OPENAI_BASE_URL="${OPENAI_BASE_URL:-}"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
CRED_ENV="${RECOVERY_CRED_ENV:-$HOME/.recovery_creds.env}"
PYTHON_BIN="${PYTHON_BIN:-python}"
: "${RECOVERY_GEN_BUNDLE:?need RECOVERY_GEN_BUNDLE (generation bundle directory)}"
TASKS="${RECOVERY_TASKS_JSON:-third_party/MyPCBench/tasks/final/all_tasks_with_grading.json}"

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
export RECOVERY_OPENAI_API_PURPOSE=realization
echo "bundle=$RECOVERY_GEN_BUNDLE key_len=${#OPENAI_API_KEY} base_url_len=${#OPENAI_BASE_URL}"
cd "$REPO_ROOT"
"$PYTHON_BIN" scripts/synthesis/realize_tasks_v1.py --bundle "$RECOVERY_GEN_BUNDLE" --tasks "$TASKS" \
  --call-model "$@" 2>&1 | tee -a "$RECOVERY_GEN_BUNDLE/realization.log"
