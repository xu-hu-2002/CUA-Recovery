#!/usr/bin/env bash
# Usage: tmux new -d -s phase4_batch 'DERAIL_RUN_ROOT=... DERAIL_FIX_ROUND=N DERAIL_GEN_OUT=... bash scripts/synthesis/run_phase4_batch.sh'
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
: "${DERAIL_VM_DB_DIR:?need DERAIL_VM_DB_DIR}"
: "${DERAIL_VM_FILES_DIR:?need DERAIL_VM_FILES_DIR}"
: "${DERAIL_TMP_ROOT:?need DERAIL_TMP_ROOT}"
: "${DERAIL_RUN_ROOT:?need DERAIL_RUN_ROOT (run dir holding reparse_02/, fix_02/, ...)}"
: "${DERAIL_FIX_ROUND:?need DERAIL_FIX_ROUND (the round to run now)}"
: "${DERAIL_GEN_OUT:?need DERAIL_GEN_OUT (generation bundle output dir)}"
TASKS="${DERAIL_TASKS_JSON:-third_party/MyPCBench/tasks/final/all_tasks_with_grading.json}"
VARIABLES="${DERAIL_VARIABLES_JSON:-third_party/MyPCBench/tasks/final/variables.json}"
WORLD_ID="${DERAIL_WORLD_ID:-$("$PYTHON_BIN" - <<'PY'
import yaml; print(yaml.safe_load(open("configs/synthesis/task_ir_v1_extractor.yaml"))["v1"]["world_id"])
PY
)}"
ACCEPTED="${DERAIL_ACCEPTED_DIR:-$(dirname "$DERAIL_RUN_ROOT")/accepted}"
PREV_ROUND=$((DERAIL_FIX_ROUND - 1))
FIX_DIR="$DERAIL_RUN_ROOT/fix_0${DERAIL_FIX_ROUND}"
PREV_DIR="$DERAIL_RUN_ROOT/fix_0${PREV_ROUND}"
cd "$REPO_ROOT"
mkdir -p "$(dirname "$DERAIL_GEN_OUT")"
LOG="$(dirname "$DERAIL_GEN_OUT")/batch.log"
STATUS="$(dirname "$DERAIL_GEN_OUT")/BATCH_STATUS"
step() { echo "[$(date -u +%FT%TZ)] $*" | tee -a "$LOG"; }
trap 'step "FAILED at step: ${CURRENT:-?}"; echo "failed:${CURRENT:-?}" > "$STATUS"' ERR
echo "running" > "$STATUS"

CURRENT="fix_round_${DERAIL_FIX_ROUND}"
if [[ -f "$FIX_DIR/manifest.json" ]]; then
  step "fix round $DERAIL_FIX_ROUND already has a manifest, skipping re-extraction"
else
  step "fix round $DERAIL_FIX_ROUND: previous=$PREV_DIR"
  DERAIL_FIX_PREVIOUS="$PREV_DIR" bash scripts/synthesis/run_task_ir_v1_fix.sh
fi

CURRENT="rubric_check_fix_0${DERAIL_FIX_ROUND}"
step "rubric check on $FIX_DIR"
"$PYTHON_BIN" scripts/synthesis/check_task_ir_rubrics.py --ir-dir "$FIX_DIR/task_ir" --tasks "$TASKS" \
  --variables "$VARIABLES" --world-id "$WORLD_ID" --out "$FIX_DIR/rubric_check" \
  > "$FIX_DIR/rubric_check.log" 2>&1
tail -1 "$FIX_DIR/rubric_check.log" | tee -a "$LOG"

CURRENT="accept"
RUNS=(--run "$DERAIL_RUN_ROOT/reparse_02")
for ((r = 2; r <= DERAIL_FIX_ROUND; r++)); do RUNS+=(--run "$DERAIL_RUN_ROOT/fix_0${r}"); done
step "accept: ${RUNS[*]}"
"$PYTHON_BIN" scripts/synthesis/accept_task_ir_v1.py "${RUNS[@]}" --out "$ACCEPTED" 2>&1 | tail -1 | tee -a "$LOG"

CURRENT="review_sheet"
step "human-review sheet for tasks without comparable rubric values"
"$PYTHON_BIN" scripts/synthesis/build_review_sheet.py "${RUNS[@]}" --tasks "$TASKS" \
  --out "$ACCEPTED/review_sheet.md" 2>&1 | tail -1 | tee -a "$LOG"

CURRENT="generate"
step "generation on $ACCEPTED/task_ir -> $DERAIL_GEN_OUT"
"$PYTHON_BIN" scripts/synthesis/generate_tasks_v1.py --ir-dir "$ACCEPTED/task_ir" --tasks "$TASKS" \
  --out "$DERAIL_GEN_OUT" > "$DERAIL_GEN_OUT.log" 2>&1
tail -1 "$DERAIL_GEN_OUT.log" | tee -a "$LOG"

step "done"
echo "done" > "$STATUS"
