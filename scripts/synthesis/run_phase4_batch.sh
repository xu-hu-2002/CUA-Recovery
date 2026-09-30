#!/usr/bin/env bash
# Usage: tmux new -d -s phase4_batch 'RECOVERY_RUN_ROOT=... RECOVERY_FIX_ROUND=N RECOVERY_GEN_OUT=... bash scripts/synthesis/run_phase4_batch.sh'
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
: "${RECOVERY_VM_DB_DIR:?need RECOVERY_VM_DB_DIR}"
: "${RECOVERY_VM_FILES_DIR:?need RECOVERY_VM_FILES_DIR}"
: "${RECOVERY_TMP_ROOT:?need RECOVERY_TMP_ROOT}"
: "${RECOVERY_RUN_ROOT:?need RECOVERY_RUN_ROOT (run dir holding reparse_02/, fix_02/, ...)}"
: "${RECOVERY_FIX_ROUND:?need RECOVERY_FIX_ROUND (the round to run now)}"
: "${RECOVERY_GEN_OUT:?need RECOVERY_GEN_OUT (generation bundle output dir)}"
TASKS="${RECOVERY_TASKS_JSON:-third_party/MyPCBench/tasks/final/all_tasks_with_grading.json}"
VARIABLES="${RECOVERY_VARIABLES_JSON:-third_party/MyPCBench/tasks/final/variables.json}"
WORLD_ID="${RECOVERY_WORLD_ID:-$("$PYTHON_BIN" - <<'PY'
import yaml; print(yaml.safe_load(open("configs/synthesis/task_ir_v1_extractor.yaml"))["v1"]["world_id"])
PY
)}"
ACCEPTED="${RECOVERY_ACCEPTED_DIR:-$(dirname "$RECOVERY_RUN_ROOT")/accepted}"
PREV_ROUND=$((RECOVERY_FIX_ROUND - 1))
FIX_DIR="$RECOVERY_RUN_ROOT/fix_0${RECOVERY_FIX_ROUND}"
PREV_DIR="$RECOVERY_RUN_ROOT/fix_0${PREV_ROUND}"
cd "$REPO_ROOT"
mkdir -p "$(dirname "$RECOVERY_GEN_OUT")"
LOG="$(dirname "$RECOVERY_GEN_OUT")/batch.log"
STATUS="$(dirname "$RECOVERY_GEN_OUT")/BATCH_STATUS"
step() { echo "[$(date -u +%FT%TZ)] $*" | tee -a "$LOG"; }
trap 'step "FAILED at step: ${CURRENT:-?}"; echo "failed:${CURRENT:-?}" > "$STATUS"' ERR
echo "running" > "$STATUS"

CURRENT="fix_round_${RECOVERY_FIX_ROUND}"
if [[ -f "$FIX_DIR/manifest.json" ]]; then
  step "fix round $RECOVERY_FIX_ROUND already has a manifest, skipping re-extraction"
else
  step "fix round $RECOVERY_FIX_ROUND: previous=$PREV_DIR"
  RECOVERY_FIX_PREVIOUS="$PREV_DIR" bash scripts/synthesis/run_task_ir_v1_fix.sh
fi

CURRENT="rubric_check_fix_0${RECOVERY_FIX_ROUND}"
step "rubric check on $FIX_DIR"
"$PYTHON_BIN" scripts/synthesis/check_task_ir_rubrics.py --ir-dir "$FIX_DIR/task_ir" --tasks "$TASKS" \
  --variables "$VARIABLES" --world-id "$WORLD_ID" --out "$FIX_DIR/rubric_check" \
  > "$FIX_DIR/rubric_check.log" 2>&1
tail -1 "$FIX_DIR/rubric_check.log" | tee -a "$LOG"

CURRENT="accept"
RUNS=(--run "$RECOVERY_RUN_ROOT/reparse_02")
for ((r = 2; r <= RECOVERY_FIX_ROUND; r++)); do RUNS+=(--run "$RECOVERY_RUN_ROOT/fix_0${r}"); done
step "accept: ${RUNS[*]}"
"$PYTHON_BIN" scripts/synthesis/accept_task_ir_v1.py "${RUNS[@]}" --out "$ACCEPTED" 2>&1 | tail -1 | tee -a "$LOG"

CURRENT="review_sheet"
step "human-review sheet for tasks without comparable rubric values"
"$PYTHON_BIN" scripts/synthesis/build_review_sheet.py "${RUNS[@]}" --tasks "$TASKS" \
  --out "$ACCEPTED/review_sheet.md" 2>&1 | tail -1 | tee -a "$LOG"

CURRENT="generate"
step "generation on $ACCEPTED/task_ir -> $RECOVERY_GEN_OUT"
"$PYTHON_BIN" scripts/synthesis/generate_tasks_v1.py --ir-dir "$ACCEPTED/task_ir" --tasks "$TASKS" \
  --out "$RECOVERY_GEN_OUT" > "$RECOVERY_GEN_OUT.log" 2>&1
tail -1 "$RECOVERY_GEN_OUT.log" | tee -a "$LOG"

step "done"
echo "done" > "$STATUS"
