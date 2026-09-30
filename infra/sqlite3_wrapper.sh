#!/usr/bin/env bash
set -u
REAL="${RECOVERY_SQLITE3_REAL:-/usr/bin/sqlite3.real}"
TRACE_DIR="${RECOVERY_TRACE_DIR:-/data/_trace}"
CURSOR_FILE="${RECOVERY_CURSOR_FILE:-/data/_recovery/cursor.json}"
LOG="$TRACE_DIR/cli.jsonl"

action_index=-1
if [ -r "$CURSOR_FILE" ]; then
  action_index=$(python3 -c 'import json,sys;print(int(json.load(open(sys.argv[1])).get("action_index",-1)))' "$CURSOR_FILE" 2>/dev/null || echo -1)
fi

db=""
sql=""
for arg in "$@"; do
  case "$arg" in
    -*) ;;
    *) if [ -z "$db" ]; then db="$arg"; else sql="$arg"; fi ;;
  esac
done

stdin_file=""
if [ ! -t 0 ]; then
  stdin_file=$(mktemp "${TMPDIR:-/tmp}/recovery-sqlite3.XXXXXX")
  cat > "$stdin_file"
  if [ -z "$sql" ]; then sql=$(cat "$stdin_file"); fi
fi

mkdir -p "$TRACE_DIR" 2>/dev/null
out_file=$(mktemp "${TMPDIR:-/tmp}/recovery-sqlite3-out.XXXXXX")
if [ -n "$stdin_file" ]; then
  "$REAL" "$@" < "$stdin_file" | tee "$out_file"
  status=${PIPESTATUS[0]}
else
  "$REAL" "$@" | tee "$out_file"
  status=${PIPESTATUS[0]}
fi
rows=$(wc -l < "$out_file" | tr -d ' ')

python3 - "$LOG" "$action_index" "$db" "$sql" "$rows" "$status" <<'PY' 2>/dev/null
import json, sys, time
log, action_index, db, sql, rows, status = sys.argv[1:7]
with open(log, "a", encoding="utf-8") as handle:
    handle.write(json.dumps({"ts": time.time(), "source": "cli", "action_index": int(action_index),
                             "db": db, "sql": sql[:4000], "rows_returned": int(rows),
                             "exit_status": int(status)}) + "\n")
PY
rm -f "$out_file" "$stdin_file"
exit "$status"
