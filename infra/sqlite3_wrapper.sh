#!/usr/bin/env bash
# DERAIL wrapper for the sqlite3 CLI (execution doc v1.2 section 4.2b).
#
# Installed as /usr/bin/sqlite3 after the real binary is renamed to /usr/bin/sqlite3.real
# (infra/README.md).  Records who read what: action index (from the control API's cursor
# file), target database, the SQL (argument or stdin), and the number of output lines, as one
# JSON line in /data/_trace/cli.jsonl; then execs the real program with the same arguments and
# stdin so behaviour is unchanged.  Writes need no handling here: the triggers live in the
# database file and log any client's changes.
#
# Environment overrides (same names as the tracer): DERAIL_TRACE_DIR, DERAIL_CURSOR_FILE,
# DERAIL_SQLITE3_REAL.
set -u
REAL="${DERAIL_SQLITE3_REAL:-/usr/bin/sqlite3.real}"
TRACE_DIR="${DERAIL_TRACE_DIR:-/data/_trace}"
CURSOR_FILE="${DERAIL_CURSOR_FILE:-/data/_derail/cursor.json}"
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
  stdin_file=$(mktemp "${TMPDIR:-/tmp}/derail-sqlite3.XXXXXX")
  cat > "$stdin_file"
  if [ -z "$sql" ]; then sql=$(cat "$stdin_file"); fi
fi

mkdir -p "$TRACE_DIR" 2>/dev/null
out_file=$(mktemp "${TMPDIR:-/tmp}/derail-sqlite3-out.XXXXXX")
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
