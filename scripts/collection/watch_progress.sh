#!/usr/bin/env bash
# Usage (inside tmux): bash scripts/collection/watch_progress.sh [COLLECTION_ID | --judge [ID] [--once]]

set -u

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
OUTPUT_ROOT="${OUTPUT_ROOT:-${REPO_ROOT}/artifacts/raw_rollouts/mypcbench}"
REFRESH="${REFRESH:-15}"
BAR_WIDTH="${BAR_WIDTH:-34}"

ONCE=0
MODE="${MODE:-rollout}"
ARGS=()
for argument in "$@"; do
  case "$argument" in
    --once) ONCE=1 ;;
    --judge) MODE=judge ;;
    --rollout) MODE=rollout ;;
    *) ARGS+=("$argument") ;;
  esac
done
set -- "${ARGS[@]+"${ARGS[@]}"}"

collection_id="${1:-}"
if [[ -z "$collection_id" ]]; then
  collection_id="$(ls -t "$OUTPUT_ROOT" 2>/dev/null | head -1)"
  [[ -n "$collection_id" ]] || { printf '找不到任何 collection：%s\n' "$OUTPUT_ROOT" >&2; exit 1; }
fi
run_root="${OUTPUT_ROOT}/${collection_id}"
if [[ ! -d "$run_root" ]]; then
  if (( ONCE )); then
    printf '目录还不存在：%s\n' "$run_root" >&2
    exit 1
  fi
  printf '\n  等待 %s 出现……（采集正在启动模型，约需 5 分钟）\n' "$collection_id"
  while [[ ! -d "$run_root" ]]; do sleep "$REFRESH"; done
fi

manifest="${run_root}/collection_manifest.json"
repeats=1
tasks_total=""
tasks_total_source="manifest"
if [[ -f "$manifest" ]]; then
  repeats="$(sed -n 's/.*"repeats": *\([0-9]*\).*/\1/p' "$manifest" | head -1)"
  [[ -n "$repeats" ]] || repeats=1
  tasks_total="$(sed -n 's/.*"tasks_total": *\([0-9]*\).*/\1/p' "$manifest" | head -1)"
fi
if [[ -n "${TASKS_TOTAL:-}" ]]; then
  tasks_total="$TASKS_TOTAL"
  tasks_total_source="TASKS_TOTAL"
elif [[ -z "$tasks_total" ]]; then
  tasks_total=184
  tasks_total_source="回退默认（manifest 没有 tasks_total；子集采集请设 TASKS_TOTAL=<条数>）"
fi
episodes_total=$((tasks_total * repeats))

run_agents=""
if [[ -f "$manifest" ]]; then
  run_agents="$(python3 - "$manifest" <<'PY' 2>/dev/null || true
import json, sys
try:
    agents = json.load(open(sys.argv[1])).get("agents") or []
except Exception:
    agents = []
print(" ".join(str(a) for a in agents))
PY
)"
fi

in_this_run() {
  [[ -z "$run_agents" ]] && return 0
  [[ " $run_agents " == *" $1 "* ]]
}

if [[ "$MODE" == "judge" ]]; then
  JUDGE_PRICE_IN="${JUDGE_PRICE_IN:-2.0}"
  JUDGE_PRICE_CACHED="${JUDGE_PRICE_CACHED:-0.2}"
  JUDGE_PRICE_OUT="${JUDGE_PRICE_OUT:-12.0}"
  judge_marker="${run_root}/.judge_started_at"
  judge_started="$(cat "$judge_marker" 2>/dev/null || echo 0)"

  while true; do
    output="$(python3 - "$run_root" "$judge_started" "$BAR_WIDTH" \
                "$JUDGE_PRICE_IN" "$JUDGE_PRICE_CACHED" "$JUDGE_PRICE_OUT" <<'PY' 2>/dev/null
import json, os, pathlib, sys, time

run_root = pathlib.Path(sys.argv[1])
started = float(sys.argv[2] or 0)
width = int(sys.argv[3])
p_in, p_cached, p_out = (float(sys.argv[i]) for i in (4, 5, 6))

def bar(done, total):
    filled = min(width, done * width // total) if total else 0
    return "█" * filled + "░" * (width - filled)

groups = {}
usage = {"calls": 0, "prompt": 0, "cached": 0, "completion": 0}
judged_since_start = 0
newest = 0.0

for bundle in run_root.rglob("rubric_bundle.json"):
    task_dir = bundle.parent
    vm = str(task_dir.parent.relative_to(run_root))
    g = groups.setdefault(vm, {"total": 0, "judged": 0, "errors": 0,
                               "scores": [], "perfect": 0})
    g["total"] += 1

    result = task_dir / "rubric_judge_result.json"
    if not result.exists():
        continue
    g["judged"] += 1
    mtime = result.stat().st_mtime
    newest = max(newest, mtime)
    if mtime >= started > 0:
        judged_since_start += 1
    try:
        raw = json.loads(result.read_text()).get("result")
    except Exception:
        raw = None
    if isinstance(raw, (int, float)) and not isinstance(raw, bool):
        score = max(0.0, min(1.0, float(raw) / 100.0))
        g["scores"].append(score)
        if score >= 0.999:
            g["perfect"] += 1
    else:
        g["errors"] += 1

    debug = task_dir / "osworld_full_traj_result.json"
    if debug.exists():
        try:
            u = json.loads(debug.read_text()).get("usage") or {}
            for k in usage:
                usage[k] += int(u.get(k, 0) or 0)
        except Exception:
            pass

lines = []
w = max([len(k) for k in groups] + [10])
t_total = t_judged = t_err = t_perfect = 0
all_scores = []
for vm in sorted(groups):
    g = groups[vm]
    pct = g["judged"] * 100 // g["total"] if g["total"] else 0
    avg = sum(g["scores"]) / len(g["scores"]) if g["scores"] else 0.0
    note = f"  ⚠ {g['errors']} 个判分失败" if g["errors"] else ""
    lines.append(f"  {vm:<{w}} [{bar(g['judged'], g['total'])}] "
                 f"{g['judged']:3d}/{g['total']:<3d} {pct:3d}%   "
                 f"Rubric {avg:.3f}  Perfect {g['perfect']}{note}")
    t_total += g["total"]; t_judged += g["judged"]
    t_err += g["errors"]; t_perfect += g["perfect"]
    all_scores += g["scores"]

sep = "  " + "─" * 62
print()
print(f"  DERAIL rubric judge   {run_root.name}")
print()
elapsed = int(time.time() - started) if started else 0
print(f"  已运行 {elapsed // 3600}h{elapsed % 3600 // 60:02d}m   "
      f"刷新间隔 {os.environ.get('REFRESH', '15')}s   {time.strftime('%H:%M:%S')}")
print()
print(sep)
for line in lines:
    print(line)
print(sep)

avg_all = sum(all_scores) / len(all_scores) if all_scores else 0.0
print(f"  合计 {t_judged}/{t_total}   Rubric {avg_all:.3f}   "
      f"Perfect {t_perfect}/{max(t_judged, 1)} "
      f"({t_perfect * 100 // max(t_judged, 1)}%)"
      + (f"   ⚠ 判分失败 {t_err}" if t_err else ""))

if judged_since_start > 0 and elapsed > 0:
    rate = elapsed / judged_since_start
    remain = int((t_total - t_judged) * rate)
    eta = time.strftime("%m-%d %H:%M", time.localtime(time.time() + remain))
    print(f"  本次已判 {judged_since_start} 个   平均 {rate:.0f}s/task   "
          f"预计还需 {remain // 3600}h{remain % 3600 // 60:02d}m（约 {eta} 结束）")
else:
    print("  本次还没有 task 判完，无法预估")
print()

if usage["calls"]:
    billed = usage["prompt"] - usage["cached"]
    cost = (billed * p_in + usage["cached"] * p_cached
            + usage["completion"] * p_out) / 1e6
    hit = usage["cached"] / usage["prompt"] * 100 if usage["prompt"] else 0
    per_task = cost / t_judged if t_judged else 0
    print(f"  token   {usage['calls']:,} 次调用   输入 {usage['prompt'] / 1e6:.1f}M"
          f"（缓存命中 {hit:.0f}%）   输出 {usage['completion'] / 1e3:.0f}k")
    print(f"  花费    ${cost:.2f} 已花   ${per_task:.3f}/task   "
          f"预计总计 ${per_task * t_total:.0f}")
else:
    print("  token   还没有 usage 数据（Gemini 路径不记账，只有 OpenAI 路径记）")
print()

idle = int(time.time() - newest) if newest else -1
if 0 <= idle < 300:
    print(f"  最近一次判分完成于 {idle}s 前")
elif newest:
    print(f"  ⚠ 已经 {idle // 60} 分钟没有新的判分结果 —— 可能在跑长 task，或者卡住了")
print()
PY
)"
    (( ONCE )) || clear 2>/dev/null || printf '\n\n'
    printf '%s\n' "$output"
    (( ONCE )) && break
    sleep "$REFRESH"
  done
  exit 0
fi

OPENAI_KEY=""
if [[ "${SHOW_OPENAI_USAGE:-1}" == "1" && -f "${REPO_ROOT}/.env" ]]; then
  OPENAI_KEY="$(sed -n 's/^OPENAI_API_KEY[=:][[:space:]]*//p' "${REPO_ROOT}/.env" \
    | tr -d '"'"'"' ' | head -1)"
fi

openai_usage_line() {
  [[ -n "$OPENAI_KEY" ]] || { printf '未配置 key，不统计'; return; }
  local day; day="$(date -u +%Y-%m-%d)"
  curl -s --max-time 8 -H "Authorization: Bearer ${OPENAI_KEY}" \
    "https://api.openai.com/v1/usage?date=${day}" 2>/dev/null | python3 -c '
import json,sys
try:
    rows = json.load(sys.stdin).get("data", [])
except Exception:
    print("查询失败（网络或权限）"); raise SystemExit
if not rows:
    print("今日 0 次请求（旧接口有延迟；project key 也可能查不到）"); raise SystemExit
req = sum(r.get("n_requests", 0) for r in rows)
ctx = sum(r.get("n_context_tokens_total", 0) for r in rows)
gen = sum(r.get("n_generated_tokens_total", 0) for r in rows)
models = ", ".join(sorted({r.get("snapshot_id", "?") for r in rows}))
print(f"{req} 次请求  输入 {ctx:,} / 输出 {gen:,} tokens  [{models}]")
' 2>/dev/null || printf '查询失败'
}

count_stalls() {
  find "$run_root" -name 'step_*.png' -printf '%p %T@\n' 2>/dev/null | python3 -c '
import sys, collections
per = collections.defaultdict(list)
for line in sys.stdin:
    path, _, stamp = line.rpartition(" ")
    vm = next((x for x in path.split("/") if x.startswith("vm")), "?")
    try:
        per[vm].append(float(stamp))
    except ValueError:
        pass
n = spent = total = 0
for stamps in per.values():
    stamps.sort()
    gaps = [b - a for a, b in zip(stamps, stamps[1:])]
    stalls = [g for g in gaps if 110 <= g <= 150]
    n += len(stalls)
    spent += sum(stalls)
    total += sum(g for g in gaps if g < 600)
print(f"{n} 次   占用 {spent/total*100:.0f}% 的步进时间" if total else "0 次")
' 2>/dev/null || printf '-'
}

human_time() {
  local s=$1
  (( s < 0 )) && s=0
  printf '%dh%02dm' $((s / 3600)) $(((s % 3600) / 60))
}

draw_bar() {
  local done=$1 total=$2 width=$3
  local filled=0
  (( total > 0 )) && filled=$((done * width / total))
  (( filled > width )) && filled=$width
  local bar=""
  local i
  for ((i = 0; i < filled; i++)); do bar+="█"; done
  for ((i = filled; i < width; i++)); do bar+="░"; done
  printf '%s' "$bar"
}

start_marker="${run_root}/.progress_started_at"
if [[ -n "$run_agents" && -f "$manifest" ]]; then
  started_at="$(stat -c %Y "$manifest" 2>/dev/null || date +%s)"
elif [[ -f "$start_marker" ]]; then
  started_at="$(cat "$start_marker")"
else
  started_at="$(find "$run_root" -type f -printf '%T@\n' 2>/dev/null \
    | sort -n | head -1 | cut -d. -f1)"
  [[ -n "$started_at" ]] || started_at="$(date +%s)"
  printf '%s\n' "$started_at" > "$start_marker" 2>/dev/null || true
fi

while true; do
  now="$(date +%s)"
  elapsed=$((now - started_at))
  grand_done=0
  grand_total=0
  grand_empty=0

  output=""
  output+=$(printf '\n  DERAIL collection   %s\n' "$collection_id")
  output+=$'\n'
  output+=$(printf '  已运行 %s   刷新间隔 %ss   %s\n' \
    "$(human_time "$elapsed")" "$REFRESH" "$(date '+%H:%M:%S')")
  output+=$'\n'
  if [[ "$tasks_total_source" != "manifest" ]]; then
    output+=$(printf '  分母 %s 条任务 × %s 轮，来源：%s\n' \
      "$tasks_total" "$repeats" "$tasks_total_source")
    output+=$'\n'
  fi
  output+=$(printf '  %s\n' "$(printf '─%.0s' $(seq 1 62))")
  output+=$'\n'

  shopt -s nullglob
  agent_dirs=("$run_root"/*/)
  shopt -u nullglob
  active_agent=""
  newest_mtime=0

  for agent_dir in "${agent_dirs[@]}"; do
    agent_id="$(basename "$agent_dir")"
    [[ "$agent_id" == .* ]] && continue
    read -r done_count empty_count < <(
      find "$agent_dir" -name result.txt -printf '%h\n' 2>/dev/null | python3 -c '
import sys, pathlib
ok = empty = 0
for line in sys.stdin:
    d = pathlib.Path(line.strip())
    if any(d.glob("step_*.png")): ok += 1
    else: empty += 1
print(ok, empty)
' 2>/dev/null || echo "0 0")
    this_run_note=""
    if in_this_run "$agent_id"; then
      grand_done=$((grand_done + done_count))
      grand_total=$((grand_total + episodes_total))
      grand_empty=$((grand_empty + empty_count))
    else
      this_run_note="  · 非本次运行"
    fi

    percent=0
    (( episodes_total > 0 )) && percent=$((done_count * 100 / episodes_total))
    empty_note=""
    (( empty_count > 0 )) && empty_note="$(printf '  ⚠ %d 个空壳' "$empty_count")"
    output+=$(printf '  %-18s [%s] %4d/%-4d %3d%%%s%s\n' \
      "$agent_id" "$(draw_bar "$done_count" "$episodes_total" "$BAR_WIDTH")" \
      "$done_count" "$episodes_total" "$percent" "$empty_note" "$this_run_note")
    output+=$'\n'

    latest="$(find "$agent_dir" -name 'step_*.png' -newermt "-5 minutes" -print -quit 2>/dev/null)"
    if [[ -n "$latest" ]]; then
      mtime="$(stat -c %Y "$latest" 2>/dev/null || echo 0)"
      if (( mtime > newest_mtime )); then
        newest_mtime=$mtime
        active_agent="$agent_id"
      fi
    fi
  done

  output+=$(printf '  %s\n' "$(printf '─%.0s' $(seq 1 62))")
  output+=$'\n'

  if (( grand_done > 0 && elapsed > 0 )); then
    rate=$((elapsed / grand_done))
    remaining=$(((grand_total - grand_done) * rate))
    output+=$(printf '  合计 %d/%d   平均 %ds/episode   预计还需 %s（约 %s 结束）\n' \
      "$grand_done" "$grand_total" "$rate" "$(human_time "$remaining")" \
      "$(date -d "+${remaining} seconds" '+%m-%d %H:%M' 2>/dev/null || echo '?')")
  else
    output+=$(printf '  合计 %d/%d   还没有 episode 完成，无法预估\n' "$grand_done" "$grand_total")
  fi
  output+=$'\n'

  if [[ -n "$active_agent" ]]; then
    steps="$(find "$run_root/$active_agent" -name 'step_*.png' -newermt "-10 minutes" 2>/dev/null | wc -l)"
    output+=$(printf '  正在跑 %s   最近 10 分钟产生 %s 张截图\n' "$active_agent" "$steps")
  else
    output+=$(printf '  最近 5 分钟没有新截图 —— 可能在加载模型、启动 VM，或者卡住了\n')
  fi
  output+=$'\n'
  if (( grand_empty > 0 )); then
    output+=$(printf '  ⚠⚠ %d 个 episode 有 result.txt 但零截图 —— endpoint 可能已经挂了，去看 collection 日志\n' "$grand_empty")
    output+=$'\n'
  fi
  output+=$(printf '  120s 超时停顿   %s\n' "$(count_stalls)")
  output+=$'\n'
  output+=$(printf '  OpenAI 用量     %s\n' "$(openai_usage_line)")
  output+=$'\n'
  output+=$(printf '                  （VM 内 NPC 回复用 gpt-5.4-mini，花的是 .env 的 key）\n')
  output+=$'\n'

  (( ONCE )) || clear 2>/dev/null || printf '\n\n'
  printf '%s\n' "$output"
  (( ONCE )) && break
  sleep "$REFRESH"
done
