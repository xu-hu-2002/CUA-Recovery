#!/usr/bin/env bash
# 显示 MyPCBench 采集（01_*）或 rubric 判分（02_*）的进度。只读文件系统，
# 不碰被监控的进程本身。
#
#   bash scripts/01_watch_progress.sh                    # 采集进度，自动挑最新 collection
#   bash scripts/01_watch_progress.sh <COLLECTION_ID>
#   bash scripts/01_watch_progress.sh --judge            # 判分进度
#   bash scripts/01_watch_progress.sh --judge v1 --once  # 判分汇总，渲染一次就退出
#   REFRESH=10 bash scripts/01_watch_progress.sh         # 改刷新间隔（默认 15s）
#
# 采集进度依据每个任务结束时落盘的 result.txt；「当前任务」的步数依据 step_*.png。
# 官方 runner 在整个 VM 批次跑完前不会输出任何东西，所以这是唯一的实时信号。
#
# 判分进度依据每个任务判完后落盘的 rubric_judge_result.json，分母是
# rubric_bundle.json 的数量。花费从 osworld_full_traj_result.json 里的 usage
# 实时累计 —— 判分烧的是 API 额度，这个数字比进度条更需要盯着。

# 刻意不用 -e / pipefail：这是个只读的监控脚本，它唯一的职责是一直显示。
# 采集跑几十小时，期间文件会被并发创建/改名，find 与 stat 之间文件消失、或者
# `find | head` 触发 SIGPIPE 都是正常现象，不该让监控自己退出（那会让 tmux
# 窗口直接关掉，看起来像"进度条不见了"）。
set -u

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
OUTPUT_ROOT="${OUTPUT_ROOT:-${REPO_ROOT}/artifacts/raw_rollouts/mypcbench}"
REFRESH="${REFRESH:-15}"
BAR_WIDTH="${BAR_WIDTH:-34}"

# --once：渲染一次就退出，不清屏。方便在别的脚本里调用或重定向到文件。
# --judge / --rollout：看哪一段流水线的进度，默认 rollout。
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
# 01_collect_all.sh 建 tmux 窗口和采集进程建目录之间有竞争：窗口常常先起来。
# 直接退出会让 tmux 窗口立刻关掉，看起来像"进度条不见了"，所以这里等它出现。
if [[ ! -d "$run_root" ]]; then
  if (( ONCE )); then
    printf '目录还不存在：%s\n' "$run_root" >&2
    exit 1
  fi
  printf '\n  等待 %s 出现……（采集正在启动模型，约需 5 分钟）\n' "$collection_id"
  while [[ ! -d "$run_root" ]]; do sleep "$REFRESH"; done
fi

# 每个 agent 的 episode 总数 = 任务数 × repeats，两个都从 manifest 拿。
#
# 任务数曾经是写死的 184，于是任何一次子集采集（smoke、重跑某几条）都会显示
# 「0/184」并按 184 去算 ETA —— 分母错了，进度条和剩余时间就全是假的。现在
# 01_collect_trajectories.sh 会把 tasks_total 写进 manifest；老的 manifest 没有
# 这个字段，用 TASKS_TOTAL 环境变量顶一下，并在界面上说明分母是猜的。
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

# 本次运行涉及哪几个 agent。
#
# COLLECTION_ID 目录是**跨多次运行累积**的：往同一个 v1 里陆续追加 agent 是正常
# 用法。以前这个脚本把目录下每个子目录都当成"本次运行"，于是 v1 会显示
# 「合计 929/1656、预计还需 19h59m」—— 分母里含着从没跑过的 claude_sonnet5 和
# open_cua_72b，以及四个几天前就跑完的 agent；ETA 因此毫无意义。更糟的是几天前
# 留下的空壳 episode 会触发「endpoint 可能已经挂了」的红色告警。
#
# manifest 的 agents 字段就是本次运行的名单。取不到（老 collection）时退回旧行为。
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

# 本次运行是否包含某个 agent。名单为空时一律算"是"，即退回旧的全目录口径。
in_this_run() {
  [[ -z "$run_agents" ]] && return 0
  [[ " $run_agents " == *" $1 "* ]]
}

# ── 判分模式 ────────────────────────────────────────────────────────────
# 和采集模式几乎没有共用逻辑（分母、单位、要盯的风险都不一样），所以单独一段。
# 计价默认按 gpt-5.6-terra：$2 输入 / $0.2 缓存输入 / $12 输出，每 M token。
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

# 一个 VM 目录 = 一个 judge_results.py 调用单位。按它分组显示。
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
    # int 0-100 是分数；list 是 judge 失败（超时 / 非零退出 / 解析失败）。
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
# avg_score 的论文口径分母是全部 attempted task，判分失败按 0.0 计。这里显示的
# 是「已判完部分」的均值，最终数字以每个 VM 的 scores.json 为准。
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

# ── OpenAI 用量 ──────────────────────────────────────────────────────────
# VM 内的 BuzzChat/WorkBuzz 用 gpt-5.4-mini 生成 NPC 回复，走的是真实 OpenAI，
# 花的是 .env 里那把 key 的钱。这些调用发生在 VM 内部，采集 harness 看不到，
# 所以只能回头问 OpenAI 自己。
#
# 注意 /v1/usage 是旧接口：project key（sk-proj-）能调通，但新的
# /v1/organization/usage 与 /v1/organization/costs 会 403（需要 admin key）。
# 而且用量上报有延迟，刚发生的调用不会立刻出现。
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

# 统计疑似被 120 秒超时卡住的步子。env.py 的 _execute_command 对 VM 的读超时是
# 120 秒，所以截图间隔落在 110-150 秒的，基本都是撞上了这个超时。
#
# 必须按 VM 分组：多个 VM 并发写截图，把所有时间戳混在一起排序会把单台 VM 的
# 长间隔填平，统计出来永远是 0。
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

# 开始时间优先用 01_collect_all.sh 写的标记；对更早启动、没有标记的 collection
# 退回到目录里最早的文件时间，否则「已运行」会从看进度的那一刻算起，ETA 全错。
# 知道本次名单时，起始时间取 manifest 的 mtime —— 它正是每次采集开跑时重写的，
# 比"目录里最早的文件"准得多：v1 里最早的文件来自几天前别的 agent，照那个算
# evocua 刚起步就会显示"已运行 25h47m"。
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
  # 分母不是从 manifest 读到的时候必须说出来：进度条和 ETA 全靠它，猜错了整屏都是假的。
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
    # 只数「真的跑过」的 episode：有 result.txt 且至少有一张截图。
    #
    # 光数 result.txt 会被骗：模型 endpoint 挂掉之后，每个任务第一次调用就失败、
    # 20 秒"完成"，照样写 result.txt=1.0。2026-08-04 那次 184/184 满格，实际
    # 只有 10 个真跑过。所以进度必须以截图为准，空壳单独列出来。
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
    # 合计、ETA、空壳告警只统计本次运行的 agent；别人的产物照常显示，但标出来
    # 且不参与计算 —— 否则几天前的空壳会伪装成"endpoint 现在挂了"。
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

    # 最近写入的截图属于哪个 agent，就认为它是当前正在跑的那个。
    # -print -quit 找到第一个就停，不经过管道，避免 SIGPIPE。
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
    rate=$((elapsed / grand_done))               # 秒/episode
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
