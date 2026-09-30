#!/usr/bin/env bash
# 对已采集的 raw trajectories 跑 MyPCBench 的 rubric 判分。
#
# 采集（01_*）只写 result.txt 这个「跑完了」的标记，不含任何分数。真正的评分是
# 这一步：每个 task 一个 judge 子进程，进程内每条 rubric 一次 LLM 调用，看完整
# 轨迹（全部动作 + 全部截图）后给 success/failure，再按权重合成 0-100。
#
# 使用方法：
#   bash scripts/02_judge_rubrics.sh                      # 最新 collection 的全部 agent
#   bash scripts/02_judge_rubrics.sh v1                   # 指定 collection
#   bash scripts/02_judge_rubrics.sh v1 qwen3_5_35b_a3b   # 再指定 agent
#   FORCE=1 bash scripts/02_judge_rubrics.sh v1           # 重判（默认跳过已判过的）
#   tmux new -s derail-judge 'bash scripts/02_judge_rubrics.sh v1'  # 后台长跑
#
# 判分结果落在每个 task 目录里：
#   rubric_judge_result.json      judge 的原始返回（0-100 或错误信息）
#   osworld_full_traj_result.json 每条 rubric 的通过/理由 + token usage
#   <vm>/scores.json              该 VM 的聚合（Perfect % / Rubric %）
#
# 这一步不占 GPU：推理在模型服务商那边，本机只读截图、编码、发 HTTP。可以和
# 下一轮采集并行跑。

set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

PYTHON_BIN="${PYTHON_BIN:-python3}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${REPO_ROOT}/artifacts/raw_rollouts/mypcbench}"
JUDGE_SCRIPT="${JUDGE_SCRIPT:-${REPO_ROOT}/third_party/MyPCBench/agent-harness/judge_results.py}"
JUDGE_LOG_ROOT="${JUDGE_LOG_ROOT:-${REPO_ROOT}/artifacts/judge_logs}"
ROOT_DOTENV="${ROOT_DOTENV:-${REPO_ROOT}/.env}"
# 判官超参数的唯一真源。以前这些值散落在 third_party 的默认常量和调用者 shell
# 的环境变量里，仓库没有记录，于是 v1 的 552 个任务跑在三套配置下（max_images
# 200/50、concurrency 4/2）都没人发现。见 configs/judges/mypcbench_rubric.yaml。
JUDGE_CONFIG="${JUDGE_CONFIG:-${REPO_ROOT}/configs/judges/mypcbench_rubric.yaml}"
# judge 模型不在上面那份里：唯一真源是 configs/judges/default.yaml（clean-start、takeover、
# EAR 三个入口共用），由 judge_model_registry.py 解析并做单一 judge / 同源检查。
REGISTRY_PY="${SCRIPT_DIR}/judge_model_registry.py"
# 判官看到的证据在 bundle 一侧补齐（纯工具轮的调用与输出、最终状态 s_T），再由
# DERAIL 包装调用上游 judge：附录 D 的 prompt 原样，user 消息多一个最终状态数据段。
BUNDLE_PY="${SCRIPT_DIR}/14_takeover_bundle_prefix.py"
JUDGE_WRAPPER="${SCRIPT_DIR}/30_full_traj_judge.py"
FORCE="${FORCE:-0}"

die() {
  printf '错误：%s\n' "$*" >&2
  exit 1
}

info() {
  printf '[DERAIL judge] %s\n' "$*"
}

# 与 01_collect_trajectories.sh 的 load_dotenv 同源：本仓库的 .env 是
# `KEY: "value"` 的 YAML 风格，不是标准 dotenv 的 `KEY=value`，两种都要认。
# 只回显变量名，绝不回显值。
load_dotenv() {
  local dotenv_path="$1"
  local parsed_env env_line env_key
  [[ -f "$dotenv_path" ]] || return 0

  parsed_env="$(
    "$PYTHON_BIN" - "$dotenv_path" <<'PY'
import ast
import re
import sys

path = sys.argv[1]
key_pattern = re.compile(r"^[A-Z][A-Z0-9_]*$")
with open(path, encoding="utf-8") as handle:
    for line_number, raw_line in enumerate(handle, start=1):
        stripped = raw_line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if stripped.startswith("export "):
            stripped = stripped[7:].lstrip()
        if "=" in stripped:
            key, value_text = stripped.split("=", 1)
        elif ":" in stripped:
            key, value_text = stripped.split(":", 1)
        else:
            raise SystemExit(f"{path}:{line_number}: 应为 KEY=VALUE 或 KEY: VALUE")
        key = key.strip()
        value_text = value_text.strip()
        if not key_pattern.fullmatch(key):
            raise SystemExit(f"{path}:{line_number}: 非法环境变量名 {key!r}")
        if value_text.startswith(("\"", "'")):
            try:
                value = ast.literal_eval(value_text)
            except (SyntaxError, ValueError) as exc:
                raise SystemExit(f"{path}:{line_number}: quoted value 无效: {exc}")
        else:
            value = value_text
        if not isinstance(value, str) or "\n" in value or "\x00" in value:
            raise SystemExit(f"{path}:{line_number}: VALUE 必须是单行字符串")
        if value:
            print(f"{key}={value}")
PY
  )"

  while IFS= read -r env_line; do
    [[ -n "$env_line" ]] || continue
    env_key="${env_line%%=*}"
    # 只加载判分需要的几个，别把采集用的 *_BASE_URLS 也带进来。
    case "$env_key" in
      OPENAI_API_KEY|OPENAI_BASE_URL|GEMINI_API_KEY|GOOGLE_API_KEY) ;;
      *) continue ;;
    esac
    export "$env_line"
    info "已从 $(basename "$dotenv_path") 加载 ${env_key}"
  done <<< "$parsed_env"
}

# 读 configs/judges/*.yaml 并把每个键映射成上游 judge 认识的环境变量。
#
# 只认下面这张白名单里的键：上游的旋钮散在 osworld_full_traj_judge.py 各处，
# 键名和环境变量名对不上（max_images -> MYPCBENCH_OSWORLD_JUDGE_MAX_IMAGES），
# 把映射写死在这里，config 才能保持可读、而且拼错的键会立刻报错而不是被忽略。
#
# config 优先于环境变量；临时试验请用 ALLOW_CONFIG_OVERRIDE=1，正式跑数改 config。
load_judge_config() {
  local config_path="$1"
  local parsed line key value
  [[ -f "$config_path" ]] || die "找不到判官配置：${config_path}（用 JUDGE_CONFIG= 覆盖）"

  parsed="$(
    "$PYTHON_BIN" - "$config_path" <<'PY'
import re
import sys

path = sys.argv[1]
# config 键 -> 上游环境变量名。值为 None 的键由 shell 直接读，不进环境。
ENV_MAP = {
    "flavor": "MYPCBENCH_JUDGE_FLAVOR",
    "provider": "MYPCBENCH_JUDGE_PROVIDER",
    "max_images": "MYPCBENCH_OSWORLD_JUDGE_MAX_IMAGES",
    "max_image_mb": "MYPCBENCH_OSWORLD_JUDGE_MAX_IMAGE_MB",
    "image_format": "MYPCBENCH_OSWORLD_JUDGE_IMAGE_FORMAT",
    "image_quality": "MYPCBENCH_OSWORLD_JUDGE_IMAGE_QUALITY",
    "reasoning_effort": "MYPCBENCH_OSWORLD_JUDGE_REASONING_EFFORT",
    "max_completion_tokens": "MYPCBENCH_OSWORLD_JUDGE_MAX_COMPLETION_TOKENS",
    "max_retries": "MYPCBENCH_OSWORLD_JUDGE_MAX_RETRIES",
    "concurrency": "MYPCBENCH_OSWORLD_JUDGE_CONCURRENCY",
    # 这几个由脚本自己消费，不传给上游进程
    "timeout_seconds": "DERAIL_JUDGE_TIMEOUT",
    "api_key_env": "DERAIL_JUDGE_API_KEY_ENV",
    "force_official_endpoint": "DERAIL_JUDGE_FORCE_OFFICIAL_ENDPOINT",
    # 纯记录用，不影响行为
    "judge_id": None,
    "status": None,
}
value_pattern = re.compile(r'^[A-Za-z0-9_.\-]+$')

with open(path, encoding="utf-8") as handle:
    for lineno, raw in enumerate(handle, start=1):
        stripped = raw.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if ":" not in stripped:
            raise SystemExit(f"{path}:{lineno}: 应为 key: value")
        key, text = stripped.split(":", 1)
        key = key.strip()
        text = text.split("#", 1)[0].strip().strip('"').strip("'")
        if key not in ENV_MAP:
            raise SystemExit(f"{path}:{lineno}: 未知配置键 {key!r}（白名单见 load_judge_config）")
        if ENV_MAP[key] is None or not text:
            continue
        if not value_pattern.fullmatch(text):
            raise SystemExit(f"{path}:{lineno}: 值 {text!r} 含非法字符")
        print(f"{ENV_MAP[key]}={text}")
PY
  )" || die "解析判官配置失败：${config_path}"

  while IFS= read -r line; do
    [[ -n "$line" ]] || continue
    key="${line%%=*}"
    value="${line#*=}"
    # config 优先，环境变量要覆盖必须显式声明 ALLOW_CONFIG_OVERRIDE=1。理由同
    # scripts/01_collect_trajectories.sh 里 load_collection_config 的注释：
    # 残留的环境变量静默压过 config，正是这套 config 要根除的问题。
    if [[ -n "${!key:-}" && "${!key}" != "$value" ]]; then
      if [[ "${ALLOW_CONFIG_OVERRIDE:-0}" == "1" ]]; then
        JUDGE_OVERRIDDEN+=("${key}=${!key}（config 为 ${value}）")
        continue
      fi
      JUDGE_IGNORED+=("${key}=${!key}")
    fi
    export "${key}=${value}"
  done <<< "$parsed"
}

collection_id="${1:-}"
requested_agent="${2:-}"

[[ -x "$PYTHON_BIN" ]] || die "找不到 python：${PYTHON_BIN}（用 PYTHON_BIN= 覆盖）"
[[ -f "$JUDGE_SCRIPT" ]] || die "找不到 judge 脚本：${JUDGE_SCRIPT}"

JUDGE_OVERRIDDEN=()
JUDGE_IGNORED=()
load_judge_config "$JUDGE_CONFIG"
# JUDGE_MODEL= 只能等于 default.yaml 的 model（试验需 ALLOW_CONFIG_OVERRIDE=1）；JUDGE_TIMEOUT= 仍可覆盖。
JUDGE_MODEL="$("$PYTHON_BIN" "$REGISTRY_PY" ${JUDGE_MODEL:+"$JUDGE_MODEL"} --print-model)" \
  || die "judge 模型校验失败（configs/judges/default.yaml 是唯一真源）"
JUDGE_TIMEOUT="${JUDGE_TIMEOUT:-${DERAIL_JUDGE_TIMEOUT:-1000}}"

if [[ -z "$collection_id" ]]; then
  collection_id="$(ls -t "$OUTPUT_ROOT" 2>/dev/null | head -1)"
  [[ -n "$collection_id" ]] || die "找不到任何 collection：${OUTPUT_ROOT}"
fi
run_root="${OUTPUT_ROOT}/${collection_id}"
[[ -d "$run_root" ]] || die "collection 不存在：${run_root}"

load_dotenv "$ROOT_DOTENV"

# ── 路由预检 ────────────────────────────────────────────────────────────
# judge 的 provider 是按模型名 + 哪把 key 存在推出来的，推错了不会报错，只会
# 悄悄用另一个模型（甚至另一种评测范式）打出一批不可比的分数。所以在这里挑明。
export MYPCBENCH_RUBRIC_JUDGE_MODEL="$JUDGE_MODEL"
case "$JUDGE_MODEL" in
  gemini*)
    [[ -n "${GEMINI_API_KEY:-}${GOOGLE_API_KEY:-}" ]] \
      || die "JUDGE_MODEL=${JUDGE_MODEL} 需要 GEMINI_API_KEY（或 GOOGLE_API_KEY）"
    ;;
  *)
    [[ -n "${OPENAI_API_KEY:-}" ]] \
      || die "JUDGE_MODEL=${JUDGE_MODEL} 需要 OPENAI_API_KEY"
    # OPENAI_BASE_URL 指向本地 vLLM 时，判分会打到那个本地模型上而不是 OpenAI。
    # 本仓库的 .env 里这一项是给采集用的，判分默认要走真实 OpenAI。
    if [[ "${DERAIL_JUDGE_FORCE_OFFICIAL_ENDPOINT:-true}" == "true" \
       && -n "${OPENAI_BASE_URL:-}" && "${KEEP_OPENAI_BASE_URL:-0}" != "1" ]]; then
      info "忽略 .env 的 OPENAI_BASE_URL（判分走官方 endpoint；要保留就设 KEEP_OPENAI_BASE_URL=1）"
      unset OPENAI_BASE_URL
    fi
    "$PYTHON_BIN" -c 'import openai' 2>/dev/null \
      || die "${PYTHON_BIN} 里没有 openai 包"
    ;;
esac

# 待判的 agent 目录。
shopt -s nullglob
agent_dirs=("$run_root"/*/)
shopt -u nullglob
(( ${#agent_dirs[@]} > 0 )) || die "${run_root} 下没有 agent 目录"

# 判分的粒度是「装着一堆 task 子目录的目录」，也就是 <agent>/<repeat>/<vm>：
# judge_results.py 的 --result_dir 会遍历它下面每个 task 并写一份 scores.json。
# find -printf '%h' 给出 task 目录，再 dirname 一层就是 VM 目录。
# 被测对象不能兼任 judge（放在进程替换外面，die 才能终止脚本）。
for agent_dir in "${agent_dirs[@]}"; do
  agent_id="$(basename "$agent_dir")"
  [[ "$agent_id" == .* ]] && continue
  [[ -n "$requested_agent" && "$agent_id" != "$requested_agent" ]] && continue
  "$PYTHON_BIN" "$REGISTRY_PY" "$JUDGE_MODEL" --agent "$agent_id" --print-model >/dev/null \
    || die "judge ${JUDGE_MODEL} 与被测 agent ${agent_id} 同源"
done

mapfile -t targets < <(
  for agent_dir in "${agent_dirs[@]}"; do
    agent_id="$(basename "$agent_dir")"
    [[ "$agent_id" == .* ]] && continue
    [[ -n "$requested_agent" && "$agent_id" != "$requested_agent" ]] && continue
    find "$agent_dir" -name rubric_bundle.json -printf '%h\n' 2>/dev/null
  done | xargs -r -n1 dirname | sort -u
)

(( ${#targets[@]} > 0 )) || die "没找到任何 rubric_bundle.json —— 这个 collection 还没采集完？"

mkdir -p "$JUDGE_LOG_ROOT"
judge_log="${JUDGE_LOG_ROOT}/${collection_id}.log"

# 给 01_watch_progress.sh --judge 算 ETA 用；每次启动都重置。
date +%s > "${run_root}/.judge_started_at" 2>/dev/null || true

total_bundles="$(find "$run_root" -name rubric_bundle.json 2>/dev/null | wc -l)"
already="$(find "$run_root" -name rubric_judge_result.json 2>/dev/null | wc -l)"

info "collection : ${collection_id}"
info "judge model: ${JUDGE_MODEL}（单 task 超时 ${JUDGE_TIMEOUT}s）"
info "判官配置   : $(realpath --relative-to="$REPO_ROOT" "$JUDGE_CONFIG")"
# 把生效的超参数全部打出来。这些值会写进每个 osworld_full_traj_result.json，
# 但那是事后才看得到；漂移必须在开跑前就能一眼发现。
info "  flavor=${MYPCBENCH_JUDGE_FLAVOR:-<上游默认>} max_images=${MYPCBENCH_OSWORLD_JUDGE_MAX_IMAGES:-<上游默认>} concurrency=${MYPCBENCH_OSWORLD_JUDGE_CONCURRENCY:-<上游默认>}"
info "  reasoning_effort=${MYPCBENCH_OSWORLD_JUDGE_REASONING_EFFORT:-<上游默认>} max_completion_tokens=${MYPCBENCH_OSWORLD_JUDGE_MAX_COMPLETION_TOKENS:-<上游默认>} max_retries=${MYPCBENCH_OSWORLD_JUDGE_MAX_RETRIES:-<上游默认>}"
info "  image=${MYPCBENCH_OSWORLD_JUDGE_IMAGE_FORMAT:-<上游默认>}/q${MYPCBENCH_OSWORLD_JUDGE_IMAGE_QUALITY:-<上游默认>} max_image_mb=${MYPCBENCH_OSWORLD_JUDGE_MAX_IMAGE_MB:-<上游默认>}"
if (( ${#JUDGE_OVERRIDDEN[@]} > 0 )); then
  info "  ★ ALLOW_CONFIG_OVERRIDE=1，以下用环境变量而非 config：${JUDGE_OVERRIDDEN[*]}"
fi
if (( ${#JUDGE_IGNORED[@]} > 0 )); then
  info "  已忽略与 config 冲突的环境变量：${JUDGE_IGNORED[*]}（要生效请设 ALLOW_CONFIG_OVERRIDE=1）"
fi
info "待判目录   : ${#targets[@]} 个"
info "task 总数  : ${total_bundles}（已判 ${already}，$( ((FORCE)) && echo '本次 FORCE=1 会全部重判' || echo '已判的会跳过，不重复计费')）"
info "日志       : ${judge_log}"
info "进度       : bash scripts/01_watch_progress.sh --judge ${collection_id}"

export MYPCBENCH_RUBRIC_JUDGE_COMMAND="$(printf '%q %q' "$PYTHON_BIN" "$JUDGE_WRAPPER")"

force_flag=()
(( FORCE )) && force_flag=(--force)

rc_total=0
{
  printf '[DERAIL judge] 开始：%s  model=%s  timeout=%ss\n' \
    "$(date '+%F %T')" "$JUDGE_MODEL" "$JUDGE_TIMEOUT"
  for vm_dir in "${targets[@]}"; do
    printf '\n[DERAIL judge] ===== %s =====\n' "${vm_dir#${run_root}/}"
    "$PYTHON_BIN" "$BUNDLE_PY" "$vm_dir" || { rc_total=$?; continue; }
    # -u：judge_results.py 每判完一个 task 就 print 一行，不加 -u 会被块缓冲攒
    # 到 8KB 才刷，日志看起来像卡住了。
    "$PYTHON_BIN" -u "$JUDGE_SCRIPT" \
      --result_dir "$vm_dir" \
      --timeout "$JUDGE_TIMEOUT" \
      "${force_flag[@]+"${force_flag[@]}"}" || rc_total=$?
  done
  printf '\n[DERAIL judge] 结束：%s  退出码 %s\n' "$(date '+%F %T')" "$rc_total"
} 2>&1 | tee -a "$judge_log"

# judge_results.py 对单个 task 的失败不会返回非零（记进 judge_errors 继续跑），
# 非零只代表结构性问题（路径不对、没有 task 目录）。
(( rc_total == 0 )) || die "有判分目录以退出码 ${rc_total} 结束，看 ${judge_log}"

info "全部完成。汇总：bash scripts/01_watch_progress.sh --judge ${collection_id} --once"
