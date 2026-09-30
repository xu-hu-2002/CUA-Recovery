#!/usr/bin/env bash
# 在 MyPCBench 的 184 个正式任务上收集 raw trajectories。
#
# 使用方法：
#   1. 修改下面的 AGENTS 数组，只保留本次要运行的 agent 名称；
#   2. 准备对应 API key（本地开源模型的 vLLM endpoint 由本脚本自动启动）；
#   3. 在 Linux + KVM 机器上执行：
#        bash scripts/01_collect_all.sh --confirm <agent name>
#
# 本地开源 agent 的 vLLM endpoint 默认由本脚本调用
# scripts/00_tmux_serve_open_source.sh 自动启动、等待 ready、跑完自动停止，
# 不需要再手动开一遍模型。三条规则：
#   * 该 agent 的 serving container 已经在跑 → 复用，且结束时不会被停掉；
#   * 由本脚本启动的 endpoint → 无论成功、失败还是 Ctrl-C 都会被停掉；
#   * 想改用自己的 endpoint（远程 / 已有服务）→ 设 DERAIL_AUTO_SERVE=0，
#     此时回到旧行为，从 .env 的 *_BASE_URLS 读取地址。
# 同一时刻只能 serve 一个模型（TP2×4 或 TP8 都会占满 8 张卡），所以 AGENTS 里
# 有多个本地模型时，会按顺序 serve 一个、跑完、停掉，再换下一个。

set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

# ======================== 只需要编辑这里 ========================
AGENTS=(
  "evocua_32b"
)
#   gpt_5_5
#   gpt_5_6_luna
#   claude_sonnet_5
#   claude_opus_4_8
#   kimi_k3
#   kimi_k3_cuabash
#   qwen3_5_35b_a3b
#   qwen3_8_27b
#   evocua_32b
#   opencua_72b
#   holo_3_1_35b_a3b
# ================================================================

# 命令行 agent 名称优先于上面的数组，便于临时运行单个或多个模型。
if (( $# > 0 )); then
  AGENTS=("$@")
fi


# 采集超参数的唯一真源。以前这一段全是脚本内的 `"${VAR:-默认}"`，谁在命令行上
# export 了什么不会留下任何痕迹，一次临时试跑和正式采集在产物里长得一样。
# 判官侧已经因为同样的原因漂过三套配置（见 configs/judges/mypcbench_rubric.yaml）。
COLLECT_CONFIG="${COLLECT_CONFIG:-${REPO_ROOT}/configs/collection/mypcbench_runtime.yaml}"
COLLECT_OVERRIDDEN=()
COLLECT_IGNORED=()
PORT_BASE="${PORT_BASE:-25000}"
# 逃生口，见 resolve_agent_num_vms。留空表示按本机 GPU 数（本地）或 yaml 的
# num_vms（托管 API）推导。
NUM_VMS_OVERRIDE="${NUM_VMS_OVERRIDE:-}"
DRY_RUN="${DRY_RUN:-0}"
FORMAL_COLLECTION="${FORMAL_COLLECTION:-0}"
PYTHON_BIN="${PYTHON_BIN:-python3}"

# 自动 serving。默认由本脚本管理本地 vLLM endpoint 的生命周期。
# SERVE_REPLICAS 留空表示与 VM 数同源（见 default_serve_replicas）。
DERAIL_AUTO_SERVE="${DERAIL_AUTO_SERVE:-1}"
SERVE_REPLICAS="${SERVE_REPLICAS:-}"
SERVE_READY_TIMEOUT="${SERVE_READY_TIMEOUT:-3600}"
SERVE_POLL_SECONDS="${SERVE_POLL_SECONDS:-10}"
# container 从 00 start 返回到出现在 docker ps 里的宽限期。
SERVE_STARTUP_GRACE="${SERVE_STARTUP_GRACE:-180}"
SERVE_SCRIPT="${SCRIPT_DIR}/00_tmux_serve_open_source.sh"
# 必须与 00_tmux_serve_open_source.sh 的同名变量一致，否则推导出的 endpoint
# 地址会指向错误的端口。
SERVING_PORT_BASE="${SERVING_PORT_BASE:-8000}"
SERVING_LOG_ROOT="${SERVING_LOG_ROOT:-${REPO_ROOT}/artifacts/serving_logs}"

# 三个上游仓库的 repository/commit/root 真源在 lib/third_party.sh，与
# scripts/setup_third_party.sh 共用一份——pin 住的 commit 抄错是不会报错的，
# 实验协议会静默漂移。
# shellcheck source=lib/third_party.sh
source "${SCRIPT_DIR}/lib/third_party.sh"
third_party_paths "$REPO_ROOT"

# 留空 = 按采集 config 的 task_source 取 configs/collection/sources.yaml 的 tasks_file；
# 显式给出时是该来源格式的任务文件（例如 ROCK 分片）。
TASKS_FILE="${TASKS_FILE:-}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${REPO_ROOT}/artifacts/raw_rollouts/mypcbench}"
COLLECTION_ID="${COLLECTION_ID:-$(date -u +%Y%m%dT%H%M%SZ)}"
ENV_YAML="${ENV_YAML:-${REPO_ROOT}/env.yaml}"
ROOT_DOTENV="${ROOT_DOTENV:-${REPO_ROOT}/.env}"
MODELS_LOCK="${MODELS_LOCK:-${REPO_ROOT}/configs/models.lock.yaml}"
DERAIL_TMUX="${DERAIL_TMUX:-1}"
DERAIL_OPENAI_API_APPROVED="${DERAIL_OPENAI_API_APPROVED:-0}"
DERAIL_OPENAI_API_PURPOSE="${DERAIL_OPENAI_API_PURPOSE:-}"
DERAIL_ANTHROPIC_API_APPROVED="${DERAIL_ANTHROPIC_API_APPROVED:-0}"
DERAIL_ANTHROPIC_API_PURPOSE="${DERAIL_ANTHROPIC_API_PURPOSE:-}"

die() {
  printf '错误：%s\n' "$*" >&2
  exit 1
}

info() {
  printf '[DERAIL collection] %s\n' "$*"
}

is_positive_integer() {
  [[ "$1" =~ ^[1-9][0-9]*$ ]]
}

# shellcheck source=lib/collection_config.sh
source "${SCRIPT_DIR}/lib/collection_config.sh"

# 本次 agent 实际使用的 VM 数，每个 agent 各算一次而不是一个全局值 ——
# 01_collect_all 会把 TP2 和 TP8 的模型排在同一个队列里串行跑。
#
# 本地 serving 的 agent：可用 GPU 数 / yaml 的 tensor_parallel_size，由
# agent_vm_count 现算。八张卡 TP2 得 4、TP8 得 1；只剩四张卡时 TP2 自动降到 2，
# 不需要人去改配置。采集 config 的 NUM_VMS 不再参与本地 agent 的推导。
# 托管 API 的 agent：不占卡，取 yaml 里显式声明的 num_vms（配额而非显存决定）。
#
# NUM_VMS_OVERRIDE 仍是手动逃生口：想少占几张卡、或想压低 API 并发时用它。
# 覆盖会打进 banner 和 manifest 的 num_vms_per_agent，不会静默生效。
resolve_agent_num_vms() {
  if [[ -n "$NUM_VMS_OVERRIDE" ]]; then
    printf '%s\n' "$NUM_VMS_OVERRIDE"
    return 0
  fi
  agent_vm_count "$REPO_ROOT" "$1"
}

# 本次 agent 实际使用的每局步数上限 / 墙钟上限。与 VM 数不同，这是**协议参数**：
# 改它轨迹就变、分数也变。论文要求所有 agent 同一预算（05:15、F:11），所以默认
# 全部取采集 config；agent yaml 里写了 max_steps / task_timeout 才会按 agent 覆盖，
# 且必须打进 banner 和 manifest（max_steps_per_agent / task_timeout_per_agent）。
# 目前没有任何 agent 覆盖（v1 的 evocua_32b=120 已撤掉）。
resolve_agent_max_steps() {
  agent_max_steps "$REPO_ROOT" "$1" "$MAX_STEPS"
}

resolve_agent_task_timeout() {
  local value
  value="$(config_scalar "$(agent_config_path "$REPO_ROOT" "$1")" task_timeout || true)"
  [[ "$value" =~ ^[1-9][0-9]*$ ]] || value="$TASK_TIMEOUT"
  printf '%s\n' "$value"
}

start_in_tmux_if_needed() {
  [[ "$DERAIL_TMUX" == "0" || -n "${TMUX:-}" ]] && return 0
  command -v tmux >/dev/null 2>&1 || die \
    "DERAIL_TMUX=1 但找不到 tmux；安装 tmux 或显式设置 DERAIL_TMUX=0"

  local session_name="${DERAIL_TMUX_SESSION:-derail-collect-${COLLECTION_ID}}"
  session_name="$(tr -c '[:alnum:]_-' '-' <<< "$session_name" | sed 's/-$//')"
  # `=` 前缀 = 精确匹配，否则 tmux 会前缀匹配到名字更长的别的 session。
  tmux has-session -t "=${session_name}" 2>/dev/null && die \
    "tmux session 已存在：${session_name}"

  # tmux 2.7 的长期 server 不保证继承任意新环境变量，所以只把非秘密的
  # experiment controls 写进子命令。API key 和 token 由子进程重新读取 .env。
  #
  # 没设置的变量必须显式 `env -u` 掉，不能只是"不传"。tmux server 活得比任何一次
  # 采集都长，它的 global environment 里留着上一次 attach 时那个 shell 的全部
  # export；只要本次没在命令行上给同名变量，新 session 就会从那份旧环境里把它补
  # 回来。2026-08-10 实测：一次 smoke 的 banner 打出 max_steps=100，而
  # configs/collection/mypcbench_runtime.yaml 写的是 150 —— server 里躺着上一轮
  # v1_bashfix_probe20 的 ALLOW_CONFIG_OVERRIDE=1 和 MAX_STEPS=100，config 那道
  # 防线被从背后绕过去了，产物里也看不出来。
  #
  # 判据和 collection config 是同一条：调用这个脚本的 shell 是唯一真源，凡是它没
  # 说的，子进程就不该有。
  local -a env_unset=()
  local -a env_assign=(DERAIL_TMUX=0)
  local -a stale_from_tmux=()
  local tmux_global_names=""
  tmux_global_names="$(tmux show-environment -g 2>/dev/null | awk -F= 'NF > 1 { print $1 }' || true)"
  local variable
  for variable in \
    REPEATS REPEAT_START_INDEX NUM_VMS_OVERRIDE MAX_STEPS DERAIL_BASH_ACCOUNTING BACKEND TIMEOUT_PER_VM TASK_TIMEOUT PORT_BASE DRY_RUN \
    CONTEXT_IMAGES TASK_SOURCE MYPCBENCH_QWEN_IMAGE_MAX \
    FORMAL_COLLECTION PYTHON_BIN MYPCBENCH_COMMIT MYPCBENCH_ROOT \
    EVOCUA_COMMIT EVOCUA_ROOT OPENCUA_OSWORLD_COMMIT OPENCUA_OSWORLD_ROOT \
    TASKS_FILE OUTPUT_ROOT COLLECTION_ID ROOT_DOTENV ENV_YAML MODELS_LOCK \
    MYPCBENCH_QCOW2 MYPCBENCH_OVMF_CODE MYPCBENCH_OVMF_VARS \
    MYPCBENCH_QEMU_BINARY \
    DERAIL_AUTO_SERVE SERVE_REPLICAS SERVE_READY_TIMEOUT SERVE_POLL_SECONDS \
    SERVE_STARTUP_GRACE SERVING_PORT_BASE SERVING_LOG_ROOT SERVING_CPUSET HF_CACHE_ROOT \
    ALLOW_BUSY_GPU CUDA_VISIBLE_DEVICES DERAIL_GPU_COUNT \
    ALLOW_NO_KVM MYPCBENCH_DIAG_DIR MYPCBENCH_QWEN_MAX_TOKENS \
    MYPCBENCH_QWEN_HISTORY_N MYPCBENCH_QWEN_CONTEXT_POLICY \
    OPENAI_BASE_URL QWEN35_BASE_URLS QWEN35_BASE_URL \
    QWEN36_BASE_URLS QWEN36_BASE_URL QWEN38_BASE_URLS QWEN38_BASE_URL QWEN38_MODEL \
    EVOCUA_BASE_URLS EVOCUA_BASE_URL \
    OPENCUA_BASE_URLS OPENCUA_BASE_URL HOLO31_BASE_URLS HOLO31_BASE_URL \
    ANTHROPIC_API_KEY ANTHROPIC_BASE_URL CLAUDE_PROMPT_CACHING_BETA \
    CLAUDE_SONNET_5_MODEL CLAUDE_OPUS_4_8_MODEL GPT55_MODEL GPT56_LUNA_MODEL \
    OPENAI_ZDR_STATELESS OPENAI_ZDR_KEEP_IMAGES \
    OPENAI_RATE_LIMIT_RETRIES ANTHROPIC_RATE_LIMIT_RETRIES \
    DERAIL_ANTHROPIC_API_APPROVED DERAIL_ANTHROPIC_API_PURPOSE \
    DERAIL_OPENAI_API_APPROVED DERAIL_OPENAI_API_PURPOSE \
    COLLECT_CONFIG ALLOW_CONFIG_OVERRIDE; do
    if [[ -n "${!variable+x}" ]]; then
      env_assign+=("${variable}=${!variable}")
    else
      env_unset+=(-u "$variable")
      if grep -qx -- "$variable" <<< "$tmux_global_names"; then
        stale_from_tmux+=("$variable")
      fi
    fi
  done
  if (( ${#stale_from_tmux[@]} > 0 )); then
    # 报出来而不是静默清掉：残留值本身说明有人在这个 tmux server 里跑过别的批次，
    # 值得看一眼是不是还有别的东西没清。
    info "已丢弃 tmux server global environment 里的残留变量（本次命令未设置）：${stale_from_tmux[*]}"
    info "  它们来自这个 tmux server 上更早的 session；要彻底清掉：tmux set-environment -gu <名字>"
  fi
  local -a command=(env "${env_unset[@]}" "${env_assign[@]}" bash "$0" "$@")

  local command_text
  printf -v command_text '%q ' "${command[@]}"
  command_text+=$'; status=$?; printf "\\n[DERAIL collection] exit=%s\\n" "$status"; exec bash'
  tmux new-session -d -s "$session_name" -n collection -c "$REPO_ROOT" "$command_text"
  tmux new-window -d -t "$session_name" -n monitor -c "$REPO_ROOT" \
    "watch -n 2 nvidia-smi"
  info "已在 tmux 启动；没有在当前 terminal 中后台裸跑"
  info "查看 collection: tmux attach -t ${session_name}"
  info "查看 GPU: tmux select-window -t ${session_name}:monitor"
  exit 0
}

sha256_file() {
  local path="$1"
  if command -v sha256sum >/dev/null 2>&1; then
    sha256sum "$path" | awk '{print $1}'
  elif command -v shasum >/dev/null 2>&1; then
    shasum -a 256 "$path" | awk '{print $1}'
  else
    die "找不到 sha256sum 或 shasum，无法记录 prompt/image hash"
  fi
}

# 读取简单且安全的顶层 YAML 环境变量映射。这里不使用 source，也不执行文件内容。
# 空值会被跳过，便于长期保留一个包含所有字段的本地模板。
load_yaml_env() {
  local yaml_path="$1"
  local parsed_env
  local env_line
  local env_key
  [[ -f "$yaml_path" ]] || return 0

  parsed_env="$(
    "$PYTHON_BIN" - "$yaml_path" <<'PY'
import json
import re
import sys

path = sys.argv[1]
key_pattern = re.compile(r"^[A-Z][A-Z0-9_]*$")

with open(path, encoding="utf-8") as handle:
    for line_number, raw_line in enumerate(handle, start=1):
        stripped = raw_line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if ":" not in raw_line:
            raise SystemExit(f"{path}:{line_number}: 应为 KEY: VALUE")
        key, raw_value = raw_line.split(":", 1)
        key = key.strip()
        value_text = raw_value.strip()
        if not key_pattern.fullmatch(key):
            raise SystemExit(f"{path}:{line_number}: 非法环境变量名 {key!r}")
        if not value_text or value_text in {"null", "~"}:
            continue
        if value_text.startswith('"'):
            try:
                value = json.loads(value_text)
            except json.JSONDecodeError as exc:
                raise SystemExit(f"{path}:{line_number}: 双引号字符串无效: {exc}")
        elif value_text.startswith("'"):
            if len(value_text) < 2 or not value_text.endswith("'"):
                raise SystemExit(f"{path}:{line_number}: 单引号字符串没有闭合")
            value = value_text[1:-1].replace("''", "'")
        else:
            value = value_text
        if not isinstance(value, str):
            raise SystemExit(f"{path}:{line_number}: VALUE 必须是字符串")
        if "\n" in value or "\x00" in value:
            raise SystemExit(f"{path}:{line_number}: VALUE 只能占一行")
        if value:
            print(f"{key}={value}")
PY
  )"

  while IFS= read -r env_line; do
    [[ -n "$env_line" ]] || continue
    env_key="${env_line%%=*}"
    export "$env_line"
    info "已从 env.yaml 加载 ${env_key}"
  done <<< "$parsed_env"
}

# 读 configs/collection/*.yaml，把协议参数落到脚本变量上。
#
# 与 load_yaml_env 的区别：那个吃的是 KEY: VALUE 形式的环境变量清单，这个吃的是
# configs/ 的描述式小写键，映射表写死在下面 —— 拼错的键会报错而不是被静默忽略。
# 屏幕分辨率从 environment 指向的那份 config 里取，不在这里重复声明。
load_collection_config() {
  local config_path="$1"
  local parsed line key value
  [[ -f "$config_path" ]] || die "找不到采集配置：${config_path}（用 COLLECT_CONFIG= 覆盖）"

  parsed="$(
    "$PYTHON_BIN" - "$config_path" "$REPO_ROOT" <<'PY'
import re
import sys
from pathlib import Path

path, repo_root = Path(sys.argv[1]), Path(sys.argv[2])
VAR_MAP = {
    "repeats": "REPEATS",
    "max_steps": "MAX_STEPS",
    "backend": "BACKEND",
    "timeout_per_vm": "TIMEOUT_PER_VM",
    "task_timeout": "TASK_TIMEOUT",
    "context_images": "CONTEXT_IMAGES",
    "task_source": "TASK_SOURCE",
    "environment": None,      # 单独展开成 SCREEN_WIDTH / SCREEN_HEIGHT / ENVIRONMENT_CONFIG
    "collection_id": None,    # 纯记录
    "status": None,
}
value_pattern = re.compile(r"^[A-Za-z0-9_.\-]+$")


def parse_flat(p):
    out = {}
    for lineno, raw in enumerate(p.read_text(encoding="utf-8").splitlines(), start=1):
        stripped = raw.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if ":" not in stripped:
            raise SystemExit(f"{p}:{lineno}: 应为 key: value")
        k, text = stripped.split(":", 1)
        out[k.strip()] = text.split("#", 1)[0].strip().strip('"').strip("'")
    return out


cfg = parse_flat(path)
for key, text in cfg.items():
    if key not in VAR_MAP:
        raise SystemExit(f"{path}: 未知配置键 {key!r}（白名单见 load_collection_config）")
    if VAR_MAP[key] is None or not text:
        continue
    if not value_pattern.fullmatch(text):
        raise SystemExit(f"{path}: 键 {key} 的值 {text!r} 含非法字符")
    print(f"{VAR_MAP[key]}={text}")

env_id = cfg.get("environment", "")
if env_id:
    env_path = repo_root / "configs" / "environments" / f"{env_id}.yaml"
    if not env_path.is_file():
        raise SystemExit(f"{path}: environment={env_id} 对应的 {env_path} 不存在")
    env_cfg = parse_flat(env_path)
    for field, var in (("screen_width", "SCREEN_WIDTH"), ("screen_height", "SCREEN_HEIGHT")):
        text = env_cfg.get(field, "")
        if not text or not value_pattern.fullmatch(text):
            raise SystemExit(f"{env_path}: 缺少可用的 {field}")
        print(f"{var}={text}")
    print(f"ENVIRONMENT_CONFIG={env_path}")
PY
  )" || die "解析采集配置失败：${config_path}"

  while IFS= read -r line; do
    [[ -n "$line" ]] || continue
    key="${line%%=*}"
    value="${line#*=}"
    # config 优先，环境变量要覆盖必须显式声明 ALLOW_CONFIG_OVERRIDE=1。
    #
    # 反过来（环境变量优先）看着更方便，但那正是这个文件要解决的问题：实测
    # tmux server 的全局环境里躺着几周前某次运行留下的 MAX_STEPS=100 / REPEATS=1，
    # 会静默压过 config。旧代码里它同样静默压过硬编码默认值，且不留任何痕迹。
    if [[ -n "${!key:-}" && "${!key}" != "$value" ]]; then
      if [[ "${ALLOW_CONFIG_OVERRIDE:-0}" == "1" ]]; then
        COLLECT_OVERRIDDEN+=("${key}=${!key}（config 为 ${value}）")
        continue
      fi
      COLLECT_IGNORED+=("${key}=${!key}")
    fi
    printf -v "$key" '%s' "$value"
  done <<< "$parsed"
}

# 安全读取 KEY=VALUE 格式的 .env。不会 source、eval 或展开命令替换。
load_dotenv() {
  local dotenv_path="$1"
  local parsed_env
  local env_line
  local env_key
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
            # Backward-compatible with the existing YAML-style local file
            # even when it is named .env. Values are still parsed as data.
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
        # Keep command-line/tmux values when a long-lived local template has
        # an intentionally blank field.  This matches load_yaml_env().
        if value:
            print(f"{key}={value}")
PY
  )"

  while IFS= read -r env_line; do
    [[ -n "$env_line" ]] || continue
    env_key="${env_line%%=*}"
    export "$env_line"
    info "已从 $(basename "$dotenv_path") 加载 ${env_key}"
  done <<< "$parsed_env"
}

# 将论文中的稳定 agent_id 映射到 MyPCBench 官方 runner 参数。
# 未完成 adapter 的模型必须 hard fail，不能偷偷套用相似模型的 prompt/action space。
resolve_agent() {
  local agent_id="$1"
  RESOLVED_AGENT_TYPE=""
  RESOLVED_MODEL=""
  RESOLVED_REQUIRED_ENV=""
  RESOLVED_BASE_URLS_ENV=""
  RESOLVED_BASE_URL_ENV=""
  case "$agent_id" in
    gpt_5_5)
      RESOLVED_AGENT_TYPE="openai_cuabash"
      # routify 等 API 网关要求带 provider 前缀的模型名（如 openai.gpt-5.5）；
      # 直连 OpenAI 官方 API 时保持默认 gpt-5.5。通过 GPT55_MODEL 显式覆盖。
      RESOLVED_MODEL="${GPT55_MODEL:-gpt-5.5}"
      RESOLVED_REQUIRED_ENV="OPENAI_API_KEY"
      ;;
    gpt_5_6_luna)
      # 与 gpt_5_5 完全同一条代码路径（官方 openai_cuabash：Responses API +
      # 内置 computer/shell），只换 model 名 —— 因此两者互为可比的原生 baseline。
      RESOLVED_AGENT_TYPE="openai_cuabash"
      RESOLVED_MODEL="${GPT56_LUNA_MODEL:-gpt-5.6-luna}"
      RESOLVED_REQUIRED_ENV="OPENAI_API_KEY"
      ;;
    kimi_k3)
      # GUI-only 对照组，不是论文的 Kimi-K3（论文用下面的 kimi_k3_cuabash，开放命令行）。
      # chat completions + DERAIL 冻结 schema scaffold（非原生 CUA，论文中属
      # scaffold 组）。kimi 拒绝 temperature 参数，由 kimi_k3_protocol() 处理；
      # reasoning tokens 计入 max_tokens，预算已调大。走 routify 网关直连，
      # 无本地 serving。
      RESOLVED_AGENT_TYPE="derail_kimi_k3"
      RESOLVED_MODEL="${KIMI_K3_MODEL:-kimi-k3}"
      RESOLVED_REQUIRED_ENV="OPENAI_API_KEY"
      ;;
    kimi_k3_cuabash)
      # kimi_k3 的 GUI+bash 对照组：同一 derail_tool_agent 装配路径、同一
      # KIMI_K3_MODEL（两对比的是工具面不是模型），仅 enable_bash=true —— bash
      # 在 predict 内部执行、不占 runner 步数（语义见 configs/agents/
      # kimi_k3_cuabash.yaml 头部注释）。
      RESOLVED_AGENT_TYPE="derail_kimi_k3_cuabash"
      RESOLVED_MODEL="${KIMI_K3_MODEL:-kimi-k3}"
      RESOLVED_REQUIRED_ENV="OPENAI_API_KEY"
      ;;
    qwen3_5_35b_a3b)
      RESOLVED_AGENT_TYPE="qwen_cuabash"
      RESOLVED_MODEL="${QWEN35_MODEL:-Qwen/Qwen3.5-35B-A3B}"
      RESOLVED_REQUIRED_ENV="LOCAL_ENDPOINT"
      RESOLVED_BASE_URLS_ENV="QWEN35_BASE_URLS"
      RESOLVED_BASE_URL_ENV="QWEN35_BASE_URL"
      ;;
    claude_sonnet_5)
      # model ID 由 CLAUDE_SONNET_5_MODEL 冻结（routify 上为 claude-sonnet-5；
      # 直连 Anthropic 时为官方快照名）。走官方 claude_cuabash：Anthropic
      # Messages API + 原生 computer_20251124 工具 —— 与 gpt 同为原生 baseline。
      [[ -n "${CLAUDE_SONNET_5_MODEL:-}" ]] || die \
        "claude_sonnet_5 尚未冻结真实 API model ID；请先设置 CLAUDE_SONNET_5_MODEL"
      RESOLVED_AGENT_TYPE="claude_cuabash"
      RESOLVED_MODEL="${CLAUDE_SONNET_5_MODEL}"
      RESOLVED_REQUIRED_ENV="ANTHROPIC_API_KEY"
      ;;
    claude_opus_4_8)
      [[ -n "${CLAUDE_OPUS_4_8_MODEL:-}" ]] || die \
        "claude_opus_4_8 尚未冻结真实 API model ID；请先设置 CLAUDE_OPUS_4_8_MODEL"
      RESOLVED_AGENT_TYPE="claude_cuabash"
      RESOLVED_MODEL="${CLAUDE_OPUS_4_8_MODEL}"
      RESOLVED_REQUIRED_ENV="ANTHROPIC_API_KEY"
      ;;
    qwen3_6_27b)
      RESOLVED_AGENT_TYPE="derail_qwen36"
      RESOLVED_MODEL="${QWEN36_MODEL:-Qwen/Qwen3.6-27B}"
      RESOLVED_REQUIRED_ENV="LOCAL_ENDPOINT"
      RESOLVED_BASE_URLS_ENV="QWEN36_BASE_URLS"
      RESOLVED_BASE_URL_ENV="QWEN36_BASE_URL"
      ;;
    qwen3_8_27b)
      RESOLVED_AGENT_TYPE="derail_qwen38"
      RESOLVED_MODEL="${QWEN38_MODEL:-Qwen/Qwen3.8-27B}"
      RESOLVED_REQUIRED_ENV="LOCAL_ENDPOINT"
      RESOLVED_BASE_URLS_ENV="QWEN38_BASE_URLS"
      RESOLVED_BASE_URL_ENV="QWEN38_BASE_URL"
      ;;
    evocua_32b)
      RESOLVED_AGENT_TYPE="derail_evocua"
      # 必须与官方 vLLM --served-model-name 一致。
      RESOLVED_MODEL="${EVOCUA_MODEL:-EvoCUA}"
      RESOLVED_REQUIRED_ENV="LOCAL_ENDPOINT"
      RESOLVED_BASE_URLS_ENV="EVOCUA_BASE_URLS"
      RESOLVED_BASE_URL_ENV="EVOCUA_BASE_URL"
      ;;
    opencua_72b)
      RESOLVED_AGENT_TYPE="derail_opencua"
      RESOLVED_MODEL="${OPENCUA_MODEL:-opencua-72b}"
      RESOLVED_REQUIRED_ENV="LOCAL_ENDPOINT"
      RESOLVED_BASE_URLS_ENV="OPENCUA_BASE_URLS"
      RESOLVED_BASE_URL_ENV="OPENCUA_BASE_URL"
      ;;
    holo_3_1_35b_a3b)
      RESOLVED_AGENT_TYPE="derail_holo31"
      RESOLVED_MODEL="${HOLO31_MODEL:-Hcompany/Holo-3.1-35B-A3B}"
      RESOLVED_REQUIRED_ENV="LOCAL_ENDPOINT"
      RESOLVED_BASE_URLS_ENV="HOLO31_BASE_URLS"
      RESOLVED_BASE_URL_ENV="HOLO31_BASE_URL"
      ;;
    dummy)
      # 仅用于无 API 的环境 smoke test，不属于论文七个 agent。
      RESOLVED_AGENT_TYPE="dummy"
      RESOLVED_MODEL="dummy"
      RESOLVED_REQUIRED_ENV=""
      RESOLVED_BASE_URLS_ENV=""
      RESOLVED_BASE_URL_ENV=""
      ;;
    *)
      die "未知 agent_id：${agent_id}；请使用 configs/agents/ 中定义的名称"
      ;;
  esac
}

# ==================== 自动 serving（本地开源 agent） ====================

# endpoint 数与 VM 数同源（官方 runner 按 VM round-robin 分配 endpoint，每个 VM
# 一个 replica 就够）。上限由 agent_vm_count 按 GPU 数算，00 那边用同一个函数。
default_serve_replicas() {
  resolve_agent_num_vms "$1"
}

# 始终打印一个数字：调用方会拿它做算术，空字符串在 set -e 下会直接炸掉。
serving_container_count() {
  local names
  names="$(docker ps --filter label=derail.project=DERAIL \
    --filter "label=derail.agent_id=$1" --format '{{.Names}}' 2>/dev/null || true)"
  if [[ -z "$names" ]]; then
    printf '0\n'
  else
    printf '%s\n' "$names" | grep -c .
  fi
}

serving_base_urls() {
  local replicas="$1"
  local urls=""
  local index
  for ((index = 0; index < replicas; index++)); do
    [[ -z "$urls" ]] || urls+=","
    urls+="http://127.0.0.1:$((SERVING_PORT_BASE + index))/v1"
  done
  printf '%s\n' "$urls"
}

# 这个端口是不是本次自动启动的 container 发布的。只在 AUTO_SERVED_AGENT 非空时
# 有意义：复用 .env 里别人维护的 endpoint 时，本来就不该要求有本地 container。
#
# 用 docker 的 label + 端口映射判定，而不是比对 /v1/models 里的模型名 —— 后者的
# 真源在 00 的 --served-model-name，比对就等于把同一个名字抄第二遍。
assert_endpoint_is_ours() {
  local agent_id="$1" url="$2"
  [[ -n "$AUTO_SERVED_AGENT" ]] || return 0
  local port="${url##*:}"
  port="${port%%/*}"
  [[ "$port" =~ ^[0-9]+$ ]] || return 0
  local published
  published="$(docker ps --filter label=derail.project=DERAIL \
    --filter "label=derail.agent_id=${agent_id}" --format '{{.Ports}}' 2>/dev/null || true)"
  grep -q "127\.0\.0\.1:${port}->" <<< "$published" || die \
    "端口 ${port} 上有服务在应答，但它不是 ${agent_id} 的 container（本次发布的端口：${published//$'\n'/ }）；换一段端口重跑：SERVING_PORT_BASE=8100"
}

# 只探 /v1/models 判断 endpoint 是否 ready，不发任何生成请求，因此不消耗
# 采集预算，也不构成模型验证。
wait_for_endpoints() {
  local agent_id="$1"
  local urls="$2"
  local deadline=$((SECONDS + SERVE_READY_TIMEOUT))
  # 00 start 只是建好 tmux window 就返回，窗口里的 docker run 还要几秒才会让
  # container 出现在 docker ps 里。在见到 container 之前不能把「数量为 0」当作
  # 崩溃，否则每次都会在启动瞬间误判。
  local startup_grace=$((SECONDS + SERVE_STARTUP_GRACE))
  local seen_container=0
  local container_count
  local -a url_list=()
  IFS=',' read -r -a url_list <<< "$urls"
  local url
  local http_code
  for url in "${url_list[@]}"; do
    info "等待 endpoint ready：${url}（最多 ${SERVE_READY_TIMEOUT}s）"
    while true; do
      http_code="$(curl -s -o /dev/null -w '%{http_code}' --max-time 5 \
        "${url}/models" || true)"
      if [[ "$http_code" == "200" ]]; then
        # 200 只说明"有人在这个端口上应答"，不说明应答的是我们的模型。自动启动
        # 时必须确认该端口确实由本 agent 的 container 发布，否则一个占着同一端口
        # 的别人的 vLLM 会被当成 ready，整批轨迹会记在错误的模型名下。
        assert_endpoint_is_ours "$agent_id" "$url"
        info "endpoint ready：${url}"
        break
      fi
      container_count="$(serving_container_count "$agent_id")"
      (( container_count > 0 )) && seen_container=1
      # 见过 container 之后又消失 = 崩了，立刻失败，不要空等到 timeout。
      if (( seen_container == 1 && container_count == 0 )); then
        die "${agent_id} 的 serving container 已退出；日志见 ${SERVING_LOG_ROOT} 下最新的 endpoint_*.log"
      fi
      # container 始终没出现 = 00 start 起失败了（镜像、GPU 占用、tmux 等）。
      if (( seen_container == 0 && SECONDS > startup_grace )); then
        die "${agent_id} 的 serving container 在 ${SERVE_STARTUP_GRACE}s 内没有出现；检查 tmux session derail-serve-* 和 ${SERVING_LOG_ROOT}"
      fi
      (( SECONDS < deadline )) || die \
        "${agent_id} 的 endpoint 在 ${SERVE_READY_TIMEOUT}s 内没有 ready：${url}"
      sleep "$SERVE_POLL_SECONDS"
    done
  done
}

# 只有本脚本亲手启动的 endpoint 才会被自动停止；复用已有 serving 时留空。
AUTO_SERVED_AGENT=""

auto_serve_stop() {
  [[ -n "$AUTO_SERVED_AGENT" ]] || return 0
  local agent_id="$AUTO_SERVED_AGENT"
  # 先清空再停，避免 stop 自身失败时 trap 递归。
  AUTO_SERVED_AGENT=""
  info "停止自动启动的 endpoint：${agent_id}"
  bash "$SERVE_SCRIPT" stop "$agent_id" || \
    printf '[DERAIL collection] 警告：停止 %s 失败，请手动检查 docker ps\n' "$agent_id" >&2
  # container 退出后显存释放还有延迟；下一个模型立刻起会被 00 的抢卡检查拦下。
  sleep "${SERVE_SETTLE_SECONDS:-15}"
}

# ==================== endpoint 中途死亡的看门狗 ====================
# wait_for_endpoints 只在启动时探一次。endpoint 在采集途中消失时（2026-08-08
# 就发生过：holo31 的 smoke10 还剩 3 个任务，下一轮 collection 把它的显卡收走
# 了），runner 不会停 —— 它给剩下的每个任务写一条 PREDICT_CRASH "Connection
# error." 然后照常写 result.txt=1.0，20 秒“跑完”一个。事后只能靠「有 result.txt
# 但零截图」把空壳挑出来，而 GPU 时间已经白烧了。
#
# 这里在 runner 跑的全程盯着 /v1/models，连续失败若干次就把 runner 打掉并让整个
# collection 失败退出，把「静默产出垃圾数据」换成「响亮地停下」。
ENDPOINT_WATCH_INTERVAL="${ENDPOINT_WATCH_INTERVAL:-60}"
ENDPOINT_WATCH_FAILURES="${ENDPOINT_WATCH_FAILURES:-3}"
ENDPOINT_WATCHDOG_PID=""
ENDPOINT_WATCHDOG_TRIP=""

endpoint_watchdog_start() {
  local urls="$1"
  local runner_pattern="$2"
  ENDPOINT_WATCHDOG_PID=""
  ENDPOINT_WATCHDOG_TRIP=""
  [[ -n "$urls" ]] || return 0
  (( ENDPOINT_WATCH_INTERVAL > 0 )) || return 0

  ENDPOINT_WATCHDOG_TRIP="$(mktemp -t derail-endpoint-trip.XXXXXX)"
  rm -f "$ENDPOINT_WATCHDOG_TRIP"
  local trip_file="$ENDPOINT_WATCHDOG_TRIP"
  local interval="$ENDPOINT_WATCH_INTERVAL"
  local max_failures="$ENDPOINT_WATCH_FAILURES"

  (
    # 这是子 shell 而不是函数体，不能用 local。
    declare -a url_list=()
    IFS=',' read -r -a url_list <<< "$urls"
    declare -A failures=()
    declare url http_code
    for url in "${url_list[@]}"; do failures["$url"]=0; done
    while true; do
      sleep "$interval"
      for url in "${url_list[@]}"; do
        http_code="$(curl -s -o /dev/null -w '%{http_code}' --max-time 10 \
          "${url}/models" || true)"
        if [[ "$http_code" == "200" ]]; then
          failures["$url"]=0
          continue
        fi
        failures["$url"]=$(( failures["$url"] + 1 ))
        if (( failures["$url"] >= max_failures )); then
          printf '%s\n' "$url" > "$trip_file"
          printf '[DERAIL collection] endpoint %s 连续 %d 次探测失败，正在中止 runner\n' \
            "$url" "${failures["$url"]}" >&2
          # 与脚本对外公布的中止方式一致：按命令行 pattern 精确杀 runner，
          # 绝不杀进程组 —— VM 子进程是 start_new_session=True，杀组会把
          # QEMU 连同临时磁盘一起留成孤儿。
          pkill -TERM -f "run_parallel_tasks.py.*${runner_pattern}" || true
          exit 0
        fi
      done
    done
  ) &
  ENDPOINT_WATCHDOG_PID="$!"
}

endpoint_watchdog_stop() {
  [[ -n "$ENDPOINT_WATCHDOG_PID" ]] || return 0
  local pid="$ENDPOINT_WATCHDOG_PID"
  ENDPOINT_WATCHDOG_PID=""
  kill "$pid" 2>/dev/null || true
  wait "$pid" 2>/dev/null || true
}

# endpoint 是否在本次 runner 期间死过。调用方负责在读完之后清掉 trip 文件。
endpoint_watchdog_tripped() {
  [[ -n "$ENDPOINT_WATCHDOG_TRIP" && -s "$ENDPOINT_WATCHDOG_TRIP" ]]
}

collection_cleanup() {
  endpoint_watchdog_stop
  # 注意不要写成 `[[ ... ]] && rm`：条件为假时整条 AND-list 返回 1，在
  # set -e 下会让 cleanup 在这里就断掉，auto_serve_stop 不会跑，显卡收不回来。
  if [[ -n "$ENDPOINT_WATCHDOG_TRIP" ]]; then
    rm -f "$ENDPOINT_WATCHDOG_TRIP"
  fi
  auto_serve_stop
}

# collection 中途失败或 Ctrl-C 时也要把 GPU 还回去，并收掉看门狗。
trap collection_cleanup EXIT INT TERM

# 启动（或复用）某个 agent 的 endpoint，并把地址写进它的 *_BASE_URLS 变量。
auto_serve_start() {
  local agent_id="$1"
  local urls_variable="$2"
  local replicas
  local running

  running="$(serving_container_count "$agent_id")"
  if (( running > 0 )); then
    replicas="$running"
    info "${agent_id} 已有 ${replicas} 个 serving container 在跑；复用，结束时不会停掉它"
  else
    replicas="${SERVE_REPLICAS:-$(default_serve_replicas "$agent_id")}"
    # SERVE_REPLICAS 是手动逃生口，但不能超过该 agent 声明的物理上限
    # （TP8 模型八张卡只拼得出一个 endpoint）。超了就夹回去并说明，别让 00
    # 在几分钟的镜像检查之后才报错。
    local ceiling
    ceiling="$(default_serve_replicas "$agent_id")"
    if (( replicas > ceiling )); then
      info "${agent_id} 在本机的 endpoint 上限是 ${ceiling}（按可用 GPU 推导）；忽略 SERVE_REPLICAS=${replicas}"
      replicas="$ceiling"
    fi
    info "自动启动 ${agent_id} 的 vLLM endpoint（${replicas} 个 replica）"
    bash "$SERVE_SCRIPT" start "$agent_id" "$replicas"
    # 只在真正由本脚本启动之后才登记，复用的 endpoint 不归本脚本管。
    AUTO_SERVED_AGENT="$agent_id"
  fi

  local urls
  urls="$(serving_base_urls "$replicas")"
  # auto-serve 拥有 endpoint 地址：.env 里的 *_BASE_URLS 是给手动流程用的，
  # 这里必须覆盖，否则自动起的 endpoint 会被旧地址盖掉。
  if [[ -n "$urls_variable" ]]; then
    if [[ -n "${!urls_variable:-}" && "${!urls_variable}" != "$urls" ]]; then
      info "覆盖 ${urls_variable}：${!urls_variable} → ${urls}（DERAIL_AUTO_SERVE=0 可保留 .env 的值）"
    fi
    export "${urls_variable}=${urls}"
  fi
  wait_for_endpoints "$agent_id" "$urls"
}

start_in_tmux_if_needed "$@"

load_collection_config "$COLLECT_CONFIG"

is_positive_integer "$REPEATS" || die "REPEATS 必须是正整数"
is_positive_integer "$MAX_STEPS" || die "MAX_STEPS 必须是正整数"
is_positive_integer "$TIMEOUT_PER_VM" || die "TIMEOUT_PER_VM 必须是正整数"
is_positive_integer "$TASK_TIMEOUT" || die "TASK_TIMEOUT 必须是正整数"
is_positive_integer "$CONTEXT_IMAGES" || die "CONTEXT_IMAGES 必须是正整数"
(( TASK_TIMEOUT < TIMEOUT_PER_VM )) || die "TASK_TIMEOUT 必须小于 TIMEOUT_PER_VM，否则单任务上限形同虚设"
(( ${#AGENTS[@]} > 0 )) || die "AGENTS 不能为空"
[[ "$DRY_RUN" == "0" || "$DRY_RUN" == "1" ]] || die "DRY_RUN 只能是 0 或 1"
[[ "$FORMAL_COLLECTION" == "0" || "$FORMAL_COLLECTION" == "1" ]] || \
  die "FORMAL_COLLECTION 只能是 0 或 1"
[[ "$DERAIL_TMUX" == "0" || "$DERAIL_TMUX" == "1" ]] || \
  die "DERAIL_TMUX 只能是 0 或 1"
[[ "$DERAIL_OPENAI_API_APPROVED" == "0" || "$DERAIL_OPENAI_API_APPROVED" == "1" ]] || \
  die "DERAIL_OPENAI_API_APPROVED 只能是 0 或 1"
[[ "$BACKEND" == "qemu" || "$BACKEND" == "docker" ]] || \
  die "BACKEND 只能是 qemu 或 docker"
[[ "$DERAIL_AUTO_SERVE" == "0" || "$DERAIL_AUTO_SERVE" == "1" ]] || \
  die "DERAIL_AUTO_SERVE 只能是 0 或 1"
is_positive_integer "$SERVE_READY_TIMEOUT" || die "SERVE_READY_TIMEOUT 必须是正整数"
is_positive_integer "$SERVE_POLL_SECONDS" || die "SERVE_POLL_SECONDS 必须是正整数"
is_positive_integer "$SERVE_STARTUP_GRACE" || die "SERVE_STARTUP_GRACE 必须是正整数"
if [[ -n "$SERVE_REPLICAS" ]]; then
  is_positive_integer "$SERVE_REPLICAS" || die "SERVE_REPLICAS 必须是正整数"
fi
if [[ -n "$NUM_VMS_OVERRIDE" ]]; then
  is_positive_integer "$NUM_VMS_OVERRIDE" || die "NUM_VMS_OVERRIDE 必须是正整数"
fi
if [[ "$DERAIL_AUTO_SERVE" == "1" && "$DRY_RUN" == "0" ]]; then
  [[ -x "$SERVE_SCRIPT" || -f "$SERVE_SCRIPT" ]] || die \
    "DERAIL_AUTO_SERVE=1 但找不到 serving 脚本：${SERVE_SCRIPT}"
  command -v curl >/dev/null 2>&1 || die \
    "DERAIL_AUTO_SERVE=1 需要 curl 探测 endpoint ready；请安装 curl 或设置 DERAIL_AUTO_SERVE=0"
  command -v docker >/dev/null 2>&1 || die \
    "DERAIL_AUTO_SERVE=1 需要 docker；请安装 docker 或设置 DERAIL_AUTO_SERVE=0"
fi
command -v "$PYTHON_BIN" >/dev/null 2>&1 || die "找不到 Python：${PYTHON_BIN}"
command -v git >/dev/null 2>&1 || die "找不到 git"
[[ -f "$MODELS_LOCK" ]] || die "找不到模型锁文件：${MODELS_LOCK}"
export PYTHONPATH="${REPO_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"

# 上下文截图数 K（论文 02:12，K=20）落到上游 agent 各自读的环境变量上：
# openai_cuabash 的 ZDR 回放（上游默认只留 4 张）与 qwen_cuabash 的 image_max。
# 与采集 config 冲突的显式值同样只有 ALLOW_CONFIG_OVERRIDE=1 才生效。
for image_var in OPENAI_ZDR_KEEP_IMAGES MYPCBENCH_QWEN_IMAGE_MAX; do
  if [[ -n "${!image_var:-}" && "${!image_var}" != "$CONTEXT_IMAGES" ]]; then
    if [[ "${ALLOW_CONFIG_OVERRIDE:-0}" == "1" ]]; then
      COLLECT_OVERRIDDEN+=("${image_var}=${!image_var}（config context_images 为 ${CONTEXT_IMAGES}）")
      continue
    fi
    COLLECT_IGNORED+=("${image_var}=${!image_var}")
  fi
  export "${image_var}=${CONTEXT_IMAGES}"
done

# 环境钩子（src/derail/rollout/state_probe.py）：官方 runner 起的每个
# run_mypcbench.py 进程经 site_hook/sitecustomize.py 在 reset 后关自动更新 / 通知
# 守护（论文 02:6），在每步后把状态指纹写进任务目录的 state_probes.jsonl。
# 命令与开关都在 environment config 里。
export DERAIL_ENVIRONMENT_CONFIG="$ENVIRONMENT_CONFIG"
export PYTHONPATH="${REPO_ROOT}/src/derail/rollout/site_hook:${PYTHONPATH}"

# .gitignore 可以被 git add -f 绕过，因此正式运行前再检查一次 Git index。
if git -C "$REPO_ROOT" rev-parse --is-inside-work-tree >/dev/null 2>&1; then
  for secret_file in .env env.yaml; do
    if git -C "$REPO_ROOT" ls-files --error-unmatch "$secret_file" >/dev/null 2>&1; then
      die "${secret_file} 已进入 Git index；请先将它从 index 移除，禁止上传敏感信息"
    fi
    git -C "$REPO_ROOT" check-ignore --quiet "$secret_file" || \
      die "${secret_file} 没有被 .gitignore 保护"
  done
fi

# 默认自动获取冻结的官方 runner。已有目录不会被覆盖或自动 checkout。
# 想在跑采集之前单独把 third_party 建起来，用 scripts/setup_third_party.sh。
setup_mypcbench
# manifest 记录的是 checkout 的实际 HEAD 而不是 pin 住的常量：setup_mypcbench 已经
# 校验过两者一致，但产物里应当留下当时真正跑的那个 commit。
actual_commit="$(git -C "$MYPCBENCH_ROOT" rev-parse HEAD)"

RUNNER="${MYPCBENCH_ROOT}/agent-harness/run_parallel_tasks.py"
PROMPT_SOURCE="${MYPCBENCH_ROOT}/agent-harness/agents/prompts.py"
[[ -f "$RUNNER" ]] || die "找不到官方 runner：${RUNNER}"
[[ -f "$PROMPT_SOURCE" ]] || die "找不到官方 prompt source：${PROMPT_SOURCE}"

# 任务来源（configs/collection/sources.yaml）。mypcbench 的文件就是 runner 格式；
# 其余来源（ReRail 组合 workflow）由 derail.rollout.tasks 转成 runner 格式另存。
SOURCE_TASKS_FILE="${TASKS_FILE:-$("$PYTHON_BIN" -m derail.rollout.tasks --source "$TASK_SOURCE")}" \
  || die "解析 task_source=${TASK_SOURCE} 失败"
[[ -f "$SOURCE_TASKS_FILE" ]] || die "找不到 ${TASK_SOURCE} 的任务文件：${SOURCE_TASKS_FILE}"
if [[ "$TASK_SOURCE" == "mypcbench" ]]; then
  TASKS_FILE="$SOURCE_TASKS_FILE"
else
  if [[ "$DRY_RUN" == "1" ]]; then
    converted_tasks="$(mktemp -d)/${TASK_SOURCE}.json"
  else
    converted_tasks="${OUTPUT_ROOT}/${COLLECTION_ID}/_task_source/${TASK_SOURCE}.json"
  fi
  task_source_summary="$("$PYTHON_BIN" -m derail.rollout.tasks --source "$TASK_SOURCE" \
    --tasks-file "$SOURCE_TASKS_FILE" --out "$converted_tasks")" \
    || die "转换 ${TASK_SOURCE} 任务失败"
  info "task_source=${TASK_SOURCE}：${task_source_summary}"
  # 组合 workflow 的 grading 是合成 bundle 里的组合 rubric（sources.yaml 的 rubrics_dir）。
  TASKS_FILE="$converted_tasks"
fi
[[ -f "$TASKS_FILE" ]] || die "找不到 task 文件：${TASKS_FILE}"

# 完整 184-task 文件是正式采集边界。真实启动必须同时得到命令行与锁文件授权；
# dry-run 只打印命令，不构成模型验证，也不需要解锁。
full_tasks_path="$($PYTHON_BIN - "${MYPCBENCH_ROOT}/tasks/final/all_tasks_with_grading.json" <<'PY'
from pathlib import Path
import sys
print(Path(sys.argv[1]).resolve())
PY
)"
selected_tasks_path="$($PYTHON_BIN - "$SOURCE_TASKS_FILE" <<'PY'
from pathlib import Path
import sys
print(Path(sys.argv[1]).resolve())
PY
)"
IS_FULL_TASK_SET=0
if [[ "$selected_tasks_path" == "$full_tasks_path" ]]; then
  IS_FULL_TASK_SET=1
fi
LOCK_FORMAL_AUTHORIZED="$($PYTHON_BIN - "$MODELS_LOCK" <<'PY'
from pathlib import Path
import sys
from derail.mypcbench.launch_contract import formal_collection_authorized
print("1" if formal_collection_authorized(Path(sys.argv[1]).read_text(encoding="utf-8")) else "0")
PY
)"
export FORMAL_COLLECTION LOCK_FORMAL_AUTHORIZED IS_FULL_TASK_SET
# manifest 的 python 片段要从环境里读这两个预算，以及本次的任务清单。
export TASK_TIMEOUT TIMEOUT_PER_VM TASKS_FILE CONTEXT_IMAGES TASK_SOURCE SOURCE_TASKS_FILE
if [[ "$DRY_RUN" == "0" && "$IS_FULL_TASK_SET" == "1" ]]; then
  [[ "$FORMAL_COLLECTION" == "1" ]] || die \
    "检测到完整正式 task 文件；必须在得到用户确认后显式设置 FORMAL_COLLECTION=1"
  [[ "$LOCK_FORMAL_AUTHORIZED" == "1" ]] || die \
    "models.lock.yaml 仍为 formal_collection_authorized: false；禁止启动 184×3"
fi

# 可选加载官方 .env。使用安全 parser，不执行文件内容。
if [[ -f "${MYPCBENCH_ROOT}/.env" ]]; then
  load_dotenv "${MYPCBENCH_ROOT}/.env"
fi

# env.yaml 仅为向后兼容；仓库根目录 .env 最后加载，是当前推荐入口。
load_yaml_env "$ENV_YAML"
load_dotenv "$ROOT_DOTENV"

# 配置 credential 不等于授权调用。任何可能使用真实 OpenAI key 的 collection
# 都要求用户在本次命令上显式确认用途；不要把 approval 写进 .env。
if [[ "$DRY_RUN" != "1" && -n "${OPENAI_API_KEY:-}" ]]; then
  [[ "$DERAIL_OPENAI_API_APPROVED" == "1" ]] || die \
    "检测到 OPENAI_API_KEY，但本次运行未获 API 授权；请先说明用途并在获批后仅对本次命令设置 DERAIL_OPENAI_API_APPROVED=1"
  case "$DERAIL_OPENAI_API_PURPOSE" in
    mypcbench_npc_replies)
      info "本次已获批的 OpenAI API 用途：MyPCBench NPC replies；不会用于 judge"
      ;;
    mypcbench_collection_agent)
      # gpt_5_5 等 API agent 的推理本身就是 OpenAI API 调用（经 OPENAI_BASE_URL
      # 网关路由），与 NPC replies 是两种不同的获批用途，必须显式声明。
      info "本次已获批的 OpenAI API 用途：MyPCBench collection agent 推理；不会用于 judge"
      ;;
    *)
      die "collection 的 API 用途必须显式为 DERAIL_OPENAI_API_PURPOSE=mypcbench_npc_replies 或 mypcbench_collection_agent"
      ;;
  esac
fi
if [[ "$DRY_RUN" != "1" && "$IS_FULL_TASK_SET" == "1" ]]; then
  [[ -n "${OPENAI_API_KEY:-}" ]] || die \
    "完整 MyPCBench collection 需要 .env 中的 OPENAI_API_KEY 供 NPC replies 使用"
fi

# Anthropic key 与 OpenAI key 是两条独立的计费通道，各自单独授权：claude_* agent
# 的每一步推理都是付费调用，不能靠 OpenAI 那道门代批。
if [[ "$DRY_RUN" != "1" && -n "${ANTHROPIC_API_KEY:-}" ]]; then
  [[ "$DERAIL_ANTHROPIC_API_APPROVED" == "1" ]] || die \
    "检测到 ANTHROPIC_API_KEY，但本次运行未获 API 授权；请先说明用途并在获批后仅对本次命令设置 DERAIL_ANTHROPIC_API_APPROVED=1"
  case "$DERAIL_ANTHROPIC_API_PURPOSE" in
    mypcbench_collection_agent)
      info "本次已获批的 Anthropic API 用途：MyPCBench collection agent 推理；不会用于 judge"
      ;;
    *)
      die "collection 的 Anthropic API 用途必须显式为 DERAIL_ANTHROPIC_API_PURPOSE=mypcbench_collection_agent"
      ;;
  esac
fi
GENERIC_OPENAI_BASE_URL="${OPENAI_BASE_URL:-}"

# 加载本地配置后验证 agent，并按需取得冻结的官方 agent loop。
for agent_id in "${AGENTS[@]}"; do
  resolve_agent "$agent_id"
  case "$agent_id" in
    evocua_32b) setup_evocua ;;
    opencua_72b) setup_opencua_osworld ;;
  esac
done

export DERAIL_REPO_ROOT="$REPO_ROOT"
export DERAIL_EVOCUA_ROOT="$EVOCUA_ROOT"
export DERAIL_OPENCUA_OSWORLD_ROOT="$OPENCUA_OSWORLD_ROOT"
# 默认值；agent yaml 声明了自己的 max_steps 时，会在 agent 循环里按 agent 覆盖。
export DERAIL_AGENT_MAX_STEPS="$MAX_STEPS"

# Qwen3.5 官方 wrapper 默认请求 32768 tokens，但锁定的 probe endpoint 只有
# 12288 context。真实运行前必须显式选择生成预算和历史窗口，并向每个 endpoint
# 核对 model ID 与 max_model_len；launcher 不会自动采用 probe override。
#
# 配置检查在这里就做完，不要等 endpoint 启动几十分钟之后才因为少设一个变量失败。
# 需要真的连 endpoint 的那一半推迟到 endpoint ready 之后（见 qwen35_endpoint_preflight）。
QWEN35_ENDPOINT_CONTRACTS_JSON="[]"
if [[ "$DRY_RUN" == "0" ]]; then
  for agent_id in "${AGENTS[@]}"; do
    [[ "$agent_id" == "qwen3_5_35b_a3b" ]] || continue
    [[ -n "${MYPCBENCH_QWEN_MAX_TOKENS:-}" ]] || die \
      "Qwen3.5 真实运行必须显式设置 MYPCBENCH_QWEN_MAX_TOKENS；probe 的 4096 不会自动转为正式配置"
    [[ -n "${MYPCBENCH_QWEN_HISTORY_N:-}" ]] || die \
      "Qwen3.5 真实运行必须显式设置 MYPCBENCH_QWEN_HISTORY_N；probe 的历史策略不会自动转为正式配置"
    [[ "${MYPCBENCH_QWEN_CONTEXT_POLICY:-}" == "tokenize_oldest_first_v1" ]] || die \
      "Qwen3.5 真实运行必须显式设置 MYPCBENCH_QWEN_CONTEXT_POLICY=tokenize_oldest_first_v1"
  done
fi
export QWEN35_ENDPOINT_CONTRACTS_JSON

# 向每个 endpoint 核对 model ID 与 max_model_len。必须在 endpoint ready 之后调用。
qwen35_endpoint_preflight() {
  local endpoint_urls="$1"
  if ! QWEN35_ENDPOINT_CONTRACTS_JSON="$($PYTHON_BIN - \
    "$endpoint_urls" "$RESOLVED_MODEL" "$MYPCBENCH_QWEN_MAX_TOKENS" \
    "$MYPCBENCH_QWEN_HISTORY_N" <<'PY'
import json
import os
import sys
from derail.mypcbench.launch_contract import LaunchContractError, fetch_vllm_contract

urls = [url.strip() for url in sys.argv[1].split(",") if url.strip()]
if not urls:
    raise SystemExit("Qwen3.5 endpoint list is empty")
try:
    contracts = [
        fetch_vllm_contract(
            url,
            sys.argv[2],
            sys.argv[3],
            sys.argv[4],
            api_key=os.environ.get("QWEN35_API_KEY"),
        )
        for url in urls
    ]
except LaunchContractError as exc:
    raise SystemExit(f"Qwen3.5 endpoint contract failed: {exc}") from exc
print(json.dumps(contracts, separators=(",", ":")))
PY
  )"; then
    die "Qwen3.5 endpoint/context preflight 未通过"
  fi
  export QWEN35_ENDPOINT_CONTRACTS_JSON
  info "Qwen3.5 endpoint/context preflight 通过：${QWEN35_ENDPOINT_CONTRACTS_JSON}"
}

# preflight 现在发生在 manifest 落盘之后，所以把 contract 回填进 manifest，
# 保持 collection_manifest.json 的 schema 与旧流程一致。
record_qwen35_contracts() {
  local manifest_path="$1"
  [[ -f "$manifest_path" ]] || return 0
  "$PYTHON_BIN" - "$manifest_path" <<'PY'
import json
import os
import pathlib
import sys

path = pathlib.Path(sys.argv[1])
manifest = json.loads(path.read_text(encoding="utf-8"))
manifest.setdefault("launch_contract", {})["qwen35_endpoint_contracts"] = json.loads(
    os.environ.get("QWEN35_ENDPOINT_CONTRACTS_JSON", "[]")
)
path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
PY
}

if [[ "$DRY_RUN" != "1" ]]; then
  if [[ "$BACKEND" == "qemu" && "${MYPCBENCH_REMOTE_ATTACH:-0}" != "1" ]]; then
    # REMOTE_ATTACH（ROCK proxy 拓扑）下本机不起 VM：KVM 在沙箱侧，
    # driver 起沙时已逐次探针，这里不再要求本机 /dev/kvm。
    [[ "$(uname -s)" == "Linux" ]] || die "QEMU 正式运行需要 Linux host"
    if [[ ! -e /dev/kvm && "${ALLOW_NO_KVM:-0}" != "1" ]]; then
      die "没有 /dev/kvm；如确实接受纯软件模拟，请显式设置 ALLOW_NO_KVM=1"
    fi
  elif [[ "$BACKEND" != "qemu" ]]; then
    command -v docker >/dev/null 2>&1 || die "BACKEND=docker 但找不到 docker"
  fi
fi

prompt_sha256="$(sha256_file "$PROMPT_SOURCE")"
info "MyPCBench commit: ${actual_commit}"
info "prompt SHA-256: ${prompt_sha256}"
info "tasks: ${TASKS_FILE}"
agent_vms_summary=""
agent_steps_summary=""
agent_timeout_summary=""
for banner_agent in "${AGENTS[@]}"; do
  [[ -z "$agent_vms_summary" ]] || agent_vms_summary+=" "
  agent_vms_summary+="${banner_agent}=$(resolve_agent_num_vms "$banner_agent")"
  [[ -z "$agent_steps_summary" ]] || agent_steps_summary+=" "
  agent_steps_summary+="${banner_agent}=$(resolve_agent_max_steps "$banner_agent")"
  [[ -z "$agent_timeout_summary" ]] || agent_timeout_summary+=" "
  agent_timeout_summary+="${banner_agent}=$(resolve_agent_task_timeout "$banner_agent")"
done
info "agents: ${AGENTS[*]}；repeats: ${REPEATS}；VMs: ${agent_vms_summary}"
info "采集配置: $(realpath --relative-to="$REPO_ROOT" "$COLLECT_CONFIG")"
# 协议参数全部打出来。这些值决定轨迹长什么样、因而决定分数；一次临时试跑和正式
# 采集在产物里长得一模一样，漂移必须在开跑前就看得见。
info "  max_steps/agent: ${agent_steps_summary}（采集 config 默认 ${MAX_STEPS}）"
info "  task_timeout/agent: ${agent_timeout_summary}（采集 config 默认 ${TASK_TIMEOUT}s）"
info "  timeout_per_vm=${TIMEOUT_PER_VM}s backend=${BACKEND} screen=${SCREEN_WIDTH}x${SCREEN_HEIGHT} context_images=${CONTEXT_IMAGES}"
info "  task_source=${TASK_SOURCE} environment_hooks=$(realpath --relative-to="$REPO_ROOT" "$DERAIL_ENVIRONMENT_CONFIG")"
if (( ${#COLLECT_OVERRIDDEN[@]} > 0 )); then
  info "  ★ ALLOW_CONFIG_OVERRIDE=1，以下用环境变量而非 config：${COLLECT_OVERRIDDEN[*]}"
fi
if (( ${#COLLECT_IGNORED[@]} > 0 )); then
  info "  已忽略与 config 冲突的环境变量：${COLLECT_IGNORED[*]}（要生效请设 ALLOW_CONFIG_OVERRIDE=1）"
fi

# QEMU image 只在整个 collection 开始时解析一次，随后显式传给每个 repeat，
# 防止跨日期的 daily image refresh 让三个重复实验落在不同初态上。
QCOW2_ARGS=()
image_sha256="not-applicable"
if [[ "$BACKEND" == "qemu" && "${MYPCBENCH_REMOTE_ATTACH:-0}" == "1" ]]; then
  # ROCK proxy 拓扑：guest 的 qcow2 在沙箱内（driver prep 拉取并 sha256
  # 对照 lock），本机不下载不校验；runner 的 remote-attach 分支不会碰这个参数。
  image_sha256="remote-attach-qcow2-in-sandbox"
  info "REMOTE_ATTACH=1：跳过本机 qcow2 解析（guest 资产在沙箱侧）"
elif [[ "$BACKEND" == "qemu" ]]; then
  qcow2_path="${MYPCBENCH_QCOW2:-${MYPCBENCH_ROOT}/mypcbench-vm/mypcbench.qcow2}"
  if [[ "$DRY_RUN" != "1" && ! -f "$qcow2_path" ]]; then
    info "首次下载 MyPCBench QEMU image；文件较大，请耐心等待"
    bash "${MYPCBENCH_ROOT}/scripts/get-eval-image.sh" \
      --out "${MYPCBENCH_ROOT}/mypcbench-vm"
  fi
  if [[ "$DRY_RUN" != "1" ]]; then
    [[ -f "$qcow2_path" ]] || die "找不到 QEMU image：${qcow2_path}"
    qcow2_path="$(cd "$(dirname "$qcow2_path")" && pwd)/$(basename "$qcow2_path")"
    image_sha256="$(sha256_file "$qcow2_path")"
  fi
  QCOW2_ARGS=(--qcow2-path "$qcow2_path")
fi

run_root="${OUTPUT_ROOT}/${COLLECTION_ID}"
if [[ "$DRY_RUN" != "1" ]]; then
  mkdir -p "$run_root"
  # 每个 agent 的 VM 数各自记一笔。只记一个全局 num_vms 会在混排队列里说谎：
  # opencua_72b 跑 1 个 VM、其余跑 4 个，而 manifest 会声称全部都是 4。
  NUM_VMS_PER_AGENT_JSON="$(
    for manifest_agent in "${AGENTS[@]}"; do
      printf '%s\t%s\n' "$manifest_agent" "$(resolve_agent_num_vms "$manifest_agent")"
    done | "$PYTHON_BIN" -c '
import json
import sys

print(json.dumps({
    agent: int(vms)
    for agent, vms in (line.rstrip("\n").split("\t") for line in sys.stdin if line.strip())
}))
'
  )"
  # max_steps / task_timeout 同理逐 agent 记一笔：agent yaml 可以覆盖采集 config，
  # 只记一个全局值会让 manifest 说谎（v1 的 evocua_32b 就是 120 而其余 150）。
  TASK_TIMEOUT_PER_AGENT_JSON="$(
    for manifest_agent in "${AGENTS[@]}"; do
      printf '%s\t%s\n' "$manifest_agent" "$(resolve_agent_task_timeout "$manifest_agent")"
    done | "$PYTHON_BIN" -c 'import json, sys; print(json.dumps({a: int(v) for a, v in (l.rstrip("\n").split("\t") for l in sys.stdin if l.strip())}))'
  )"
  MAX_STEPS_PER_AGENT_JSON="$(
    for manifest_agent in "${AGENTS[@]}"; do
      printf '%s\t%s\n' "$manifest_agent" "$(resolve_agent_max_steps "$manifest_agent")"
    done | "$PYTHON_BIN" -c '
import json
import sys

print(json.dumps({
    agent: int(steps)
    for agent, steps in (line.rstrip("\n").split("\t") for line in sys.stdin if line.strip())
}))
'
  )"
  export NUM_VMS_PER_AGENT_JSON MAX_STEPS_PER_AGENT_JSON TASK_TIMEOUT_PER_AGENT_JSON
  "$PYTHON_BIN" - "$run_root/collection_manifest.json" \
    "$actual_commit" "$prompt_sha256" "$image_sha256" "$REPEATS" \
    "$MAX_STEPS" "$REPO_ROOT" "$MYPCBENCH_PATCH" "$EVOCUA_ROOT" \
    "$OPENCUA_OSWORLD_ROOT" "$EVOCUA_COMMIT" "$OPENCUA_OSWORLD_COMMIT" \
    "${AGENTS[@]}" <<'PY'
import hashlib
import importlib.metadata
import json
import os
import pathlib
import platform
import sys

path = pathlib.Path(sys.argv[1])
repo_root = pathlib.Path(sys.argv[7])
mypcbench_patch = pathlib.Path(sys.argv[8])
evocua_root = pathlib.Path(sys.argv[9])
opencua_root = pathlib.Path(sys.argv[10])

source_files = [
    mypcbench_patch,
    repo_root / "third_party/MyPCBench/agent-harness/agents/qwen_cua.py",
    repo_root / "third_party/MyPCBench/agent-harness/agents/vendored_paper_results/qwen35vl_agent.py",
    *sorted((repo_root / "src/derail/mypcbench").glob("*.py")),
    # `*.txt` 而不是 `*_mypcbench_system.txt`：共享的 mypcbench_env_block.txt 匹配
    # 不上后者，而它恰恰是所有 agent 都拿到的那一块（persona + 17 应用端口表），
    # 漏哈希就等于产物里证明不了这次发出去的环境描述是哪一版。
    *sorted((repo_root / "prompts/agents").glob("*.txt")),
    # agent 的行为参数现在由这些 yaml 决定（见 derail/mypcbench/agent_config.py），
    # 不哈希它们就没法证明这次采集用的是哪一版配置。
    *sorted((repo_root / "configs/agents").glob("*.yaml")),
    # 环境钩子（确定性命令 + 每步状态探针）及其配置、任务来源配置。
    *sorted((repo_root / "src/derail/rollout").rglob("*.py")),
    pathlib.Path(os.environ["DERAIL_ENVIRONMENT_CONFIG"]),
    # sqlite_digest 探针送进 guest 的 digest 脚本与易变列规则。
    repo_root / "infra/snapshot/changelog_replay.py",
    repo_root / "infra/volatile_columns.json",
    repo_root / "configs/collection/sources.yaml",
]
for external_file in (
    evocua_root / "mm_agents/evocua/evocua_agent.py",
    evocua_root / "mm_agents/evocua/prompts.py",
    evocua_root / "mm_agents/evocua/utils.py",
    opencua_root / "mm_agents/opencua/opencua_agent.py",
    opencua_root / "mm_agents/opencua/prompts.py",
    opencua_root / "mm_agents/opencua/utils.py",
):
    if external_file.is_file():
        source_files.append(external_file)

source_hashes = {
    str(source.relative_to(repo_root) if source.is_relative_to(repo_root) else source):
        hashlib.sha256(source.read_bytes()).hexdigest()
    for source in source_files
}

# 把解析后的 live 配置直接摊进 manifest：采样参数以前藏在 EVOCUA_*/OPENCUA_*
# 环境变量里，事后无从追查。只有走 DERAIL factory 的 agent 有 live 配置，其余
# 由 MyPCBench 自己构造，这里如实留空。
from derail.mypcbench.agent_config import AGENT_ID_BY_TYPE, load_agent_config

_factory_agents = set(AGENT_ID_BY_TYPE.values())
agent_live_config = {
    agent_id: dict(load_agent_config(agent_id).live)
    for agent_id in sys.argv[13:]
    if agent_id in _factory_agents
}
tasks_file = pathlib.Path(os.environ["TASKS_FILE"]).resolve()
# 不 try/except：脚本早就 `[[ -f "$TASKS_FILE" ]] || die` 过了，清单要是坏的，
# 几秒后 runner 一样会崩，在这里静默吞掉只会让 manifest 说谎。
tasks_total = len(json.loads(tasks_file.read_text(encoding="utf-8")))

manifest = {
    "source_benchmark": "mypcbench",
    "mypcbench_commit": sys.argv[2],
    "prompt_sha256": sys.argv[3],
    "image_sha256": sys.argv[4],
    # 任务清单是"这次到底该跑多少个 episode"的唯一依据。以前 manifest 只记
    # launch_contract.full_task_set 这个布尔，一次 20 条的 smoke 和一次 184 条的
    # 正式采集在产物里分不出分母，01_watch_progress.sh 也只能把 184 写死。
    "tasks_file": str(tasks_file),
    "tasks_total": tasks_total,
    # 任务来源（configs/collection/sources.yaml）与转换前的原始文件。
    "task_source": os.environ["TASK_SOURCE"],
    "task_source_file": str(pathlib.Path(os.environ["SOURCE_TASKS_FILE"]).resolve()),
    "repeats": int(sys.argv[5]),
    # 只记按 agent 的表。全局 num_vms 已经删掉：本地 serving 的并发度由本机
    # GPU 数除以 tensor_parallel_size 现算，混排队列里没有"一个全局值"可言。
    "num_vms_per_agent": json.loads(os.environ.get("NUM_VMS_PER_AGENT_JSON", "{}")),
    # 采集 config 的默认值；实际生效的是下面这张按 agent 的表。
    "max_steps_per_agent": json.loads(os.environ.get("MAX_STEPS_PER_AGENT_JSON", "{}")),
    "max_steps": int(sys.argv[6]),
    # 预算三件套一起记：光有 max_steps 无法解释一次采集为什么在某个任务上停下，
    # 也无法区分「跑满 100 步」和「撞上墙钟上限被截断」。
    "task_timeout_seconds": int(os.environ.get("TASK_TIMEOUT", "0")) or None,
    "task_timeout_per_agent": json.loads(os.environ.get("TASK_TIMEOUT_PER_AGENT_JSON", "{}")),
    # 上下文截图数 K；导出给上游 agent 的实际值一并记下。
    "context_images": int(os.environ["CONTEXT_IMAGES"]),
    "context_image_env": {
        name: os.environ.get(name)
        for name in ("OPENAI_ZDR_KEEP_IMAGES", "MYPCBENCH_QWEN_IMAGE_MAX")
    },
    "environment_config": os.environ["DERAIL_ENVIRONMENT_CONFIG"],
    "timeout_per_vm_seconds": int(os.environ.get("TIMEOUT_PER_VM", "0")) or None,
    "evocua_agent_commit": sys.argv[11],
    "opencua_osworld_agent_commit": sys.argv[12],
    "agents": sys.argv[13:],
    "launch_contract": {
        "full_task_set": os.environ.get("IS_FULL_TASK_SET") == "1",
        "formal_collection_requested": os.environ.get("FORMAL_COLLECTION") == "1",
        "formal_collection_authorized": os.environ.get("LOCK_FORMAL_AUTHORIZED") == "1",
        "qwen35_endpoint_contracts": json.loads(
            os.environ.get("QWEN35_ENDPOINT_CONTRACTS_JSON", "[]")
        ),
        "qwen35_context_policy": os.environ.get("MYPCBENCH_QWEN_CONTEXT_POLICY"),
    },
    "adapter_policy": {
        "runner_owns_max_steps": True,
        "official_upstream_guard_offset": 1,
        "invalid_native_tool_call": "one_schema_repair_then_FAIL",
        "qwen36_action_loop": "one_replan_then_FAIL",
        "qwen38_action_loop": "one_replan_then_FAIL",
    },
    "harness_runtime": {
        "python_executable": sys.executable,
        "python_version": platform.python_version(),
        "packages": {
            package: importlib.metadata.version(package)
            for package in ("backoff", "openai", "Pillow", "requests")
        },
    },
    "adapter_source_sha256": source_hashes,
    "agent_live_config": agent_live_config,
}
path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
PY
fi

for agent_id in "${AGENTS[@]}"; do
  resolve_agent "$agent_id"
  agent_vms="$(resolve_agent_num_vms "$agent_id")"
  agent_max_steps_value="$(resolve_agent_max_steps "$agent_id")"
  agent_task_timeout_value="$(resolve_agent_task_timeout "$agent_id")"
  (( agent_task_timeout_value < TIMEOUT_PER_VM )) || die \
    "${agent_id} 的 task_timeout(${agent_task_timeout_value}) 必须小于 TIMEOUT_PER_VM"
  # 上游 guard 从这个环境变量取值（factory._upstream_step_budget 取它 +1），
  # 所以必须在起 runner 之前按 agent 重设，不能沿用循环外的全局默认。
  export DERAIL_AGENT_MAX_STEPS="$agent_max_steps_value"
  if [[ "$agent_max_steps_value" != "$MAX_STEPS" ]]; then
    info "${agent_id}：max_steps 取 agent config 声明的 ${agent_max_steps_value}（采集 config 默认 ${MAX_STEPS}）"
  fi
  if [[ -n "$NUM_VMS_OVERRIDE" ]]; then
    info "${agent_id}：★ VM 数被 NUM_VMS_OVERRIDE 压到 ${agent_vms}（不覆盖时是 $(agent_vm_count "$REPO_ROOT" "$agent_id")）"
  else
    info "${agent_id}：VM 数 ${agent_vms}（本地 serving 按可用 GPU / tensor_parallel_size 推导，托管 API 取 yaml 的 num_vms）"
  fi
  VLLM_ARGS=()
  # 必须逐个 agent 清空：只有 LOCAL_ENDPOINT 分支会给它赋值，不清空的话走托管
  # API 的 agent 会继承上一个本地模型的 URL，看门狗就会去盯一个已经停掉的
  # endpoint，把一次完全正常的采集掐死。
  endpoint_urls=""
  if [[ "$RESOLVED_REQUIRED_ENV" == "LOCAL_ENDPOINT" ]]; then
    # 同一时刻只能 serve 一个模型，所以 serve/stop 放在 agent 循环里：
    # 起当前 agent 的 endpoint → 跑完它的全部 repeat → 停掉 → 换下一个。
    if [[ "$DERAIL_AUTO_SERVE" == "1" && "$DRY_RUN" == "0" ]]; then
      auto_serve_start "$agent_id" "$RESOLVED_BASE_URLS_ENV"
    fi
    endpoint_urls="${!RESOLVED_BASE_URLS_ENV:-}"
    if [[ -z "$endpoint_urls" ]]; then
      endpoint_urls="${!RESOLVED_BASE_URL_ENV:-$GENERIC_OPENAI_BASE_URL}"
    fi
    [[ -n "$endpoint_urls" ]] || die \
      "${agent_id} 需要 ${RESOLVED_BASE_URLS_ENV}、${RESOLVED_BASE_URL_ENV} 或 OPENAI_BASE_URL"
    # 官方 parallel runner 会按 VM round-robin，每个 child 只看到自己的 URL。
    VLLM_ARGS=(--vllm-base-urls "$endpoint_urls")
    endpoint_count="$(awk -F',' '{print NF}' <<< "$endpoint_urls")"
    if (( agent_vms > endpoint_count )); then
      info "注意：${agent_id} 的 ${agent_vms} 个 VM 将共享 ${endpoint_count} 个推理 endpoint"
    fi
    # endpoint 此时才保证 ready，contract 核对放在这里，并回填进 manifest。
    if [[ "$agent_id" == "qwen3_5_35b_a3b" && "$DRY_RUN" == "0" ]]; then
      qwen35_endpoint_preflight "$endpoint_urls"
      record_qwen35_contracts "${run_root}/collection_manifest.json"
    fi
  elif [[ -n "$RESOLVED_REQUIRED_ENV" && -z "${!RESOLVED_REQUIRED_ENV:-}" ]]; then
    die "${agent_id} 需要环境变量 ${RESOLVED_REQUIRED_ENV}"
  fi

  # 官方 runner 只接受 --agent-type，configs/agents/*.yaml 却按 agent_id 命名。
  # factory 自己有一张 agent_type -> agent_id 表并以它为准；这里导出 agent_id 只
  # 作交叉核对，两张表漂移时 agent_config 会 hard fail 而不是加载错配置。
  export DERAIL_AGENT_ID="$agent_id"

  repeat_start_index="${REPEAT_START_INDEX:-1}"
  is_positive_integer "$repeat_start_index" || die "REPEAT_START_INDEX 必须是正整数"
  repeat_end_index=$((repeat_start_index + REPEATS - 1))
  for repeat_index in $(seq "$repeat_start_index" "$repeat_end_index"); do
    result_dir="${run_root}/${agent_id}/repeat_${repeat_index}"
    container_base="derail-${COLLECTION_ID}-${agent_id}-r${repeat_index}"
    command=(
      "$PYTHON_BIN" "$RUNNER"
      --backend "$BACKEND"
      --tasks-file "$TASKS_FILE"
      --num-vms "$agent_vms"
      --agent-type "$RESOLVED_AGENT_TYPE"
      --model "$RESOLVED_MODEL"
      --max-steps "$agent_max_steps_value"
      --timeout-per-vm "$TIMEOUT_PER_VM"
      --task-timeout "$agent_task_timeout_value"
      --port-base "$PORT_BASE"
      --container-base "$container_base"
      --result-dir "$result_dir"
      --screen-width "$SCREEN_WIDTH"
      --screen-height "$SCREEN_HEIGHT"
      "${VLLM_ARGS[@]}"
      "${QCOW2_ARGS[@]}"
    )

    info "${agent_id}，repeat ${repeat_index}/${REPEATS}"
    if [[ "$DRY_RUN" == "1" ]]; then
      printf '  '
      printf '%q ' "${command[@]}"
      printf '\n'
    else
      # 只有本地 endpoint 才需要看门狗；托管 API 的瞬时 5xx 不该掐掉整个采集。
      endpoint_watchdog_start "${endpoint_urls:-}" "$container_base"
      runner_status=0
      "${command[@]}" || runner_status=$?
      endpoint_watchdog_stop
      if endpoint_watchdog_tripped; then
        dead_endpoint="$(cat "$ENDPOINT_WATCHDOG_TRIP")"
        rm -f "$ENDPOINT_WATCHDOG_TRIP"
        die "${agent_id} repeat ${repeat_index}：endpoint ${dead_endpoint} 在采集途中消失，
     runner 已被中止。${result_dir} 下这一刻之后的 episode 都是空壳（PREDICT_CRASH +
     result.txt=1.0），必须删掉重跑，不要拿去判分。
     先确认没有别的 collection 在抢同一批显卡，再看 ${SERVING_LOG_ROOT} 下最新的 endpoint_*.log。"
      fi
      if [[ -n "$ENDPOINT_WATCHDOG_TRIP" ]]; then
        rm -f "$ENDPOINT_WATCHDOG_TRIP"
      fi
      (( runner_status == 0 )) || die \
        "${agent_id} repeat ${repeat_index}：runner 以退出码 ${runner_status} 结束"
    fi
  done

  # 换下一个 agent 之前先把显卡还回去。复用的 endpoint 不属于本脚本，不会被停。
  auto_serve_stop

  # 每个 agent 各留一份具名 manifest 快照。
  #
  # collection_manifest.json 是「最后写的赢」：往同一个 COLLECTION_ID 里追加
  # agent 时，后跑的会把前面那份整个覆盖掉。v1 就是这么丢掉的 —— 里面有
  # gpt5_5 / claude_opus_4_8 / qwen3_5 / holo 四个 agent 各 184 条轨迹，而
  # manifest 只剩最后跑的 holo 那份，其余三个用的什么 max_steps、什么 prompt
  # SHA、什么 VM image 已经无从查证。
  #
  # 不改 collection_manifest.json 的名字：03/04/09 和 01_watch_progress 都读它。
  # 这里只是额外追加，且放在 agent 循环末尾 —— 那时 qwen35 endpoint contract 之类
  # 的后置写入已经落盘，快照才是完整的。
  if [[ "$DRY_RUN" != "1" ]]; then
    "$PYTHON_BIN" - "$run_root" "$agent_id" <<'PY'
import json
import pathlib
import sys

run_root = pathlib.Path(sys.argv[1])
agent_id = sys.argv[2]
shared = run_root / "collection_manifest.json"
manifest = json.loads(shared.read_text(encoding="utf-8"))

# 窄化到这一个 agent：留着全量的 agents 列表会让快照声称自己覆盖了别人。
manifest["agents"] = [agent_id]
for field in ("num_vms_per_agent", "max_steps_per_agent", "task_timeout_per_agent", "agent_live_config"):
    table = manifest.get(field)
    if isinstance(table, dict):
        manifest[field] = {agent_id: table[agent_id]} if agent_id in table else {}

target = run_root / f"collection_manifest.{agent_id}.json"
target.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
print(f"[DERAIL collection] 已留存 {target.name}")
PY
  fi
done

if [[ "$DRY_RUN" == "1" ]]; then
  info "dry run 完成，没有启动 VM 或调用模型"
else
  info "collection 完成：${run_root}"
fi
