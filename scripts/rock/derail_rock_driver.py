#!/usr/bin/env python3
# =============================================================================
# DERAIL MyPCBench 采集 — ROCK 沙箱 driver（在 Nebula job 内运行）。
#
# 职责（照 MCUA submit_single_rock.py 的 legacy closed-model 模式）：
#   1. 建 ROCK 沙箱（429 指数退避 ≤8 次）
#   2. /dev/kvm 探针，缺失则回收重建（≤14 次；不降配——guest 要 8G RAM）
#   3. upload_dir 传 DERAIL repo -> /opt/derail（harness 由 OSS tar 恢复）
#   4. 0600 secrets 注入沙箱（OSS AK + routify key，永不上 argv）
#   5. 沙箱内 prep：ossutil 拉 qcow2/OVMF/harness tar + python venv 装依赖
#   6. nohup 跑 scripts/01_collect_trajectories.sh（DERAIL_TMUX=0 NUM_VMS_OVERRIDE=1）
#   7. 诊断探针 + 结果目录 ossutil 上 OSS
#   8. finally: sandbox.stop()（僵尸沙箱 = 429 之源）
#   9. 沙箱中途死亡容错（2026-08-15）：采集期每 1800s 增量上 OSS 抢救；
#      arun 报 "sandbox not started" → 重建沙箱 + 强制 RESUME_SEED 续采（≤2 轮）
#
# DERAIL_ROCK_TOPOLOGY=proxy（开源模型 full 阶段）：仿 MCUA OSWorld 拓扑。
# 沙箱→Nebula pod IP 不可达（实证），改为 agent loop 上 Nebula（L2），沙箱
# 只留 guest QEMU + gateway(:8080)（L3）；本 driver 在 Nebula 侧起 bridge
# （scripts/rock/rock_proxy_bridge.py）把 127.0.0.1:PORT_BASE 系端口映射到
# ROCK Proxy 路径前缀端点，01 脚本带 MYPCBENCH_REMOTE_ATTACH=1 跑，模型调用
# 走 localhost vLLM。产物结构不变（traj/截图/result.txt），经 OSS FUSE 挂载上云。
#
# DRY_RUN=1 只打印计划（默认 0：entry 提交即执行；干跑验证走 submit 侧）。
# =============================================================================
"""ROCK sandbox driver for the DERAIL MyPCBench trajectory collection."""

from __future__ import annotations

import asyncio
import json
import os
import re
import shlex
import sys
import tempfile
import time
from pathlib import Path
from typing import Optional

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent


def _env(name: str, default: str) -> str:
    """空字符串回退默认值（.rock_run.env 会把未设置的 knob 落成 export FOO=''）。"""
    value = os.environ.get(name)
    return value if value is not None and value.strip() != "" else default


RUNTIME_CONFIG = REPO / "configs/collection/mypcbench_runtime.yaml"


def _runtime_default(key: str) -> str:
    """采集协议参数的默认值只认 configs/collection/mypcbench_runtime.yaml（论文统一
    预算 150 步 / 3600 秒 / 3 次 / K=20），解析规则与 01 的 load_collection_config
    相同。driver 不再自带数字，免得和 01 的 config 各说各话。"""
    for raw in RUNTIME_CONFIG.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if line and not line.startswith("#") and ":" in line:
            name, text = line.split(":", 1)
            if name.strip() == key:
                return text.split("#", 1)[0].strip().strip('"').strip("'")
    raise KeyError(f"{RUNTIME_CONFIG} 缺少 {key}")


# --- 采集参数 -------------------------------------------------------------------
AGENT_ID = _env("AGENT_ID", "gpt_5_5")
DERAIL_WORKLOAD = _env("DERAIL_WORKLOAD", "collection")
SHARD_FILE = _env("SHARD_FILE", "configs/mypcbench_task_shards/smoke_one.json")
_shard_tag = Path(SHARD_FILE).stem[-24:]
COLLECTION_ID = _env(
    "COLLECTION_ID",
    time.strftime("%Y%m%dT%H%M%SZ", time.gmtime()) + f"-{AGENT_ID}-{_shard_tag}",
)
GPT55_MODEL = _env("GPT55_MODEL", "openai.gpt-5.5")
GPT56_LUNA_MODEL = _env("GPT56_LUNA_MODEL", "gpt-5.6-luna")
CLAUDE_OPUS_4_8_MODEL = _env("CLAUDE_OPUS_4_8_MODEL", "claude-opus-4-8")
CLAUDE_SONNET_5_MODEL = _env("CLAUDE_SONNET_5_MODEL", "claude-sonnet-5")
# kimi_k3 走 routify chat completions（同一把 OPENAI key）+ DERAIL 冻结 schema
# scaffold。行为参数（max_tokens、temperature 省略等）全部读
# configs/agents/kimi_k3.yaml，沙箱里没有对应的环境变量旋钮。
KIMI_K3_MODEL = _env("KIMI_K3_MODEL", "kimi-k3")
# Bedrock 拒绝 computer-use + prompt-caching 的组合 beta 头（实测 400
# invalid beta flag）；原生 computer 工具不能让，所以默认关 caching。
CLAUDE_PROMPT_CACHING_BETA = _env("CLAUDE_PROMPT_CACHING_BETA", "0")
# 网关 RPM=60（按 API key，跨模型共享）；429 有独立指数退避预算。
OPENAI_RATE_LIMIT_RETRIES = _env("OPENAI_RATE_LIMIT_RETRIES", "8")
ANTHROPIC_RATE_LIMIT_RETRIES = _env("ANTHROPIC_RATE_LIMIT_RETRIES", "8")
MYPCBENCH_OPENAI_REASONING_EFFORT = _env(
    "MYPCBENCH_OPENAI_REASONING_EFFORT", "high"
)
MYPCBENCH_OPENAI_EMPTY_OUTPUT_RETRIES = _env(
    "MYPCBENCH_OPENAI_EMPTY_OUTPUT_RETRIES", "2"
)
MYPCBENCH_SCREENSHOT_RETRIES = _env("MYPCBENCH_SCREENSHOT_RETRIES", "3")
REPEATS = _env("REPEATS", _runtime_default("repeats"))
# 落盘目录 repeat_<k> 的起始编号（补跑 repeat_2/3 时设 2）。
REPEAT_START_INDEX = _env("REPEAT_START_INDEX", "1")
# 沙箱是按单 VM 规格开的，所以走 01 的逃生口把 agent config 推导出的并发度压到 1
# —— 采集 config 那个全局 NUM_VMS 已经删掉了，导出它不会有任何效果。
NUM_VMS_OVERRIDE = _env("NUM_VMS_OVERRIDE", "1")
# 默认取采集 config（论文 150 步）。F1-F4 旧数据均为 100 口径，跨批次对比须交代。
MAX_STEPS = _env("MAX_STEPS", _runtime_default("max_steps"))
# bash 记账模式（tool_agent.py 读同名环境变量）：internal=bash 轮步内循环
# 不占步数（默认）；steps=统一记账，bash 轮每轮计为 TOOL_CALL 步。注意下面
# env_exports 是白名单——不落进白名单的环境变量到不了沙箱里的 agent，
# 2026-08-24 steps smoke 就因漏了这里白跑成 internal（traj 0 个 TOOL_CALL 步）。
DERAIL_BASH_ACCOUNTING = _env("DERAIL_BASH_ACCOUNTING", "internal")
# 单任务墙钟预算，默认取采集 config（论文 3600 秒，所有 agent 相同）。v1 曾默认
# 7200s（2026-08-11 截断专项：sonnet 396s/step 在 3600s 内只走 ~9 步），F4/Kimi
# 等批次是 7200 口径，跨批次对比须交代。
TASK_TIMEOUT = _env("TASK_TIMEOUT", _runtime_default("task_timeout"))
TIMEOUT_PER_VM = _env("TIMEOUT_PER_VM", "259200")
FORMAL_COLLECTION = _env("FORMAL_COLLECTION", "0")
DERAIL_OPENAI_API_APPROVED = _env("DERAIL_OPENAI_API_APPROVED", "1")
DERAIL_OPENAI_API_PURPOSE = _env("DERAIL_OPENAI_API_PURPOSE", "mypcbench_collection_agent")
# Anthropic 是独立计费通道，01 脚本对它有单独的授权门。
DERAIL_ANTHROPIC_API_APPROVED = _env("DERAIL_ANTHROPIC_API_APPROVED", "1")
DERAIL_ANTHROPIC_API_PURPOSE = _env(
    "DERAIL_ANTHROPIC_API_PURPOSE", "mypcbench_collection_agent"
)

# agent 家族：决定走哪条凭证/依赖/协议路径（见 docs/ROUTIFY_API_MATRIX.md）。
# kimi_k3_cuabash 与 kimi_k3 同网关同模型（KIMI_K3_MODEL），只翻工具面。
OPENAI_AGENTS = ("gpt_5_5", "gpt_5_6_luna", "kimi_k3", "kimi_k3_cuabash")
# claude_opus_4_7 只是 F4 重采当时的 AGENT_ID（配 MODEL 覆盖成 4.8）；其 yaml 已
# 改名为 claude_opus_4_8.yaml，这里只列现存配置。
ANTHROPIC_AGENTS = ("claude_opus_4_8", "claude_sonnet_5")
# 开源模型家族：不经任何 API 网关，采样直连同 Nebula job 内的 vLLM（entry
# 起服务后把 OPENCUA_BASE_URLS=http://<JOB_IP>:<port>/v1 导出进来）。沙箱侧
# 链路与 API agent 完全一致，只是 endpoint 地址不同（MCUA legacy 已验拓扑）。
OPENCUA_AGENTS = ("opencua_72b", "qwen3_8_27b", "qwen3_5_35b_a3b", "evocua_32b")
IS_ANTHROPIC_AGENT = AGENT_ID in ANTHROPIC_AGENTS
IS_OPENCUA_AGENT = AGENT_ID in OPENCUA_AGENTS
OPENCUA_BASE_URLS = _env("OPENCUA_BASE_URLS", "")
OPENCUA_MODEL = _env("OPENCUA_MODEL", "opencua-72b")
QWEN38_MODEL = _env("QWEN38_MODEL", "Qwen/Qwen3.8-27B")
QWEN35_MODEL = _env("QWEN35_MODEL", "Qwen/Qwen3.5-35B-A3B")
EVOCUA_MODEL = _env("EVOCUA_MODEL", "EvoCUA")

# --- ROCK proxy 拓扑（开源模型 full 阶段；仿 MCUA OSWorld 三层架构） --------------
# sandbox 内跑 guest + gateway，agent loop 在 Nebula；入口开关由 entry 导出。
DERAIL_ROCK_TOPOLOGY = _env("DERAIL_ROCK_TOPOLOGY", "in-sandbox")
IS_PROXY_TOPOLOGY = DERAIL_ROCK_TOPOLOGY == "proxy"
# 端口布局（与 derail_guest_boot.sh / harness proxy patch 三方对齐）：
#   Nebula bridge :PORT_BASE       -> Proxy /pcapi -> gateway -> guest 控制面 :5000
#   Nebula bridge :PORT_BASE+10+i  -> Proxy /app<cp> -> gateway -> guest app <cp>
PROXY_PORT_BASE = int(_env("PROXY_PORT_BASE", "15000"))
_PHASE5_SUPERVISOR_TASK = None
PROXY_LIFECYCLE_PORT = int(_env("PROXY_LIFECYCLE_PORT", "19090"))
# Nebula 侧产物先落本地，再经 OSS FUSE 挂载上云（同 bucket 路径）。
PROXY_RESULTS_LOCAL = _env("PROXY_RESULTS_LOCAL", "/tmp/derail-results")
PROXY_RESULTS_OSS_MOUNT = _env(
    "PROXY_RESULTS_OSS_MOUNT",
    f"/data/oss_bucket_0/{_env('OSS_PREFIX', '<oss-prefix>')}/DERAIL/results/raw/mypcbench",
)
TAKEOVER_INPUT_ROOT = Path(_env(
    "TAKEOVER_INPUT_ROOT",
    f"/data/oss_bucket_0/{_env('OSS_PREFIX', '<oss-prefix>')}/DERAIL/"
    "takeover_inputs/failure_prefix_v1_full_v3",
))
TAKEOVER_SOURCE_AGENT = _env("TAKEOVER_SOURCE_AGENT", "opencua_72b")
TAKEOVER_TARGET_AGENT = _env("TAKEOVER_TARGET_AGENT", AGENT_ID)
TAKEOVER_ANNOTATOR = _env("TAKEOVER_ANNOTATOR", "dingyi")
TAKEOVER_BUILD_DIR = _env(
    "TAKEOVER_BUILD_DIR", "artifacts/derail_builds/opencua72b_dingyi_takeover"
)
TAKEOVER_DEPTH = _env("TAKEOVER_DEPTH", "0")
TAKEOVER_CONDITION = _env("TAKEOVER_CONDITION", "unaware")
TAKEOVER_SHARD_COUNT = _env("TAKEOVER_SHARD_COUNT", "1")
TAKEOVER_SHARD_OFFSET = _env("TAKEOVER_SHARD_OFFSET", "0")
TAKEOVER_TRAJECTORY_ID_FILTER = _env("TAKEOVER_TRAJECTORY_ID_FILTER", "")
TAKEOVER_TRAJECTORY_ID_FILE = _env("TAKEOVER_TRAJECTORY_ID_FILE", "")
TAKEOVER_TOKENIZE_BASE_URL = _env("TAKEOVER_TOKENIZE_BASE_URL", "")
TAKEOVER_TOKENIZE_MODE = _env("TAKEOVER_TOKENIZE_MODE", "vllm")
TAKEOVER_CONTEXT_CAP = _env("TAKEOVER_CONTEXT_CAP", "0")
PHASE5_SHARD_COUNT = _env("PHASE5_SHARD_COUNT", "1")
PHASE5_SHARD_INDEX = _env("PHASE5_SHARD_INDEX", "0")
PHASE5_SMOKE_MODE = _env("PHASE5_SMOKE_MODE", "single_e2e")

# --- ROCK 沙箱 ------------------------------------------------------------------
# 端点与个人 ID 不入库：跑 driver 前必须 export ROCK_BASE_URL / ROCK_USER_ID。
ROCK_BASE_URL = _env("ROCK_BASE_URL", "<rock-endpoint>")
ROCK_API_KEY = _env("ROCK_API_KEY", "")  # 空 => 自建无鉴权
ROCK_USER_ID = _env("ROCK_USER_ID", "<rock-user-id>")
ROCK_EXPERIMENT_ID = _env("ROCK_EXPERIMENT_ID", "derail-mypcbench")
ROCK_SANDBOX_IMAGE = _env("ROCK_SANDBOX_IMAGE", "<rock-sandbox-image>")
ROCK_CLUSTER = _env("ROCK_CLUSTER", "<rock-cluster>")
ROCK_MEMORY = _env("ROCK_MEMORY", "16g")
ROCK_CPUS = float(_env("ROCK_CPUS", "8"))
ROCK_DISK = _env("ROCK_DISK", "100g")
# 单片 46 任务中位 ~23h；沙箱 TTL 48h 兜底自愈（被杀的 job 走不到 finally stop）。
ROCK_AUTO_CLEAR_SECONDS = int(_env("ROCK_AUTO_CLEAR_SECONDS", str(48 * 3600)))
ROCK_STARTUP_TIMEOUT = float(_env("ROCK_STARTUP_TIMEOUT", "600"))
# arun wait 上限默认 30h（中位 23h + 裕量）；harness 内部还有 72h VM 级兜底。
ROCK_RUN_TIMEOUT = int(_env("ROCK_RUN_TIMEOUT", str(30 * 3600)))
ROCK_RESULT_TIMEOUT = int(_env("ROCK_RESULT_TIMEOUT", "1800"))
# 断点续采（RESUME_SEED=1）：起跑前先把 OSS 上同 COLLECTION_ID 的部分结果
# 回灌沙箱，harness 内置 result.txt 跳过守卫（run_mypcbench.py）只重跑未完
# 成任务；无 result.txt 的任务目录在回灌后删除，避免截断残留与新轨迹混杂。
# _run 的沙箱死亡重建路径会把它就地置 True（故不能冻结成局部常量）。
RESUME_SEED = _env("RESUME_SEED", "0") == "1"
# 重跑强制清场（2026-08-18 F4 opus 4.8 rerun 实证教训）：续采守卫语义是
# 「有 result.txt 就跳过」，而 rerun 的目标恰恰是已完成、带 result.txt 的
# 任务——不清场则整个 rerun 清单被守卫静默跳过（s1/s2/s3 共 86 任务 38s
# 「完成」，job 照常 exit 0，OSS 上仍是旧模型轨迹）。RERUN_PURGE=1：seed
# 回灌前先删 OSS 上清单任务的旧目录（cp -r -f 只增不删，不清则旧截图会
# 与新轨迹混存在同一任务目录，2026-08-19 实证 3501 张旧图残留），再回灌、
# 对沙箱内清单任务目录 rm -rf 逼 harness 重跑。nonce 标记文件随增量抢救
# 上 OSS：沙箱死亡重建的第 2 轮 seed 见到标记即跳过清场，保住第 1 轮已
# 跑出的新轨迹；新一轮提交是新进程新 nonce，旧标记被清除。
RERUN_PURGE = _env("RERUN_PURGE", "0") == "1"
RERUN_PURGE_NONCE = f"{os.getpid()}{int(time.time())}"
RERUN_PURGE_MARKER_GLOB = ".derail_rerun_purged_*"
_TASK_ID_RE = re.compile(r"^[A-Za-z0-9_.\-]+$")


def _shard_task_ids() -> list:
    """SHARD_FILE 里的任务 id 列表（RERUN_PURGE 清场用）。解析失败或 id
    形状异常一律硬断：清场名单错了比不跑更危险（会误删无关轨迹）。"""
    path = REPO / SHARD_FILE
    with open(path, encoding="utf-8") as fh:
        data = json.load(fh)
    ids = [task["id"] for task in data]
    bad = [t for t in ids if not _TASK_ID_RE.match(str(t))]
    if bad:
        raise SystemExit(f"[driver] ERROR: RERUN_PURGE 拒绝异常 task id: {bad[:5]}")
    if not ids:
        raise SystemExit("[driver] ERROR: RERUN_PURGE 但 SHARD_FILE 任务清单为空")
    return ids
# 沙箱内增量抢救上传间隔：沙箱会在长任务中途死亡（2026-08-15 opus_s3 五提 /
# kimi CRASH-5 三提两 job 独立复现），死则本地结果不可达，只能靠运行期
# 周期性抢上 OSS 保产物；重建后 RESUME_SEED 从 OSS 续采，最多损失一个周期。
SHIP_INCREMENT_SECONDS = int(_env("SHIP_INCREMENT_SECONDS", "1800"))
# 沙箱死亡重建上限：每轮重建（起沙+prep+seed+续采）成本约 10-15 分钟，
# 2 轮足够吃掉偶发死亡；连续死说明集群/镜像系统性故障，再试只是烧队列。
SANDBOX_REBUILD_MAX = int(_env("SANDBOX_REBUILD_MAX", "2"))
# 采集脚本后台运行时前台轮询间隔：给 rocklet 一个持续活着的 stdout，
# 同时把进展射进 Nebula job 日志（单片 ~23h，没这个就是盲跑）。
RUN_POLL_SECONDS = int(_env("RUN_POLL_SECONDS", "300"))

# --- 沙箱内路径 -------------------------------------------------------------------
SANDBOX_REPO = _env("SANDBOX_REPO", "/opt/derail")
SANDBOX_VENV_PY = _env("SANDBOX_VENV_PY", "/opt/derail-venv/bin/python")
SANDBOX_VM_DIR = _env("SANDBOX_VM_DIR", "/storage/mypcbench-vm")
SANDBOX_RESULTS = _env("SANDBOX_RESULTS", "/storage/results")
SECRETS_REMOTE_PATH = _env("SECRETS_REMOTE_PATH", "/tmp/.derail_secrets")
OSS_CONFIG_REMOTE_PATH = _env("OSS_CONFIG_REMOTE_PATH", "/tmp/.derail_ossutilconfig")
MYPCBENCH_COMMIT = _env("MYPCBENCH_COMMIT", "caf9c754ffe0b41c7e629e17fff19299774af1cb")

# --- OSS -------------------------------------------------------------------------
# bucket 与个人前缀不入库：跑 driver 前必须 export OSS_ASSETS_URI /
# OSS_HARNESS_URI / OSS_RESULTS_ROOT（submit 脚本会从 OSS_BUCKET/OSS_PREFIX 拼出）。
OSS_ACCESS_ID = _env("OSS_ACCESS_ID", "")
OSS_ACCESS_KEY = _env("OSS_ACCESS_KEY", "")
OSS_ENDPOINT = _env("OSS_ENDPOINT", "<oss-endpoint>")
OSS_ASSETS_URI = _env("OSS_ASSETS_URI", "<oss-assets-uri>")
OSS_QCOW2_URI = _env("OSS_QCOW2_URI", f"{OSS_ASSETS_URI}/mypcbench.qcow2")
OSS_HARNESS_URI = _env("OSS_HARNESS_URI", "<oss-harness-uri>")
OSS_RESULTS_ROOT = _env("OSS_RESULTS_ROOT", "<oss-results-root>")
# OpenCUA-OSWorld 冻结快照（含 .git）：Nebula 的 github egress 未验，
# factory 的 adapter source 优先从 OSS FUSE 恢复（submit 脚本预置）。
OSS_OPENCUA_SNAPSHOT_URI = _env(
    "OSS_OPENCUA_SNAPSHOT_URI",
    "oss://<oss-bucket>/<oss-prefix>/DERAIL/assets/opencua-osworld/"
    "opencua-osworld-091f5ef.tar.gz",
)
OSS_EVOCUA_SNAPSHOT_URI = _env(
    "OSS_EVOCUA_SNAPSHOT_URI",
    "oss://<oss-bucket>/<oss-prefix>/DERAIL/assets/evocua/"
    "evocua-4a0ad5f.tar.gz",
)
EVOCUA_COMMIT = _env(
    "EVOCUA_COMMIT", "4a0ad5fd4eb1d5b65966e1c7cc3feaa3b534eadd"
)
OPENCUA_OSWORLD_COMMIT = _env(
    "OPENCUA_OSWORLD_COMMIT", "091f5ef1d5544bc74953c77875d5feb5bed30108"
)
OSS_RESULTS_URI = f"{OSS_RESULTS_ROOT}/{COLLECTION_ID}"
# configs/models.lock.yaml 锁定的解压后 qcow2 sha256；不一致只告警不阻塞
# （计划 Phase 1：记录实际值进 manifest，人工确认差异原因）。
LOCK_QCOW2_SHA256 = _env(
    "LOCK_QCOW2_SHA256",
    "29e6ef0230655501920ad7b08c38e16c0d39e94c5be8220de2cf723885759e90",
)

# provider 凭证（closed model）：经 0600 secrets 文件进沙箱，不上 argv。
# 两个家族共用同一把 routify key，但 SDK 读不同的环境变量名。
PROVIDER_SECRET_VARS = (
    "OPENAI_API_KEY",
    "OPENAI_BASE_URL",
    "ANTHROPIC_API_KEY",
    "ANTHROPIC_BASE_URL",
)

DRY_RUN = _env("DRY_RUN", "0") == "1"

# 沙箱内 agent venv 的依赖：harness 本体只需 openai+requests，DERAIL 额外要
# Pillow/backoff；claude_* 走 Anthropic SDK（版本下限同 harness requirements）。
# opencua 的上游 agent loop（mm_agents/opencua/opencua_agent.py）自己 import
# httpx/loguru/backoff，其余同 harness 基线。
# PyYAML：harness bootstrap 经 derail agent_config 加载采集 config 时 import yaml，
# 2026-08-15 沙箱镜像漂移后系统 python 不再自带，两次重采因此假收官，必须显式装。
AGENT_PIP_DEPS = "openai requests Pillow backoff PyYAML" + (
    " 'anthropic>=0.39.0'" if IS_ANTHROPIC_AGENT else ""
) + (
    " httpx loguru" if IS_OPENCUA_AGENT else ""
)


def _anthropic_import() -> str:
    """依赖自检的 import 片段（claude_* 才把 anthropic 算进必需）。"""
    return ", anthropic" if IS_ANTHROPIC_AGENT else ""


def _closed_model_problems() -> list[str]:
    """API agent 经沙箱内 agent loop 直连网关：key 必须有、集群必须有 egress。"""
    if AGENT_ID == "dummy":
        return []
    problems = []
    if IS_OPENCUA_AGENT:
        if not OPENCUA_BASE_URLS:
            problems.append(
                "OPENCUA_BASE_URLS unset -- opencua_* 的 endpoint 由 Nebula job 内 "
                "vLLM 提供，entry（entry_derail_opencua_nebula.sh）起服务后导出；"
                "不能直接用 submit_derail_rock_nebula.sh 提交 opencua"
            )
        elif IS_PROXY_TOPOLOGY and "127.0.0.1" not in OPENCUA_BASE_URLS \
                and "localhost" not in OPENCUA_BASE_URLS:
            problems.append(
                f"proxy 拓扑下 OPENCUA_BASE_URLS 必须是 localhost vLLM（当前 "
                f"{OPENCUA_BASE_URLS}）；沙箱回连 Nebula pod IP 不可达（实证）"
            )
    elif IS_ANTHROPIC_AGENT:
        if not _env("ANTHROPIC_API_KEY", "") or not _env("ANTHROPIC_BASE_URL", ""):
            problems.append(
                "ANTHROPIC_API_KEY/ANTHROPIC_BASE_URL unset -- claude_* agent 靠 "
                "Anthropic SDK 从 env 读网关地址（submit 脚本的 mount-secret 通道）"
            )
    elif AGENT_ID in OPENAI_AGENTS:
        if not _env("OPENAI_API_KEY", "") or not _env("OPENAI_BASE_URL", ""):
            problems.append(
                "OPENAI_API_KEY/OPENAI_BASE_URL unset -- 沙箱内 agent loop 需要经 "
                "0600 secrets 拿到 routify 凭证（submit 脚本的 mount-secret 通道）"
            )
    else:
        problems.append(
            f"AGENT_ID={AGENT_ID} 未适配 ROCK 流水线；支持：dummy / "
            f"{'/'.join(OPENAI_AGENTS)} / {'/'.join(ANTHROPIC_AGENTS)} / "
            f"{'/'.join(OPENCUA_AGENTS)}（开源模型须经 entry_derail_opencua_nebula.sh）"
        )
    if ROCK_CLUSTER.startswith("<"):
        problems.append(
            "ROCK_CLUSTER 未设置：须指向一个有公网 egress、/dev/kvm 且能拉取 ROCK_SANDBOX_IMAGE 的集群"
        )
    return problems


def _secrets_env_body() -> str:
    """沙箱内 source 的 0600 snippet：OSS AK + routify 凭证（永不上 argv/日志）。"""
    lines = ["# generated by derail_rock_driver.py; contains credentials"]
    if OSS_ACCESS_ID:
        lines.append(f"export OSS_ACCESS_ID={shlex.quote(OSS_ACCESS_ID)}")
    if OSS_ACCESS_KEY:
        lines.append(f"export OSS_ACCESS_KEY={shlex.quote(OSS_ACCESS_KEY)}")
    lines.append(f"export OSS_ENDPOINT={shlex.quote(OSS_ENDPOINT)}")
    for name in PROVIDER_SECRET_VARS:
        value = _env(name, "")
        if value:
            lines.append(f"export {name}={shlex.quote(value)}")
    return "\n".join(lines) + "\n"


async def _stage_sandbox_secrets(sandbox) -> None:
    """Transfer credentials without putting their values in ROCK action history."""
    secrets_body = _secrets_env_body()
    if not any(line.startswith("export ") for line in secrets_body.splitlines()):
        raise RuntimeError("no credentials to stage (OSS creds empty)")
    with tempfile.TemporaryDirectory(prefix="derail-secret-") as directory:
        source = Path(directory) / "credentials.env"
        source.write_text(secrets_body, encoding="utf-8")
        source.chmod(0o600)
        oss_config = Path(directory) / "ossutilconfig"
        oss_config.write_text(
            "[Credentials]\n"
            "language=EN\n"
            f"endpoint={OSS_ENDPOINT}\n"
            f"accessKeyID={OSS_ACCESS_ID}\n"
            f"accessKeySecret={OSS_ACCESS_KEY}\n",
            encoding="utf-8",
        )
        oss_config.chmod(0o600)
        await sandbox.fs.upload_dir(
            source_dir=directory,
            target_dir="/tmp/derail-secret-stage",
            extract_timeout=120,
        )
    await sandbox.arun(
        cmd=(
            "install -m 600 /tmp/derail-secret-stage/credentials.env "
            f"{shlex.quote(SECRETS_REMOTE_PATH)} && "
            "install -m 600 /tmp/derail-secret-stage/ossutilconfig "
            f"{shlex.quote(OSS_CONFIG_REMOTE_PATH)} && rm -rf /tmp/derail-secret-stage"
        ),
        session="default",
        wait_timeout=120,
    )
    print(f"[driver] staged credentials -> {SECRETS_REMOTE_PATH} (0600 file transfer)")


def _ossutil(prefix: str = "") -> str:
    """Sandbox ossutil command whose credential values never enter process argv."""
    return f"{prefix}ossutil -c {shlex.quote(OSS_CONFIG_REMOTE_PATH)}"


def _prep_command() -> str:
    """沙箱内一次性环境准备（幂等；重跑/重试安全）。

    四件事：harness 源码（OSS tar，含 .git，运行时零 github）、qcow2 + OVMF
    （OSS 资产，sha256 对照 lock 只告警）、python venv + agent 依赖、
    防御性删除可能混进来的 repo 根 .env。
    """
    qcow2 = f"{SANDBOX_VM_DIR}/mypcbench.qcow2"
    return f"""
set -Eeuo pipefail
# nohup 执行环境可能完全没有 PATH（rocklet 精简环境）：bash 靠编译默认
# PATH 仍能找到命令，但 Python getpath 依赖 environ['PATH'] 解析
# _base_executable；PATH 缺失时 `python3.12 -m venv` 报 gh-96861
# "Unable to determine path to the running Python interpreter"（已实证）。
export PATH="/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin${{PATH:+:$PATH}}"
set -a; . {shlex.quote(SECRETS_REMOTE_PATH)}; set +a
step() {{ echo "[prep $(date +%H:%M:%S)] $*"; }}

# upload_dir 与 tar 解压的文件属主可能不同于当前用户，git 会拒绝一切操作
# （"detected dubious ownership"，rev-parse 输出空）；沙箱是单次性环境，直接放开。
# 用 --system（不写 ~/.gitconfig）：arun 的 shell 里 $HOME 可能未设置；
# HOME 兜底同时供 01 脚本内的 git 调用使用。
export HOME="${{HOME:-/root}}"
git config --system --add safe.directory '*'

# 1. MyPCBench harness（pinned {MYPCBENCH_COMMIT}）：upload_dir 的代码包不含
#    third_party，从 OSS tar 恢复（含 .git，01 脚本要 rev-parse + apply patch）。
if [[ -d {SANDBOX_REPO}/third_party/MyPCBench/.git ]] && \\
   [[ "$(git -C {SANDBOX_REPO}/third_party/MyPCBench rev-parse HEAD 2>/dev/null)" == "{MYPCBENCH_COMMIT}" ]]; then
  step "harness already at {MYPCBENCH_COMMIT}"
else
  step "restoring harness from {OSS_HARNESS_URI}"
  mkdir -p {SANDBOX_REPO}/third_party
  rm -rf {SANDBOX_REPO}/third_party/MyPCBench
  {_ossutil()} cp -f {shlex.quote(OSS_HARNESS_URI)} /tmp/mypcbench-harness.tar.gz
  tar -xzf /tmp/mypcbench-harness.tar.gz -C {SANDBOX_REPO}/third_party
  rm -f /tmp/mypcbench-harness.tar.gz
  [[ "$(git -C {SANDBOX_REPO}/third_party/MyPCBench rev-parse HEAD)" == "{MYPCBENCH_COMMIT}" ]] \\
    || {{ echo "[prep] ERROR: harness commit mismatch after restore" >&2; exit 1; }}
fi

# 2. qcow2 + OVMF（OSS 资产；01 脚本经 MYPCBENCH_QCOW2/OVMF_* 指过来）。
mkdir -p {shlex.quote(SANDBOX_VM_DIR)}
if [[ -f {qcow2} ]]; then
  step "qcow2 already staged"
else
  step "pulling mypcbench.qcow2 from OSS (~10G)"
  {_ossutil()} cp -f {shlex.quote(OSS_QCOW2_URI)} {qcow2}
fi
step "qcow2 sha256 (lock expects {LOCK_QCOW2_SHA256}):"
_actual_sha="$(sha256sum {qcow2} | awk '{{print $1}}')"
echo "[prep] actual: $_actual_sha"
[[ "$_actual_sha" == "{LOCK_QCOW2_SHA256}" ]] \\
  || echo "[prep] WARNING: qcow2 sha256 differs from lock; continuing (recorded in collection manifest)"
for f in OVMF_CODE.fd OVMF_VARS.fd; do
  [[ -f {SANDBOX_VM_DIR}/$f ]] || \\
    {_ossutil()} cp -f {shlex.quote(OSS_ASSETS_URI)}/$f {SANDBOX_VM_DIR}/$f
done
step "OVMF staged: $(ls {SANDBOX_VM_DIR})"

# 3. agent venv（openai/requests/Pillow/backoff [+anthropic]）。ROCK 沙箱有
#    公网 egress 但内网镜像源不可达（实测连接超时 5次重试 ~75s）：
#    PyPI 主、环境变量 PIP_INDEX_URL 指定的镜像源兜底。claude_* agent 额外需要 anthropic SDK。
if [[ -x {shlex.quote(SANDBOX_VENV_PY)} ]] && \\
   {shlex.quote(SANDBOX_VENV_PY)} -c 'import openai, requests, PIL, backoff, yaml{_anthropic_import()}' 2>/dev/null; then
  step "venv already present"
else
  step "creating venv with python3.12"
  [[ -x {shlex.quote(SANDBOX_VENV_PY)} ]] || python3.12 -m venv /opt/derail-venv
  {shlex.quote(SANDBOX_VENV_PY)} -m pip install -q --upgrade pip
  _pip_fallback=""; [ -n "${{PIP_INDEX_URL:-}}" ] && _pip_fallback="-i ${{PIP_INDEX_URL}}"
  {shlex.quote(SANDBOX_VENV_PY)} -m pip install -q {AGENT_PIP_DEPS} \\
    || {shlex.quote(SANDBOX_VENV_PY)} -m pip install -q {AGENT_PIP_DEPS} $_pip_fallback
fi
{shlex.quote(SANDBOX_VENV_PY)} -c 'import openai, requests, PIL, backoff, yaml{_anthropic_import()}; print("[prep] agent deps OK")'

# 4. 防御：repo 根 .env 永不进沙箱（routify key 只走 0600 secrets 文件）。
rm -f {SANDBOX_REPO}/.env
step "PREP_DONE"
""".strip()


def _repeat_dirs() -> list[str]:
    """本次提交会写的 repeat_<k> 目录（01 从 REPEAT_START_INDEX 起连跑 REPEATS 轮）。"""
    start = int(REPEAT_START_INDEX)
    return [f"repeat_{k}" for k in range(start, start + int(REPEATS))]


def _seed_command() -> str:
    """断点续采种子回灌：OSS 部分结果 → 沙箱 /storage/results（幂等）。

    三步：整目录回灌（同 COLLECTION_ID，命中 harness 的 result.txt 跳过
    守卫）；删除无 result.txt 的任务目录（截断残留，会被重跑）；把上一轮
    _collect.log 改名留档，避免与本轮日志混淆。末尾打点 SEED_DONE + 已完成
    任务数，供 driver 侧判成功。RERUN_PURGE=1 时追加第四步：强制删除
    SHARD_FILE 清单内任务的回灌目录（见模块头 RERUN_PURGE 注释）。
    """
    seed_dir = f"{SANDBOX_RESULTS}/{COLLECTION_ID}"
    if RERUN_PURGE:
        ids = _shard_task_ids()
        marker = f"{seed_dir}/.derail_rerun_purged_{RERUN_PURGE_NONCE}"
        # 回灌前先把 OSS 上清单任务的旧目录删掉：否则旧截图被回灌进沙箱
        # 后虽会被下面的沙箱清场删掉，但终局/增量 ship（cp -r -f 只增不删）
        # 永远无法清除 OSS 上已存在的旧文件，新旧截图混存污染验收。
        oss_purge_paths = ' '.join(
            shlex.quote(p)
            for t in ids
            for r in _repeat_dirs()
            for p in (
                f"{COLLECTION_ID}/{AGENT_ID}/{r}/vm0/{t}/",
                f"{AGENT_ID}/{r}/vm0/{t}/",
            )
        )
        purge_block = f"""
# RERUN_PURGE 第四步：清场 rerun 清单（有 result.txt 的旧轨迹，守卫会跳过）。
# nonce 标记在 => 沙箱死亡重建轮，第 1 轮新轨迹受守卫保护，跳过清场。
if [[ -f {shlex.quote(marker)} ]]; then
  echo "[seed] rerun purge marker present; skipping purge (rebuild round)"
else
  # 4a. OSS 侧清场（两种历史目录布局都删；不存在的路径 rm 静默跳过）
  for _p in {oss_purge_paths}; do
    {_ossutil()} rm -r -f "{OSS_RESULTS_URI}/$_p" >/dev/null 2>&1 || true
  done
  echo "[seed] rerun oss purge done: {len(ids)} task ids"
  # 4b. 沙箱侧清场（回灌后的旧目录）
  rm -f {shlex.quote(seed_dir)}/{RERUN_PURGE_MARKER_GLOB} 2>/dev/null || true
  for _t in {' '.join(shlex.quote(t) for t in ids)}; do
    find {shlex.quote(seed_dir)} -mindepth 4 -maxdepth 4 -type d -name "$_t" -print0 | \
      xargs -0 -r rm -rf
  done
  touch {shlex.quote(marker)}
  echo "[seed] rerun purge applied: {len(ids)} task ids (nonce={RERUN_PURGE_NONCE})"
fi
"""
    else:
        purge_block = ""
    # fresh 起跑（OSS 无种子）也要落 marker：否则沙箱死亡重建轮 seed 见到
    # 第 1 轮抢救产物却无 marker，会误清已跑出的新轨迹。
    fresh_marker = (
        f"mkdir -p {shlex.quote(seed_dir)} && "
        f"touch {shlex.quote(seed_dir)}/.derail_rerun_purged_{RERUN_PURGE_NONCE} && "
        if RERUN_PURGE else ""
    )
    return f"""
set -Eeuo pipefail
export PATH="/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin${{PATH:+:$PATH}}"
set -a; . {shlex.quote(SECRETS_REMOTE_PATH)}; set +a
mkdir -p {shlex.quote(SANDBOX_RESULTS)}
echo "[seed] pulling partial results from {OSS_RESULTS_URI}/"
{_ossutil()} cp -r -f {shlex.quote(OSS_RESULTS_URI)}/ {shlex.quote(SANDBOX_RESULTS)}/
if [[ ! -d {shlex.quote(seed_dir)} ]]; then
  # OSS 无部分产物（如沙箱死在首个增量抢救周期前）：无物可护，fresh 起跑
  # 语义无损。driver 凭 SEED_NOTHING 标记豁免守卫（2026-08-16 opus_s3 七提实证）。
  echo "[seed] no partial results on OSS; fresh run is lossless"
  {fresh_marker}echo SEED_NOTHING
  echo SEED_DONE
  exit 0
fi
for f in {shlex.quote(seed_dir)}/_collect.log; do
  if [[ -f "$f" ]]; then mv "$f" "${{f}}.round1"; fi
done
# 任务目录在 COLLECTION_ID 根下第 4 层（<agent>/repeat_<k>/vm0/<task_id>）；
# 无 result.txt 的一律删除（截断残留，harness 会重跑）。
find {shlex.quote(seed_dir)} -mindepth 4 -maxdepth 4 -type d ! -name '_tasks' -print0 | \
  while IFS= read -r -d '' d; do
    [[ -f "$d/result.txt" ]] || rm -rf "$d"
  done
{purge_block}
echo "[seed] completed tasks seeded: $(find {shlex.quote(seed_dir)} -name result.txt | wc -l)"
echo SEED_DONE
"""


def _run_command() -> str:
    """01_collect_trajectories.sh 启动脚本体（写文件后单行 bash 调用）。

    三个实证约束（2026-08-06 两次 smoke 换来的）：
    (a) nohup 模式下含 `;` 的复合命令会被截断作用范围，rocklet 的临时输出
        文件建不起来（报 "cat /tmp/tmp_*.out: No such file"）→ 先写脚本文件；
    (b) 顶层 stdout 必须留给 rocklet：脚本里 `exec ... > log` 把 stdout 换掉
        后，rocklet 立刻判定命令结束并返回空输出，driver 会在 QEMU 还在跑时
        就上传结果 + stop 沙箱（gpt-5.5 smoke#2 实证：采集只跑了 6 步）→
        harness 放后台自己重定向，前台留一个轮询循环把 stdout 保持活着；
    (c) 日志用纯重定向不用 tee 管道（MCUA 2026-07-30 教训）。
    """
    env_exports = {
        # arun shell 里 $HOME 可能未设置（prep 阶段已实证）；01 脚本与 harness
        # 内的 git/python 调用都需要一个可写的 HOME。
        "HOME": "/root",
        # nohup 环境可能无 PATH（见 _prep_command 注释）；01 脚本内的 git、
        # qemu-img、harness subprocess 都依赖正常 PATH。
        "PATH": "/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin",
        "DERAIL_TMUX": "0",
        "REPEATS": REPEATS,
        "REPEAT_START_INDEX": REPEAT_START_INDEX,
        "NUM_VMS_OVERRIDE": NUM_VMS_OVERRIDE,
        "BACKEND": "qemu",
        "MAX_STEPS": MAX_STEPS,
        "DERAIL_BASH_ACCOUNTING": DERAIL_BASH_ACCOUNTING,
        "TASK_TIMEOUT": TASK_TIMEOUT,
        "TIMEOUT_PER_VM": TIMEOUT_PER_VM,
        # 留空 = 01 取采集 config；与 config 不同的显式值要 ALLOW_CONFIG_OVERRIDE=1 才生效。
        "TASK_SOURCE": _env("TASK_SOURCE", ""),
        "ALLOW_CONFIG_OVERRIDE": _env("ALLOW_CONFIG_OVERRIDE", "0"),
        "TASKS_FILE": f"{SANDBOX_REPO}/{SHARD_FILE}",
        "OUTPUT_ROOT": SANDBOX_RESULTS,
        "COLLECTION_ID": COLLECTION_ID,
        # Passed from the live ROCK driver so each takeover_launch.json can
        # be joined to an authoritative sandbox lifecycle record.
        "ROCK_SANDBOX_ID": os.environ.get("ROCK_SANDBOX_ID", ""),
        "MYPCBENCH_QCOW2": f"{SANDBOX_VM_DIR}/mypcbench.qcow2",
        "MYPCBENCH_OVMF_CODE": f"{SANDBOX_VM_DIR}/OVMF_CODE.fd",
        "MYPCBENCH_OVMF_VARS": f"{SANDBOX_VM_DIR}/OVMF_VARS.fd",
        "PYTHON_BIN": SANDBOX_VENV_PY,
        "GPT55_MODEL": GPT55_MODEL,
        "GPT56_LUNA_MODEL": GPT56_LUNA_MODEL,
        "CLAUDE_OPUS_4_8_MODEL": CLAUDE_OPUS_4_8_MODEL,
        "CLAUDE_SONNET_5_MODEL": CLAUDE_SONNET_5_MODEL,
        "KIMI_K3_MODEL": KIMI_K3_MODEL,
        # 开源模型 endpoint（job 内 vLLM）：01 脚本在 DERAIL_AUTO_SERVE=0 下
        # 从 OPENCUA_BASE_URLS 读地址；模型名必须与 vLLM --served-model-name 一致。
        "OPENCUA_BASE_URLS": OPENCUA_BASE_URLS,
        "OPENCUA_MODEL": OPENCUA_MODEL,
        "QWEN38_BASE_URLS": OPENCUA_BASE_URLS,
        "QWEN35_BASE_URLS": OPENCUA_BASE_URLS,
        "EVOCUA_BASE_URLS": OPENCUA_BASE_URLS,
        "QWEN38_MODEL": QWEN38_MODEL,
        "QWEN35_MODEL": QWEN35_MODEL,
        "EVOCUA_MODEL": EVOCUA_MODEL,
        "KIMI_K3_MODEL": KIMI_K3_MODEL,
        "CLAUDE_PROMPT_CACHING_BETA": CLAUDE_PROMPT_CACHING_BETA,
        "OPENAI_RATE_LIMIT_RETRIES": OPENAI_RATE_LIMIT_RETRIES,
        "ANTHROPIC_RATE_LIMIT_RETRIES": ANTHROPIC_RATE_LIMIT_RETRIES,
        "MYPCBENCH_OPENAI_REASONING_EFFORT": MYPCBENCH_OPENAI_REASONING_EFFORT,
        "MYPCBENCH_OPENAI_EMPTY_OUTPUT_RETRIES": MYPCBENCH_OPENAI_EMPTY_OUTPUT_RETRIES,
        "MYPCBENCH_SCREENSHOT_RETRIES": MYPCBENCH_SCREENSHOT_RETRIES,
        # routify 组织开了 Zero Data Retention，拒绝 previous_response_id（实测
        # 2026-08-06 step2 起全 400）；走 harness patch 的无状态回放模式。
        "OPENAI_ZDR_STATELESS": _env("OPENAI_ZDR_STATELESS", "1"),
        # 回放保留的截图张数 = 采集 config 的 context_images（论文 K=20）。上游默认
        # 只留 4 张，v1 的 GPT-5.5 就是 4 张口径。
        "OPENAI_ZDR_KEEP_IMAGES": _env(
            "OPENAI_ZDR_KEEP_IMAGES", _runtime_default("context_images")
        ),
        "FORMAL_COLLECTION": FORMAL_COLLECTION,
        "DERAIL_OPENAI_API_APPROVED": DERAIL_OPENAI_API_APPROVED,
        "DERAIL_OPENAI_API_PURPOSE": DERAIL_OPENAI_API_PURPOSE,
        "DERAIL_ANTHROPIC_API_APPROVED": DERAIL_ANTHROPIC_API_APPROVED,
        "DERAIL_ANTHROPIC_API_PURPOSE": DERAIL_ANTHROPIC_API_PURPOSE,
        # 本地开源模型的 auto-serve 在沙箱里无意义（无 GPU），gpt_5_5/dummy 用不到，
        # 显式关掉防止 01 脚本去碰 tmux serving 脚本。
        "DERAIL_AUTO_SERVE": "0",
    }
    exports = "\n".join(
        f"export {k}={shlex.quote(v)}" for k, v in env_exports.items()
    )
    log = f"{SANDBOX_RESULTS}/_collect.log"
    return f"""
set -Eeuo pipefail
set -a; . {shlex.quote(SECRETS_REMOTE_PATH)} 2>/dev/null || true; set +a
{exports}
# _collect.log 的重定向发生在 01 脚本自己 mkdir OUTPUT_ROOT 之前。
mkdir -p {shlex.quote(SANDBOX_RESULTS)}
cd {shlex.quote(SANDBOX_REPO)}

# harness 进后台，自己写 _collect.log；本脚本的 stdout 留给 rocklet（见 docstring b）。
bash scripts/01_collect_trajectories.sh {shlex.quote(AGENT_ID)} > {log} 2>&1 &
COLLECT_PID=$!
echo "[run] collection started (pid $COLLECT_PID); polling every {RUN_POLL_SECONDS}s"
while kill -0 "$COLLECT_PID" 2>/dev/null; do
  sleep {RUN_POLL_SECONDS}
  echo "[run $(date +%H:%M:%S)] alive; tail:"
  tail -3 {log} 2>/dev/null || true
done
wait "$COLLECT_PID" || COLLECT_RC=$?
echo "[run] collection exited rc=${{COLLECT_RC:-0}}"
tail -5 {log} 2>/dev/null || true
exit "${{COLLECT_RC:-0}}"
""".strip()


def _result_command() -> str:
    """结果目录 + 采集日志上 OSS（plain mode；nohup 的临时文件会提前消失）。"""
    return (
        f"set -a; . {shlex.quote(SECRETS_REMOTE_PATH)} 2>/dev/null || true; set +a; "
        f"{_ossutil()} cp -r -f {shlex.quote(SANDBOX_RESULTS)}/ "
        f"{shlex.quote(OSS_RESULTS_URI)}/ "
        f"|| echo '[driver] ossutil upload failed'"
    )


def _sandbox_config():
    """8C/16G/100G 固定规格：guest QEMU 要 -m 8G -smp 4，无法降配探 KVM。"""
    from rock.sdk.sandbox.config import SandboxConfig

    kwargs: dict = dict(
        base_url=ROCK_BASE_URL,
        image=ROCK_SANDBOX_IMAGE,
        auto_clear_seconds=ROCK_AUTO_CLEAR_SECONDS,
        startup_timeout=ROCK_STARTUP_TIMEOUT,
        memory=ROCK_MEMORY,
        cpus=ROCK_CPUS,
        disk=ROCK_DISK,
        cluster=ROCK_CLUSTER,
    )
    if ROCK_API_KEY:
        kwargs["extra_headers"] = {"XRL-Authorization": f"Bearer {ROCK_API_KEY}"}
        kwargs["user_id"] = ROCK_USER_ID
        kwargs["experiment_id"] = ROCK_EXPERIMENT_ID
    return SandboxConfig(**kwargs)


def _sandbox_dead_marker(exc: BaseException) -> bool:
    """沙箱已死的异常签名（2026-08-15 两 job 实证）：rocklet 对死沙箱的
    一切命令回 'sandbox <id> not started'；区别于 arun wait_timeout 等其他
    失败（那些不该触发重建）。"""
    return "not started" in str(exc)


async def _sandbox_alive(sandbox) -> bool:
    """廉价死活探针：echo 都跑不动即已死。"""
    try:
        await sandbox.arun(cmd="echo DRIVER_PROBE_OK", session="default", wait_timeout=60)
        return True
    except Exception:  # noqa: BLE001 -- 探针失败即视为死
        return False


def _transient_start_error(exc: BaseException) -> bool:
    text = f"{type(exc).__name__}: {exc}".lower()
    markers = (
        "429", "failed to start sandbox", "readerror", "connecterror",
        "remoteprotocolerror", "timeout", "temporarily unavailable",
        "connection reset", "server disconnected", "502", "503", "504",
    )
    return any(marker in text for marker in markers)


async def _start_sandbox_with_retry(sandbox, Sandbox):
    attempts = int(_env("ROCK_START_RETRIES", "8"))
    for attempt in range(1, attempts + 1):
        try:
            await asyncio.wait_for(sandbox.start(), ROCK_STARTUP_TIMEOUT + 30)
            return sandbox
        except Exception as exc:
            if not _transient_start_error(exc) or attempt == attempts:
                raise
            delay = min(30 * (2 ** (attempt - 1)), 300)
            print(f"[driver] transient sandbox start error ({type(exc).__name__}); "
                  f"attempt {attempt}/{attempts}, retrying in {delay}s")
            try:
                await sandbox.stop()
            except Exception as stop_exc:
                print(f"[driver] half-start stop ignored: {stop_exc}", file=sys.stderr)
            sandbox = Sandbox(_sandbox_config())
            await asyncio.sleep(delay)


async def _setup_sandbox(sandbox, CreateBashSessionRequest):
    """起沙（429 退避）+ KVM 探针回收 + repo/secrets/prep/seed 全流程。

    从原 _run 内联逻辑抽出，供首轮与沙箱死亡重建复用；除 KVM 探针外全部
    步骤幂等（prep/seed 自带 already-staged 守卫）。KVM 缺失时回收重建
    内层自消化，仍由调用方拿到活沙箱。
    """
    from rock.sdk.sandbox.client import Sandbox  # KVM 回收内层要新建沙箱对象
    print(f"[driver] starting sandbox (image={ROCK_SANDBOX_IMAGE} cluster={ROCK_CLUSTER} "
          f"shape={ROCK_CPUS}C/{ROCK_MEMORY}/{ROCK_DISK})")
    # ROCK 起沙失败退避重试：429 限流 + 起沙超时（InternalServerRockError
    # "Failed to start sandbox within Ns"，2026-08-15 opus_s3 六提实证：
    # 起沙 600s 超时直接抛，旧代码只吃 429 → 整 job 起不了步即 FAILED）。
    # 两者都是瞬态平台事件，换一次调度就可能落到健康节点。
    sandbox = await _start_sandbox_with_retry(sandbox, Sandbox)

    await sandbox.create_session(CreateBashSessionRequest(session="default"))

    # KVM 探针回收循环：/dev/kvm 是节点级随机属性；guest 必须 KVM 才能起。
    # 不降配（guest -m 8G -smp 4 需要完整 8C/16G），每次探针只要几秒，多试。
    kvm_attempts = int(_env("ROCK_KVM_RETRIES", "14"))
    for kvm_attempt in range(1, kvm_attempts + 1):
        probe = await sandbox.arun(
            # "test -e" 在沙箱 shell 下误报（MCUA 四集群验证）；ls 可靠。
            cmd="ls /dev/kvm 2>/dev/null && echo KVM_OK || echo KVM_MISSING",
            session="default",
        )
        if "KVM_OK" in str(getattr(probe, "output", probe) or ""):
            print(f"[driver] /dev/kvm present (attempt {kvm_attempt})")
            break
        print(f"[driver] /dev/kvm MISSING (attempt {kvm_attempt}/{kvm_attempts}) "
              "-- recycling sandbox")
        if kvm_attempt == kvm_attempts:
            raise RuntimeError(
                f"no /dev/kvm after {kvm_attempts} attempts; guest cannot boot"
            )
        await sandbox.stop()
        await asyncio.sleep(20)
        sandbox = await _start_sandbox_with_retry(
            Sandbox(_sandbox_config()), Sandbox
        )
        await sandbox.create_session(CreateBashSessionRequest(session="default"))

    print(f"[driver] uploading repo -> {SANDBOX_REPO}")
    await sandbox.fs.upload_dir(source_dir=str(REPO), target_dir=SANDBOX_REPO)

    # 0600 secrets（OSS AK + routify key；永不上 argv / action history）。
    await _stage_sandbox_secrets(sandbox)

    print("[driver] prep: harness/qcow2/OVMF/venv")
    # 两个约束：(a) 普通模式受 rocklet 单次请求 85s 超时限制（qcow2 拉取
    # ~90s 必撞）→ 用 nohup 模式；(b) nohup 模式下多行脚本的末尾重定向只
    # 作用于最后一行 → 先写脚本文件（quoted heredoc 防展开），再单行调用。
    await sandbox.arun(
        cmd=(
            "cat > /tmp/derail_prep.sh <<'DERAILPREPEOF'\n"
            f"{_prep_command()}\nDERAILPREPEOF\n"
        ),
        session="default",
    )
    prep = await sandbox.arun(
        cmd="bash /tmp/derail_prep.sh > /tmp/prep.log 2>&1",
        session="default",
        mode="nohup",
        wait_timeout=3600,
    )
    print(f"[driver] prep output tail:\n{prep.output}")
    # 普通模式 arun 偶发返回空输出（实测 2026-08-06：nohup 输出含
    # PREP_DONE 但后续 check 为空 → 误判失败）：判定同时看两处输出，
    # check 步骤自身也重试几次。
    prep_ok = "PREP_DONE" in str(getattr(prep, "output", "") or "")
    for check_attempt in range(1, 4):
        prep_check = await sandbox.arun(
            cmd="grep -c PREP_DONE /tmp/prep.log 2>/dev/null; tail -30 /tmp/prep.log",
            session="default",
            wait_timeout=120,
        )
        check_out = str(getattr(prep_check, "output", "") or "")
        print(f"[driver] prep log tail (attempt {check_attempt}):\n{check_out}")
        if check_out.strip():
            prep_ok = prep_ok or "PREP_DONE" in check_out
            break
        await asyncio.sleep(10)
    if not prep_ok:
        raise RuntimeError("prep did not reach PREP_DONE; see prep log tail above")

    # 断点续采：在 run 之前回灌 OSS 部分结果。失败必须硬断：若带着空
    # 结果目录起跑，会把 46 任务全量重跑一遍，覆盖掉 OSS 上已有的部分
    # 结果（cp -r -f），比不跑更糟。
    if RESUME_SEED:
        await sandbox.arun(
            cmd=(
                "cat > /tmp/derail_seed.sh <<'DERAILSEEDEOF'\n"
                f"{_seed_command()}\nDERAILSEEDEOF\n"
            ),
            session="default",
        )
        print("[driver] resume seed staged; pulling partial results from OSS")
        # nohup 模式偶发不起跑（seed.log 恒 0 字节，一提事故与 2026-08-16
        # opus_s3 七提重建轮同款）：launch+轮询整体包重试，最多 3 轮。
        seed_launch_attempts = int(_env("SEED_LAUNCH_RETRIES", "3"))
        seed_ok = False
        seed_out = ""
        for launch_round in range(1, seed_launch_attempts + 1):
            seed = await sandbox.arun(
                cmd="bash /tmp/derail_seed.sh > /tmp/seed.log 2>&1",
                session="default",
                mode="nohup",
                wait_timeout=3600,
            )
            # 同 prep 的教训（2026-08-09 续采首提 8/8 全挂于此）：rocklet 偶发
            # 提前返回空输出，单次 check 撞上会把仍在回灌的 seed 误判失败。
            # 轮询 seed.log 等 SEED_DONE（回灌含数千截图，可能跑满几十分钟）。
            seed_ok = "SEED_DONE" in str(getattr(seed, "output", "") or "")
            seed_out = ""
            for seed_attempt in range(1, 37):
                if seed_ok:
                    break
                await asyncio.sleep(60)
                seed_check = await sandbox.arun(
                    cmd="grep -c SEED_DONE /tmp/seed.log 2>/dev/null; tail -20 /tmp/seed.log 2>/dev/null; ls -la /tmp/derail_seed.sh /tmp/seed.log 2>&1",
                    session="default",
                    wait_timeout=120,
                )
                seed_out = str(getattr(seed_check, "output", "") or "")
                print(f"[driver] seed log tail (round {launch_round}, attempt {seed_attempt}):\n{seed_out}")
                seed_ok = "SEED_DONE" in seed_out
            if seed_ok:
                break
            print(f"[driver] seed never reached SEED_DONE (round {launch_round}/"
                  f"{seed_launch_attempts}); relaunching")
        if not seed_ok:
            raise RuntimeError("resume seed did not reach SEED_DONE; refusing to run "
                               "(fresh run would overwrite OSS partial results)")
        if "SEED_NOTHING" in seed_out or "SEED_NOTHING" in str(getattr(seed, "output", "") or ""):
            print("[driver] SEED_NOTHING: OSS has no partial results; fresh run is lossless")

    # KVM 回收可能换过新沙箱对象，调用方必须用返回值。
    return sandbox

def _ship_increment_command() -> str:
    """沙箱内增量抢救上传循环脚本体：每 SHIP_INCREMENT_SECONDS 把结果目录
    抢上 OSS。沙箱中途死亡时（2026-08-15 两 job 实证），本地结果不可达，
    这是唯一保产物手段；重建后 RESUME_SEED 从 OSS 续采，最多损失一个周期。
    cp -r -f 与终局 _result_command 同形状（已验）；任何单轮失败不得杀循环。"""
    return f"""
set -a; . {shlex.quote(SECRETS_REMOTE_PATH)} 2>/dev/null || true; set +a
while true; do
  sleep {SHIP_INCREMENT_SECONDS}
  {_ossutil()} cp -r -f {shlex.quote(SANDBOX_RESULTS)}/ {shlex.quote(OSS_RESULTS_URI)}/ \\
    && echo "[ship-increment $(date +%H:%M:%S)] synced -> {OSS_RESULTS_URI}/" \\
    || echo "[ship-increment $(date +%H:%M:%S)] sync FAILED (will retry next round)"
done
""".strip()


async def _collect_and_ship(sandbox, run_cmd: str) -> tuple[Optional[BaseException], bool]:
    """在活沙箱内完成：staging run 脚本 + 起增量抢救上传 + 跑采集 + 诊断 +
    终局上传。返回 (run_error, sandbox_died)：sandbox_died=True 表示沙箱在
    途中死亡（应重建续采），此时诊断/终局上传已不可达，产物靠增量抢救保底。
    """
    # run 与 prep 同模式：先写脚本文件，nohup 只执行单行 bash 调用
    #（含 `;` 的复合命令在 nohup 模式下必挂，见 _run_command 注释）。
    await sandbox.arun(
        cmd=(
            "cat > /tmp/derail_run.sh <<'DERAILRUNEOF'\n"
            f"{run_cmd}\nDERAILRUNEOF\n"
        ),
        session="default",
    )
    # 增量抢救上传：与采集同生共死的后台循环（nohup 模式单行启动）。
    await sandbox.arun(
        cmd=(
            "cat > /tmp/derail_ship_increment.sh <<'DERAILSHIPEOF'\n"
            f"{_ship_increment_command()}\nDERAILSHIPEOF\n"
        ),
        session="default",
    )
    await sandbox.arun(
        cmd="nohup bash /tmp/derail_ship_increment.sh > /tmp/ship_increment.log 2>&1 &",
        session="default",
        wait_timeout=60,
    )
    print(f"[driver] incremental ship armed (every {SHIP_INCREMENT_SECONDS}s); "
          f"launching run (timeout={ROCK_RUN_TIMEOUT}s)")

    # 外层重定向照 prep 的已验证形状（见 _run_command docstring b）。
    #
    # 采集异常必须被吃住：一个分片要跑几小时到几十小时，若 arun 超时
    # 或报错直接向上抛，就会跳过下面的诊断 + 结果上传直捣 finally 停
    # 沙箱 —— 已完成的那些任务的轨迹/截图全部丢失。部分结果比异常
    # 传递重要得多，所以这里只记录并继续走完上传流程。
    run_error: Optional[BaseException] = None
    try:
        obs = await sandbox.arun(
            cmd="bash /tmp/derail_run.sh > /tmp/run_launch.log 2>&1",
            session="default",
            mode="nohup",
            wait_timeout=ROCK_RUN_TIMEOUT,
        )
        print(f"[driver] collect output tail:\n{obs.output}")
    except Exception as exc:  # noqa: BLE001 -- 故意吃住，保住部分结果
        run_error = exc
        print(f"[driver] ERROR: collection run failed/timed out: {exc}",
              file=sys.stderr)
        if _sandbox_dead_marker(exc) and not await _sandbox_alive(sandbox):
            # 沙箱死在采集途中：诊断/终局上传都不可达，增量抢救已保底，
            # 直接上报由 _run 重建续采（旧代码在这里继续 arun 二次炸掉整 job）。
            print("[driver] sandbox died mid-run; diagnostics/ship unreachable; "
                  "relying on incremental salvage", file=sys.stderr)
            return run_error, True
        print("[driver] 继续走诊断 + 结果上传，不丢已完成的任务", file=sys.stderr)

    # 沙箱即将销毁，先把能解释失败的诊断拉出来。
    for probe_name, cmd in (
        ("run launcher log", "tail -20 /tmp/run_launch.log 2>/dev/null || echo '<no run_launch.log>'"),
        ("collect log tail", f"tail -60 {SANDBOX_RESULTS}/_collect.log 2>/dev/null || echo '<no _collect.log>'"),
        ("result tree", f"find {SANDBOX_RESULTS} -maxdepth 4 2>/dev/null | head -40 || echo '<no results>'"),
        ("completed/error count", f"grep -rc . {SANDBOX_RESULTS}/{COLLECTION_ID}/*/repeat_*/ 2>/dev/null | head -5; ls {SANDBOX_RESULTS}/{COLLECTION_ID} 2>/dev/null || true"),
        ("qemu processes", "pgrep -af qemu-system | head -3 || echo '<no qemu running>'"),
    ):
        try:
            out = await sandbox.arun(cmd=cmd, session="default", wait_timeout=120)
            print(f"[driver] --- {probe_name} ---\n{out.output}")
        except Exception as exc:
            print(f"[driver] --- {probe_name} FAILED: {exc}", file=sys.stderr)

    print(f"[driver] ship results -> {OSS_RESULTS_URI}/")
    # 同样避开 85s 单次请求超时（正式分片结果数 GB）+ 同样先写脚本文件；
    # 然后以 OSS 侧 _collect.log 存在作为上传成功判据。终局上传段任何
    # arun 抛 'not started' 同样按沙箱死亡处置（增量抢救已保底）。
    try:
        await sandbox.arun(
            cmd=(
                "cat > /tmp/derail_result.sh <<'DERAILRESEOF'\n"
                f"{_result_command()}\nDERAILRESEOF\n"
            ),
            session="default",
        )
        res = await sandbox.arun(
            cmd="bash /tmp/derail_result.sh > /tmp/result_upload.log 2>&1",
            session="default",
            mode="nohup",
            wait_timeout=ROCK_RESULT_TIMEOUT,
        )
        print(res.output)
        verify = await sandbox.arun(
            cmd=(
                f"set -a; . {shlex.quote(SECRETS_REMOTE_PATH)} 2>/dev/null || true; set +a; "
                f"{_ossutil()} ls {shlex.quote(OSS_RESULTS_URI)}/_collect.log; "
                "echo ---; tail -5 /tmp/result_upload.log"
            ),
            session="default",
            wait_timeout=120,
        )
        print(f"[driver] result upload verify:\n{verify.output}")
        if "_collect.log" not in str(getattr(verify, "output", "") or ""):
            print("[driver] WARNING: OSS result upload unverified (_collect.log missing)",
                  file=sys.stderr)
    except Exception as exc:  # noqa: BLE001
        if _sandbox_dead_marker(exc):
            print("[driver] sandbox died during ship; incremental salvage applies",
                  file=sys.stderr)
            return run_error or exc, True
        raise
    return run_error, False


# --- ROCK proxy 拓扑（开源模型 full 阶段；仿 MCUA OSWorld 三层架构） -----------------
#
# 沙箱→Nebula pod IP 不可达（2026-08-15 full 首撞实证）后的正式拓扑：
#   L1 vLLM       Nebula job 内 localhost:SERVE_PORT（entry 起）
#   L2 编排       本 driver + bridge + 01 脚本（agent loop）全部在 Nebula，
#                 agent 代码只连 127.0.0.1（harness MYPCBENCH_REMOTE_ATTACH=1）
#   L3 环境       ROCK 沙箱：guest QEMU（boot 脚本）+ gateway(:8080 按路径前缀转发)
# 方向永远只有 Nebula → ROCK Proxy → 沙箱 gateway → guest；产物结构不变，
# 经 OSS FUSE 挂载上云（不经 ossutil，Nebula 侧无凭证需求）。


def _sandbox_id(sandbox) -> str:
    """ROCK SDK 不同版本暴露 id 的属性名不一（同 MCUA submit_rock_proxy）。"""
    for name in ("sandbox_id", "id", "instance_id", "_sandbox_id"):
        value = getattr(sandbox, name, None)
        if isinstance(value, str) and value:
            return value
    for name, value in vars(sandbox).items():
        if "id" in name.lower() and isinstance(value, str) and value:
            return value
    raise RuntimeError(f"ROCK SDK did not expose the sandbox id: {vars(sandbox)!r}")


def _fuse_path(oss_uri: str) -> str:
    """oss://<bucket>/<key> -> /data/oss_bucket_0/<key>（Nebula FUSE 挂载）。"""
    stripped = oss_uri[len("oss://"):]
    return "/data/oss_bucket_0/" + stripped.split("/", 1)[1]


def _proxy_routes():
    """bridge 路由表：本地口 -> Proxy 路径前缀（port=None = gateway :8080 形态）。

    api 口（PORT_BASE）前缀 pcapi；18 个 app 口（PORT_BASE+10+i）前缀
    app<cp>，与沙箱 gateway 的路由表一一对应（见 _proxy_guest_prep_command）。
    """
    from rock_proxy_bridge import ProxyRoute

    routes = [ProxyRoute(local_port=PROXY_PORT_BASE, sandbox_port=None,
                         upstream_prefix="pcapi")]
    for i in range(18):
        cp = 3001 + i
        routes.append(ProxyRoute(local_port=PROXY_PORT_BASE + 10 + i,
                                 sandbox_port=None, upstream_prefix=f"app{cp}"))
    return routes


def _proxy_gateway_route_args() -> str:
    args = ["--route", "pcapi:127.0.0.1:15000"]
    for cp in range(3001, 3019):
        args += ["--route", f"app{cp}:127.0.0.1:{cp}"]
    return " ".join(args)


async def _stage_sandbox_file(sandbox, source: Path, target: str) -> None:
    """小文件 base64 注入沙箱（同 MCUA _stage_file；upload_dir 只为整个 repo）。"""
    import base64

    encoded = base64.b64encode(source.read_bytes()).decode("ascii")
    script = (
        "import base64,pathlib; "
        f"p=pathlib.Path({target!r}); p.parent.mkdir(parents=True,exist_ok=True); "
        f"p.write_bytes(base64.b64decode({encoded!r}))"
    )
    # osworld-rock:v7 镜像无系统 python3（只有 python3.12，首跑实证 127）；
    # 解析式取解释器，与 prep 的 venv 创建用 python3.12 同一事实。
    await sandbox.arun(
        cmd=('PY_BIN="$(command -v python3 || command -v python3.12)"; '
             f'"$PY_BIN" -c {shlex.quote(script)} && chmod +x {target}'),
        session="default", wait_timeout=120)


def _proxy_guest_prep_command() -> str:
    """沙箱侧 prep-lite（proxy 拓扑）：拉资产 → 起 gateway → boot guest → 验健康。

    与 in-sandbox 拓扑的 _prep_command 相比去掉 repo/venv（harness 在 Nebula），
    末尾的健康探针走 gateway 路径前缀（pcapi/health），等于在沙箱内就把
    L3 链（gateway → hostfwd → guest 控制面）验完；L2 链（bridge → Proxy）
    由 driver 在 Nebula 侧另验。
    """
    qcow2 = f"{SANDBOX_VM_DIR}/mypcbench.qcow2"
    return f"""
set -Eeuo pipefail
export PATH="/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin${{PATH:+:$PATH}}"
set -a; . {shlex.quote(SECRETS_REMOTE_PATH)}; set +a
step() {{ echo "[guest-prep $(date +%H:%M:%S)] $*"; }}

# 1. qcow2 + OVMF（OSS 资产；同 in-sandbox prep 的已验形状）。
mkdir -p {shlex.quote(SANDBOX_VM_DIR)}
if [[ -f {qcow2} ]]; then
  step "qcow2 already staged"
else
  step "pulling mypcbench.qcow2 from OSS (~10G)"
  {_ossutil()} cp -f {shlex.quote(OSS_QCOW2_URI)} {qcow2}
fi
for f in OVMF_CODE.fd OVMF_VARS.fd; do
  [[ -f {SANDBOX_VM_DIR}/$f ]] || \\
    {_ossutil()} cp -f {shlex.quote(OSS_ASSETS_URI)}/$f {SANDBOX_VM_DIR}/$f
done
step "assets staged: $(ls {SANDBOX_VM_DIR})"

# 2. gateway（纯 stdlib，:8080）：幂等 —— 已活则不重启。
if curl -fsS --max-time 5 http://127.0.0.1:8080/__rock_gateway_health >/dev/null 2>&1; then
  step "gateway already up"
else
  pkill -f derail_guest_gateway.py 2>/dev/null || true
  DERAIL_PY="$(command -v python3 || command -v python3.12)"
  nohup setsid "$DERAIL_PY" /opt/derail-guest/derail_guest_gateway.py \\
    --port 8080 {_proxy_gateway_route_args()} \\
    </dev/null > /tmp/derail_gateway.log 2>&1 &
  for _ in $(seq 1 12); do
    curl -fsS --max-time 3 http://127.0.0.1:8080/__rock_gateway_health >/dev/null 2>&1 && break
    sleep 2
  done
  curl -fsS http://127.0.0.1:8080/__rock_gateway_health || {{
    echo "ERROR: gateway failed to start" >&2; tail -20 /tmp/derail_gateway.log >&2; exit 1; }}
  step "gateway up"
fi

# 3. guest boot（CoW overlay + KVM；幂等：已健康直接 GUEST_ALREADY_UP）。
bash /opt/derail-guest/derail_guest_boot.sh

# 4. 经 gateway 验 guest 控制面（L3 链终验）。
curl -fsS --max-time 10 http://127.0.0.1:8080/pcapi/health >/dev/null \\
  || {{ echo "ERROR: guest health via gateway failed" >&2; exit 1; }}
echo "GUEST_PREP_DONE"
if [[ "${{DERAIL_KEEP_GUEST_SUPERVISOR:-0}}" == "1" ]]; then
  # Keep the prep session alive so ROCK does not reap QEMU/gateway children.
  sleep 86400
fi
""".strip()


async def _guest_lifecycle(sandbox, mode: str) -> str:
    """在沙箱内执行 guest 生命周期（供 LifecycleServer 转发 env.py 的 reset）。

    boot/reset 同形：boot 脚本幂等，非健康时杀残留→重建 overlay→重启，
    与 env.py 本地 _reset_qemu 的干净态语义一致。nohup + 轮询日志：guest
    自举 2-4min 必撞 rocklet 85s 单请求上限（同 prep 的已验形状）。
    """
    if mode == "stop":
        out = await sandbox.arun(
            cmd="pkill -f qemu-system 2>/dev/null || true; "
                "rm -f /tmp/mypcbench-guest.pid; echo LIFECYCLE_STOPPED",
            session="default", wait_timeout=120,
        )
        return str(getattr(out, "output", out) or "")
    if mode not in ("boot", "reset"):
        raise ValueError(f"unknown lifecycle mode: {mode}")
    # 重定向必须走 SDK 的 output_file 参数，不能内嵌在 cmd 里：nohup 模式
    # 会把 cmd 包成 `nohup {cmd} < /dev/null > {tmp} 2>&1 &`，bash 重定向
    # 后者生效，内嵌的 `> log` 只被截断成 0 字节、永远写不进东西（2026-08-16
    # 正式全量首跑实证：每次 reset 其实都 GUEST_ALREADY_UP，输出全进了 SDK
    # 的 tmp_*.out，轮询却 grep 空 log，720s 假超时 ×3 重试，每任务白烧 1h）。
    await sandbox.arun(
        cmd="bash /opt/derail-guest/derail_guest_boot.sh",
        session="default", mode="nohup", wait_timeout=1800,
        output_file="/tmp/guest_lifecycle.log",
    )
    for attempt in range(1, 37):
        check = await sandbox.arun(
            cmd="grep -E 'GUEST_READY|GUEST_ALREADY_UP|GUEST_DIED|GUEST_READY_TIMEOUT' "
                "/tmp/guest_lifecycle.log 2>/dev/null | tail -1",
            session="default", wait_timeout=60,
        )
        out = str(getattr(check, "output", check) or "")
        if "GUEST_READY" in out or "GUEST_ALREADY_UP" in out:
            return out
        if "GUEST_DIED" in out or "GUEST_READY_TIMEOUT" in out:
            raise RuntimeError(f"guest lifecycle {mode} failed: {out}")
        await asyncio.sleep(20)
    raise RuntimeError(f"guest lifecycle {mode} timed out waiting for GUEST_READY")


class _LifecycleServer:
    """Nebula 127.0.0.1:PROXY_LIFECYCLE_PORT：env.py 的 _reset_qemu 在
    REMOTE_ATTACH 下 POST 到这里（harness proxy patch）；持沙箱句柄的
    driver 替它在沙箱内重建 overlay 重启 guest。sandbox 句柄经 holder
    间接持有，沙箱重建后不用重启服务。
    """

    def __init__(self, holder):
        self.holder = holder  # SimpleNamespace(sandbox=...)
        self.runner = None

    async def start(self) -> None:
        from aiohttp import web

        app = web.Application()
        for mode in ("boot", "reset", "stop"):
            app.router.add_post(f"/{mode}", self._handler(mode))
        self.runner = web.AppRunner(app)
        await self.runner.setup()
        await web.TCPSite(self.runner, "127.0.0.1", PROXY_LIFECYCLE_PORT).start()
        print(f"[driver:proxy] lifecycle server on 127.0.0.1:{PROXY_LIFECYCLE_PORT}")

    async def close(self) -> None:
        if self.runner:
            await self.runner.cleanup()

    def _handler(self, mode: str):
        from aiohttp import web

        async def handle(_request):
            try:
                result = await _guest_lifecycle(self.holder.sandbox, mode)
                return web.Response(text=result)
            except Exception as exc:  # noqa: BLE001 -- 以 HTTP 500 暴露给 env.py
                return web.Response(status=500, text=str(exc))
        return handle


async def _setup_sandbox_proxy(sandbox, CreateBashSessionRequest):
    """proxy 拓扑的起沙流程：429 退避 + KVM 回收（同 in-sandbox）+ secrets
    + gateway/boot 脚本注入 + prep-lite。不 upload_dir repo（harness 在 Nebula）。
    """
    from rock.sdk.sandbox.client import Sandbox

    print(f"[driver:proxy] starting sandbox (image={ROCK_SANDBOX_IMAGE} "
          f"cluster={ROCK_CLUSTER})")
    start_attempts = int(_env("ROCK_START_RETRIES", "8"))
    for attempt in range(1, start_attempts + 1):
        try:
            await asyncio.wait_for(
                sandbox.start(), timeout=ROCK_STARTUP_TIMEOUT + 30
            )
            break
        except Exception as exc:  # noqa: BLE001 -- 只对限流重试
            transient = "429" in str(exc) or "Failed to start sandbox" in str(exc)
            if not transient or attempt == start_attempts:
                raise
            delay = min(120 * attempt, 600)
            reason = "429 throttled" if "429" in str(exc) else "start timeout"
            print(f"[driver:proxy] start failed ({reason}), attempt {attempt}/"
                  f"{start_attempts}; retrying in {delay}s")
            try:
                await sandbox.stop()
            except Exception as stop_exc:  # noqa: BLE001 -- half-start cleanup
                print(f"[driver:proxy] stop(half-started sandbox) ignored: {stop_exc}")
            await asyncio.sleep(delay)
            sandbox = Sandbox(_sandbox_config())

    await sandbox.create_session(CreateBashSessionRequest(session="default"))

    kvm_attempts = int(_env("ROCK_KVM_RETRIES", "14"))
    for kvm_attempt in range(1, kvm_attempts + 1):
        probe = await sandbox.arun(
            cmd="ls /dev/kvm 2>/dev/null && echo KVM_OK || echo KVM_MISSING",
            session="default",
        )
        if "KVM_OK" in str(getattr(probe, "output", probe) or ""):
            print(f"[driver:proxy] /dev/kvm present (attempt {kvm_attempt})")
            break
        print(f"[driver:proxy] /dev/kvm MISSING (attempt {kvm_attempt}/"
              f"{kvm_attempts}) -- recycling sandbox")
        if kvm_attempt == kvm_attempts:
            raise RuntimeError(
                f"no /dev/kvm after {kvm_attempts} attempts; guest cannot boot")
        await sandbox.stop()
        await asyncio.sleep(20)
        sandbox = Sandbox(_sandbox_config())
        await sandbox.start()
        await sandbox.create_session(CreateBashSessionRequest(session="default"))

    # 0600 secrets（OSS AK；guest prep 拉资产用。不上 argv）。
    await _stage_sandbox_secrets(sandbox)

    print("[driver:proxy] staging gateway + boot scripts into sandbox")
    await _stage_sandbox_file(
        sandbox, HERE / "derail_guest_gateway.py", "/opt/derail-guest/derail_guest_gateway.py")
    await _stage_sandbox_file(
        sandbox, HERE / "derail_guest_boot.sh", "/opt/derail-guest/derail_guest_boot.sh")

    print("[driver:proxy] guest prep: assets + gateway + guest boot")
    await sandbox.arun(
        cmd=(
            "cat > /tmp/derail_guest_prep.sh <<'DERAILGUESTPREPEOF'\n"
            f"{_proxy_guest_prep_command()}\nDERAILGUESTPREPEOF\n"
        ),
        session="default",
    )
    global _PHASE5_SUPERVISOR_TASK
    supervisor_mode = os.environ.get("PHASE5_KEEP_SUPERVISOR") == "1"
    if supervisor_mode:
        await sandbox.create_session(CreateBashSessionRequest(session="supervisor"))
        # The command remains active through the sleep; run it asynchronously.
        _PHASE5_SUPERVISOR_TASK = asyncio.create_task(
            sandbox.arun(
                cmd="export DERAIL_KEEP_GUEST_SUPERVISOR=1; bash /tmp/derail_guest_prep.sh > /tmp/guest_prep.log 2>&1",
                session="supervisor", mode="normal", wait_timeout=86400,
            )
        )
        prep = None
    else:
        prep = await sandbox.arun(
            cmd="bash /tmp/derail_guest_prep.sh > /tmp/guest_prep.log 2>&1",
            session="default", mode="nohup", wait_timeout=3600,
        )
    print(f"[driver:proxy] guest prep output tail:\n{getattr(prep, 'output', '')}")
    prep_ok = "GUEST_PREP_DONE" in str(getattr(prep, "output", "") or "")
    for check_attempt in range(1, 37):
        if prep_ok:
            break
        await asyncio.sleep(30)
        try:
            prep_check = await sandbox.arun(
                cmd="grep -c GUEST_PREP_DONE /tmp/guest_prep.log 2>/dev/null; "
                    "tail -20 /tmp/guest_prep.log",
                session="default", wait_timeout=120,
            )
        except Exception as exc:  # noqa: BLE001 -- 单次探测网络抖动不判死
            print(f"[driver:proxy] guest prep check failed (attempt {check_attempt}): {exc}")
            continue
        check_out = str(getattr(prep_check, "output", "") or "")
        print(f"[driver:proxy] guest prep log (attempt {check_attempt}):\n{check_out}")
        prep_ok = "GUEST_PREP_DONE" in check_out
    if not prep_ok:
        raise RuntimeError("guest prep did not reach GUEST_PREP_DONE; see log above")
    return sandbox


def _proxy_nebula_setup() -> None:
    """Nebula 侧一次性准备（同步 bash）：harness（OSS tar 经 FUSE）+ proxy
    patch + agent 依赖 + OpenCUA-OSWorld（OSS 快照优先，github clone 兜底）。
    失败即抛：没有这些 01 脚本必挂，早断早重提。
    """
    import subprocess

    harness_fuse = _fuse_path(OSS_HARNESS_URI)
    setup_script = f"""
set -Eeuo pipefail
cd {shlex.quote(str(REPO))}
step() {{ echo "[nebula-setup $(date +%H:%M:%S)] $*"; }}

# tar 解出的 .git 属主与容器运行用户不同，git 会以 "detected dubious
# ownership" 拒绝一切操作（rev-parse 输出空 → commit 校验假失败，首跑实证）。
# 容器单次性环境直接放开；--system 不依赖 $HOME（同沙箱 prep 的处置）。
export HOME="${{HOME:-/root}}"
git config --system --add safe.directory '*' 2>/dev/null || true

# 1. harness：third_party 不进代码包（IGNORE_LIST），从 OSS FUSE 恢复
#    （同沙箱 prep 的 tar，含 .git + 已打的 agents patch + 本地适配）。
if [[ -d third_party/MyPCBench/.git ]] && \\
   [[ "$(git -C third_party/MyPCBench rev-parse HEAD 2>/dev/null)" == "{MYPCBENCH_COMMIT}" ]]; then
  step "harness already at {MYPCBENCH_COMMIT}"
else
  step "restoring harness from FUSE: {harness_fuse}"
  [[ -f {shlex.quote(harness_fuse)} ]] || {{ echo "ERROR: harness tar missing on FUSE mount" >&2; exit 1; }}
  mkdir -p third_party
  rm -rf third_party/MyPCBench
  tar -xzf {shlex.quote(harness_fuse)} -C third_party
  [[ "$(git -C third_party/MyPCBench rev-parse HEAD)" == "{MYPCBENCH_COMMIT}" ]] \\
    || {{ echo "ERROR: harness commit mismatch after restore" >&2; exit 1; }}
fi

# 2. proxy patch（幂等；tar 里若已含会被 reverse-check 跳过）。third_party.sh
#    的 info/die 由调用方定义，这里先补 shim。
info() {{ echo "[nebula-setup] $*"; }}
die() {{ echo "[nebula-setup] ERROR: $*" >&2; exit 1; }}
source scripts/lib/third_party.sh
third_party_paths "$(pwd)"
setup_mypcbench_proxy

# 3. agent 依赖（Nebula 系统 python；镜像自带大部分，只补缺）。
{shlex.quote(sys.executable)} -c 'import openai, requests, PIL, backoff, yaml, httpx, loguru, aiohttp' 2>/dev/null \\
  || {shlex.quote(sys.executable)} -m pip install -q openai requests Pillow backoff PyYAML httpx loguru aiohttp \\
  || {shlex.quote(sys.executable)} -m pip install -q openai requests Pillow backoff PyYAML httpx loguru aiohttp -i "${{PIP_INDEX_URL:-https://pypi.org/simple}}"
{shlex.quote(sys.executable)} -c 'import openai, requests, PIL, backoff, yaml, httpx, loguru, aiohttp; print("[nebula-setup] agent deps OK")'

# 4. EvoCUA（factory 的 adapter source）：OSS 快照优先（含 .git）。
if [[ "{AGENT_ID}" == "evocua_32b" ]]; then
  if [[ -f third_party/EvoCUA/mm_agents/evocua/evocua_agent.py ]] && \
     [[ "$(git -C third_party/EvoCUA rev-parse HEAD 2>/dev/null)" == "{EVOCUA_COMMIT}" ]]; then
    step "EvoCUA already staged"
  else
    _evocua_snap="${{OSS_EVOCUA_SNAPSHOT_FUSE:-{_fuse_path(OSS_EVOCUA_SNAPSHOT_URI)}}}"
    if [[ -f "$_evocua_snap" ]]; then
      step "restoring EvoCUA from FUSE: $_evocua_snap"
      rm -rf third_party/EvoCUA
      tar -xzf "$_evocua_snap" -C third_party
    else
      step "EvoCUA OSS snapshot missing; falling back to frozen github clone"
      setup_evocua
    fi
    [[ "$(git -C third_party/EvoCUA rev-parse HEAD)" == "{EVOCUA_COMMIT}" ]] \
      || die "EvoCUA commit mismatch after restore"
  fi
fi

# 5. OpenCUA-OSWorld（factory 的 adapter source）：OSS 快照优先（含 .git，
#    01 脚本的 commit 校验直接过，零 github）；快照缺失才 fallback github clone。
if [[ -f third_party/OpenCUA-OSWorld/mm_agents/opencua/opencua_agent.py ]] && \\
   [[ "$(git -C third_party/OpenCUA-OSWorld rev-parse HEAD 2>/dev/null)" == "{OPENCUA_OSWORLD_COMMIT}" ]]; then
  step "OpenCUA-OSWorld already staged"
else
  _snap="${{OSS_OPENCUA_SNAPSHOT_FUSE:-{_fuse_path(OSS_OPENCUA_SNAPSHOT_URI)}}}"
  if [[ -f "$_snap" ]]; then
    step "restoring OpenCUA-OSWorld from FUSE: $_snap"
    rm -rf third_party/OpenCUA-OSWorld
    tar -xzf "$_snap" -C third_party
  else
    step "OSS snapshot missing; falling back to github clone (egress 未验)"
    info() {{ echo "[nebula-setup] $*"; }}
    setup_opencua_osworld
  fi
  [[ "$(git -C third_party/OpenCUA-OSWorld rev-parse HEAD)" == "{OPENCUA_OSWORLD_COMMIT}" ]] \\
    || die "OpenCUA-OSWorld commit mismatch after restore"
fi
echo "NEBULA_SETUP_DONE"
""".strip()
    print("[driver:proxy] nebula-side setup: harness + proxy patch + deps + OpenCUA-OSWorld")
    proc = subprocess.run(["bash", "-c", setup_script], check=False,
                          stdout=sys.stdout, stderr=sys.stderr)
    if proc.returncode != 0:
        raise RuntimeError(f"nebula-side setup failed rc={proc.returncode}")


def _proxy_collect_env() -> dict:
    """01 脚本在 proxy 拓扑下的环境变量：harness 只连 127.0.0.1（bridge），
    模型采样走 localhost vLLM；产物落 PROXY_RESULTS_LOCAL（后接 FUSE 上云）。
    """
    env = os.environ.copy()
    env.update({
        "DERAIL_TMUX": "0",
        "REPEATS": REPEATS,
        "NUM_VMS_OVERRIDE": NUM_VMS_OVERRIDE,
        "BACKEND": "qemu",
        "MAX_STEPS": MAX_STEPS,
        "DERAIL_BASH_ACCOUNTING": DERAIL_BASH_ACCOUNTING,
        "TASK_TIMEOUT": TASK_TIMEOUT,
        "TIMEOUT_PER_VM": TIMEOUT_PER_VM,
        "TASKS_FILE": str(REPO / SHARD_FILE),
        "OUTPUT_ROOT": PROXY_RESULTS_LOCAL,
        "COLLECTION_ID": COLLECTION_ID,
        "PYTHON_BIN": sys.executable,
        "OPENCUA_BASE_URLS": OPENCUA_BASE_URLS,
        "OPENCUA_MODEL": OPENCUA_MODEL,
        "QWEN38_BASE_URLS": OPENCUA_BASE_URLS,
        "QWEN35_BASE_URLS": OPENCUA_BASE_URLS,
        "QWEN38_MODEL": QWEN38_MODEL,
        "QWEN35_MODEL": QWEN35_MODEL,
        "FORMAL_COLLECTION": FORMAL_COLLECTION,
        "DERAIL_OPENAI_API_APPROVED": DERAIL_OPENAI_API_APPROVED,
        "DERAIL_OPENAI_API_PURPOSE": DERAIL_OPENAI_API_PURPOSE,
        "DERAIL_ANTHROPIC_API_APPROVED": DERAIL_ANTHROPIC_API_APPROVED,
        "DERAIL_ANTHROPIC_API_PURPOSE": DERAIL_ANTHROPIC_API_PURPOSE,
        "DERAIL_AUTO_SERVE": "0",
        # proxy 拓扑三开关（harness proxy patch 的契约）：
        #   REMOTE_ATTACH：runner 不本地起 VM、端口不滑窗；env.py reset 走 lifecycle
        #   VM_HOST=localhost + REUSE：attach 分支连 bridge 本地口
        #   PORT_BASE：bridge 监听段（与 _proxy_routes 对齐）
        "MYPCBENCH_REMOTE_ATTACH": "1",
        "MYPCBENCH_VM_HOST": "127.0.0.1",
        "MYPCBENCH_REUSE_CONTAINER": "1",
        "MYPCBENCH_LIFECYCLE_PORT": str(PROXY_LIFECYCLE_PORT),
        "PORT_BASE": str(PROXY_PORT_BASE),
        "MYPCBENCH_HOST_API_PORT": str(PROXY_PORT_BASE),
    })
    # reuse-attach 路径不走 _discover_ports_qemu，app 口需显式告知
    #（bridge 本地口 PORT_BASE+10+i <-> guest <cp>，经 gateway app<cp> 前缀）。
    for i in range(18):
        env[f"MYPCBENCH_HOST_APP_PORT_{3001 + i}"] = str(PROXY_PORT_BASE + 10 + i)
    return env


def _takeover_path_mappings() -> dict[str, str]:
    manifest = TAKEOVER_INPUT_ROOT / "bundle_manifest.json"
    raw = json.loads(manifest.read_text(encoding="utf-8"))
    mappings = {
        source: str(TAKEOVER_INPUT_ROOT / relative)
        for source, relative in raw["path_mappings"].items()
    }
    return mappings


def _takeover_path_mappings_file() -> str:
    mappings_path = Path("/tmp/derail_takeover_path_mappings.json")
    mappings_path.write_text(
        json.dumps(_takeover_path_mappings(), separators=(",", ":")),
        encoding="utf-8",
    )
    return str(mappings_path)


def _takeover_env() -> dict:
    env = _proxy_collect_env()
    # 采集预算不是 takeover 预算：run_takeover.sh 自己的默认（接管后最多 100 步，
    # 论文 05:12）只有在调用方没显式给值时才该生效，不能被 clean-start 的 150 步盖掉。
    for name in ("MAX_STEPS", "TASK_TIMEOUT", "REPEATS"):
        if not os.environ.get(name, "").strip():
            env.pop(name, None)
    env.update({
        "SOURCE_AGENT": TAKEOVER_SOURCE_AGENT,
        "TARGET_AGENT": TAKEOVER_TARGET_AGENT,
        "ANNOTATOR_ID": TAKEOVER_ANNOTATOR,
        "BUILD_DIR": str(TAKEOVER_INPUT_ROOT / TAKEOVER_BUILD_DIR),
        "HUMAN_LABELS_DIR": str(TAKEOVER_INPUT_ROOT / "takeovewr_annotation/human_labels"),
        "QCOW2": _fuse_path(OSS_ASSETS_URI) + "/mypcbench.qcow2",
        "QCOW2_SHA256": LOCK_QCOW2_SHA256,
        "QCOW2_HASH_PREVERIFIED": "1",
        "CONDITIONS": TAKEOVER_CONDITION,
        "DEPTHS": TAKEOVER_DEPTH,
        "NUM_WORKERS": "1",
        "SHARD_COUNT": TAKEOVER_SHARD_COUNT,
        "SHARD_OFFSET": TAKEOVER_SHARD_OFFSET,
        "TRAJECTORY_ID_FILTER": TAKEOVER_TRAJECTORY_ID_FILTER,
        "TRAJECTORY_ID_FILE": TAKEOVER_TRAJECTORY_ID_FILE,
        "TARGET_BASE_URLS": OPENCUA_BASE_URLS or _env("OPENAI_BASE_URL", ""),
        "TOKENIZE_BASE_URL": TAKEOVER_TOKENIZE_BASE_URL,
        "TOKENIZE_MODE": TAKEOVER_TOKENIZE_MODE,
        "EXPERIMENT_CONTEXT_CAP": TAKEOVER_CONTEXT_CAP,
        # Keep every submission below its immutable collection ID. Shipping the
        # parent then produces the documented d<depth>/<collection_id>/ layout.
        "OUTPUT_ROOT": str(Path(PROXY_RESULTS_LOCAL) / COLLECTION_ID),
        "RUN_JUDGE": "0",
        # 1,822 mappings are about 469 KiB and exceed execve's environment
        # limit when passed inline. Keep the JSON fallback for small local runs.
        "DERAIL_PATH_REMAP_FILE": _takeover_path_mappings_file(),
    })
    return env


def _phase5_env() -> dict:
    env = _proxy_collect_env()
    env.update({
        "PHASE5_MODEL": AGENT_ID,
        "PHASE5_BATCH": COLLECTION_ID,
        "PHASE5_SHARD_COUNT": PHASE5_SHARD_COUNT,
        "PHASE5_SHARD_INDEX": PHASE5_SHARD_INDEX,
        "PHASE5_SMOKE_MODE": PHASE5_SMOKE_MODE,
        "DERAIL_PHASE5_OUT": str(Path(PROXY_RESULTS_LOCAL) / COLLECTION_ID),
        "DERAIL_MYPCBENCH_ROOT": str(REPO / "third_party/MyPCBench"),
        "DERAIL_CONTROL_API_URL": f"http://127.0.0.1:{PROXY_PORT_BASE}",
        "DERAIL_IMAGE_DIGEST": LOCK_QCOW2_SHA256,
    })
    return env


def _proxy_workload() -> tuple[list[str], dict, str]:
    if DERAIL_WORKLOAD == "takeover":
        command = ["bash", str(REPO / "scripts/rock/run_takeover.sh")]
        return command, _takeover_env(), "takeover"
    if DERAIL_WORKLOAD == "phase5":
        command = ["bash", str(REPO / "scripts/rock/run_phase5_rollout.sh")]
        return command, _phase5_env(), "phase5"
    command = ["bash", str(REPO / "scripts/01_collect_trajectories.sh"), AGENT_ID]
    return command, _proxy_collect_env(), "collection"


async def _proxy_ship(local_dir: str, dest_dir: str) -> None:
    """结果经 OSS FUSE 上云（同 bucket 路径，无凭证需求）。cp -a 保时间戳；
    FUSE 写入慢但 smoke 体量小；失败只告警（下一轮/终局会再拷）。
    """
    proc = await asyncio.create_subprocess_exec(
        "bash", "-c",
        f"mkdir -p {shlex.quote(dest_dir)} && "
        f"cp -a {shlex.quote(local_dir)}/. {shlex.quote(dest_dir)}/",
        stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.STDOUT,
    )
    out, _ = await proc.communicate()
    if proc.returncode != 0:
        print(f"[driver:proxy] ship FAILED rc={proc.returncode}: "
              f"{out.decode(errors='replace')[-400:]}", file=sys.stderr)


def _write_rollout_heartbeat() -> None:
    collection = Path(PROXY_RESULTS_LOCAL) / COLLECTION_ID
    collection.mkdir(parents=True, exist_ok=True)
    files = [path for path in collection.rglob("*") if path.is_file()]
    payload = {
        "schema_version": 1,
        "collection_id": COLLECTION_ID,
        "generated_at_unix": int(time.time()),
        "result_count": sum(path.name == "result.txt" for path in files),
        "trajectory_count": sum(path.name == "traj.jsonl" for path in files),
        "screenshot_count": sum(
            path.name.startswith("step_") and path.suffix == ".png"
            for path in files
        ),
        "strict_admission": "result.txt + traj.jsonl + screenshot + real action",
    }
    target = collection / "_rollout_heartbeat.json"
    temporary = target.with_suffix(".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temporary.replace(target)


SHIP_LEDGER_NAME = "ship_ledger.json"


def _relative_file_sizes(root: Path) -> dict[str, int]:
    return {
        str(path.relative_to(root)): path.stat().st_size
        for path in root.rglob("*")
        if path.is_file() and path.name != SHIP_LEDGER_NAME
    }


def _build_ship_ledger(local: Path, remote: Path) -> dict:
    local_files = _relative_file_sizes(local)
    remote_files = _relative_file_sizes(remote) if remote.is_dir() else {}
    missing = sorted(set(local_files) - set(remote_files))
    unexpected = sorted(set(remote_files) - set(local_files))
    mismatched = sorted(
        path for path in set(local_files) & set(remote_files)
        if local_files[path] != remote_files[path]
    )
    result_count = sum(path.endswith("/result.txt") for path in local_files)
    return {
        "schema_version": 1,
        "collection_id": COLLECTION_ID,
        "generated_at_unix": int(time.time()),
        "local_object_count": len(local_files),
        "remote_object_count": len(remote_files),
        "completed_result_count": result_count,
        "missing_remote_objects": missing,
        "unexpected_remote_objects": unexpected,
        "size_mismatches": mismatched,
        "object_parity": not (missing or unexpected or mismatched),
        "local_objects": local_files,
    }


def _write_ship_ledger(local: Path, remote: Path) -> dict:
    ledger = _build_ship_ledger(local, remote)
    payload = json.dumps(ledger, indent=2, sort_keys=True) + "\n"
    (local / SHIP_LEDGER_NAME).write_text(payload, encoding="utf-8")
    (remote / SHIP_LEDGER_NAME).write_text(payload, encoding="utf-8")
    return ledger


async def _verify_proxy_ship() -> bool:
    local = Path(PROXY_RESULTS_LOCAL) / COLLECTION_ID
    remote = Path(PROXY_RESULTS_OSS_MOUNT) / COLLECTION_ID
    ledger = await asyncio.to_thread(_write_ship_ledger, local, remote)
    summary = {key: ledger[key] for key in (
        "local_object_count", "remote_object_count", "completed_result_count",
        "object_parity",
    )}
    print(f"[driver:proxy] ship ledger: {json.dumps(summary, sort_keys=True)}")
    return bool(ledger["object_parity"] and ledger["completed_result_count"])


async def _proxy_collect() -> int:
    """Nebula 侧跑 01 脚本（子进程前台，stdout 直通 job 日志）+ 周期 FUSE
    抢救上传。返回 01 脚本退出码。
    """
    os.makedirs(PROXY_RESULTS_LOCAL, exist_ok=True)
    # 断点续采：先把 FUSE 上同 COLLECTION_ID 的部分结果回灌本地（harness
    # result.txt 守卫只重跑未完成任务），并清无 result.txt 的截断残留
    #（与 _seed_command 同语义，只是改在 Nebula 本地执行）。RERUN_PURGE=1
    # 时同 _seed_command 第四步：清场 rerun 清单（nonce marker 防重建轮误清）。
    if RESUME_SEED:
        seed_dir = f"{PROXY_RESULTS_OSS_MOUNT}/{COLLECTION_ID}"
        local_cid_dir = PROXY_RESULTS_LOCAL + "/" + COLLECTION_ID
        if RERUN_PURGE:
            ids = _shard_task_ids()
            marker = f"{local_cid_dir}/.derail_rerun_purged_{RERUN_PURGE_NONCE}"
            # proxy 拓扑产物经 FUSE 直写 OSS：先删 FUSE 上清单任务旧目录
            #（同沙箱拓扑 4a，cp -a 只增不删），再回灌、清场。
            fuse_purge_paths = ' '.join(
                shlex.quote(p)
                for t in ids
                for r in _repeat_dirs()
                for p in (
                    f"{seed_dir}/{COLLECTION_ID}/{AGENT_ID}/{r}/vm0/{t}",
                    f"{seed_dir}/{AGENT_ID}/{r}/vm0/{t}",
                )
            )
            purge_cmd = (
                f"if [[ ! -f {shlex.quote(marker)} ]]; then "
                f"for _p in {fuse_purge_paths}; do rm -rf \"$_p\" 2>/dev/null || true; done; "
                f"rm -f {shlex.quote(local_cid_dir)}/{RERUN_PURGE_MARKER_GLOB} 2>/dev/null || true; "
                f"for _t in {' '.join(shlex.quote(t) for t in ids)}; do "
                f"find {shlex.quote(local_cid_dir)} -mindepth 4 -maxdepth 4 -type d -name \"$_t\" -print0 | "
                f"xargs -0 -r rm -rf; done; "
                f"touch {shlex.quote(marker)}; "
                f"echo '[seed] rerun purge applied: {len(ids)} task ids'; "
                f"else echo '[seed] rerun purge marker present; skipping purge'; fi && "
            )
        else:
            purge_cmd = ""
        seed_cmd = (
            f"if [[ -d {shlex.quote(seed_dir)} ]]; then "
            f"{purge_cmd}"
            f"mkdir -p {shlex.quote(local_cid_dir)} && "
            f"cp -a {shlex.quote(seed_dir)}/. {shlex.quote(local_cid_dir)}/ && "
            f"if [[ -f {shlex.quote(local_cid_dir + '/_collect.log')} ]]; then "
            f"mv {shlex.quote(local_cid_dir + '/_collect.log')} "
            f"{shlex.quote(local_cid_dir + '/_collect.log.round1')}; fi && "
            f"find {shlex.quote(local_cid_dir)} "
            f"-mindepth 4 -maxdepth 4 -type d ! -name '_tasks' -print0 | "
            f"while IFS= read -r -d '' d; do [[ -f \"$d/result.txt\" ]] || rm -rf \"$d\"; done; "
            f"echo \"SEED_DONE kept=$(find {shlex.quote(local_cid_dir)} "
            f"-mindepth 4 -maxdepth 4 -type d ! -name '_tasks' | wc -l)\"; "
            f"else echo SEED_SKIP_NO_SEED_DIR; fi"
        )
        proc = await asyncio.create_subprocess_exec(
            "bash", "-c", seed_cmd,
            stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.STDOUT)
        out, _ = await proc.communicate()
        print(f"[driver:proxy] resume seed: {out.decode(errors='replace')[-300:]}")

    _write_rollout_heartbeat()

    async def _incremental_ship():
        while True:
            await asyncio.sleep(SHIP_INCREMENT_SECONDS)
            _write_rollout_heartbeat()
            await _proxy_ship(PROXY_RESULTS_LOCAL, PROXY_RESULTS_OSS_MOUNT)
            print(f"[driver:proxy] incremental ship -> {PROXY_RESULTS_OSS_MOUNT}/")

    ship_task = asyncio.create_task(_incremental_ship())
    command, workload_env, workload_name = _proxy_workload()
    print(f"[driver:proxy] launching {workload_name} for {AGENT_ID} "
          f"(output_root={PROXY_RESULTS_LOCAL})")
    try:
        proc = await asyncio.create_subprocess_exec(
            *command, cwd=str(REPO), env=workload_env,
            stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.STDOUT,
        )
        assert proc.stdout is not None
        while True:
            line = await proc.stdout.readline()
            if not line:
                break
            sys.stdout.write(line.decode(errors="replace"))
            sys.stdout.flush()
        rc = await proc.wait()
    finally:
        ship_task.cancel()

    print(f"[driver:proxy] collection exited rc={rc}; final ship -> FUSE")
    await _proxy_ship(PROXY_RESULTS_LOCAL, PROXY_RESULTS_OSS_MOUNT)
    ship_is_complete = await _verify_proxy_ship()
    if rc == 0 and not ship_is_complete:
        print("[driver:proxy] ERROR: workload exited successfully but ship parity failed", file=sys.stderr)
        return 1
    return rc


async def _verify_bridge_health() -> None:
    """bridge → ROCK Proxy → gateway → guest 全链终验（Nebula 侧）。
    不过则不跑采集：agent loop 起来也只会撞 bridge 502。
    阻塞式 urlopen 必须丢线程：bridge 与本函数共用同一事件循环，
    在 loop 里同步等 30s 会让 aiohttp bridge 无法处理该请求，
    确定性自锁超时（首跑实证 12/12 timed out）。
    """
    import urllib.request

    url = f"http://127.0.0.1:{PROXY_PORT_BASE}/health"

    def _probe() -> bytes:
        with urllib.request.urlopen(url, timeout=30) as resp:
            return resp.read(200)

    for attempt in range(1, 13):
        try:
            body = await asyncio.to_thread(_probe)
            print(f"[driver:proxy] bridge chain healthy: {url} -> {body[:120]!r}")
            return
        except Exception as exc:  # noqa: BLE001
            print(f"[driver:proxy] bridge health attempt {attempt}/12 failed: {exc}")
            await asyncio.sleep(15)
    raise RuntimeError(f"bridge chain unhealthy after retries: {url}")


async def _run_proxy() -> int:
    """proxy 拓扑主流程：Nebula 侧准备 → 起沙（guest+gateway）→ bridge +
    lifecycle → 01 脚本采集 → FUSE 上云。外层包沙箱死亡重建循环
    （同 in-sandbox 拓扑：重建后强制 RESUME_SEED 从 FUSE 抢救产物续采）。
    """
    global RESUME_SEED
    import types

    from rock.actions import CreateBashSessionRequest
    from rock.sdk.sandbox.client import Sandbox

    # nebula setup 先跑：它会 pip install aiohttp（bridge 依赖；镜像未必有）。
    _proxy_nebula_setup()

    sys.path.insert(0, str(HERE))  # rock_proxy_bridge 与本 driver 同目录
    from rock_proxy_bridge import RockProxyBridge

    rebuilds = 0
    run_error: Optional[BaseException] = None
    holder = types.SimpleNamespace(sandbox=None)
    control = _LifecycleServer(holder)
    bridge = None
    sandbox = Sandbox(_sandbox_config())
    try:
        await control.start()
        while True:
            sandbox = await _setup_sandbox_proxy(sandbox, CreateBashSessionRequest)
            holder.sandbox = sandbox
            sandbox_id = _sandbox_id(sandbox)
            os.environ["ROCK_SANDBOX_ID"] = sandbox_id
            bridge = RockProxyBridge(
                ROCK_BASE_URL, ROCK_API_KEY, sandbox_id, _proxy_routes())
            await bridge.start()
            print(f"[driver:proxy] bridge ready for sandbox {sandbox_id} "
                  f"(api 127.0.0.1:{PROXY_PORT_BASE} + 18 app ports)")
            try:
                await _verify_bridge_health()
                rc = await _proxy_collect()
            except Exception as exc:  # noqa: BLE001 -- 吃住，先判沙箱死活
                run_error = exc
                print(f"[driver:proxy] ERROR during collect: {exc}", file=sys.stderr)
                rc = 1
            await bridge.close()
            bridge = None
            if rc == 0:
                break
            run_error = run_error or RuntimeError(f"collection exited rc={rc}")
            if await _sandbox_alive(sandbox) or rebuilds >= SANDBOX_REBUILD_MAX:
                # 沙箱活着说明失败在采集/模型侧，重建无益；重建额度用完同断。
                break
            rebuilds += 1
            print(f"[driver:proxy] sandbox died mid-run; rebuilding "
                  f"(round {rebuilds}/{SANDBOX_REBUILD_MAX}); RESUME_SEED 从 FUSE 续采",
                  file=sys.stderr)
            try:
                await sandbox.stop()
            except Exception as exc:  # noqa: BLE001
                print(f"[driver:proxy] stop(dead sandbox) ignored: {exc}",
                      file=sys.stderr)
            RESUME_SEED = True
            sandbox = Sandbox(_sandbox_config())
    finally:
        if bridge is not None:
            try:
                await bridge.close()
            except Exception:  # noqa: BLE001
                pass
        await control.close()
        print("[driver:proxy] stopping sandbox")
        try:
            await sandbox.stop()
        except Exception as exc:  # noqa: BLE001
            print(f"[driver:proxy] final stop ignored: {exc}", file=sys.stderr)
        # Persist a joinable lifecycle record with the shipped collection.
        # This is evidence about the sandbox only; it never changes result
        # classification or the human-annotation denominator.
        try:
            lifecycle_path = Path(PROXY_RESULTS_LOCAL) / COLLECTION_ID / "rock_lifecycle.json"
            lifecycle_path.parent.mkdir(parents=True, exist_ok=True)
            lifecycle_path.write_text(json.dumps({
                "collection_id": COLLECTION_ID,
                "rock_sandbox_id": os.environ.get("ROCK_SANDBOX_ID") or None,
                "stop_requested": True,
                "is_alive_after_stop": await _sandbox_alive(sandbox),
            }, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        except Exception as exc:  # noqa: BLE001
            print(f"[driver:proxy] lifecycle ledger write failed: {exc}", file=sys.stderr)
    if run_error is not None:
        print(f"[driver:proxy] collection did not finish cleanly: {run_error}",
              file=sys.stderr)
        return 3
    return 0


async def _run() -> int:
    """主流程：起沙→采集→上传，外层包沙箱死亡重建循环（≤SANDBOX_REBUILD_MAX）。

    重建路径（2026-08-15 新增）：沙箱在长任务中途死亡属平台侧随机事件
    （opus_s3 五提 / kimi CRASH-5 三提两 job 独立复现），旧代码无容错致
    整 job 炸且产物全丢。现在：增量抢救保住已完成任务 → 新沙箱 +
    强制 RESUME_SEED 从 OSS 回灌续采（harness result.txt 守卫只重跑未完
    成任务）。COLLECTION_ID 为模块级常量，重建后 OSS 目录不变。
    """
    global RESUME_SEED
    from rock.actions import CreateBashSessionRequest
    from rock.sdk.sandbox.client import Sandbox

    run_cmd = _run_command()
    rebuilds = 0
    run_error: Optional[BaseException] = None
    sandbox = Sandbox(_sandbox_config())
    try:
        sandbox = await _setup_sandbox(sandbox, CreateBashSessionRequest)
        while True:
            run_error, sandbox_died = await _collect_and_ship(sandbox, run_cmd)
            if not sandbox_died:
                break
            if rebuilds >= SANDBOX_REBUILD_MAX:
                print(f"[driver] FATAL: sandbox died {rebuilds + 1} times; giving up "
                      f"(salvaged partials -> {OSS_RESULTS_URI}/)", file=sys.stderr)
                run_error = run_error or RuntimeError("sandbox died too many times")
                break
            rebuilds += 1
            print(f"[driver] rebuilding sandbox (round {rebuilds}/{SANDBOX_REBUILD_MAX}); "
                  "forcing RESUME_SEED from OSS salvage", file=sys.stderr)
            try:
                await sandbox.stop()  # 死沙箱 stop 失败无所谓，防僵尸配额占用
            except Exception as exc:  # noqa: BLE001
                print(f"[driver] stop(dead sandbox) ignored: {exc}", file=sys.stderr)
            RESUME_SEED = True  # _setup_sandbox 的 seed 段据此回灌 OSS 抢救产物
            sandbox = await _setup_sandbox(Sandbox(_sandbox_config()), CreateBashSessionRequest)
    finally:
        print("[driver] stopping sandbox")
        try:
            await sandbox.stop()
        except Exception as exc:  # noqa: BLE001 -- 重建路径可能已 stop 过死沙箱
            print(f"[driver] final stop ignored: {exc}", file=sys.stderr)
    if run_error is not None:
        # 结果已落 OSS，现在才把采集失败反映到退出码（否则 job 会
        # 显示 SUCCESSFUL，掩盖一个跑了几十小时才断掉的分片）。
        print(f"[driver] collection did not finish cleanly: {run_error}",
              file=sys.stderr)
        return 3
    return 0


def _print_plan() -> None:
    print("[driver] DRY_RUN=1 — plan only.\n")
    print("ROCK sandbox:")
    print(f"  image   = {ROCK_SANDBOX_IMAGE}")
    print(f"  cluster = {ROCK_CLUSTER}  cpus={ROCK_CPUS}  memory={ROCK_MEMORY}  disk={ROCK_DISK}")
    print(f"  ttl     = {ROCK_AUTO_CLEAR_SECONDS}s  run_timeout={ROCK_RUN_TIMEOUT}s")
    print("Collection:")
    print(f"  workload={DERAIL_WORKLOAD}")
    _model_label = {
        "gpt_5_5": GPT55_MODEL,
        "gpt_5_6_luna": GPT56_LUNA_MODEL,
        "claude_opus_4_8": CLAUDE_OPUS_4_8_MODEL,
        "claude_sonnet_5": CLAUDE_SONNET_5_MODEL,
        "kimi_k3": KIMI_K3_MODEL,
        "kimi_k3_cuabash": KIMI_K3_MODEL,
    }.get(AGENT_ID, "-")
    print(f"  agent={AGENT_ID}  model={_model_label}")
    _protocol_label = (
        "anthropic-messages" if IS_ANTHROPIC_AGENT
        else "openai-chat-completions-scaffold"
        if AGENT_ID in ("kimi_k3", "kimi_k3_cuabash")
        else "openai-responses"
    )
    print(f"  protocol={_protocol_label}"
          f"  pip_deps={AGENT_PIP_DEPS}")
    print(f"  shard={SHARD_FILE}  collection_id={COLLECTION_ID}")
    print(f"  resume_seed={RESUME_SEED}"
          + (f"  rerun_purge={len(_shard_task_ids())} tasks (nonce={RERUN_PURGE_NONCE})"
             if RERUN_PURGE else ""))
    print(f"  formal={FORMAL_COLLECTION}  repeats={REPEATS}  vms={NUM_VMS_OVERRIDE}  "
          f"max_steps={MAX_STEPS}  task_timeout={TASK_TIMEOUT}")
    if IS_PROXY_TOPOLOGY:
        print("Topology: ROCK proxy (agent loop on Nebula; guest+gateway in sandbox)")
        print(f"  bridge api=127.0.0.1:{PROXY_PORT_BASE} apps={PROXY_PORT_BASE + 10}-{PROXY_PORT_BASE + 27}")
        print(f"  lifecycle=127.0.0.1:{PROXY_LIFECYCLE_PORT}  model={OPENCUA_BASE_URLS}")
        print(f"  results {PROXY_RESULTS_LOCAL} -> FUSE {PROXY_RESULTS_OSS_MOUNT}/{COLLECTION_ID}/")
        print(f"  osworld snapshot={OSS_OPENCUA_SNAPSHOT_URI}")
        if AGENT_ID == "evocua_32b":
            print(f"  evocua snapshot={OSS_EVOCUA_SNAPSHOT_URI}")
        print(f"\n  guest prep (sandbox):\n{_proxy_guest_prep_command()}\n")
        print(f"  collect env: {_proxy_collect_env().__class__.__name__} "
              f"(MYPCBENCH_REMOTE_ATTACH=1 PORT_BASE={PROXY_PORT_BASE})")
        if DERAIL_WORKLOAD == "takeover":
            print(
                f"  takeover={TAKEOVER_SOURCE_AGENT}->{TAKEOVER_TARGET_AGENT} "
                f"depth={TAKEOVER_DEPTH} condition={TAKEOVER_CONDITION} "
                f"shard={TAKEOVER_SHARD_OFFSET}/{TAKEOVER_SHARD_COUNT}"
            )
        return
    print(f"  results -> {OSS_RESULTS_URI}/")
    print(f"\n  prep:\n{_prep_command()}\n")
    if RESUME_SEED:
        print(f"  seed:\n{_seed_command()}\n")
    print(f"  run: {_run_command()}\n")
    print(f"  result: {_result_command()}")


def main() -> int:
    problems = _closed_model_problems()
    for problem in problems:
        print(f"[driver] closed-model problem: {problem}", file=sys.stderr)
    if problems:
        return 2
    if DERAIL_WORKLOAD not in {"collection", "takeover", "phase5"}:
        print(f"[driver] ERROR: unknown DERAIL_WORKLOAD={DERAIL_WORKLOAD}", file=sys.stderr)
        return 2
    if DERAIL_WORKLOAD == "collection" and not (REPO / SHARD_FILE).is_file():
        print(f"[driver] ERROR: shard file missing in code package: {REPO / SHARD_FILE}",
              file=sys.stderr)
        return 2
    if RERUN_PURGE:
        # 两个硬断：无 RESUME_SEED 则根本不回灌、清场无处发生（rerun 又会
        # 静默落空）；清单解析/形状校验提前到起跑前（_shard_task_ids 内）。
        if not RESUME_SEED:
            print("[driver] ERROR: RERUN_PURGE=1 必须搭配 RESUME_SEED=1（清场发生"
                  "在 seed 回灌之后；无回灌则 rerun 清单又会被守卫静默跳过）",
                  file=sys.stderr)
            return 2
        purge_ids = _shard_task_ids()
        print(f"[driver] RERUN_PURGE armed: 将强制清场 {len(purge_ids)} 个任务目录"
              f"（nonce={RERUN_PURGE_NONCE}）")
    if DRY_RUN:
        _print_plan()
        return 0
    unset = [name for name, value in (
        ("ROCK_BASE_URL", ROCK_BASE_URL),
        ("ROCK_USER_ID", ROCK_USER_ID),
        ("OSS_ASSETS_URI", OSS_ASSETS_URI),
        ("OSS_HARNESS_URI", OSS_HARNESS_URI),
        ("OSS_RESULTS_ROOT", OSS_RESULTS_ROOT),
    ) if not value.strip() or value.startswith("<")]
    # 这两个默认值也是占位符，但只有对应链路会读，故按开关分别拦。
    if IS_PROXY_TOPOLOGY and "<" in PROXY_RESULTS_OSS_MOUNT:
        unset.append("PROXY_RESULTS_OSS_MOUNT/OSS_PREFIX")
    if IS_OPENCUA_AGENT and "<" in OSS_OPENCUA_SNAPSHOT_URI:
        unset.append("OSS_OPENCUA_SNAPSHOT_URI")
    if AGENT_ID == "evocua_32b" and "<" in OSS_EVOCUA_SNAPSHOT_URI:
        unset.append("OSS_EVOCUA_SNAPSHOT_URI")
    if unset:
        print(f"[driver] ERROR: {', '.join(unset)} 仍是占位符；先 set -a; . <env文件>; "
              "set +a 再真跑（DRY_RUN=1 可免）", file=sys.stderr)
        return 2
    if IS_PROXY_TOPOLOGY:
        return asyncio.run(_run_proxy())
    return asyncio.run(_run())


if __name__ == "__main__":
    raise SystemExit(main())
