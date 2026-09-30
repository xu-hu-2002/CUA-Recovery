#!/usr/bin/env bash
# =============================================================================
# DERAIL opencua_72b smoke — Nebula job entry（开源模型形态）。
#
# 与 closed-model entry 的区别：job 内起 vLLM（L1 推理），加载 OSS FUSE 挂载上
# 的 xlangai/OpenCUA-72B 权重；ROCK 侧链路不变（derail_rock_driver.py），
# 沙箱内 01 脚本经 OPENCUA_BASE_URLS=http://<JOB_IP>:<port>/v1 采样。
# 该拓扑在 MCUA 项目同一 AMD MI308X 队列上对 opencua-72b 已有 OSWorld 出分实证。
#
# 关键移植点（全部来自 MCUA entry_rock_nebula.sh 的踩坑记录）：
#   * 权重 shim：OSS 挂载目录名与模型别名不一致，用 symlink 映射，不拷 137G；
#   * transformers 必须 4.x：OpenCUA remote code 用 4.x API（5.x 删了
#     bytes_to_unicode / ProcessorMixin 改版，vLLM 启动即崩）；
#   * vLLM 无该架构原生实现时加 --model-impl transformers（ROCm build 不动）；
#   * ready 判定必须校验 served-model-name，防同 netns 别人端口的假阳性；
#   * 退出清理要杀整个进程组：TP>1 的 worker 残留显存会让平台判 job 失败。
#
# SMOKE_PHASE：
#   probe     只验证「OSS 权重能加载 + 采样能返回合法 GUI 动作」（默认）；
#   full      探针过后 exec derail_rock_driver.py 走 ROCK 单任务 smoke。
# =============================================================================
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/../.." && pwd)"
cd "$REPO"
export PYTHONPATH="$REPO${PYTHONPATH:+:$PYTHONPATH}"

# --- source 运行配置（submit_derail_opencua_smoke.sh 生成） ---------------------
if [[ -f "$REPO/.opencua_run.env" ]]; then
  set -a; # shellcheck disable=SC1091
  source "$REPO/.opencua_run.env"; set +a
  echo "[opencua-smoke] sourced run-config .opencua_run.env"
else
  echo "[opencua-smoke] ERROR: .opencua_run.env missing (submit script writes it)" >&2
  exit 2
fi

SMOKE_PHASE="${SMOKE_PHASE:-probe}"
OPENCUA_MODEL="${OPENCUA_MODEL:-opencua-72b}"
SERVE_PORT="${SERVE_PORT:-8000}"
SERVE_READY_TIMEOUT="${SERVE_READY_TIMEOUT:-3600}"
OPENCUA_WEIGHTS_OSS_DIR="${OPENCUA_WEIGHTS_OSS_DIR:-/data/oss_bucket_0/${OSS_PREFIX:-<oss-prefix>}/MCUA/OpenCUA-72B}"
LOCAL_MODEL_CACHE_DIR="${LOCAL_MODEL_CACHE_DIR:-}"
PROBE_EVIDENCE_DIR="${PROBE_EVIDENCE_DIR:-${OSS_SMOKE_EVIDENCE_DIR:-/data/oss_bucket_0/${OSS_PREFIX:-<oss-prefix>}/DERAIL/results/smoke/opencua72b}}"

# --- source 0600 mount-secret 并自删（同 closed-model entry 通道） --------------
if [[ -n "${DERAIL_SECRET_MOUNT:-}" && -f "${DERAIL_SECRET_MOUNT}" ]]; then
  set -a; # shellcheck disable=SC1090
  source "${DERAIL_SECRET_MOUNT}"; set +a
  rm -f "${DERAIL_SECRET_MOUNT}" || true
  echo "[opencua-smoke] secrets loaded from staged mount file (and removed)"
else
  echo "[opencua-smoke] ERROR: DERAIL_SECRET_MOUNT not found: ${DERAIL_SECRET_MOUNT:-<unset>}" >&2
  exit 2
fi
export OSS_ACCESS_ID="${OSS_ACCESS_ID:-${OSS_ID:-${OSS_ACCESS_KEY_ID:-}}}"
export OSS_ACCESS_KEY="${OSS_ACCESS_KEY:-${OSS_KEY:-${OSS_ACCESS_KEY_SECRET:-}}}"

# --- python 解释器 ---------------------------------------------------------------
PY="${DERAIL_ENTRY_PYTHON:-python3}"
if ! command -v "$PY" >/dev/null 2>&1; then
  for _p in python python3.12 python3.11 python3.10; do
    if command -v "$_p" >/dev/null 2>&1; then PY="$_p"; break; fi
  done
fi

JOB_IP="$(hostname -i 2>/dev/null | awk '{print $1}')"
[[ -z "$JOB_IP" ]] && JOB_IP="$(hostname -I 2>/dev/null | awk '{print $1}')"
[[ -z "$JOB_IP" ]] && JOB_IP="127.0.0.1"
echo "[opencua-smoke] job_ip=${JOB_IP} phase=${SMOKE_PHASE}"

# --- 0b. 前置 fail-fast：rl-rock SDK（深度 import 探针 + 3 次重试） -------------------
# 必须在 75min 权重加载之前探：SDK 装不上的话 full 阶段必死，早探早重提，不烧权重加载。
# 镜像内 SDK 可能半安装/缺失（full 首跑即撞 ModuleNotFoundError: No module named 'rock'）。
if [[ "$SMOKE_PHASE" == "full" || "${SANDBOX_REACH_PROBE:-1}" == "1" ]]; then
  ROCK_SDK_PROBE='from rock.sdk.sandbox.config import SandboxConfig; from rock.actions import CreateBashSessionRequest; from rock.sdk.sandbox.client import Sandbox'
  rock_sdk_ok() { "$PY" -c "$ROCK_SDK_PROBE" 2>/dev/null; }
  if ! rock_sdk_ok; then
    for attempt in 1 2 3; do
      echo "[opencua-smoke] ROCK SDK not importable -- install attempt ${attempt}/3 with $PY -m pip"
      extra=""
      [ "$attempt" -gt 1 ] && extra="--force-reinstall --no-cache-dir"
      _pip_index=""; [ -n "${PIP_INDEX_URL:-}" ] && _pip_index="-i $PIP_INDEX_URL"
      # Install only the SDK wheel; never let its dependency resolver mutate the
      # vLLM/transformers/protobuf runtime that was baked into the production image.
      "$PY" -m pip install -q --no-deps $extra "rl-rock" $_pip_index \
        || "$PY" -m pip install -q --no-deps $extra "rock-rl" $_pip_index \
        || "$PY" -m pip install -q --no-deps $extra "rl-rock" \
        || "$PY" -m pip install -q --no-deps $extra "rock-rl" \
        || echo "[opencua-smoke] pip install attempt ${attempt} returned non-zero"
      rock_sdk_ok && { echo "[opencua-smoke] ROCK SDK deep import OK after attempt ${attempt}"; break; }
      [ "$attempt" -eq 3 ] && {
        if [[ "$SMOKE_PHASE" == "full" ]]; then
          echo "[opencua-smoke] ERROR: ROCK SDK still not importable after 3 attempts (fail-fast，不烧权重加载)" >&2
          "$PY" -c "$ROCK_SDK_PROBE" 2>&1 | tail -5 >&2
          exit 1
        fi
        echo "[opencua-smoke] WARNING: ROCK SDK 装不上；稍后跳过沙箱探针（不阻塞 probe 阶段）" >&2
      }
      sleep 20
    done
  fi
  rock_sdk_ok && echo "[opencua-smoke] ROCK SDK ready: $("$PY" -c 'import rock; print(getattr(rock,"__version__","?"))' 2>/dev/null || echo '?')"
fi

# --- 1. 权重 shim：OSS 挂载路径 -> vLLM 模型路径 ----------------------------------
[[ -d "$OPENCUA_WEIGHTS_OSS_DIR" ]] || {
  echo "[opencua-smoke] ERROR: weights dir missing on OSS mount: $OPENCUA_WEIGHTS_OSS_DIR" >&2
  exit 2; }
[[ -f "$OPENCUA_WEIGHTS_OSS_DIR/config.json" ]] || {
  echo "[opencua-smoke] ERROR: config.json absent under $OPENCUA_WEIGHTS_OSS_DIR" >&2
  exit 2; }
if [[ -n "$LOCAL_MODEL_CACHE_DIR" ]]; then
  mkdir -p "$LOCAL_MODEL_CACHE_DIR"
  if [[ ! -f "$LOCAL_MODEL_CACHE_DIR/.derail_model_ready" ]]; then
    echo "[opencua-smoke] warming model to local cache: $LOCAL_MODEL_CACHE_DIR"
    # Read independent safetensors shards concurrently from OSS/FUSE.  The
    # marker is written only after every shard and metadata file are present.
    warm_workers="${MODEL_WARM_WORKERS:-4}"
    [[ "$warm_workers" =~ ^[1-9][0-9]*$ ]] || {
      echo "[opencua-smoke] invalid MODEL_WARM_WORKERS: $warm_workers" >&2
      exit 2
    }
    warm_pids=()
    while IFS= read -r -d '' src; do
      cp -f "$src" "$LOCAL_MODEL_CACHE_DIR/$(basename "$src")" &
      warm_pids+=("$!")
      if (( ${#warm_pids[@]} >= warm_workers )); then
        for pid in "${warm_pids[@]}"; do wait "$pid"; done
        warm_pids=()
      fi
    done < <(find "$OPENCUA_WEIGHTS_OSS_DIR" -maxdepth 1 -type f \
      \( -name '*.safetensors' -o -name '*.json' -o -name '*.jinja' -o -name '*.model' \) \
      -print0)
    for pid in "${warm_pids[@]}"; do wait "$pid"; done
    [[ -f "$LOCAL_MODEL_CACHE_DIR/config.json" ]] || { echo "[opencua-smoke] local cache missing config.json" >&2; exit 2; }
    _missing=0
    while IFS= read -r -d '' _src; do
      [[ -f "$LOCAL_MODEL_CACHE_DIR/$(basename "$_src")" ]] || { _missing=1; break; }
    done < <(find "$OPENCUA_WEIGHTS_OSS_DIR" -maxdepth 1 -type f -name '*.safetensors' -print0)
    [[ "$_missing" -eq 0 ]] || { echo "[opencua-smoke] local cache missing safetensors shard" >&2; exit 2; }
    touch "$LOCAL_MODEL_CACHE_DIR/.derail_model_ready"
  fi
  OPENCUA_WEIGHTS_OSS_DIR="$LOCAL_MODEL_CACHE_DIR"
  echo "[opencua-smoke] using local model cache: $LOCAL_MODEL_CACHE_DIR"
fi
LINK_ROOT="${WEIGHTS_LINK_ROOT:-/tmp/derail-weights}"
mkdir -p "$LINK_ROOT"
MODEL_LINK_NAME="${OPENCUA_MODEL//\//--}"
ln -sfn "$OPENCUA_WEIGHTS_OSS_DIR" "${LINK_ROOT}/${MODEL_LINK_NAME}"
MODEL_PATH="${LINK_ROOT}/${MODEL_LINK_NAME}"
echo "[opencua-smoke] weights shim: ${MODEL_PATH} -> ${OPENCUA_WEIGHTS_OSS_DIR}"

# --- 2. opencua transformers pin（4.x；vLLM 0.17 要求 <5） ------------------------
_tf_cur="$("$PY" -c 'import transformers; print(transformers.__version__)' 2>/dev/null || echo unknown)"
if [[ "${AGENT_ID:-opencua_72b}" == "opencua_72b" ]]; then
  case "${_tf_cur}" in
    4.*) echo "[opencua-smoke] transformers ${_tf_cur} already 4.x" ;;
    *)
      echo "[opencua-smoke] downgrading transformers ${_tf_cur} -> 4.56.2 for opencua"
      "$PY" -m pip install -q "transformers==4.56.2" -i "${PIP_INDEX_URL:-https://pypi.org/simple}" 2>&1 | tail -3 \
        || "$PY" -m pip install -q "transformers<5" -i "${PIP_INDEX_URL:-https://pypi.org/simple}" 2>&1 | tail -3 \
        || echo "[opencua-smoke] WARNING: transformers downgrade failed" >&2
      ;;
  esac
fi
echo "[opencua-smoke] runtime: vllm=$("$PY" -c 'import vllm; print(vllm.__version__)' 2>/dev/null || echo '?') \
transformers=$("$PY" -c 'import transformers; print(transformers.__version__)' 2>/dev/null || echo '?')"

# --- 3. 架构守卫：vLLM 无原生实现 -> --model-impl transformers ---------------------
VLLM_EXTRA_ARGS="${VLLM_EXTRA_ARGS:-}"
if [[ -z "$VLLM_EXTRA_ARGS" ]]; then
  _impl="$("$PY" - <<PYEOF 2>&1
import importlib, json, sys
try:
    archs = json.load(open('${MODEL_PATH}/config.json')).get('architectures') or []
except Exception as e:
    archs = []; print(f'DIAG config unreadable: {e}', file=sys.stderr)
supported, err = set(), None
for mod in ('vllm.model_executor.models.registry', 'vllm.model_executor.models'):
    try:
        supported = set(getattr(importlib.import_module(mod), 'ModelRegistry').get_supported_archs())
        break
    except Exception as e:
        err = f'{mod}: {type(e).__name__}: {e}'
print(f'DIAG archs={archs} n_supported={len(supported)} err={err}', file=sys.stderr)
if archs and supported and not any(a in supported for a in archs):
    print('transformers')
PYEOF
)"
  echo "[opencua-smoke] arch probe: $(echo "$_impl" | tr '\n' ' ')"
  if grep -q '^transformers$' <<<"$_impl"; then
    VLLM_EXTRA_ARGS="--model-impl transformers"
    echo "[opencua-smoke] vLLM 无该架构原生实现 -> --model-impl transformers"
  fi
fi

# --- 4. GPU 数与 TP ---------------------------------------------------------------
GPU_COUNT="$(DERAIL_GPU_COUNT="${DERAIL_GPU_COUNT:-}" "$PY" - <<'PYEOF' 2>/dev/null || echo 0
import os
override = os.environ.get('DERAIL_GPU_COUNT', '').strip()
if override:
    print(int(override)); raise SystemExit
try:
    import torch
    print(torch.cuda.device_count()); raise SystemExit
except Exception:
    pass
try:
    import subprocess
    out = subprocess.run(['rocm-smi', '--showid'], capture_output=True, text=True, timeout=20)
    print(sum(1 for line in out.stdout.splitlines() if line.strip().startswith('GPU')))
except Exception:
    print(8)
PYEOF
)"
[[ "$GPU_COUNT" =~ ^[1-9][0-9]*$ ]] || GPU_COUNT=8
echo "[opencua-smoke] GPU_COUNT=${GPU_COUNT} (TP=${GPU_COUNT})"

# --- 5. 端口防碰撞（同 netns 可能有别人的 vLLM） ------------------------------------
if curl -s --max-time 3 "http://127.0.0.1:${SERVE_PORT}/v1/models" 2>/dev/null | grep -q '"data"'; then
  _busy="$SERVE_PORT"
  SERVE_PORT="$("$PY" -c 'import socket; s=socket.socket(); s.bind(("127.0.0.1",0)); print(s.getsockname()[1]); s.close()' 2>/dev/null || echo 8123)"
  echo "[opencua-smoke] WARNING: port ${_busy} busy in this netns -> switching to ${SERVE_PORT}"
fi

# --- 6. 起 vLLM（bind 0.0.0.0：沙箱内 runner 要经 JOB_IP 访问） --------------------
VLLM_LOG=/tmp/derail_opencua_vllm.log
"$PY" -m vllm.entrypoints.openai.api_server \
  --model "$MODEL_PATH" \
  --served-model-name "$OPENCUA_MODEL" \
  --trust-remote-code \
  --tensor-parallel-size "$GPU_COUNT" \
  --host 0.0.0.0 --port "$SERVE_PORT" \
  $VLLM_EXTRA_ARGS \
  >"$VLLM_LOG" 2>&1 &
VLLM_PID=$!
# TP>1 会 fork worker 且持续占显存；平台对“显存未释放”判 job 失败，必须杀进程组。
_vllm_cleanup() {
  kill -TERM -"$VLLM_PID" 2>/dev/null || kill -TERM "$VLLM_PID" 2>/dev/null || true
  for _ in 1 2 3 4 5 6 7 8 9 10; do
    kill -0 "$VLLM_PID" 2>/dev/null || break
    sleep 2
  done
  kill -KILL -"$VLLM_PID" 2>/dev/null || kill -KILL "$VLLM_PID" 2>/dev/null || true
  pkill -KILL -f 'vllm.entrypoints.openai.api_server' 2>/dev/null || true
}
trap _vllm_cleanup EXIT

deadline=$(( $(date +%s) + SERVE_READY_TIMEOUT ))
next_log=$(( $(date +%s) + 60 ))
until curl -s "http://127.0.0.1:${SERVE_PORT}/v1/models" 2>/dev/null | grep -q "\"${OPENCUA_MODEL}\""; do
  if ! kill -0 "$VLLM_PID" 2>/dev/null; then
    echo "[opencua-smoke] ERROR: vLLM exited early; log tail:" >&2
    tail -80 "$VLLM_LOG" >&2 || true
    exit 1
  fi
  now=$(date +%s)
  if (( now >= next_log )); then
    echo "[opencua-smoke] vLLM still starting; latest:"
    tail -8 "$VLLM_LOG" 2>/dev/null | sed 's/^/[vllm] /' || true
    next_log=$(( now + 60 ))
  fi
  if (( now >= deadline )); then
    echo "[opencua-smoke] ERROR: vLLM not ready in ${SERVE_READY_TIMEOUT}s; log tail:" >&2
    tail -80 "$VLLM_LOG" >&2 || true
    exit 1
  fi
  sleep 5
done
echo "[opencua-smoke] vLLM ready: served=${OPENCUA_MODEL} port=${SERVE_PORT} job_ip=${JOB_IP}"

# --- 7. 采样探针：文本 + 图片各一条，证据落 OSS -------------------------------------
mkdir -p "$PROBE_EVIDENCE_DIR"
OPENCUA_BASE_URL="http://127.0.0.1:${SERVE_PORT}/v1" \
OPENCUA_PROBE_MODEL="$OPENCUA_MODEL" \
OPENCUA_PROBE_AGENT="${AGENT_ID:-opencua_72b}" \
OPENCUA_PROBE_OUT="$PROBE_EVIDENCE_DIR/probe_$(date -u +%Y%m%dT%H%M%SZ).json" \
"$PY" - <<'PYEOF'
import base64, io, json, os, struct, sys, time, urllib.error, urllib.request, zlib

base = os.environ["OPENCUA_BASE_URL"]
model = os.environ["OPENCUA_PROBE_MODEL"]
agent = os.environ["OPENCUA_PROBE_AGENT"]
out_path = os.environ["OPENCUA_PROBE_OUT"]

def post(payload, timeout=600):
    req = urllib.request.Request(
        base.rstrip("/") + "/chat/completions",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json", "Authorization": "Bearer EMPTY"},
    )
    t0 = time.time()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            body = json.load(resp)
    except urllib.error.HTTPError as error:
        response_body = error.read().decode("utf-8", errors="replace")[:2000]
        raise RuntimeError(
            f"chat completions probe failed: HTTP {error.code}: {response_body}"
        ) from error
    return body, time.time() - t0

def tiny_png(rgb):
    # 32x32 纯色 PNG：只依赖标准库，避免探针引入第三方依赖。
    def chunk(tag, data):
        block = tag + data
        return struct.pack(">I", len(data)) + block + struct.pack(">I", zlib.crc32(block))
    raw = b"".join(b"\x00" + bytes(rgb) * 32 for _ in range(32))
    png = (b"\x89PNG\r\n\x1a\n"
           + chunk(b"IHDR", struct.pack(">IIBBBBB", 32, 32, 8, 2, 0, 0, 0))
           + chunk(b"IDAT", zlib.compress(raw))
           + chunk(b"IEND", b""))
    return base64.b64encode(png).decode()

record = {"model": model, "base_url": base, "probes": []}

if agent in {"qwen3_8_27b", "qwen3_5_35b_a3b"}:
    tokenize_payload = json.dumps({"model": model, "prompt": "DERAIL_TOKENIZE_PROBE"}).encode()
    tokenize_error = None
    for tokenize_url in (base.removesuffix("/v1") + "/tokenize", base + "/tokenize"):
        try:
            request = urllib.request.Request(
                tokenize_url,
                data=tokenize_payload,
                headers={"Content-Type": "application/json", "Authorization": "Bearer EMPTY"},
            )
            with urllib.request.urlopen(request, timeout=120) as response:
                tokenize_body = json.load(response)
            count = tokenize_body.get("count")
            if count is None:
                count = len(tokenize_body.get("tokens") or [])
            record["probes"].append({
                "name": "tokenize", "url": tokenize_url, "count": count, "ok": count > 0,
            })
            break
        except Exception as error:
            tokenize_error = f"{type(error).__name__}: {error}"
    else:
        record["probes"].append({
            "name": "tokenize", "ok": False, "error": tokenize_error,
        })

text_max_tokens = 4096 if agent == "qwen3_5_35b_a3b" else 256
body, dt = post({
    "model": model,
    "messages": [{"role": "user", "content":
        "Reply with exactly one line: OPENCUA_PROBE_OK"}],
    "max_tokens": text_max_tokens, "temperature": 0.0,
})
text_message = body["choices"][0]["message"]
text_out = text_message.get("content") or ""
record["probes"].append({
    "name": "text_only", "latency_s": round(dt, 1), "content": text_out,
    "reasoning_content": text_message.get("reasoning_content"),
    "finish_reason": body["choices"][0].get("finish_reason"),
    "usage": body.get("usage"),
    "requested_max_tokens": text_max_tokens,
    "ok": "OPENCUA_PROBE_OK" in text_out,
})

if agent in {"qwen3_8_27b", "qwen3_5_35b_a3b"}:
    body, dt = post({
        "model": model,
        "messages": [{"role": "user", "content": "Call the click tool at x=16, y=16."}],
        "tools": [{
            "type": "function",
            "function": {
                "name": "click",
                "description": "Click a desktop coordinate.",
                "parameters": {
                    "type": "object",
                    "properties": {"x": {"type": "integer"}, "y": {"type": "integer"}},
                    "required": ["x", "y"],
                },
            },
        }],
        "tool_choice": "required", "max_tokens": 128, "temperature": 0.0,
    })
    message = body["choices"][0]["message"]
    tool_calls = message.get("tool_calls") or []
    record["probes"].append({
        "name": "structured_tool_call", "latency_s": round(dt, 1),
        "tool_calls": tool_calls, "ok": bool(tool_calls),
    })

img = tiny_png((200, 30, 30))
body, dt = post({
    "model": model,
    "messages": [{
        "role": "user",
        "content": [
            {"type": "text", "text":
                "You are operating a 1280x800 desktop. This is a screenshot. "
                "Output ONE PyAutoGUI action line to click near the center of the screen."},
            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img}"}},
        ],
    }],
    "max_tokens": 256, "temperature": 0.0,
})
gui_message = body["choices"][0]["message"]
gui_out = gui_message.get("content") or ""
looks_like_action = any(k in gui_out for k in ("pyautogui", "click", "moveTo"))
record["probes"].append({
    "name": "multimodal_action", "latency_s": round(dt, 1), "content": gui_out,
    "reasoning_content": gui_message.get("reasoning_content"),
    "ok": bool(gui_out and looks_like_action),
})

record["verdict"] = "PASS" if all(p["ok"] for p in record["probes"]) else "PARTIAL"
os.makedirs(os.path.dirname(out_path), exist_ok=True)
with open(out_path, "w") as fh:
    json.dump(record, fh, ensure_ascii=False, indent=2)
print(f"[opencua-smoke] PROBE_VERDICT={record['verdict']} -> {out_path}")
for p in record["probes"]:
    preview = (p.get("content") or json.dumps(p.get("tool_calls") or []))[:160]
    preview = preview.replace("\n", " | ")
    print(
        f"[opencua-smoke]   {p['name']}: ok={p['ok']} "
        f"latency={p.get('latency_s', '-')}s :: {preview}"
    )
PYEOF

# --- 8. 可选：沙箱 -> job vLLM 可达性探针（legacy 拓扑遗留；proxy 拓扑下
#         vLLM 走 localhost，探针无意义，默认关） ------------------------------
if [[ "${SANDBOX_REACH_PROBE:-0}" == "1" ]]; then
  echo "[opencua-smoke] sandbox reachability probe: creating minimal ROCK sandbox"
  OPENCUA_PROBE_URL="http://${JOB_IP}:${SERVE_PORT}/v1/models" \
  "$PY" - "$REPO" <<'PYEOF' || echo "[opencua-smoke] WARNING: sandbox reach probe failed (不阻塞 probe 阶段)"
import asyncio, os, sys
sys.path.insert(0, sys.argv[1])

async def main():
    from rock.actions import CreateBashSessionRequest
    from rock.sdk.sandbox.client import Sandbox
    from rock.sdk.sandbox.config import SandboxConfig
    kwargs = dict(
        base_url=os.environ["ROCK_BASE_URL"],
        image=os.environ["ROCK_SANDBOX_IMAGE"],
        auto_clear_seconds=1800, startup_timeout=600,
        memory="2g", cpus=1.0, disk="10g",
        cluster=os.environ["ROCK_CLUSTER"],
    )
    api_key = os.environ.get("ROCK_API_KEY", "")
    if api_key:
        kwargs["extra_headers"] = {"XRL-Authorization": f"Bearer {api_key}"}
        kwargs["user_id"] = os.environ.get("ROCK_USER_ID", "")
        kwargs["experiment_id"] = os.environ.get("ROCK_EXPERIMENT_ID", "derail-mypcbench")
    sandbox = Sandbox(SandboxConfig(**kwargs))
    await sandbox.start()
    try:
        await sandbox.create_session(CreateBashSessionRequest(session="default"))
        url = os.environ["OPENCUA_PROBE_URL"]
        out = await sandbox.arun(
            cmd=(f"curl -s --max-time 15 -o /tmp/reach.json -w 'http=%{{http_code}}' '{url}' "
                 f"> /tmp/reach.code 2>&1; cat /tmp/reach.code; echo; head -c 300 /tmp/reach.json"),
            session="default", wait_timeout=120)
        print(f"[opencua-smoke] sandbox->vLLM: {getattr(out, 'output', out)}")
    finally:
        await sandbox.stop()

asyncio.run(main())
PYEOF
fi

# --- 9. 分流 -----------------------------------------------------------------------
if [[ "$SMOKE_PHASE" == "probe" ]]; then
  echo "[opencua-smoke] SMOKE_PHASE=probe complete（vLLM ready + 采样探针完成）"
  exit 0
fi

# full 阶段：ROCK proxy 拓扑（仿 MCUA OSWorld）——沙箱回连 Nebula pod IP 不可达
#（2026-08-15 full 首撞实证），改为 agent loop 上 Nebula：模型调用走 localhost
# vLLM，截图/动作经 driver 内 bridge 打进沙箱（见 derail_rock_driver.py 头注）。
export DERAIL_ROCK_TOPOLOGY="proxy"
export OPENCUA_BASE_URLS="http://127.0.0.1:${SERVE_PORT}/v1"
export QWEN38_BASE_URLS="$OPENCUA_BASE_URLS"
export QWEN35_BASE_URLS="$OPENCUA_BASE_URLS"
export EVOCUA_BASE_URLS="$OPENCUA_BASE_URLS"
export QWEN38_MODEL="$OPENCUA_MODEL"
export QWEN35_MODEL="$OPENCUA_MODEL"
export EVOCUA_MODEL="$OPENCUA_MODEL"
export TAKEOVER_TOKENIZE_BASE_URL="${TAKEOVER_TOKENIZE_BASE_URL:-$OPENCUA_BASE_URLS}"
# Open-model jobs share the same ROCK proxy driver and need the same per-job
# localhost isolation as hosted-model jobs.
# shellcheck disable=SC1091
source "$HERE/allocate_proxy_ports.sh"
echo "[opencua-smoke] SMOKE_PHASE=full topology=proxy -> OPENCUA_BASE_URLS=${OPENCUA_BASE_URLS}"
echo "[opencua-smoke] launching derail_rock_driver.py (AGENT_ID=${AGENT_ID:-opencua_72b})"
"$PY" -u "$REPO/scripts/rock/derail_rock_driver.py"
rc=$?
echo "[opencua-smoke] driver exited rc=${rc}; cleaning up vLLM via EXIT trap"
exit "$rc"
