#!/usr/bin/env bash
# 构建 EvoCUA 的 vLLM runtime image。它不在任何 registry 上，所以每台机器都要自己建。
#
# 可复现的是配方，不是产物：Dockerfile 从一个固定 digest 的 base FROM 出来，只多装
# 一个钉死版本的 transformers；但 `docker build` 出来的 image ID 由本机构建过程决定，
# 换台机器必然不同。00_tmux_serve_open_source.sh 因此校验 Dockerfile 哈希与 base
# layer 前缀，而不是 image ID。

set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
DOCKERFILE="${SCRIPT_DIR}/evocua-vllm.Dockerfile"
IMAGE_TAG="${EVOCUA_IMAGE_TAG:-derail/evocua-vllm:0.11.0-transformers4.57.3}"
MODELS_LOCK="${REPO_ROOT}/configs/models.lock.yaml"

die() {
  printf '错误：%s\n' "$*" >&2
  exit 1
}

info() {
  printf '[DERAIL serving] %s\n' "$*"
}

# shellcheck source=../scripts/lib/collection_config.sh
source "${REPO_ROOT}/scripts/lib/collection_config.sh"

command -v docker >/dev/null 2>&1 || die "找不到 docker"
[[ -f "$DOCKERFILE" ]] || die "找不到 ${DOCKERFILE}"

# 先确认手上的配方就是 lock 记的那份，免得建出一个校验不过的镜像。
want_dockerfile="$(config_scalar "$MODELS_LOCK" dockerfile_sha256 || true)"
actual_dockerfile="$(sha256sum "$DOCKERFILE" | awk '{print $1}')"
if [[ -n "$want_dockerfile" && "$actual_dockerfile" != "$want_dockerfile" ]]; then
  die "Dockerfile 与 models.lock.yaml 不一致：期望 ${want_dockerfile}，实际 ${actual_dockerfile}
     改了 Dockerfile 就要同步更新 lock 里的 dockerfile_sha256"
fi

# base 用 digest 钉死在 FROM 里；--pull 只是让缺失时去拉，digest 决定拉到什么。
# build context 用一个空目录：Dockerfile 里没有 COPY/ADD，不需要把仓库送进 daemon。
context_dir="$(mktemp -d)"
trap 'rm -rf "$context_dir"' EXIT

info "构建 ${IMAGE_TAG}（base digest 由 Dockerfile 的 FROM 固定）"
docker build --pull --file "$DOCKERFILE" --tag "$IMAGE_TAG" "$context_dir"

image_id="$(docker image inspect "$IMAGE_TAG" --format '{{.Id}}')"
info "完成：${IMAGE_TAG}"
info "本机 image ID：${image_id}"
info "这个 ID 因机器而异，属正常；启动脚本校验的是 Dockerfile 哈希与 base layer 前缀"
