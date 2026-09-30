#!/usr/bin/env bash
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

want_dockerfile="$(config_scalar "$MODELS_LOCK" dockerfile_sha256 || true)"
actual_dockerfile="$(sha256sum "$DOCKERFILE" | awk '{print $1}')"
if [[ -n "$want_dockerfile" && "$actual_dockerfile" != "$want_dockerfile" ]]; then
  die "Dockerfile 与 models.lock.yaml 不一致：期望 ${want_dockerfile}，实际 ${actual_dockerfile}
     改了 Dockerfile 就要同步更新 lock 里的 dockerfile_sha256"
fi

context_dir="$(mktemp -d)"
trap 'rm -rf "$context_dir"' EXIT

info "构建 ${IMAGE_TAG}（base digest 由 Dockerfile 的 FROM 固定）"
docker build --pull --file "$DOCKERFILE" --tag "$IMAGE_TAG" "$context_dir"

image_id="$(docker image inspect "$IMAGE_TAG" --format '{{.Id}}')"
info "完成：${IMAGE_TAG}"
info "本机 image ID：${image_id}"
info "这个 ID 因机器而异，属正常；启动脚本校验的是 Dockerfile 哈希与 base layer 前缀"
