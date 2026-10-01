#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
DOCKERFILE="${SCRIPT_DIR}/evocua-vllm.Dockerfile"
IMAGE_TAG="${EVOCUA_IMAGE_TAG:-recovery/evocua-vllm:0.11.0-transformers4.57.3}"
MODELS_LOCK="${REPO_ROOT}/configs/models.lock.yaml"

die() {
  printf 'error: %s\n' "$*" >&2
  exit 1
}

info() {
  printf '[RECOVERY serving] %s\n' "$*"
}

# shellcheck source=../scripts/lib/collection_config.sh
source "${REPO_ROOT}/scripts/lib/collection_config.sh"

command -v docker >/dev/null 2>&1 || die "docker not found"
[[ -f "$DOCKERFILE" ]] || die "${DOCKERFILE} not found"

want_dockerfile="$(config_scalar "$MODELS_LOCK" dockerfile_sha256 || true)"
actual_dockerfile="$(sha256sum "$DOCKERFILE" | awk '{print $1}')"
if [[ -n "$want_dockerfile" && "$actual_dockerfile" != "$want_dockerfile" ]]; then
  die "Dockerfile does not match models.lock.yaml: expected ${want_dockerfile}, got ${actual_dockerfile}
     If you change the Dockerfile, update dockerfile_sha256 in the lock too"
fi

context_dir="$(mktemp -d)"
trap 'rm -rf "$context_dir"' EXIT

info "Building ${IMAGE_TAG} (base digest pinned by the Dockerfile FROM)"
docker build --pull --file "$DOCKERFILE" --tag "$IMAGE_TAG" "$context_dir"

image_id="$(docker image inspect "$IMAGE_TAG" --format '{{.Id}}')"
info "Done: ${IMAGE_TAG}"
info "Local image ID: ${image_id}"
info "This ID varies by machine, which is expected; the launch script checks the Dockerfile hash and base layer prefix"
