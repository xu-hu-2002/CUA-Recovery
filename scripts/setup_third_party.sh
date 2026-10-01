#!/usr/bin/env bash

set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

die() {
  printf 'error: %s\n' "$*" >&2
  exit 1
}

info() {
  printf '[RECOVERY setup] %s\n' "$*"
}

# shellcheck source=lib/third_party.sh
source "${SCRIPT_DIR}/lib/third_party.sh"
third_party_paths "$REPO_ROOT"

TARGETS=()
for arg in "$@"; do
  case "$arg" in
    --all) TARGETS=(mypcbench evocua opencua) ;;
    mypcbench|evocua|opencua) TARGETS+=("$arg") ;;
    -h|--help)
      sed -n '2,20p' "${BASH_SOURCE[0]}"
      exit 0
      ;;
    *) die "Unknown argument: ${arg} (valid: mypcbench / evocua / opencua / --all)" ;;
  esac
done
(( ${#TARGETS[@]} )) || TARGETS=(mypcbench)

command -v git >/dev/null 2>&1 || die "git not found"

for target in "${TARGETS[@]}"; do
  case "$target" in
    mypcbench)
      setup_mypcbench
      setup_mypcbench_judge
      ;;
    evocua) setup_evocua ;;
    opencua) setup_opencua_osworld ;;
  esac
done

info "third_party ready: $(printf '%s ' "${TARGETS[@]}")"

if [[ " ${TARGETS[*]} " == *" mypcbench "* && ! -f "${MYPCBENCH_ROOT}/mypcbench-vm/mypcbench.qcow2" ]]; then
  cat <<EOF

VM image still missing (about 16.5G, only needed for rollouts):
  bash ${MYPCBENCH_ROOT}/scripts/get-eval-image.sh --out ${MYPCBENCH_ROOT}/mypcbench-vm

The image needs KVM. Check whether this host supports it:
  [ -r /dev/kvm ] && [ -w /dev/kvm ] && echo "qemu backend available" || echo "use the docker backend"
If unavailable, set backend to docker in configs/collection/mypcbench_runtime.yaml.
EOF
fi
