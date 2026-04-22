#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

export HOST_UID="$(id -u)"
export HOST_GID="$(id -g)"
export HOST_USER="$(id -un)"

ORIGINAL_ARGS=("$@")
OUTPUT_ROOT="data/interim/splits/hc3"
while [[ $# -gt 0 ]]; do
  case "$1" in
    --output-root)
      if [[ $# -lt 2 ]]; then
        echo "Missing value for --output-root" >&2
        exit 1
      fi
      OUTPUT_ROOT="$2"
      shift 2
      ;;
    *)
      shift
      ;;
  esac
done

OUTPUT_ROOT="${OUTPUT_ROOT#./}"
OUTPUT_PARENT="$(dirname "$OUTPUT_ROOT")"

# Ensure output parent is writable by host user even if created earlier by root-owned containers.
docker compose run --rm --user 0:0 apm sh -lc \
  "mkdir -p '/workspace/$OUTPUT_PARENT' && chown -R $HOST_UID:$HOST_GID '/workspace/$OUTPUT_PARENT'"

CUDA_VISIBLE_DEVICES=0 NVIDIA_VISIBLE_DEVICES=0 docker compose run --rm apm \
  python -m apm.experiments.augmented_hc3_materialize \
  --project-root . \
  "${ORIGINAL_ARGS[@]}"
