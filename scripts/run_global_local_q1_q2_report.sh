#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

export HOST_UID="$(id -u)"
export HOST_GID="$(id -g)"
export HOST_USER="$(id -un)"

CUDA_VISIBLE_DEVICES=0 NVIDIA_VISIBLE_DEVICES=0 docker compose run --rm apm \
  python -m apm.experiments.global_local_q1_q2_report \
  --project-root . \
  "$@"
