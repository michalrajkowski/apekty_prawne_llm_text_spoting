#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

export HOST_UID="$(id -u)"
export HOST_GID="$(id -g)"
export HOST_USER="$(id -un)"

CUDA_VISIBLE_DEVICES=0 NVIDIA_VISIBLE_DEVICES=0 docker compose run --rm apm \
  python -m apm.experiments.global_local_runner \
  --project-root . \
  --model-runs aigc_detector_env3 seqxgpt:gpt2_medium seqxgpt:gpt_j_6b \
  --hc3-splits all_train finance_train medicine_train open_qa_train reddit_eli5_train wiki_csai_train \
  --grid-splits filtered unfiltered \
  --threshold-objective balanced_accuracy \
  --batch-size 8
