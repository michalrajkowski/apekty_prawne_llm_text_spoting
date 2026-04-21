#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

CUDA_VISIBLE_DEVICES=0 NVIDIA_VISIBLE_DEVICES=0 docker compose run --rm apm \
  python -m apm.experiments.split_materialize \
  --project-root . \
  --datasets hc3:all_train hc3:finance_train hc3:medicine_train hc3:open_qa_train hc3:reddit_eli5_train hc3:wiki_csai_train grid:filtered grid:unfiltered \
  --train-ratio 0.5 \
  --seed 42

CUDA_VISIBLE_DEVICES=0 NVIDIA_VISIBLE_DEVICES=0 docker compose run --rm apm \
  python -m apm.experiments.global_local_runner \
  --project-root . \
  --model-runs aigc_detector_env3 seqxgpt:gpt2_medium seqxgpt:gpt_j_6b \
  --hc3-splits all_train finance_train medicine_train open_qa_train reddit_eli5_train wiki_csai_train \
  --grid-splits filtered unfiltered \
  --threshold-objective balanced_accuracy \
  --batch-size 8
