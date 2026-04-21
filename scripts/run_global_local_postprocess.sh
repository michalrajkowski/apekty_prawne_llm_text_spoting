#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 <raw_scores_jsonl_path>" >&2
  exit 1
fi

RAW_SCORES_PATH="$1"

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

CUDA_VISIBLE_DEVICES=0 NVIDIA_VISIBLE_DEVICES=0 docker compose run --rm apm \
  python -m apm.experiments.global_local_postprocess \
  --project-root . \
  --raw-scores-path "$RAW_SCORES_PATH" \
  --model-runs aigc_detector_env3 seqxgpt:gpt2_medium seqxgpt:gpt_j_6b \
  --hc3-splits all_train finance_train medicine_train open_qa_train reddit_eli5_train wiki_csai_train \
  --grid-splits filtered unfiltered \
  --threshold-objective balanced_accuracy
