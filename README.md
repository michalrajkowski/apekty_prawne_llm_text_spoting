# Aspekty prawne - 

## Architecture

Current repository structure and architecture decisions are documented in `ARCHITECTURE.md`.

## Submodule Initialization

This repository vendors detector implementations as Git submodules under `external/detectors/`.

Clone with submodules initialized:

```bash
git clone --recurse-submodules <repo-url>
```

If you already cloned without `--recurse-submodules`, run:

```bash
./scripts/init_submodules.sh
```

## Reproducible Docker Runtime (GPU)

Recommended runtime is Docker with GPU support and pinned dependencies.

What is pinned:
- Python image: `3.13.2`
- direct dependencies: `requirements.txt` (exact versions)
- full lock: `requirements.lock.txt` (transitive freeze used by Docker build)

Day-to-day code/config edits do not require rebuilding because the whole repository is bind-mounted into the container.
Rebuild only when `Dockerfile`/`requirements*.txt` change.

Build and open shell:

```bash
docker compose build apm
docker compose run --rm apm
```

Or via `make` wrappers:

```bash
make docker-build
make docker-shell
```

Inside container, `PYTHONPATH=/workspace/src` is set automatically.
Docker runtime is pinned to `GPU:0` (`CUDA_VISIBLE_DEVICES=0`, `NVIDIA_VISIBLE_DEVICES=0`) so `GPU:1` stays free.

### Kaggle Credentials in Docker

Create local credential file (gitignored) and keep it in repository root under `.kaggle/`:

```bash
mkdir -p .kaggle
cp configs/credentials/kaggle.example.json .kaggle/kaggle.json
chmod 600 .kaggle/kaggle.json
```

### Initialize All Datasets (Docker)

```bash
docker compose run --rm apm \
  python -m apm.data.materialize_all \
  --project-root . \
  --config-dir configs/datasets \
  --sample-size 100 \
  --seed 42
```

or:

```bash
make docker-materialize-all
```

### Run Detector Scoring Experiment (Docker)

```bash
docker compose run --rm apm \
  python scratch/detector_scoring/run_detector_scores.py \
  --project-root . \
  --examples-per-label 30 \
  --model-runs aigc_detector_env3 seqxgpt:gpt2_medium seqxgpt:gpt_j_6b

docker compose run --rm apm \
  python scratch/detector_scoring/summarize_scores.py \
  --project-root .

docker compose run --rm apm \
  python scratch/detector_scoring/plot_detector_scores.py \
  --project-root .
```

or:

```bash
make docker-score
make docker-summarize
make docker-plot
```

`plot_detector_scores.py` now creates separate PNG plots for each model under
`scratch/detector_scoring/results/by_model/`, with bars ordered as first 30 human samples and then 30 ai samples.
`run_detector_scores.py` now defaults to HC3 + GriD sources (Kaggle skipped) and these three run ids:
`aigc_detector_env3`, `seqxgpt:gpt2_medium`, `seqxgpt:gpt_j_6b`.

### Fast-DetectGPT / Ghostbuster Smoke Validation (Docker)

Each command validates score separation on HC3 interim data using 10 `human` and 10 `ai` samples
from `data/interim/datasets/hc3/all_train/`, and writes JSON output under `runs/detectors/`.

```bash
docker compose run --rm apm \
  python -m apm.detectors.adapters.fast_detectgpt_smoke \
  --project-root . \
  --samples-per-label 10 \
  --hc3-split all_train

docker compose run --rm apm \
  python -m apm.detectors.adapters.ghostbuster_smoke \
  --project-root . \
  --samples-per-label 10 \
  --hc3-split all_train
```

## Dataset Ingestion Foundation

Universal dataset ingestion foundations are implemented in `src/apm/data/`:

- canonical record schema and typed load request/result models (`src/apm/types.py`),
- registry-backed universal loader (`src/apm/data/hf_loader.py`),
- deterministic default random sampling with explicit seed (`src/apm/data/sampling.py`),
- canonical validation helpers (`src/apm/data/validation.py`),
- storage path/metadata helpers for normalized artifacts (`src/apm/data/storage.py`).

Dataset config schema and template:

- `configs/datasets/schema.md`
- `configs/datasets/template.dataset.json`

Normalized artifacts convention:

- raw snapshot directories: `data/raw/datasets/<dataset_id>/<split>/`
- raw class-partitioned snapshots: `data/raw/datasets/<dataset_id>/<split>/<label>/sampled_records.jsonl`
- normalized split outputs: `data/interim/datasets/<dataset_id>/<split>.parquet`
- normalized class-partitioned outputs: `data/interim/datasets/<dataset_id>/<split>/<label>/sampled_records.parquet`
- metadata sidecar: `data/interim/datasets/<dataset_id>/<split>.metadata.json`

## HC3 Materialization

HC3 adapter configuration and field audit:

- `configs/datasets/hc3.dataset.json`
- `configs/datasets/hc3_field_audit.md`

Materialize sampled HC3 outputs (default: balanced `100 human + 100 ai` per selector where available, deterministic `seed=42`):

```bash
PYTHONPATH=src python -m apm.data.adapters.hc3_materialize \
  --project-root . \
  --config configs/datasets/hc3.dataset.json \
  --sample-size 100 \
  --seed 42
```

## Kaggle LLM Detect AI Generated Text Materialization

Kaggle adapter configuration and field audit:

- `configs/datasets/kaggle_llm_detect_ai_generated_text.dataset.json`
- `configs/datasets/kaggle_llm_detect_ai_generated_text_field_audit.md`

Materialize sampled Kaggle outputs (default: `100` per configured split, deterministic `seed=42`):

```bash
PYTHONPATH=src python -m apm.data.adapters.kaggle_llm_detect_ai_generated_text_materialize \
  --project-root . \
  --config configs/datasets/kaggle_llm_detect_ai_generated_text.dataset.json \
  --sample-size 100 \
  --seed 42
```

## GriD Materialization

GriD adapter configuration and field audit:

- `configs/datasets/grid.dataset.json`
- `configs/datasets/grid_field_audit.md`

Download-only bootstrap command:

```bash
PYTHONPATH=src python -m apm.data.adapters.grid_download \
  --project-root . \
  --config configs/datasets/grid.dataset.json
```

Materialize sampled GriD outputs (default: balanced `100 human + 100 ai` per split where available):

```bash
PYTHONPATH=src python -m apm.data.adapters.grid_materialize \
  --project-root . \
  --config configs/datasets/grid.dataset.json \
  --sample-size 100 \
  --seed 42
```

Kaggle/HC3 materializers support `--sampling-strategy`:

- `balanced_random` (default): target `sample-size` per label (`human`, `ai`)
- `random`: legacy total-random sampling behavior

If configured source files are missing, Kaggle adapter/materializer will auto-download them using the
`download` block from dataset config. Prerequisites:

- `kaggle` CLI installed,
- valid Kaggle credentials in `~/.kaggle/kaggle.json`.

Credential bootstrap for this repository:

1. Copy template:
```bash
cp configs/credentials/kaggle.example.json configs/credentials/kaggle.json
```
2. Fill real values in `configs/credentials/kaggle.json` (this file is gitignored).
3. Install to Kaggle CLI location:
```bash
mkdir -p ~/.kaggle
cp configs/credentials/kaggle.json ~/.kaggle/kaggle.json
chmod 600 ~/.kaggle/kaggle.json
```

Download-only bootstrap command:

```bash
PYTHONPATH=src python -m apm.data.adapters.kaggle_llm_detect_ai_generated_text_download \
  --project-root . \
  --config configs/datasets/kaggle_llm_detect_ai_generated_text.dataset.json
```

## Bulk Dataset Init

Initialize/materialize all supported datasets discovered in `configs/datasets/*.dataset.json`:

```bash
PYTHONPATH=src python -m apm.data.materialize_all \
  --project-root . \
  --config-dir configs/datasets \
  --sample-size 100 \
  --seed 42
```

In bulk mode, `--sample-size` is interpreted as per-label target for balanced sampling.

Limit execution to selected dataset ids:

```bash
PYTHONPATH=src python -m apm.data.materialize_all \
  --project-root . \
  --config-dir configs/datasets \
  --datasets hc3
```

Or provide a dataset-id list file (`#` comments allowed):

```bash
PYTHONPATH=src python -m apm.data.materialize_all \
  --project-root . \
  --config-dir configs/datasets \
  --datasets-file configs/datasets/datasets_to_init.example.txt
```

## Propozycja tematu prowadzącego

10. Detekcja treści generowanych przez AI
Narzędzie lub eksperyment badający wykrywalność treści wygenerowanych przez AI.

- **Base**: Porównanie 2-3 detektorów (GPTZero, Originality.ai, watermarking) na zbiorze tekstów ludzkich i AI-generowanych. Metryki.
- **Good**: + analiza jak parafrazowanie/prompty wpływają na wykrywalność, confusion matrix, wnioski prawne (plagiat, prawo autorskie).
- **Excellent**: + własny prosty klasyfikator (fine-tuned lub statystyczny), porównanie z komercyjnymi, analiza implikacji dla edukacji/publikacji.
