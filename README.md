# Aspekty prawne - 

## Architecture

Current repository structure and architecture decisions are documented in `ARCHITECTURE.md`.

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
- normalized split outputs: `data/interim/datasets/<dataset_id>/<split>.parquet`
- metadata sidecar: `data/interim/datasets/<dataset_id>/<split>.metadata.json`

## HC3 Materialization

HC3 adapter configuration and field audit:

- `configs/datasets/hc3.dataset.json`
- `configs/datasets/hc3_field_audit.md`

Materialize sampled HC3 outputs (default: `100` per selector, deterministic `seed=42`):

```bash
PYTHONPATH=src python -m apm.data.adapters.hc3_materialize \
  --project-root . \
  --config configs/datasets/hc3.dataset.json \
  --sample-size 100 \
  --seed 42
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

## Prompt History Viewer

Task history viewer is isolated under `prompt_history/` and generates a static HTML chat view from `prompt_history/PROMPTS_HISTORY.md`.

1. Run:
```bash
python3 prompt_history/build_task_history.py
```
2. This runner:
- creates/uses hidden `prompt_history/.task_history_venv`,
- installs dependencies from `prompt_history/requirements.txt`,
- runs the builder CLI.
3. Default outputs:
- `prompt_history/runs/prompt_history.html`

Optional arguments:
```bash
python3 prompt_history/build_task_history.py \
  --input prompt_history/PROMPTS_HISTORY.md \
  --html-output prompt_history/runs/prompt_history.html \
  --user-avatar-url prompt_history/assets/michalrajkowski.png \
  --title "Prompt History"
```


## Propozycja tematu prowadzącego

10. Detekcja treści generowanych przez AI
Narzędzie lub eksperyment badający wykrywalność treści wygenerowanych przez AI.

- **Base**: Porównanie 2-3 detektorów (GPTZero, Originality.ai, watermarking) na zbiorze tekstów ludzkich i AI-generowanych. Metryki.
- **Good**: + analiza jak parafrazowanie/prompty wpływają na wykrywalność, confusion matrix, wnioski prawne (plagiat, prawo autorskie).
- **Excellent**: + własny prosty klasyfikator (fine-tuned lub statystyczny), porównanie z komercyjnymi, analiza implikacji dla edukacji/publikacji.
