# Docker Setup

Minimal command flow to initialize and run this repository with Docker (GPU).

## 1. Clone and submodules

```bash
git clone --recurse-submodules <repo-url>
cd <repo-dir>
./scripts/init_submodules.sh
```

## 2. Kaggle credentials (required for Kaggle dataset bootstrap)

```bash
mkdir -p .kaggle
cp configs/credentials/kaggle.example.json .kaggle/kaggle.json
chmod 600 .kaggle/kaggle.json
```

Fill real Kaggle credentials in `.kaggle/kaggle.json`.

## 3. Build Docker image (Python 3.13.2, pinned deps)

```bash
docker compose build apm
```

## 4. Initialize datasets

```bash
docker compose run --rm apm \
  python -m apm.data.materialize_all \
  --project-root . \
  --config-dir configs/datasets \
  --sample-size 100 \
  --seed 42
```

## 5. Run detector scoring experiment pipeline

```bash
docker compose run --rm apm \
  python scratch/detector_scoring/run_detector_scores.py \
  --project-root . \
  --examples-per-label 3

docker compose run --rm apm \
  python scratch/detector_scoring/summarize_scores.py \
  --project-root .

docker compose run --rm apm \
  python scratch/detector_scoring/plot_detector_scores.py \
  --project-root .
```

Outputs:
- `scratch/detector_scoring/results/raw_scores.jsonl`
- `scratch/detector_scoring/results/summary_scores.json`
- `scratch/detector_scoring/results/detector_scores_barplot.png`
