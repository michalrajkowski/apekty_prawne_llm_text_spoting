# Dataset Config Schema (Foundation)

This schema defines how dataset-specific adapters are configured in `configs/datasets/`.

## Required Fields

- `dataset_id`: stable dataset key used by the registry.
- `source_type`: one of `huggingface`, `kaggle`, `local`, `git_repo`.
- `source_uri`: primary source location (HF id, URL, repo path, etc.).
- `default_split`: split selected when no split is explicitly requested.
- `sampling`:
  - `strategy`: currently `random`.
  - `seed`: default integer seed (project default: `42`).

## Split Definition

- `splits`: mapping from split name to source-specific selector.

Each split entry should define:
- `selector`: source-specific split key/file reference.
- `text_field`: source field used as primary text.
- `label_field`: source field used to map labels.
- `label_mapping`: explicit mapping from source labels to canonical labels `human` / `ai`.

## Notes

- Adapters must output canonical records with fields:
  - `dataset_id`, `split`, `sample_id`, `text`, `label`, `source_fields`.
- Sampling default is `random`, meaning sampled subset order is randomized deterministically by `seed`.
- Normalized outputs are stored as:
  - `data/interim/datasets/<dataset_id>/<split>.parquet`
  - `data/interim/datasets/<dataset_id>/<split>.metadata.json`
