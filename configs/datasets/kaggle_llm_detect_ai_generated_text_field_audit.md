# Kaggle LLM Detect AI Generated Text - Field Audit

Dataset source:
- https://www.kaggle.com/competitions/llm-detect-ai-generated-text/data

Primary file used by adapter:
- `train_essays.csv`

Observed/expected columns in `train_essays.csv`:
- `id`: essay identifier (string-like).
- `prompt_id`: prompt reference identifier (integer-like).
- `text`: essay content (string).
- `generated`: binary label (`0` human-written, `1` AI-generated).

Label mapping used by canonical adapter:
- `generated = 0` -> `human`
- `generated = 1` -> `ai`

Notes:
- Adapter is file-path based for reproducible reruns, but includes an auto-download bootstrap:
  - when required files are missing, adapter/materializer can invoke `kaggle competitions download`
    using the `download` block from dataset config.
- Prompt metadata files (for example `train_prompts.csv`) are not required for canonical text/label ingestion in Task 004.
