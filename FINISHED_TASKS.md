## 2026-04-02

### Task 001 - prompt history viewer
- Implemented typed parser for `PROMPTS_HISTORY.md` with robust handling of nested fenced examples, clear malformed-input errors, and newline-preserving message extraction.
- Added HTML/CSS chat renderer with deterministic tag color mapping, day separators, USER/AGENT alignment, AI text avatar, and long-message shortening (`20` head lines + `...` + `20` tail lines).
- Isolated all prompt-history tooling under `prompt_history/`:
  - `prompt_history/src/prompt_history/`
  - `prompt_history/build_task_history.py`
  - `prompt_history/PROMPTS_HISTORY.md`
  - `prompt_history/assets/michalrajkowski.png`
  - hidden venv `prompt_history/.task_history_venv`
- Added Python runner that installs `prompt_history/requirements.txt` and generates:
  - `prompt_history/runs/prompt_history.html`
- Added automated tests for parser behavior, newline preservation, nested fence handling, and deterministic tag colors.

### Task 002 - universal dataset ingestion foundation
- Implemented typed canonical ingestion models in `src/apm/types.py`:
  - `dataset_id`, `split`, `sample_id`, `text`, `label`, `source_fields`
  - load request/result models with default `sampling_strategy=random` and `seed=42`.
- Added universal ingestion infrastructure in `src/apm/data/`:
  - adapter protocol (`base.py`),
  - registry (`dataset_registry.py`),
  - shared loader (`hf_loader.py`),
  - deterministic sampling utility (`sampling.py`),
  - canonical validation helpers (`validation.py`),
  - storage path and metadata helpers (`storage.py`),
  - local JSONL custom loader (`custom_loader.py`).
- Added dataset config foundation docs:
  - `configs/datasets/schema.md`
  - `configs/datasets/template.dataset.json`
- Added `data/raw/.gitkeep` and documented artifact conventions in `README.md`.
- Added targeted tests for sampling, validation, registry, and universal loader (`9 passed`).

### Task 003 - HC3 dataset adapter integration
- Implemented HC3 adapter using the Hugging Face `datasets` library:
  - adapter module moved into dataset-specific adapters folder:
    - `src/apm/data/adapters/hc3_adapter.py`
  - source loading uses `datasets.load_dataset(...)` for each configured selector.
- Kept HC3 integration config-driven:
  - `configs/datasets/hc3.dataset.json`
  - `configs/datasets/hc3_field_audit.md`
- Implemented HC3 sample materialization runner:
  - `src/apm/data/adapters/hc3_materialize.py`
  - outputs deterministic sampled records (`sample_size=100`, `seed=42`) per selector.
- Added HC3 adapter tests with mock dataset loader:
  - `tests/test_hc3_adapter.py` (`12` total targeted tests passed with existing data tests).
- Generated validation report and persisted sampled outputs for all configured HC3 selectors.
