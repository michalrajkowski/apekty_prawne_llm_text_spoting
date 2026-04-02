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

### Task 007 - dataset initialization and bulk materialization CLI
- Added bulk dataset initialization/materialization CLI:
  - `src/apm/data/materialize_all.py`
  - command: `python -m apm.data.materialize_all`
- Implemented config-driven discovery from `configs/datasets/*.dataset.json` with `dataset_id` resolution.
- Added dataset selection controls:
  - `--datasets` CLI argument (space-separated dataset ids),
  - `--datasets-file` text file support (one dataset id per line, `#` comments allowed).
- Implemented orchestration logic:
  - runs all supported datasets by default,
  - skips unsupported discovered datasets in default mode and reports them,
  - fails fast for explicitly requested missing/unsupported datasets.
- Reused per-dataset materializers through registry dispatch (currently `hc3`).
- Added tests for orchestration and selection logic:
  - `tests/test_materialize_all.py`
- Added docs for bulk init workflow and example dataset-id file:
  - `README.md`
  - `configs/datasets/datasets_to_init.example.txt`

### Task 004 - Kaggle LLM Detect AI Generated Text adapter integration
- Added Kaggle dataset config and field audit:
  - `configs/datasets/kaggle_llm_detect_ai_generated_text.dataset.json`
  - `configs/datasets/kaggle_llm_detect_ai_generated_text_field_audit.md`
- Implemented Kaggle CSV adapter using the universal dataset interface:
  - `src/apm/data/adapters/kaggle_llm_detect_ai_generated_text_adapter.py`
  - config-driven split file selection and explicit `0 -> human`, `1 -> ai` label mapping.
- Added Kaggle materialization runner:
  - `src/apm/data/adapters/kaggle_llm_detect_ai_generated_text_materialize.py`
  - persists sampled outputs with canonical parquet/jsonl + metadata artifacts.
- Wired Kaggle into bulk materialization defaults:
  - `src/apm/data/materialize_all.py`
  - `configs/datasets/datasets_to_init.example.txt`
- Added tests for config parsing, mapping, deterministic sampling, and artifact writes:
  - `tests/test_kaggle_adapter.py`
  - targeted suite result: `20 passed`.

### Task 008 - balanced per-class sampling and split class subfolders
- Added balanced per-class sampling strategy (`balanced_random`) in shared ingestion stack:
  - `src/apm/types.py`
  - `src/apm/data/sampling.py`
  - `src/apm/data/hf_loader.py`
  - `src/apm/data/custom_loader.py`
- New behavior supports target `X` per canonical class (`human`, `ai`) with availability cap.
- Updated HC3/Kaggle materializers to default to balanced sampling while keeping legacy `random` mode via CLI flag:
  - `src/apm/data/adapters/hc3_materialize.py`
  - `src/apm/data/adapters/kaggle_llm_detect_ai_generated_text_materialize.py`
- Added split-level class subfolders in persisted artifacts:
  - raw: `data/raw/datasets/<dataset>/<split>/<label>/sampled_records.jsonl`
  - interim: `data/interim/datasets/<dataset>/<split>/<label>/sampled_records.parquet`
  - helpers in `src/apm/data/storage.py`.
- Metadata now records:
  - requested per-label target,
  - realized per-label counts,
  - label-partitioned artifact paths.
- Added/extended tests:
  - `tests/test_data_sampling.py`
  - `tests/test_hf_loader.py`
  - `tests/test_kaggle_adapter.py`
  - targeted suite result: `26 passed`.
- End-to-end verification from clean outputs:
  - removed existing HC3/Kaggle raw + interim outputs,
  - ran one command `python -m apm.data.materialize_all --sample-size 100 --seed 42`,
  - verified per-split label outputs:
    - HC3 selectors: `100 human + 100 ai`,
    - Kaggle train: `100 human + 3 ai` (availability cap from source class imbalance).

### Task 005 - GriD dataset adapter integration
- Added GriD dataset config and field audit:
  - `configs/datasets/grid.dataset.json`
  - `configs/datasets/grid_field_audit.md`
- Implemented GriD CSV adapter with config-driven split mapping and GitHub raw auto-download bootstrap:
  - `src/apm/data/adapters/grid_adapter.py`
  - canonical mapping: `Data -> text`, `Labels 0 -> human`, `Labels 1 -> ai`.
- Added GriD download and materialization CLIs:
  - `src/apm/data/adapters/grid_download.py`
  - `src/apm/data/adapters/grid_materialize.py`
- Wired GriD into bulk initializer registry:
  - `src/apm/data/materialize_all.py`
  - `configs/datasets/datasets_to_init.example.txt`
- Added focused adapter/downloader/materialization tests:
  - `tests/test_grid_adapter.py`
  - targeted suite result after integration updates: `33 passed`.
- Finishing validation (live run):
  - cleaned previous GriD artifacts, ran download CLI and materialize CLI,
  - verified both configured splits/sectors (`filtered`, `unfiltered`),
  - verified per-split `human/` and `ai/` JSONL outputs with `100` lines each,
  - verified per-split label-partitioned parquet outputs and metadata counts.
