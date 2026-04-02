# Architecture

## Purpose

This repository is a research framework for evaluating AI-text detectors on:

1. Benchmark datasets (including Hugging Face datasets).
2. Custom text collections.
3. Transformed versions of texts (obfuscation/modification operations).

The objective is to measure how transformations and AI-content ratio influence detector confidence and classification quality.

## Design Principles

1. Adapter-based integrations: each new dataset or detector is added through a small adapter layer.
2. Config-driven experiments: experiment matrices are described in `configs/`, not hardcoded.
3. Reproducibility first: runs are deterministic and seed-controlled.
4. Docker-first execution: environment setup should be portable across machines.
5. Research-ready initialization: setup/init scripts should allow project bootstrapping from a clean machine.

## Repository Layout

- `configs/datasets/`: dataset definitions and column mappings.
- `configs/detectors/`: detector configuration and runtime parameters.
- `configs/transforms/`: text transformation configurations.
- `configs/experiments/`: experiment matrices and run plans.
- `src/apm/data/`: shared ingestion infrastructure (registry, sampling, validation, storage, common loaders).
- `src/apm/data/adapters/`: dataset-specific adapters and dataset-specific materialization scripts (for example HC3).
- `src/apm/detectors/`: detector interface, registry, and per-detector adapters.
- `src/apm/transforms/`: transform interface, registry, and operation implementations.
- `src/apm/experiments/`: runner and matrix expansion/execution logic.
- `src/apm/metrics/`: evaluation metrics (classification and robustness-related metrics).
- `src/apm/reporting/`: aggregation and export logic for experiment outputs.
- `external/detectors/`: detector repositories added as git submodules.
- `data/custom/`: user-provided datasets/collections.
- `data/interim/`: normalized intermediate datasets.
- `data/processed/`: transformed inputs used for experiments.
- `runs/`: per-run outputs, configs, metadata, and metrics.
- `tests/`: unit/integration tests.

## Extension Workflow

### Add a new dataset

1. Add dataset config in `configs/datasets/`.
2. If needed, implement or extend a loader adapter in `src/apm/data/`.
3. Register the dataset in `dataset_registry`.

### Add a new detector

1. Add detector repository as a git submodule under `external/detectors/`.
2. Implement an adapter in `src/apm/detectors/adapters/`.
3. Add detector config in `configs/detectors/`.
4. Register in detector `registry`.

### Add a new transform

1. Add transform config in `configs/transforms/`.
2. Implement operation in `src/apm/transforms/ops/`.
3. Register in transform `registry`.

## Reproducibility Requirements

1. All experiment entry points must accept and persist explicit random seeds.
2. Every run should save resolved configs and execution metadata in `runs/<run_id>/`.
3. Results should be reproducible when using the same code revision, submodule revisions, configs, and seed values.

## Environment and Initialization

1. The project should be runnable from initialization scripts on a clean machine.
2. Dataset initialization should be available via a bulk materialization entry point driven by dataset configs (currently `python -m apm.data.materialize_all`).
3. Dockerized workflows should mount host directories for:
   - input data (`data/`)
   - run artifacts (`runs/`)
   - optional caches
4. Initialization scripts should prepare dependencies, required folders, and baseline config state.
