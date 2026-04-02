# TASKS.md

Active execution queue for the chunking research repository.

For any new dataset integration task, follow
`docs/INTEGRATING_NEW_DATASET.md` before implementation.

## How To Use

1. Select the top item from `Queue`.
2. Move it to `In Progress`.
3. If user says `discuss next task`, run ideation phase first (problem framing, options, tradeoffs, solution direction).
4. Record settled decisions directly in the task entry (`Decision Notes`).
5. Discuss and agree unit test strategy (`Test Plan`) before implementation.
6. Write tests first (red), run tests, and confirm expected failing behavior.
7. Implement the smallest complete slice (green), then refactor safely.
8. Run checker/linter/formatter; fix all issues.
9. Run relevant validation:
- Python tests: `pytest -q` (or targeted tests)
- Bash syntax for touched scripts: `bash -n <script>`
10. If tests still fail, fix root cause; do not “cheat” tests by weakening assertions or bypassing behavior.
11. Update docs for behavior/workflow/interface changes.
12. Move completed item to `FINISHED_TASKS.md` with date + short outcome.

## Task Writing Rules

1. Every task has a unique incrementing ID (`NNN`).
2. Tasks must include scope and acceptance criteria.
3. Keep tasks small and independently shippable.
4. If blocked, move task to `Blocked` with explicit blocker and unblock condition.
5. If task changes experiment reproducibility, include acceptance criteria for config and run traceability.
6. Each task should include:
- `Decision Notes`: agreed design/approach from ideation.
- `Test Plan`: agreed red-green test scope and validation strategy.

## Queue

### Task 012 - detector abstraction layer and unified interface

Scope:
- Add an abstract detector base class in `src/apm/detectors/` that standardizes detector usage.
- Required methods:
  - single prediction: input `text` -> output `float` score,
  - batch prediction: input text batch -> output list/batch of `float` scores,
  - initialization/constructor flow for creating a configured detector instance.
- Add detector-facing types/protocols needed to keep implementations typed and consistent.

Acceptance Criteria:
- A shared abstract detector contract exists and is used as the required parent/protocol for detector adapters.
- Method signatures and return types are explicit and validated by tests.
- A small test suite validates contract behavior and shape assumptions for single/batch predictions.
- Follow-up detector tasks (009-011) are explicitly aligned to this interface.

Decision Notes:
- Establish interface first to avoid glue-code drift across detector integrations.
- Keep scope strict to abstraction + basic contract validation (no model integration in this task).

Test Plan:
- Add unit tests for abstract contract compliance and example stub implementation.
- Run targeted detector-interface tests and type checks on touched files.

### Task 009 - dirty detector integration: AIGC Detector V3-Short

Scope:
- Add detector config and minimal runtime wiring for `yuchuantian/AIGC_detector_env3short`.
- Implement an adapter that subclasses/implements the shared detector abstraction from Task 012.
- Implement abstraction methods for this detector:
  - single prediction: `text -> float`,
  - batch prediction: `batch -> batch[float]`,
  - configured initialization flow.
- Add a smoke-run entry point that scores a small text file/jsonl sample and prints JSON outputs for quick manual validation.
- Keep integration intentionally minimal ("dirty"): no calibration, no advanced optimization, no experiment-matrix wiring yet.

Acceptance Criteria:
- Model loads and runs inference on GPU with 8GB VRAM (with safe low-memory defaults).
- Adapter implements the shared abstraction and returns stable float score outputs for single and batch methods.
- Smoke command runs end-to-end on at least 10 sample texts and saves outputs under `runs/`.
- Basic usage is documented in `README.md`.

Decision Notes:
- Chosen as first detector because checkpoint size is small (~499 MB) and should comfortably fit 8GB VRAM.
- Prioritize "works end-to-end" over correctness benchmarking in this task.

Test Plan:
- Add deterministic adapter tests for single/batch abstraction methods with mocked model outputs.
- Run one real smoke test on local sample texts and verify output artifact shape.

### Task 010 - dirty detector integration: GLTR (gpt2-small backend)

Scope:
- Add GLTR repository as a git submodule under `external/detectors/`.
- Implement minimal wrapper/adapter that subclasses/implements the shared detector abstraction from Task 012.
- Create or extend a single initialization script that initializes all git submodules recursively (required for detector repos).
- Add a smoke-run command for batch scoring of local sample texts.

Acceptance Criteria:
- `external/detectors/` contains GLTR as a reproducible pinned submodule reference.
- One init script can bootstrap all submodules with one command (recursive init/update).
- GLTR adapter exposes abstraction-compatible single and batch score methods and runs on 8GB VRAM with gpt2-small backend.
- Smoke command writes runnable outputs under `runs/`.

Decision Notes:
- Chosen as lightweight baseline because default backend (`gpt2-small`) is small and fast.
- Keep GLTR integration read-only (inference only), without UI/server setup.

Test Plan:
- Add tests for abstraction method compliance/output shape and submodule-init script invocation behavior.
- Execute one local smoke run on sample texts and verify output file creation.

### Task 011 - dirty detector integration: DetectGPT (light config)

Scope:
- Add DetectGPT repository as a git submodule under `external/detectors/`.
- Extend detector init/bootstrap script to include DetectGPT submodule (same one-command recursive init flow).
- Implement a minimal DetectGPT adapter that subclasses/implements the shared detector abstraction from Task 012, with conservative runtime defaults (small batch/perturbation settings) to fit 8GB VRAM.
- Add a smoke-run command that scores sample texts and persists raw + normalized outputs.

Acceptance Criteria:
- DetectGPT submodule is pinned and initialized by the shared submodule bootstrap script.
- Adapter runs end-to-end on a small sample set without OOM on 8GB VRAM under documented defaults.
- Adapter implements standardized single and batch float-score prediction methods from the abstraction layer.
- README docs include run command and known "dirty integration" limitations.

Decision Notes:
- Chosen as a third lightweight-enough research baseline from available candidates (~2.25 GB default weights path).
- This phase validates operability only; robust thresholding/quality evaluation is deferred.

Test Plan:
- Add abstraction method compliance tests and adapter output-shape tests.
- Run targeted smoke inference on at least 10 texts and verify artifacts and exit status.

### Task 006 - final integration and reproducibility hardening

Scope:
- Add missing glue code, docs, and sanity checks after 002-005.
- Ensure all dataset adapters are runnable through one entry point.
- Ensure reproducibility metadata is captured consistently.

Acceptance Criteria:
- One CLI/programmatic flow can run all configured dataset adapters.
- Docs describe end-to-end flow: source -> normalized dataset -> experiments.
- Reproducibility metadata (seed, source, config, revision/timestamp) is persisted.
- Any remaining gaps from adapter tasks are closed.

Decision Notes:
- Keep this task strictly for integration hardening, not new dataset features.

Test Plan:
- Add integration smoke tests for registry-driven loading of all adapters.
- Verify normalized artifacts and metadata files are emitted as expected.

## In Progress

- (empty)

## Blocked

- (empty)
