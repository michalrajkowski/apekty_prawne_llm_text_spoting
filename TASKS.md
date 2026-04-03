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

### Task 017 - adapter fidelity contract v2 (native outputs + canonical AI direction)

Scope:
- Update the detector base contract to support model-intent fidelity instead of forcing everything into one probability-like float.
- Introduce structured detector output that keeps native detector output separate from optional canonicalized score(s).
- Add required semantics metadata fields for each detector output:
  - `score_kind` (for example: `binary_probability`, `curvature_statistic`, `token_rank_signal`, `multiclass_logits`),
  - `score_range` (for example: `(0,1)`, `unbounded`, `depends`),
  - optional canonical score field used for cross-detector comparison.
- Enforce project-wide rule: canonical comparison score must always follow `higher = more AI-like`.
- Update detector integration docs to make this contract mandatory for all newly added detectors.

Acceptance Criteria:
- Base detector abstractions/types are updated to represent native outputs + semantics metadata.
- Docs (`docs/INTEGRATING_NEW_DETECTOR.md`) explicitly require preserving native detector output semantics.
- Canonical comparison score rule (`higher = AI`) is documented and validated in code paths that expose canonical scores.
- No adapter is forced to pretend its native output is a calibrated Human/AI probability.

Decision Notes:
- Native outputs are the source of truth for scientific fidelity.
- Canonical AI-direction score is a secondary layer for cross-detector comparability, not a replacement for native output.

Test Plan:
- Add unit tests for new output types/metadata validation and canonical direction enforcement.
- Run targeted detector-base tests and type checks on touched files.

### Task 018 - migrate existing detector adapters to fidelity contract v2

Scope:
- Refactor already integrated detectors to comply with Task 017 output contract.
- Remove forced probability-style mappings where they are not native detector outputs.
- Preserve each detector’s native scoring behavior and attach required semantics metadata.
- Where needed, add canonical comparison score adapters with enforced `higher = AI` direction.
- Correct known direction/index mapping mistakes (for example, RADAR label index mismatch).

Acceptance Criteria:
- Existing adapters return contract-compliant structured outputs with semantics metadata.
- Native score behavior is preserved per detector intended use.
- Canonical `higher = AI` score is present only as derived optional field where applicable.
- Detector smoke scripts and README examples are updated for new output structure.

Decision Notes:
- Migration is separate from contract definition to keep the change reviewable and lower risk.
- Direction fixes are part of migration because wrong label orientation invalidates downstream experiments.

Test Plan:
- Update existing detector adapter tests for new output schema.
- Run detector-special targeted tests for all migrated adapters.
- Run smoke checks for at least one representative input per migrated detector.

### Task 019 - universal real-data direction/index validation test for new detectors

Scope:
- Add a reusable detector validation test utility that runs on real dataset samples:
  - select 10 human and 10 AI texts from configured dataset materialization outputs,
  - score with detector under test,
  - compare average canonical AI-direction score between classes.
- If `mean(ai) <= mean(human)`, test must fail with clear diagnostic indicating likely label/direction/index configuration issue.
- Require every newly implemented detector task to include this validation test directly in its detector-special test module.

Acceptance Criteria:
- One universal test helper exists and is reusable across detector integrations.
- New-detector test template includes mandatory real-data direction check.
- Failing output explicitly points to probable direction/index mismatch (for example `ai_label_index` inversion).
- Test documentation explains prerequisites (materialized dataset availability, required markers).

Decision Notes:
- This is an automated guardrail against silent orientation bugs like RADAR index inversion.
- Validation is statistical sanity check, not full benchmark replacement.

Test Plan:
- Add tests for helper behavior (pass/fail diagnostics) with controlled synthetic scores.
- Run at least one real detector-special validation on actual materialized data.

### Task 020 - remove SynthID watermarking detector integration

Scope:
- Remove watermarking-focused SynthID detector integration from repository runtime surface.
- Remove associated adapter/config/smoke/tests and disable any workflow paths depending on SynthID detector.
- Remove SynthID detector references from detector lists/docs where it is currently presented as a text detector option.
- Keep repository history intact (no destructive git rewrite).

Acceptance Criteria:
- `synthid_text` adapter is no longer runnable through detector registry/config workflows.
- No active detector config points to SynthID.
- Detector docs/model lists no longer present SynthID as supported experiment detector.
- Test suite no longer expects SynthID integration.

Decision Notes:
- Watermark-presence detection is out of scope for current AI-vs-human detector comparison goals.

Test Plan:
- Run targeted tests for detector registry/config loading after removal.
- Run smoke path for remaining detectors to confirm no broken imports/registration side effects.

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

### Task 014 - detector integration: Binoculars

Scope:
- Verify whether the repository/upstream Binoculars integration target exposes multiple detector models/checkpoints that should be treated as separate detectors.
- If multiple models are available, define and implement a separate adapter module for each model under `src/apm/detectors/adapters/` (shared utilities allowed, no merged single adapter path).
- Integrate Binoculars under `src/apm/detectors/adapters/` with native detector output preserved per fidelity contract (Task 017).
- Add config, smoke runner, and detector-special tests.
- Ensure integration is fully compatible with current detector spec in `docs/INTEGRATING_NEW_DETECTOR.md`.

Acceptance Criteria:
- Verification result is documented in task notes/code comments: either "single model" or explicit list of supported models.
- For multi-model support, each model has its own adapter file and its own detector config in `configs/detectors/`.
- Adapter implements required interface (`initialize`, `predict_single`, `predict_batch`, `delete`) per `docs/INTEGRATING_NEW_DETECTOR.md`.
- Config exists in `configs/detectors/` and smoke runner exists in `src/apm/detectors/adapters/`.
- Detector-special tests exist in `tests/detectors/` and validate output shape/type, single-vs-batch consistency, and cleanup path.
- Adapter output follows fidelity contract: native score semantics metadata + optional canonical `higher = AI` score.

Decision Notes:
- Keep implementation focused on inference path and reproducible runtime configuration without forcing probability mapping.

Test Plan:
- Add `tests/detectors/test_binoculars.py` with marker `detector_special`.
- Run targeted suite: `pytest -q -m detector_special tests/detectors/test_binoculars.py`.

### Task 015 - detector integration: Fast-DetectGPT

Scope:
- Integrate Fast-DetectGPT under `src/apm/detectors/adapters/` with native detector output preserved per fidelity contract (Task 017).
- Add config, smoke runner, and detector-special tests.
- Ensure integration is fully compatible with current detector spec in `docs/INTEGRATING_NEW_DETECTOR.md`.

Acceptance Criteria:
- Adapter implements required interface (`initialize`, `predict_single`, `predict_batch`, `delete`) per `docs/INTEGRATING_NEW_DETECTOR.md`.
- Config exists in `configs/detectors/` and smoke runner exists in `src/apm/detectors/adapters/`.
- Detector-special tests validate output shape/type, single-vs-batch consistency, and cleanup path.
- Adapter output follows fidelity contract: native score semantics metadata + optional canonical `higher = AI` score.

Decision Notes:
- Prioritize stable adapter behavior and deterministic config-driven execution without reinterpreting native outputs as probabilities.

Test Plan:
- Add `tests/detectors/test_fast_detectgpt.py` with marker `detector_special`.
- Run targeted suite: `pytest -q -m detector_special tests/detectors/test_fast_detectgpt.py`.

### Task 016 - detector integration: Ghostbuster

Scope:
- Integrate Ghostbuster under `src/apm/detectors/adapters/` with native detector output preserved per fidelity contract (Task 017).
- Add config, smoke runner, and detector-special tests.
- Ensure integration is fully compatible with current detector spec in `docs/INTEGRATING_NEW_DETECTOR.md`.

Acceptance Criteria:
- Adapter implements required interface (`initialize`, `predict_single`, `predict_batch`, `delete`) per `docs/INTEGRATING_NEW_DETECTOR.md`.
- Config exists in `configs/detectors/` and smoke runner exists in `src/apm/detectors/adapters/`.
- Detector-special tests validate output shape/type, single-vs-batch consistency, and cleanup path.
- Adapter output follows fidelity contract: native score semantics metadata + optional canonical `higher = AI` score.

Decision Notes:
- If upstream requires API-backed calls, keep adapter contract unchanged and surface required credentials/config explicitly.

Test Plan:
- Add `tests/detectors/test_ghostbuster.py` with marker `detector_special`.
- Run targeted suite: `pytest -q -m detector_special tests/detectors/test_ghostbuster.py`.

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
