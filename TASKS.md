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
