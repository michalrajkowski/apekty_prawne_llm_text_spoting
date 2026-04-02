# Integrating A New Dataset (Universal Instruction)

Use this checklist for every new dataset adapter task.

1. Analyze source dataset
- Confirm source link/access method (HF/Kaggle/local repo/files).
- Document split names, key fields, and label semantics (human vs ai).

2. Define dataset config
- Add `configs/datasets/<dataset_id>.dataset.json`.
- Define canonical mapping (`id`, `text`, label mapping, optional metadata fields).
- Set sampling strategy to deterministic random with explicit seed support.

3. Implement adapter + materialization entrypoint
- Add dataset-specific loader/adapter in `src/apm/data/adapters/`.
- Keep parsing/mapping logic in adapter and orchestration in materialize script.
- Add source bootstrap behavior: if required source files are missing, loader/materializer should
  trigger configured download step (unless source/provider makes this impossible).

4. Validate output
- Run loader/materialization and check canonical schema compatibility.
- Materialize a smoke sample (default: balanced target 100 per label per configured split/selector).
- Save artifacts and validation report in project-standard locations.

5. Add tests
- Test mapping correctness (especially human/ai mapping).
- Test deterministic random sampling behavior.
- Add basic integration/smoke test for adapter entrypoint.

6. Update docs and task state
- Update relevant docs (`README.md`, `ARCHITECTURE.md`) if behavior/workflow changed.
- Move task status in `TASKS.md` / `FINISHED_TASKS.md` with short outcome.
