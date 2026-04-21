# Agents.md

The repository's main goal is to compare GPT/AI text detectors on benchmark datasets and custom documents, apply techniques that obfuscate/modify LLM-generated text, and verify whether those techniques make GPT detectors less confident that the text is human-written.

## Development Workflow

1. Pull the next task from `TASKS.md` (`Queue` section).
2. Move the task to `In Progress` and keep scope minimal.
3. Implement the change with tests where practical (`tests/`), especially for deterministic logic.
4. Run linting/checking only for touched Python files, and never run these tools on GitHub submodule scripts.
5. Update docs when behavior, interfaces, workflow, or experiment conventions change.
6. Move completed tasks to `FINISHED_TASKS.md` with completion date and short outcome.

## Linting and Checking Scope

1. Run linters/checkers only on Python files.
2. Exclude GitHub submodule scripts from linting/checking runs.

## Research and Reproducibility Constraints

1. Treat the repository as scientific research infrastructure, not a one-off script collection.
2. Keep project initialization reproducible via initialization files/scripts so a clean machine can bootstrap the environment.
3. Prioritize Dockerized workflows with mounted host folders (at minimum `data/` and `runs/`) so collaborators can run experiments locally with consistent setup.
4. Ensure experiments and text operations are reproducible by explicit seeds; do not add nondeterministic experiment paths without seed control.

## Task Files Policy

1. `TASKS.md` is the active execution queue and planning board.
2. `FINISHED_TASKS.md` is the immutable archive of completed tasks.
3. Keep task IDs stable once assigned.

## Python Code Rules (Non-Negotiable)

These rules must never be violated in newly created or modified Python files:

1. All code must be correctly typed, including function/method signatures and return types.
2. Imports must be declared only at the top of the file. Do not place imports inside functions/methods/branches.
3. Every Python file must have a concise, up-to-date module docstring at the top of the file.
4. Avoid `try`/`except` unless truly necessary. Do not add broad or flow-obscuring exception handling that makes program behavior unpredictable.
