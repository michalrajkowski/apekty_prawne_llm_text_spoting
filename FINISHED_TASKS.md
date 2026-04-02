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
