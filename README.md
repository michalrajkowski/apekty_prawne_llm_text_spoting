# Aspekty prawne - 

## Architecture

Current repository structure and architecture decisions are documented in `ARCHITECTURE.md`.

## Prompt History Viewer

Task history viewer is isolated under `prompt_history/` and generates a static HTML chat view from `prompt_history/PROMPTS_HISTORY.md`.

1. Run:
```bash
python3 prompt_history/build_task_history.py
```
2. This runner:
- creates/uses hidden `prompt_history/.task_history_venv`,
- installs dependencies from `prompt_history/requirements.txt`,
- runs the builder CLI.
3. Default outputs:
- `prompt_history/runs/prompt_history.html`

Optional arguments:
```bash
python3 prompt_history/build_task_history.py \
  --input prompt_history/PROMPTS_HISTORY.md \
  --html-output prompt_history/runs/prompt_history.html \
  --user-avatar-url prompt_history/assets/michalrajkowski.png \
  --title "Prompt History"
```


## Propozycja tematu prowadzącego

10. Detekcja treści generowanych przez AI
Narzędzie lub eksperyment badający wykrywalność treści wygenerowanych przez AI.

- **Base**: Porównanie 2-3 detektorów (GPTZero, Originality.ai, watermarking) na zbiorze tekstów ludzkich i AI-generowanych. Metryki.
- **Good**: + analiza jak parafrazowanie/prompty wpływają na wykrywalność, confusion matrix, wnioski prawne (plagiat, prawo autorskie).
- **Excellent**: + własny prosty klasyfikator (fine-tuned lub statystyczny), porównanie z komercyjnymi, analiza implikacji dla edukacji/publikacji.
