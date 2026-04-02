# GriD (GPT Reddit Dataset) - Field Audit

Dataset source:
- https://github.com/madlab-ucr/GriD

Source files used by adapter:
- `reddit_datasets/reddit_filtered_dataset.csv`
- `reddit_datasets/reddit_unfiltered_data.csv`

Configured logical splits:
- `filtered` -> `reddit_filtered_dataset.csv`
- `unfiltered` -> `reddit_unfiltered_data.csv`

Observed columns:
- `Data`: text snippet.
- `Labels`: binary class label (`0` human-written, `1` GPT-generated).

Canonical label mapping:
- `Labels = 0` -> `human`
- `Labels = 1` -> `ai`

Notes:
- Adapter auto-downloads source CSV files from GitHub raw URLs when missing.
- Balanced sampling target is applied after canonical mapping; per-class availability cap is respected.
