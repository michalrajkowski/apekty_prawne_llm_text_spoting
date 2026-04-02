# HC3 Field Audit

Source: `Hello-SimpleAI/HC3`

Checked selectors:

- `all/train`
- `finance/train`
- `medicine/train`
- `open_qa/train`
- `reddit_eli5/train`
- `wiki_csai/train`

## Observed Fields

Common required fields in all checked selectors:

- `id` (`str`)
- `question` (`str`)
- `human_answers` (`list[str]`)
- `chatgpt_answers` (`list[str]`)

Optional field:

- `source` (`str`) present in `all/train`.

## Label Mapping Decision

- Human text source field: `human_answers`
- AI text source field: `chatgpt_answers`

Canonical mapping:

- records from `human_answers[*]` -> `label = "human"`
- records from `chatgpt_answers[*]` -> `label = "ai"`

Each row can emit multiple canonical text records because both answer fields are arrays.

## 100-Row Load Check Per Selector

- `all/train`: loaded 100 rows (dataset size: 24,322)
- `finance/train`: loaded 100 rows (dataset size: 3,933)
- `medicine/train`: loaded 100 rows (dataset size: 1,248)
- `open_qa/train`: loaded 100 rows (dataset size: 1,187)
- `reddit_eli5/train`: loaded 100 rows (dataset size: 17,112)
- `wiki_csai/train`: loaded 100 rows (dataset size: 842)

For all checks above:

- required fields were present in all 100 sampled rows,
- `human_answers` had no empty lists,
- `chatgpt_answers` had no empty lists.
