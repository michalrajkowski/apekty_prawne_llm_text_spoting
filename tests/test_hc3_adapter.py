"""Tests for HC3 adapter config parsing and canonical mapping behavior."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterator, Mapping

from apm.data.adapters.hc3_adapter import HC3Adapter, load_hc3_config


class FakeDataset:
    """Minimal dataset object compatible with HC3 adapter protocol."""

    def __init__(self, rows: list[dict[str, Any]]) -> None:
        self._rows = rows

    def __len__(self) -> int:
        return len(self._rows)

    def __iter__(self) -> Iterator[Mapping[str, Any]]:
        return iter(self._rows)

    def select(self, indices: list[int]) -> "FakeDataset":
        return FakeDataset([self._rows[index] for index in indices])


def _write_hc3_config(path: Path) -> None:
    payload = {
        "dataset_id": "hc3",
        "source_type": "huggingface",
        "source_uri": "Hello-SimpleAI/HC3",
        "default_selector": {"config": "all", "split": "train"},
        "sampling": {"strategy": "random", "seed": 42},
        "selectors": {
            "all_train": {"config": "all", "split": "train"},
            "finance_train": {"config": "finance", "split": "train"},
        },
        "mapping": {
            "id_field": "id",
            "prompt_field": "question",
            "human_answers_field": "human_answers",
            "ai_answers_field": "chatgpt_answers",
            "optional_source_field": "source",
            "explode_answer_lists": True,
            "canonical_label_mapping": {
                "human_answers": "human",
                "chatgpt_answers": "ai",
            },
            "sample_id_pattern": "{config}:{split}:{id}:{label}:{answer_index}",
        },
    }
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_load_hc3_config_parses_selectors_and_mapping(tmp_path: Path) -> None:
    config_path = tmp_path / "hc3.dataset.json"
    _write_hc3_config(config_path)

    config = load_hc3_config(config_path)

    assert config.dataset_id == "hc3"
    assert config.source_uri == "Hello-SimpleAI/HC3"
    assert tuple(config.selectors.keys()) == ("all_train", "finance_train")
    assert config.mapping.human_answers_field == "human_answers"
    assert config.mapping.ai_answers_field == "chatgpt_answers"


def test_hc3_adapter_load_split_maps_human_and_ai_answers(tmp_path: Path) -> None:
    config_path = tmp_path / "hc3.dataset.json"
    _write_hc3_config(config_path)

    datasets_by_selector: dict[tuple[str, str], FakeDataset] = {
        (
            "all",
            "train",
        ): FakeDataset(
            [
                {
                    "id": "0",
                    "question": "Q0",
                    "human_answers": ["human-0a", "human-0b"],
                    "chatgpt_answers": ["ai-0"],
                    "source": "reddit_eli5",
                },
                {
                    "id": "1",
                    "question": "Q1",
                    "human_answers": ["human-1"],
                    "chatgpt_answers": ["ai-1a", "ai-1b"],
                    "source": "finance",
                },
            ]
        ),
        (
            "finance",
            "train",
        ): FakeDataset([]),
    }

    def dataset_loader(source_uri: str, config: str, split: str) -> FakeDataset:
        assert source_uri == "Hello-SimpleAI/HC3"
        return datasets_by_selector[(config, split)]

    adapter = HC3Adapter(config=load_hc3_config(config_path), dataset_loader=dataset_loader)

    records = adapter.load_split("all_train")

    assert len(records) == 6
    labels = [record.label for record in records]
    assert labels.count("human") == 3
    assert labels.count("ai") == 3
    assert records[0].sample_id == "all:train:0:human:0"
    assert records[0].split == "all_train"
    assert records[0].source_fields["question"] == "Q0"
    assert records[0].source_fields["source"] == "reddit_eli5"


def test_hc3_adapter_validate_selectors_reports_row_check_summary(tmp_path: Path) -> None:
    config_path = tmp_path / "hc3.dataset.json"
    _write_hc3_config(config_path)

    dataset = FakeDataset(
        [
            {
                "id": "0",
                "question": "Q0",
                "human_answers": ["h0"],
                "chatgpt_answers": ["a0"],
            }
        ]
    )

    def dataset_loader(source_uri: str, config: str, split: str) -> FakeDataset:
        assert source_uri == "Hello-SimpleAI/HC3"
        if config == "all":
            return dataset
        return FakeDataset([])

    adapter = HC3Adapter(config=load_hc3_config(config_path), dataset_loader=dataset_loader)

    report = adapter.validate_selectors(sample_rows=1)

    assert report.dataset_id == "hc3"
    assert report.requested_rows_per_selector == 1
    assert len(report.summaries) == 2
    assert all(summary.loaded_rows == 1 or summary.loaded_rows == 0 for summary in report.summaries)
    assert all(summary.missing_required_rows == 0 for summary in report.summaries)
