"""Tests for universal registry-backed dataset loader behavior."""

from __future__ import annotations

import pytest

from apm.data.dataset_registry import DatasetRegistry
from apm.data.hf_loader import load_dataset
from apm.types import CanonicalDatasetRecord, DatasetLoadRequest


class InMemoryAdapter:
    """Test adapter exposing two splits from in-memory records."""

    def __init__(self, dataset_id: str, split_to_records: dict[str, list[CanonicalDatasetRecord]]) -> None:
        self.dataset_id = dataset_id
        self._split_to_records = split_to_records

    def list_splits(self) -> tuple[str, ...]:
        return tuple(self._split_to_records.keys())

    def load_split(self, split: str) -> list[CanonicalDatasetRecord]:
        return list(self._split_to_records[split])


def _build_records(dataset_id: str, split: str, count: int) -> list[CanonicalDatasetRecord]:
    return [
        CanonicalDatasetRecord(
            dataset_id=dataset_id,
            split=split,
            sample_id=f"{split}-{index}",
            text=f"text-{index}",
            label="human" if index % 2 == 0 else "ai",
            source_fields={"index": index},
        )
        for index in range(count)
    ]


def test_load_dataset_uses_default_random_sampling() -> None:
    dataset_id = "demo_dataset"
    registry = DatasetRegistry()
    adapter = InMemoryAdapter(
        dataset_id=dataset_id,
        split_to_records={
            "train": _build_records(dataset_id, "train", 5),
            "test": _build_records(dataset_id, "test", 3),
        },
    )
    registry.register(dataset_id=dataset_id, adapter=adapter)

    result = load_dataset(
        request=DatasetLoadRequest(dataset_id=dataset_id, sample_size=3, seed=42),
        registry=registry,
    )

    assert result.split == "train"
    assert result.total_available == 5
    assert result.sampled_count == 3
    assert result.sampling_strategy == "random"
    assert [record.sample_id for record in result.records] == ["train-3", "train-1", "train-2"]


def test_load_dataset_rejects_unknown_split() -> None:
    dataset_id = "demo_dataset"
    registry = DatasetRegistry()
    adapter = InMemoryAdapter(dataset_id=dataset_id, split_to_records={"train": _build_records(dataset_id, "train", 2)})
    registry.register(dataset_id=dataset_id, adapter=adapter)

    with pytest.raises(ValueError, match="Unknown split"):
        load_dataset(
            request=DatasetLoadRequest(dataset_id=dataset_id, split="missing", sample_size=1, seed=42),
            registry=registry,
        )
