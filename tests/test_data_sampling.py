"""Tests for deterministic default-random record sampling."""

from __future__ import annotations

from apm.data.sampling import sample_records
from apm.types import CanonicalDatasetRecord


def _build_records(count: int) -> list[CanonicalDatasetRecord]:
    return [
        CanonicalDatasetRecord(
            dataset_id="demo",
            split="train",
            sample_id=f"sample-{index}",
            text=f"text-{index}",
            label="human" if index % 2 == 0 else "ai",
            source_fields={"original_index": index},
        )
        for index in range(count)
    ]


def test_default_sampling_strategy_random_is_deterministic() -> None:
    records = _build_records(10)

    sampled_once = sample_records(records=records, sample_size=4, seed=42)
    sampled_twice = sample_records(records=records, sample_size=4, seed=42)

    sampled_ids_once = [record.sample_id for record in sampled_once]
    sampled_ids_twice = [record.sample_id for record in sampled_twice]
    assert sampled_ids_once == sampled_ids_twice
    assert sampled_ids_once == ["sample-7", "sample-3", "sample-2", "sample-8"]


def test_sampling_without_size_keeps_source_order() -> None:
    records = _build_records(5)

    sampled = sample_records(records=records, sample_size=None, seed=42)

    sampled_ids = [record.sample_id for record in sampled]
    assert sampled_ids == ["sample-0", "sample-1", "sample-2", "sample-3", "sample-4"]
