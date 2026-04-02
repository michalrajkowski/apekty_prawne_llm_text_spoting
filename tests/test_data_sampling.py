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

    sampled_once = sample_records(records=records, sample_size=4, per_label_sample_size=None, seed=42)
    sampled_twice = sample_records(records=records, sample_size=4, per_label_sample_size=None, seed=42)

    sampled_ids_once = [record.sample_id for record in sampled_once]
    sampled_ids_twice = [record.sample_id for record in sampled_twice]
    assert sampled_ids_once == sampled_ids_twice
    assert sampled_ids_once == ["sample-7", "sample-3", "sample-2", "sample-8"]


def test_sampling_without_size_keeps_source_order() -> None:
    records = _build_records(5)

    sampled = sample_records(records=records, sample_size=None, per_label_sample_size=None, seed=42)

    sampled_ids = [record.sample_id for record in sampled]
    assert sampled_ids == ["sample-0", "sample-1", "sample-2", "sample-3", "sample-4"]


def test_balanced_sampling_returns_per_label_quota_or_less() -> None:
    records = _build_records(9)

    sampled = sample_records(
        records=records,
        sample_size=None,
        per_label_sample_size=3,
        seed=7,
        sampling_strategy="balanced_random",
    )

    labels = [record.label for record in sampled]
    assert labels.count("human") == 3
    assert labels.count("ai") == 3


def test_balanced_sampling_caps_to_available_class_count() -> None:
    records = [
        CanonicalDatasetRecord(
            dataset_id="demo",
            split="train",
            sample_id=f"h-{index}",
            text=f"text-h-{index}",
            label="human",
            source_fields={},
        )
        for index in range(4)
    ] + [
        CanonicalDatasetRecord(
            dataset_id="demo",
            split="train",
            sample_id="a-0",
            text="text-a-0",
            label="ai",
            source_fields={},
        )
    ]

    sampled = sample_records(
        records=records,
        sample_size=None,
        per_label_sample_size=3,
        seed=11,
        sampling_strategy="balanced_random",
    )

    labels = [record.label for record in sampled]
    assert labels.count("human") == 3
    assert labels.count("ai") == 1
