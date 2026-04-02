"""Deterministic sampling helpers for canonical dataset records."""

from __future__ import annotations

import random
from collections.abc import Sequence

from apm.types import CanonicalDatasetRecord, SamplingStrategy

DEFAULT_SAMPLING_STRATEGY: SamplingStrategy = "random"
BALANCED_SAMPLING_STRATEGY: SamplingStrategy = "balanced_random"

_BALANCED_LABEL_ORDER: tuple[str, ...] = ("human", "ai")


def _sample_records_random(
    records: Sequence[CanonicalDatasetRecord],
    sample_size: int | None,
    seed: int,
) -> list[CanonicalDatasetRecord]:
    """Sample records randomly with deterministic seed."""

    if sample_size is None:
        return list(records)
    if sample_size < 0:
        raise ValueError("sample_size must be >= 0.")
    if sample_size == 0 or not records:
        return []

    indices: list[int] = list(range(len(records)))
    random_generator = random.Random(seed)
    random_generator.shuffle(indices)
    selected_indices: list[int] = indices[: min(sample_size, len(indices))]
    return [records[index] for index in selected_indices]


def _sample_records_balanced_random(
    records: Sequence[CanonicalDatasetRecord],
    per_label_sample_size: int | None,
    seed: int,
) -> list[CanonicalDatasetRecord]:
    """Sample records deterministically per label (`human` and `ai`)."""

    if per_label_sample_size is None:
        raise ValueError("per_label_sample_size must be provided for balanced_random strategy.")
    if per_label_sample_size < 0:
        raise ValueError("per_label_sample_size must be >= 0.")
    if per_label_sample_size == 0 or not records:
        return []

    indices_by_label: dict[str, list[int]] = {label: [] for label in _BALANCED_LABEL_ORDER}
    for index, record in enumerate(records):
        if record.label in indices_by_label:
            indices_by_label[record.label].append(index)

    selected_indices: list[int] = []
    for label_offset, label in enumerate(_BALANCED_LABEL_ORDER):
        label_indices = list(indices_by_label[label])
        label_rng = random.Random(seed + label_offset)
        label_rng.shuffle(label_indices)
        selected_indices.extend(label_indices[: min(per_label_sample_size, len(label_indices))])

    merged_rng = random.Random(seed)
    merged_rng.shuffle(selected_indices)
    return [records[index] for index in selected_indices]


def sample_records(
    records: Sequence[CanonicalDatasetRecord],
    sample_size: int | None,
    per_label_sample_size: int | None,
    seed: int,
    sampling_strategy: SamplingStrategy = DEFAULT_SAMPLING_STRATEGY,
) -> list[CanonicalDatasetRecord]:
    """Sample records using the configured strategy and explicit seed."""

    if sampling_strategy == "random":
        return _sample_records_random(records=records, sample_size=sample_size, seed=seed)
    if sampling_strategy == "balanced_random":
        return _sample_records_balanced_random(
            records=records,
            per_label_sample_size=per_label_sample_size,
            seed=seed,
        )
    raise ValueError(f"Unsupported sampling strategy: {sampling_strategy!r}")
