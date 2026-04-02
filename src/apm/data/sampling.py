"""Deterministic sampling helpers for canonical dataset records."""

from __future__ import annotations

import random
from collections.abc import Sequence

from apm.types import CanonicalDatasetRecord, SamplingStrategy

DEFAULT_SAMPLING_STRATEGY: SamplingStrategy = "random"


def sample_records(
    records: Sequence[CanonicalDatasetRecord],
    sample_size: int | None,
    seed: int,
    sampling_strategy: SamplingStrategy = DEFAULT_SAMPLING_STRATEGY,
) -> list[CanonicalDatasetRecord]:
    """Sample records using the configured strategy and explicit seed."""

    if sampling_strategy != "random":
        raise ValueError(f"Unsupported sampling strategy: {sampling_strategy!r}")
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
