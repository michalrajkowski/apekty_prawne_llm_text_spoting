"""Shared typed models for dataset ingestion and normalization workflows."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

DatasetLabel = Literal["human", "ai"]
SamplingStrategy = Literal["random"]


@dataclass(frozen=True, slots=True)
class CanonicalDatasetRecord:
    """Canonical text record used across all dataset adapters."""

    dataset_id: str
    split: str
    sample_id: str
    text: str
    label: DatasetLabel
    source_fields: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class DatasetLoadRequest:
    """Request object for loading a dataset split through an adapter."""

    dataset_id: str
    split: str | None = None
    sample_size: int | None = None
    seed: int = 42
    sampling_strategy: SamplingStrategy = "random"


@dataclass(frozen=True, slots=True)
class DatasetLoadResult:
    """Normalized loader output after optional deterministic sampling."""

    dataset_id: str
    split: str
    total_available: int
    sampled_count: int
    seed: int
    sampling_strategy: SamplingStrategy
    records: tuple[CanonicalDatasetRecord, ...]
