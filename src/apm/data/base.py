"""Interfaces for dataset adapters used by ingestion loaders."""

from __future__ import annotations

from typing import Protocol

from apm.types import CanonicalDatasetRecord


class DatasetAdapter(Protocol):
    """Protocol implemented by dataset-specific adapter integrations."""

    dataset_id: str

    def list_splits(self) -> tuple[str, ...]:
        """Return supported split names for the adapter."""

    def load_split(self, split: str) -> list[CanonicalDatasetRecord]:
        """Load one split and return canonical records."""
