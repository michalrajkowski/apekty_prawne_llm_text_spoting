"""Validation utilities for canonical dataset records."""

from __future__ import annotations

from collections.abc import Iterable

from apm.types import CanonicalDatasetRecord, DatasetLabel

ALLOWED_LABELS: tuple[DatasetLabel, ...] = ("human", "ai")
REQUIRED_RECORD_FIELDS: tuple[str, ...] = (
    "dataset_id",
    "split",
    "sample_id",
    "text",
    "label",
    "source_fields",
)


def validate_canonical_records(records: Iterable[CanonicalDatasetRecord]) -> None:
    """Validate canonical records and raise ValueError for invalid entries."""

    for index, record in enumerate(records):
        if not record.dataset_id.strip():
            raise ValueError(f"Record {index}: dataset_id cannot be empty.")
        if not record.split.strip():
            raise ValueError(f"Record {index}: split cannot be empty.")
        if not record.sample_id.strip():
            raise ValueError(f"Record {index}: sample_id cannot be empty.")
        if not record.text.strip():
            raise ValueError(f"Record {index}: text cannot be empty.")
        if record.label not in ALLOWED_LABELS:
            raise ValueError(f"Record {index}: unsupported label {record.label!r}.")
        if not isinstance(record.source_fields, dict):
            raise ValueError(f"Record {index}: source_fields must be a dict.")
