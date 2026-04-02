"""Tests for canonical dataset record validation rules."""

from __future__ import annotations

import pytest

from apm.data.validation import validate_canonical_records
from apm.types import CanonicalDatasetRecord


def test_validate_canonical_records_accepts_valid_input() -> None:
    records = [
        CanonicalDatasetRecord(
            dataset_id="demo",
            split="train",
            sample_id="1",
            text="hello",
            label="human",
            source_fields={"raw_label": "human"},
        )
    ]

    validate_canonical_records(records)


def test_validate_canonical_records_rejects_empty_text() -> None:
    records = [
        CanonicalDatasetRecord(
            dataset_id="demo",
            split="train",
            sample_id="1",
            text="   ",
            label="human",
            source_fields={},
        )
    ]

    with pytest.raises(ValueError, match="text cannot be empty"):
        validate_canonical_records(records)
