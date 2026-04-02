"""Tests for dataset adapter registry behavior."""

from __future__ import annotations

import pytest

from apm.data.dataset_registry import DatasetRegistry
from apm.types import CanonicalDatasetRecord


class DummyAdapter:
    """Simple in-memory adapter used for registry unit tests."""

    def __init__(self, dataset_id: str) -> None:
        self.dataset_id = dataset_id

    def list_splits(self) -> tuple[str, ...]:
        return ("train",)

    def load_split(self, split: str) -> list[CanonicalDatasetRecord]:
        return [
            CanonicalDatasetRecord(
                dataset_id=self.dataset_id,
                split=split,
                sample_id="1",
                text="x",
                label="human",
                source_fields={},
            )
        ]


def test_registry_register_and_resolve_adapter() -> None:
    registry = DatasetRegistry()
    adapter = DummyAdapter(dataset_id="hc3")

    registry.register(dataset_id="hc3", adapter=adapter)

    assert registry.resolve("hc3") is adapter
    assert registry.list_dataset_ids() == ("hc3",)


def test_registry_rejects_duplicate_registration_without_override() -> None:
    registry = DatasetRegistry()
    registry.register(dataset_id="hc3", adapter=DummyAdapter(dataset_id="hc3"))

    with pytest.raises(KeyError, match="already registered"):
        registry.register(dataset_id="hc3", adapter=DummyAdapter(dataset_id="hc3"))


def test_registry_rejects_dataset_id_mismatch() -> None:
    registry = DatasetRegistry()

    with pytest.raises(ValueError, match="mismatch"):
        registry.register(dataset_id="hc3", adapter=DummyAdapter(dataset_id="other"))
