"""Dataset adapter registry for name-based adapter lookup."""

from __future__ import annotations

from dataclasses import dataclass, field

from apm.data.base import DatasetAdapter


@dataclass(slots=True)
class DatasetRegistry:
    """In-memory registry of dataset adapters keyed by dataset id."""

    _adapters: dict[str, DatasetAdapter] = field(default_factory=dict)

    def register(self, dataset_id: str, adapter: DatasetAdapter, *, override: bool = False) -> None:
        """Register adapter under `dataset_id` with optional override."""

        normalized_dataset_id = dataset_id.strip()
        if not normalized_dataset_id:
            raise ValueError("dataset_id cannot be empty.")

        adapter_dataset_id = adapter.dataset_id.strip()
        if adapter_dataset_id != normalized_dataset_id:
            raise ValueError(
                f"Adapter dataset_id mismatch: expected {normalized_dataset_id!r}, got {adapter_dataset_id!r}."
            )
        if normalized_dataset_id in self._adapters and not override:
            raise KeyError(f"Dataset adapter already registered: {normalized_dataset_id!r}")
        self._adapters[normalized_dataset_id] = adapter

    def resolve(self, dataset_id: str) -> DatasetAdapter:
        """Resolve dataset adapter by id."""

        normalized_dataset_id = dataset_id.strip()
        if normalized_dataset_id not in self._adapters:
            raise KeyError(f"Unknown dataset adapter: {normalized_dataset_id!r}")
        return self._adapters[normalized_dataset_id]

    def list_dataset_ids(self) -> tuple[str, ...]:
        """Return a sorted tuple with all registered dataset ids."""

        return tuple(sorted(self._adapters.keys()))

    def clear(self) -> None:
        """Remove all registered adapters."""

        self._adapters.clear()


DATASET_REGISTRY = DatasetRegistry()


def register_dataset_adapter(dataset_id: str, adapter: DatasetAdapter, *, override: bool = False) -> None:
    """Register a dataset adapter in the global registry."""

    DATASET_REGISTRY.register(dataset_id=dataset_id, adapter=adapter, override=override)


def get_dataset_adapter(dataset_id: str) -> DatasetAdapter:
    """Resolve a dataset adapter from the global registry."""

    return DATASET_REGISTRY.resolve(dataset_id)
