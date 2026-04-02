"""Universal registry-backed dataset loader for adapter integrations."""

from __future__ import annotations

from apm.data.dataset_registry import DATASET_REGISTRY, DatasetRegistry
from apm.data.sampling import sample_records
from apm.data.validation import validate_canonical_records
from apm.types import CanonicalDatasetRecord, DatasetLoadRequest, DatasetLoadResult


def _resolve_split(available_splits: tuple[str, ...], requested_split: str | None) -> str:
    """Resolve split name from explicit request or adapter defaults."""

    if not available_splits:
        raise ValueError("Adapter did not expose any splits.")
    if requested_split is None:
        return available_splits[0]
    if requested_split not in available_splits:
        raise ValueError(f"Unknown split {requested_split!r}. Available splits: {available_splits!r}")
    return requested_split


def _validate_record_context(records: list[CanonicalDatasetRecord], dataset_id: str, split: str) -> None:
    """Validate adapter records against request-level dataset context."""

    for index, record in enumerate(records):
        if record.dataset_id != dataset_id:
            raise ValueError(
                f"Record {index} has dataset_id={record.dataset_id!r}, expected {dataset_id!r}."
            )
        if record.split != split:
            raise ValueError(f"Record {index} has split={record.split!r}, expected {split!r}.")


def load_dataset(request: DatasetLoadRequest, registry: DatasetRegistry = DATASET_REGISTRY) -> DatasetLoadResult:
    """Load dataset split through a registered adapter with deterministic sampling."""

    adapter = registry.resolve(request.dataset_id)
    available_splits = adapter.list_splits()
    selected_split = _resolve_split(available_splits, request.split)
    records = adapter.load_split(selected_split)

    validate_canonical_records(records)
    _validate_record_context(records, dataset_id=request.dataset_id, split=selected_split)
    sampled_records = sample_records(
        records=records,
        sample_size=request.sample_size,
        seed=request.seed,
        sampling_strategy=request.sampling_strategy,
    )

    return DatasetLoadResult(
        dataset_id=request.dataset_id,
        split=selected_split,
        total_available=len(records),
        sampled_count=len(sampled_records),
        seed=request.seed,
        sampling_strategy=request.sampling_strategy,
        records=tuple(sampled_records),
    )
