"""Custom local dataset loader for JSONL sources."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, cast

from apm.data.sampling import sample_records
from apm.data.validation import validate_canonical_records
from apm.types import CanonicalDatasetRecord, DatasetLabel, DatasetLoadRequest, DatasetLoadResult


def _parse_label(raw_label: str) -> DatasetLabel:
    """Normalize and validate textual label into canonical label domain."""

    normalized = raw_label.strip().lower()
    if normalized not in {"human", "ai"}:
        raise ValueError(f"Unsupported label value: {raw_label!r}")
    return cast(DatasetLabel, normalized)


def _build_record(dataset_id: str, split: str, line_number: int, payload: dict[str, Any]) -> CanonicalDatasetRecord:
    """Build canonical record from one JSONL object."""

    text_value = payload.get("text")
    if not isinstance(text_value, str):
        raise ValueError(f"Line {line_number}: `text` must be a string.")

    label_value = payload.get("label")
    if not isinstance(label_value, str):
        raise ValueError(f"Line {line_number}: `label` must be a string.")

    sample_id_value = payload.get("sample_id", f"{split}-{line_number}")
    if not isinstance(sample_id_value, str):
        raise ValueError(f"Line {line_number}: `sample_id` must be a string when provided.")

    source_fields_value = payload.get("source_fields", {})
    if not isinstance(source_fields_value, dict):
        raise ValueError(f"Line {line_number}: `source_fields` must be a dict when provided.")

    return CanonicalDatasetRecord(
        dataset_id=dataset_id,
        split=split,
        sample_id=sample_id_value,
        text=text_value,
        label=_parse_label(label_value),
        source_fields=source_fields_value,
    )


def load_custom_jsonl(input_path: Path, request: DatasetLoadRequest) -> DatasetLoadResult:
    """Load local JSONL file into canonical records and apply deterministic sampling."""

    split = request.split or "default"
    raw_lines = input_path.read_text(encoding="utf-8").splitlines()

    records: list[CanonicalDatasetRecord] = []
    for line_number, raw_line in enumerate(raw_lines, start=1):
        if not raw_line.strip():
            continue
        payload = json.loads(raw_line)
        if not isinstance(payload, dict):
            raise ValueError(f"Line {line_number}: JSON object is required.")
        records.append(_build_record(request.dataset_id, split, line_number, payload))

    validate_canonical_records(records)
    sampled_records = sample_records(
        records=records,
        sample_size=request.sample_size,
        seed=request.seed,
        sampling_strategy=request.sampling_strategy,
    )
    return DatasetLoadResult(
        dataset_id=request.dataset_id,
        split=split,
        total_available=len(records),
        sampled_count=len(sampled_records),
        seed=request.seed,
        sampling_strategy=request.sampling_strategy,
        records=tuple(sampled_records),
    )
