"""Dataset ingestion interfaces, loaders, registry, and utilities."""

from apm.data.base import DatasetAdapter
from apm.data.custom_loader import load_custom_jsonl
from apm.data.dataset_registry import DATASET_REGISTRY, DatasetRegistry, get_dataset_adapter, register_dataset_adapter
from apm.data.hf_loader import load_dataset
from apm.data.sampling import DEFAULT_SAMPLING_STRATEGY, sample_records
from apm.data.storage import (
    DatasetArtifactPaths,
    build_metadata_payload,
    ensure_artifact_parent_dirs,
    resolve_dataset_artifact_paths,
    write_metadata_json,
)
from apm.data.validation import ALLOWED_LABELS, REQUIRED_RECORD_FIELDS, validate_canonical_records

__all__ = [
    "ALLOWED_LABELS",
    "DATASET_REGISTRY",
    "DEFAULT_SAMPLING_STRATEGY",
    "DatasetAdapter",
    "DatasetArtifactPaths",
    "DatasetRegistry",
    "REQUIRED_RECORD_FIELDS",
    "build_metadata_payload",
    "ensure_artifact_parent_dirs",
    "get_dataset_adapter",
    "load_custom_jsonl",
    "load_dataset",
    "register_dataset_adapter",
    "resolve_dataset_artifact_paths",
    "sample_records",
    "validate_canonical_records",
    "write_metadata_json",
]
