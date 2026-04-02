"""Path and persistence helpers for dataset storage artifacts."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping

import pandas as pd

from apm.types import CanonicalDatasetRecord
from apm.types import SamplingStrategy


@dataclass(frozen=True, slots=True)
class DatasetArtifactPaths:
    """Filesystem paths used for one dataset split lifecycle."""

    raw_snapshot_dir: Path
    normalized_parquet_path: Path
    metadata_json_path: Path


def resolve_dataset_artifact_paths(project_root: Path, dataset_id: str, split: str) -> DatasetArtifactPaths:
    """Resolve deterministic paths for raw snapshots and normalized artifacts."""

    normalized_dataset_id = dataset_id.strip()
    normalized_split = split.strip()
    if not normalized_dataset_id:
        raise ValueError("dataset_id cannot be empty.")
    if not normalized_split:
        raise ValueError("split cannot be empty.")

    raw_snapshot_dir = project_root / "data" / "raw" / "datasets" / normalized_dataset_id / normalized_split
    normalized_parquet_path = (
        project_root / "data" / "interim" / "datasets" / normalized_dataset_id / f"{normalized_split}.parquet"
    )
    metadata_json_path = (
        project_root / "data" / "interim" / "datasets" / normalized_dataset_id / f"{normalized_split}.metadata.json"
    )
    return DatasetArtifactPaths(
        raw_snapshot_dir=raw_snapshot_dir,
        normalized_parquet_path=normalized_parquet_path,
        metadata_json_path=metadata_json_path,
    )


def ensure_artifact_parent_dirs(paths: DatasetArtifactPaths) -> None:
    """Create required directories for raw, normalized, and metadata outputs."""

    paths.raw_snapshot_dir.mkdir(parents=True, exist_ok=True)
    paths.normalized_parquet_path.parent.mkdir(parents=True, exist_ok=True)
    paths.metadata_json_path.parent.mkdir(parents=True, exist_ok=True)


def build_metadata_payload(
    dataset_id: str,
    split: str,
    seed: int,
    sampling_strategy: SamplingStrategy,
    source_uri: str,
    source_revision: str | None,
) -> dict[str, Any]:
    """Build canonical metadata payload for persisted normalized artifacts."""

    metadata: dict[str, Any] = {
        "dataset_id": dataset_id,
        "split": split,
        "seed": seed,
        "sampling_strategy": sampling_strategy,
        "source_uri": source_uri,
        "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }
    if source_revision is not None:
        metadata["source_revision"] = source_revision
    return metadata


def write_metadata_json(metadata_path: Path, metadata: Mapping[str, Any]) -> None:
    """Persist metadata payload in deterministic JSON representation."""

    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    serialized = json.dumps(dict(metadata), indent=2, ensure_ascii=False, sort_keys=True) + "\n"
    metadata_path.write_text(serialized, encoding="utf-8")


def write_raw_snapshot_jsonl(raw_snapshot_dir: Path, records: list[CanonicalDatasetRecord]) -> Path:
    """Persist canonical records as JSONL raw snapshot for auditability."""

    raw_snapshot_dir.mkdir(parents=True, exist_ok=True)
    snapshot_path = raw_snapshot_dir / "sampled_records.jsonl"

    lines: list[str] = []
    for record in records:
        payload = {
            "dataset_id": record.dataset_id,
            "split": record.split,
            "sample_id": record.sample_id,
            "text": record.text,
            "label": record.label,
            "source_fields": record.source_fields,
        }
        lines.append(json.dumps(payload, ensure_ascii=False))
    snapshot_path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
    return snapshot_path


def write_normalized_parquet(parquet_path: Path, records: list[CanonicalDatasetRecord]) -> Path:
    """Persist canonical records into normalized Parquet format."""

    parquet_path.parent.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    for record in records:
        rows.append(
            {
                "dataset_id": record.dataset_id,
                "split": record.split,
                "sample_id": record.sample_id,
                "text": record.text,
                "label": record.label,
                "source_fields": json.dumps(record.source_fields, ensure_ascii=False, sort_keys=True),
            }
        )
    dataframe = pd.DataFrame(rows)
    dataframe.to_parquet(parquet_path, index=False)
    return parquet_path
