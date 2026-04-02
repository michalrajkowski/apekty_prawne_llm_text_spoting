"""GriD dataset adapter for GitHub-hosted CSV sources with auto-download bootstrap."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping
from urllib.request import urlretrieve

import pandas as pd

from apm.types import CanonicalDatasetRecord, DatasetLabel


@dataclass(frozen=True, slots=True)
class GridSplit:
    """Source file selector for one logical split."""

    path: Path


@dataclass(frozen=True, slots=True)
class GridMapping:
    """Field mapping from source CSV rows into canonical records."""

    id_field: str | None
    text_field: str
    label_field: str
    optional_fields: tuple[str, ...]
    canonical_label_mapping: dict[str, DatasetLabel]
    sample_id_pattern: str


@dataclass(frozen=True, slots=True)
class GridDownloadFile:
    """One source file mapping for downloader bootstrap."""

    name: str
    url: str


@dataclass(frozen=True, slots=True)
class GridDownloadConfig:
    """Download/bootstrap configuration for local source files."""

    output_dir: Path
    files: tuple[GridDownloadFile, ...]
    auto_download_if_missing: bool
    force_download: bool


@dataclass(frozen=True, slots=True)
class GridConfig:
    """Parsed adapter config for GriD CSV source files."""

    dataset_id: str
    source_uri: str
    splits: dict[str, GridSplit]
    mapping: GridMapping
    download: GridDownloadConfig | None


def _parse_label(value: Any, field_name: str) -> DatasetLabel:
    """Parse canonical label values from config."""

    if value not in {"human", "ai"}:
        raise ValueError(f"{field_name} must be one of ['human', 'ai'].")
    return value


def _is_missing_scalar(value: Any) -> bool:
    """Return True for scalar missing values (None/NaN/pd.NA)."""

    missing = pd.isna(value)
    return isinstance(missing, bool) and missing


def _normalize_label_key(raw_value: Any) -> str:
    """Normalize raw label value from CSV to mapping key string."""

    if _is_missing_scalar(raw_value):
        return ""
    if isinstance(raw_value, bool):
        return "1" if raw_value else "0"
    if isinstance(raw_value, int):
        return str(raw_value)
    if isinstance(raw_value, float):
        if raw_value.is_integer():
            return str(int(raw_value))
        return str(raw_value).strip()
    return str(raw_value).strip()


def _coerce_row_id(raw_value: Any) -> str:
    """Convert row identifier value to non-empty string."""

    if _is_missing_scalar(raw_value):
        raise ValueError("Source row id is missing.")
    if isinstance(raw_value, str):
        normalized = raw_value.strip()
    elif isinstance(raw_value, int):
        normalized = str(raw_value)
    elif isinstance(raw_value, float) and raw_value.is_integer():
        normalized = str(int(raw_value))
    else:
        normalized = str(raw_value).strip()
    if not normalized:
        raise ValueError("Source row id cannot be empty.")
    return normalized


def _resolve_base_dir(path_value: Path, project_root: Path) -> Path:
    """Resolve a config path against project root when relative."""

    if path_value.is_absolute():
        return path_value
    return project_root / path_value


def _parse_download_config(raw_download: Mapping[str, Any]) -> GridDownloadConfig:
    """Parse optional download/bootstrap config."""

    output_dir = raw_download.get("output_dir")
    raw_files = raw_download.get("files")
    auto_download_if_missing = raw_download.get("auto_download_if_missing", True)
    force_download = raw_download.get("force_download", False)

    if not isinstance(output_dir, str) or not output_dir.strip():
        raise ValueError("download.output_dir must be a non-empty string.")
    if not isinstance(raw_files, list) or not raw_files:
        raise ValueError("download.files must be a non-empty list.")
    if not isinstance(auto_download_if_missing, bool):
        raise ValueError("download.auto_download_if_missing must be boolean.")
    if not isinstance(force_download, bool):
        raise ValueError("download.force_download must be boolean.")

    files: list[GridDownloadFile] = []
    for index, raw_file in enumerate(raw_files):
        if not isinstance(raw_file, dict):
            raise ValueError(f"download.files[{index}] must be an object.")
        name = raw_file.get("name")
        url = raw_file.get("url")
        if not isinstance(name, str) or not name.strip():
            raise ValueError(f"download.files[{index}].name must be a non-empty string.")
        if not isinstance(url, str) or not url.strip():
            raise ValueError(f"download.files[{index}].url must be a non-empty string.")
        files.append(GridDownloadFile(name=name, url=url))

    return GridDownloadConfig(
        output_dir=Path(output_dir),
        files=tuple(files),
        auto_download_if_missing=auto_download_if_missing,
        force_download=force_download,
    )


def load_grid_config(config_path: Path) -> GridConfig:
    """Load GriD adapter config from JSON file."""

    payload = json.loads(config_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("GriD config root must be a JSON object.")

    dataset_id = payload.get("dataset_id")
    source_uri = payload.get("source_uri")
    if not isinstance(dataset_id, str) or not dataset_id.strip():
        raise ValueError("dataset_id must be a non-empty string.")
    if not isinstance(source_uri, str) or not source_uri.strip():
        raise ValueError("source_uri must be a non-empty string.")

    raw_splits = payload.get("splits")
    if not isinstance(raw_splits, dict) or not raw_splits:
        raise ValueError("splits must be a non-empty object.")
    splits: dict[str, GridSplit] = {}
    for split_name, raw_split in raw_splits.items():
        if not isinstance(split_name, str) or not split_name.strip():
            raise ValueError("Split key must be a non-empty string.")
        if not isinstance(raw_split, dict):
            raise ValueError(f"Split {split_name!r} must be an object.")
        split_path = raw_split.get("path")
        if not isinstance(split_path, str) or not split_path.strip():
            raise ValueError(f"splits[{split_name!r}].path must be a non-empty string.")
        splits[split_name] = GridSplit(path=Path(split_path))

    raw_mapping = payload.get("mapping")
    if not isinstance(raw_mapping, dict):
        raise ValueError("mapping must be an object.")
    id_field = raw_mapping.get("id_field")
    text_field = raw_mapping.get("text_field")
    label_field = raw_mapping.get("label_field")
    optional_fields = raw_mapping.get("optional_fields", [])
    sample_id_pattern = raw_mapping.get("sample_id_pattern")
    if id_field is not None and (not isinstance(id_field, str) or not id_field.strip()):
        raise ValueError("mapping.id_field must be a non-empty string or null.")
    if not isinstance(text_field, str) or not text_field.strip():
        raise ValueError("mapping.text_field must be a non-empty string.")
    if not isinstance(label_field, str) or not label_field.strip():
        raise ValueError("mapping.label_field must be a non-empty string.")
    if not isinstance(optional_fields, list) or not all(
        isinstance(field_name, str) and field_name.strip() for field_name in optional_fields
    ):
        raise ValueError("mapping.optional_fields must be a list of non-empty strings.")
    if not isinstance(sample_id_pattern, str) or not sample_id_pattern.strip():
        raise ValueError("mapping.sample_id_pattern must be a non-empty string.")

    raw_label_mapping = raw_mapping.get("canonical_label_mapping")
    if not isinstance(raw_label_mapping, dict) or not raw_label_mapping:
        raise ValueError("mapping.canonical_label_mapping must be a non-empty object.")
    canonical_label_mapping: dict[str, DatasetLabel] = {}
    for source_label_key, canonical_label in raw_label_mapping.items():
        if not isinstance(source_label_key, str) or not source_label_key.strip():
            raise ValueError("mapping.canonical_label_mapping keys must be non-empty strings.")
        canonical_label_mapping[source_label_key] = _parse_label(
            canonical_label,
            f"mapping.canonical_label_mapping[{source_label_key!r}]",
        )

    mapping = GridMapping(
        id_field=id_field,
        text_field=text_field,
        label_field=label_field,
        optional_fields=tuple(optional_fields),
        canonical_label_mapping=canonical_label_mapping,
        sample_id_pattern=sample_id_pattern,
    )

    raw_download = payload.get("download")
    if raw_download is None:
        download = None
    elif isinstance(raw_download, dict):
        download = _parse_download_config(raw_download)
    else:
        raise ValueError("download must be an object when provided.")

    return GridConfig(
        dataset_id=dataset_id,
        source_uri=source_uri,
        splits=splits,
        mapping=mapping,
        download=download,
    )


def ensure_grid_sources_available(config: GridConfig, project_root: Path) -> None:
    """Ensure required local source files exist; auto-download when configured."""

    download_config = config.download
    if download_config is None:
        return

    output_dir = _resolve_base_dir(download_config.output_dir, project_root=project_root)
    output_dir.mkdir(parents=True, exist_ok=True)

    for file_spec in download_config.files:
        target_path = output_dir / file_spec.name
        if target_path.exists() and not download_config.force_download:
            continue
        if not download_config.auto_download_if_missing and not target_path.exists():
            raise FileNotFoundError(f"Missing required source file: {target_path}")
        urlretrieve(file_spec.url, target_path)

    for split_name, split_payload in config.splits.items():
        split_path = _resolve_base_dir(split_payload.path, project_root=project_root)
        if not split_path.exists():
            raise FileNotFoundError(f"Missing source file for split {split_name!r}: {split_path}")


class GridAdapter:
    """Dataset adapter for GriD CSV source files."""

    def __init__(self, config: GridConfig, project_root: Path | None = None) -> None:
        self._config = config
        self._project_root = project_root if project_root is not None else Path.cwd()
        self.dataset_id = config.dataset_id

    @classmethod
    def from_config_path(cls, config_path: Path, project_root: Path | None = None) -> "GridAdapter":
        """Create adapter from config path."""

        return cls(config=load_grid_config(config_path), project_root=project_root)

    @property
    def source_uri(self) -> str:
        """Return configured source URI."""

        return self._config.source_uri

    def list_splits(self) -> tuple[str, ...]:
        """List configured logical split names."""

        return tuple(self._config.splits.keys())

    def _resolve_split_path(self, split: str) -> Path:
        """Resolve configured split file path against project root when relative."""

        if split not in self._config.splits:
            raise ValueError(f"Unknown split {split!r}.")
        split_path = self._config.splits[split].path
        return _resolve_base_dir(split_path, project_root=self._project_root)

    def resolve_split_path(self, split: str) -> Path:
        """Return resolved filesystem path for one configured split."""

        return self._resolve_split_path(split)

    def ensure_sources_available(self) -> None:
        """Ensure required local files exist for all configured splits."""

        split_to_path = {split: self._resolve_split_path(split) for split in self.list_splits()}
        if all(path.exists() for path in split_to_path.values()):
            return

        ensure_grid_sources_available(config=self._config, project_root=self._project_root)
        for split, source_path in split_to_path.items():
            if not source_path.exists():
                raise FileNotFoundError(f"Missing source file for split {split!r}: {source_path}")

    def _build_source_fields(
        self,
        row_payload: Mapping[str, Any],
        row_id: str,
        row_index: int,
        split: str,
        source_path: Path,
        raw_label: str,
    ) -> dict[str, Any]:
        """Build canonical source_fields payload from row context and optional columns."""

        source_fields: dict[str, Any] = {
            "source_dataset": self._config.source_uri,
            "source_file": str(source_path),
            "source_split": split,
            "source_id": row_id,
            "row_index": row_index,
            "raw_label": raw_label,
        }
        for field_name in self._config.mapping.optional_fields:
            if field_name not in row_payload:
                continue
            field_value = row_payload[field_name]
            if _is_missing_scalar(field_value):
                continue
            source_fields[field_name] = field_value
        return source_fields

    def _build_row_id(self, row_payload: Mapping[str, Any], row_index: int) -> str:
        """Build canonical row id from configured field or row index fallback."""

        id_field = self._config.mapping.id_field
        if id_field is None:
            return str(row_index)
        if id_field not in row_payload:
            raise ValueError(f"Missing id field {id_field!r} in row {row_index}.")
        return _coerce_row_id(row_payload[id_field])

    def _build_record(
        self,
        split: str,
        row_payload: Mapping[str, Any],
        row_index: int,
        source_path: Path,
    ) -> CanonicalDatasetRecord:
        """Map one CSV row into canonical dataset record."""

        mapping = self._config.mapping
        if mapping.text_field not in row_payload:
            raise ValueError(f"Missing text field {mapping.text_field!r} in row {row_index}.")
        if mapping.label_field not in row_payload:
            raise ValueError(f"Missing label field {mapping.label_field!r} in row {row_index}.")

        row_id = self._build_row_id(row_payload=row_payload, row_index=row_index)
        raw_text_value = row_payload[mapping.text_field]
        if not isinstance(raw_text_value, str):
            raise ValueError(f"Row {row_index} field {mapping.text_field!r} must be string.")
        text_value = raw_text_value.strip()
        if not text_value:
            raise ValueError(f"Row {row_index} field {mapping.text_field!r} cannot be empty.")

        raw_label_key = _normalize_label_key(row_payload[mapping.label_field])
        if raw_label_key not in mapping.canonical_label_mapping:
            raise ValueError(
                f"Row {row_index} has unmapped label value {row_payload[mapping.label_field]!r}."
            )
        label = mapping.canonical_label_mapping[raw_label_key]
        sample_id = mapping.sample_id_pattern.format(
            split=split,
            id=row_id,
            row_index=row_index,
            label=label,
            raw_label=raw_label_key,
        )
        source_fields = self._build_source_fields(
            row_payload=row_payload,
            row_id=row_id,
            row_index=row_index,
            split=split,
            source_path=source_path,
            raw_label=raw_label_key,
        )
        return CanonicalDatasetRecord(
            dataset_id=self.dataset_id,
            split=split,
            sample_id=sample_id,
            text=text_value,
            label=label,
            source_fields=source_fields,
        )

    def load_split(self, split: str) -> list[CanonicalDatasetRecord]:
        """Load configured split CSV and map rows into canonical records."""

        self.ensure_sources_available()
        source_path = self._resolve_split_path(split)
        dataframe = pd.read_csv(source_path)
        source_rows = dataframe.to_dict(orient="records")
        records: list[CanonicalDatasetRecord] = []
        for row_index, row_payload in enumerate(source_rows):
            records.append(
                self._build_record(
                    split=split,
                    row_payload=row_payload,
                    row_index=row_index,
                    source_path=source_path,
                )
            )
        return records
