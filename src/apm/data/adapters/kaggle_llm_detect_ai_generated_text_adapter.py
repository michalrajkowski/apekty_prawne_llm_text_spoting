"""Kaggle LLM Detect AI Generated Text adapter with local-file bootstrap support."""

from __future__ import annotations

import json
import shutil
import subprocess
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import pandas as pd

from apm.types import CanonicalDatasetRecord, DatasetLabel


@dataclass(frozen=True, slots=True)
class KaggleSplit:
    """Source file selector for one logical split."""

    path: Path


@dataclass(frozen=True, slots=True)
class KaggleMapping:
    """Field mapping from source CSV rows into canonical records."""

    id_field: str
    text_field: str
    label_field: str
    optional_fields: tuple[str, ...]
    canonical_label_mapping: dict[str, DatasetLabel]
    sample_id_pattern: str


@dataclass(frozen=True, slots=True)
class KaggleDownloadConfig:
    """Download/bootstrap configuration for Kaggle local source files."""

    competition: str
    output_dir: Path
    archive_filename: str
    required_files: tuple[str, ...]
    auto_download_if_missing: bool
    force_download: bool


@dataclass(frozen=True, slots=True)
class KaggleConfig:
    """Parsed adapter config for Kaggle CSV source files."""

    dataset_id: str
    source_uri: str
    splits: dict[str, KaggleSplit]
    mapping: KaggleMapping
    download: KaggleDownloadConfig | None


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


def _parse_download_config(raw_download: Mapping[str, Any], splits: Mapping[str, KaggleSplit]) -> KaggleDownloadConfig:
    """Parse optional Kaggle download/bootstrap config."""

    competition = raw_download.get("competition")
    output_dir = raw_download.get("output_dir")
    archive_filename = raw_download.get("archive_filename")
    required_files = raw_download.get("required_files")
    auto_download_if_missing = raw_download.get("auto_download_if_missing", True)
    force_download = raw_download.get("force_download", False)

    if not isinstance(competition, str) or not competition.strip():
        raise ValueError("download.competition must be a non-empty string.")
    if not isinstance(output_dir, str) or not output_dir.strip():
        raise ValueError("download.output_dir must be a non-empty string.")
    if archive_filename is None:
        normalized_archive_filename = f"{competition}.zip"
    elif isinstance(archive_filename, str) and archive_filename.strip():
        normalized_archive_filename = archive_filename
    else:
        raise ValueError("download.archive_filename must be a non-empty string when provided.")
    if not isinstance(auto_download_if_missing, bool):
        raise ValueError("download.auto_download_if_missing must be boolean.")
    if not isinstance(force_download, bool):
        raise ValueError("download.force_download must be boolean.")

    default_required_files = tuple(dict.fromkeys(split.path.name for split in splits.values()))
    if required_files is None:
        normalized_required_files = default_required_files
    elif isinstance(required_files, list) and all(
        isinstance(file_name, str) and file_name.strip() for file_name in required_files
    ):
        normalized_required_files = tuple(required_files)
    else:
        raise ValueError("download.required_files must be a list of non-empty strings when provided.")
    if not normalized_required_files:
        raise ValueError("download.required_files cannot be empty.")

    return KaggleDownloadConfig(
        competition=competition,
        output_dir=Path(output_dir),
        archive_filename=normalized_archive_filename,
        required_files=normalized_required_files,
        auto_download_if_missing=auto_download_if_missing,
        force_download=force_download,
    )


def _resolve_base_dir(path_value: Path, project_root: Path) -> Path:
    """Resolve a config path against project root when relative."""

    if path_value.is_absolute():
        return path_value
    return project_root / path_value


def _missing_required_paths(base_dir: Path, required_files: tuple[str, ...]) -> tuple[Path, ...]:
    """Return required file paths that are currently missing."""

    missing = [base_dir / file_name for file_name in required_files if not (base_dir / file_name).exists()]
    return tuple(missing)


def _run_kaggle_download(download_dir: Path, competition: str, force_download: bool) -> None:
    """Run Kaggle CLI download command for a competition dataset bundle."""

    kaggle_cli_path = shutil.which("kaggle")
    if kaggle_cli_path is None:
        raise RuntimeError(
            "kaggle CLI not found. Install it and configure ~/.kaggle/kaggle.json to enable auto-download."
        )
    command: list[str] = [
        kaggle_cli_path,
        "competitions",
        "download",
        "-c",
        competition,
        "-p",
        str(download_dir),
    ]
    if force_download:
        command.append("--force")
    subprocess.run(command, check=True)


def _resolve_archive_path(download_dir: Path, archive_filename: str) -> Path:
    """Resolve archive path from explicit filename or single-zip fallback."""

    archive_path = download_dir / archive_filename
    if archive_path.exists():
        return archive_path

    zip_candidates = sorted(download_dir.glob("*.zip"))
    if len(zip_candidates) == 1:
        return zip_candidates[0]
    raise FileNotFoundError(
        f"Expected archive {archive_path} not found and could not infer unique fallback zip in {download_dir}."
    )


def ensure_kaggle_sources_available(config: KaggleConfig, project_root: Path) -> None:
    """Ensure required local source files exist; auto-download from Kaggle when configured."""

    download_config = config.download
    if download_config is None:
        return

    download_dir = _resolve_base_dir(download_config.output_dir, project_root=project_root)
    missing_before = _missing_required_paths(download_dir, download_config.required_files)
    if not missing_before:
        return

    if not download_config.auto_download_if_missing:
        missing_rendered = ", ".join(str(path) for path in missing_before)
        raise FileNotFoundError(f"Missing required Kaggle source files: {missing_rendered}")

    download_dir.mkdir(parents=True, exist_ok=True)
    _run_kaggle_download(
        download_dir=download_dir,
        competition=download_config.competition,
        force_download=download_config.force_download,
    )
    archive_path = _resolve_archive_path(download_dir=download_dir, archive_filename=download_config.archive_filename)
    with zipfile.ZipFile(archive_path) as archive:
        archive.extractall(download_dir)

    missing_after = _missing_required_paths(download_dir, download_config.required_files)
    if missing_after:
        missing_rendered = ", ".join(str(path) for path in missing_after)
        raise FileNotFoundError(
            f"Kaggle download completed but required files are still missing: {missing_rendered}"
        )


def load_kaggle_config(config_path: Path) -> KaggleConfig:
    """Load Kaggle adapter config from JSON file."""

    payload = json.loads(config_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("Kaggle config root must be a JSON object.")

    dataset_id = payload.get("dataset_id")
    source_uri = payload.get("source_uri")
    if not isinstance(dataset_id, str) or not dataset_id.strip():
        raise ValueError("dataset_id must be a non-empty string.")
    if not isinstance(source_uri, str) or not source_uri.strip():
        raise ValueError("source_uri must be a non-empty string.")

    raw_splits = payload.get("splits")
    if not isinstance(raw_splits, dict) or not raw_splits:
        raise ValueError("splits must be a non-empty object.")
    splits: dict[str, KaggleSplit] = {}
    for split_name, raw_split in raw_splits.items():
        if not isinstance(split_name, str) or not split_name.strip():
            raise ValueError("Split key must be a non-empty string.")
        if not isinstance(raw_split, dict):
            raise ValueError(f"Split {split_name!r} must be an object.")
        split_path = raw_split.get("path")
        if not isinstance(split_path, str) or not split_path.strip():
            raise ValueError(f"splits[{split_name!r}].path must be a non-empty string.")
        splits[split_name] = KaggleSplit(path=Path(split_path))

    raw_mapping = payload.get("mapping")
    if not isinstance(raw_mapping, dict):
        raise ValueError("mapping must be an object.")
    id_field = raw_mapping.get("id_field")
    text_field = raw_mapping.get("text_field")
    label_field = raw_mapping.get("label_field")
    optional_fields = raw_mapping.get("optional_fields", [])
    sample_id_pattern = raw_mapping.get("sample_id_pattern")
    if not isinstance(id_field, str) or not id_field.strip():
        raise ValueError("mapping.id_field must be a non-empty string.")
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

    mapping = KaggleMapping(
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
        download = _parse_download_config(raw_download, splits=splits)
    else:
        raise ValueError("download must be an object when provided.")

    return KaggleConfig(
        dataset_id=dataset_id,
        source_uri=source_uri,
        splits=splits,
        mapping=mapping,
        download=download,
    )


class KaggleLlmDetectAiGeneratedTextAdapter:
    """Dataset adapter for Kaggle LLM Detect AI Generated Text CSV exports."""

    def __init__(self, config: KaggleConfig, project_root: Path | None = None) -> None:
        self._config = config
        self._project_root = project_root if project_root is not None else Path.cwd()
        self.dataset_id = config.dataset_id

    @classmethod
    def from_config_path(
        cls,
        config_path: Path,
        project_root: Path | None = None,
    ) -> "KaggleLlmDetectAiGeneratedTextAdapter":
        """Create adapter from config path."""

        return cls(config=load_kaggle_config(config_path), project_root=project_root)

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

        ensure_kaggle_sources_available(config=self._config, project_root=self._project_root)
        for split, source_path in split_to_path.items():
            if not source_path.exists():
                raise FileNotFoundError(f"Missing source file for split {split!r}: {source_path}")

    def _build_source_fields(self, row_payload: Mapping[str, Any], row_id: str, row_index: int, split: str, source_path: Path, raw_label: str) -> dict[str, Any]:
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

    def _build_record(self, split: str, row_payload: Mapping[str, Any], row_index: int, source_path: Path) -> CanonicalDatasetRecord:
        """Map one CSV row into canonical dataset record."""

        mapping = self._config.mapping
        if mapping.id_field not in row_payload:
            raise ValueError(f"Missing id field {mapping.id_field!r} in row {row_index}.")
        if mapping.text_field not in row_payload:
            raise ValueError(f"Missing text field {mapping.text_field!r} in row {row_index}.")
        if mapping.label_field not in row_payload:
            raise ValueError(f"Missing label field {mapping.label_field!r} in row {row_index}.")

        row_id = _coerce_row_id(row_payload[mapping.id_field])
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
