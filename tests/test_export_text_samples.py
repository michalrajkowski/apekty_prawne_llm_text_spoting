"""Tests for deterministic HC3 text sample export."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from apm.data.export_text_samples import ExportTextSamplesRequest, export_hc3_text_samples


def _write_label_source(path: Path, label: str, count: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(
        {
            "sample_id": [f"{label}:{index}" for index in range(count)],
            "text": [f"{label} text {index}" for index in range(count)],
        }
    )
    frame.to_parquet(path, index=False)


def _build_request(project_root: Path, per_label_sample_size: int) -> ExportTextSamplesRequest:
    return ExportTextSamplesRequest(
        project_root=project_root,
        dataset_id="hc3",
        source_split="all_train",
        per_label_sample_size=per_label_sample_size,
        seed=42,
        output_root=Path("data/to_export"),
    )


def _manifest_records(path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def test_export_hc3_text_samples_exports_expected_counts_and_is_deterministic(tmp_path: Path) -> None:
    for subset_root in (
        tmp_path / "data" / "interim" / "datasets" / "hc3" / "all_train",
        tmp_path / "data" / "interim" / "splits" / "hc3" / "all_train" / "test",
    ):
        _write_label_source(subset_root / "human" / "sampled_records.parquet", label="human", count=150)
        _write_label_source(subset_root / "ai" / "sampled_records.parquet", label="ai", count=150)

    request = _build_request(tmp_path, per_label_sample_size=100)
    first = export_hc3_text_samples(request)
    second = export_hc3_text_samples(request)

    assert first.export_root == second.export_root
    assert first.manifest_path.exists()
    assert first.metadata_path.exists()

    all_train_human = sorted((first.export_root / "all_train" / "human").glob("*.txt"))
    all_train_ai = sorted((first.export_root / "all_train" / "ai").glob("*.txt"))
    test_human = sorted((first.export_root / "test" / "human").glob("*.txt"))
    test_ai = sorted((first.export_root / "test" / "ai").glob("*.txt"))

    assert len(all_train_human) == 100
    assert len(all_train_ai) == 100
    assert len(test_human) == 100
    assert len(test_ai) == 100

    first_records = _manifest_records(first.manifest_path)
    second_records = _manifest_records(second.manifest_path)
    assert len(first_records) == 400
    assert first_records == second_records

    by_subset_label: dict[tuple[str, str], int] = {}
    for record in first_records:
        key = (str(record["subset_name"]), str(record["label"]))
        by_subset_label[key] = by_subset_label.get(key, 0) + 1

    assert by_subset_label == {
        ("all_train", "human"): 100,
        ("all_train", "ai"): 100,
        ("test", "human"): 100,
        ("test", "ai"): 100,
    }


def test_export_hc3_text_samples_rejects_when_requested_count_exceeds_available(tmp_path: Path) -> None:
    for subset_root in (
        tmp_path / "data" / "interim" / "datasets" / "hc3" / "all_train",
        tmp_path / "data" / "interim" / "splits" / "hc3" / "all_train" / "test",
    ):
        _write_label_source(subset_root / "human" / "sampled_records.parquet", label="human", count=12)
        _write_label_source(subset_root / "ai" / "sampled_records.parquet", label="ai", count=12)

    request = _build_request(tmp_path, per_label_sample_size=20)

    with pytest.raises(ValueError, match="Requested 20 examples"):
        export_hc3_text_samples(request)
