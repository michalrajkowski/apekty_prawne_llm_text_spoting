"""Tests for bulk dataset materialization orchestration."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from apm.data.adapters.hc3_materialize import MaterializedSplitOutput
from apm.data.materialize_all import materialize_all_datasets, read_dataset_ids_file


def _write_dataset_config(config_path: Path, dataset_id: str) -> None:
    payload = {
        "dataset_id": dataset_id,
        "source_type": "test",
        "source_uri": f"source://{dataset_id}",
    }
    config_path.write_text(json.dumps(payload), encoding="utf-8")


def test_read_dataset_ids_file_parses_non_empty_non_comment_lines(tmp_path: Path) -> None:
    datasets_file = tmp_path / "datasets.txt"
    datasets_file.write_text("\n# comment\nhc3\n\nfinance_dataset\n", encoding="utf-8")

    dataset_ids = read_dataset_ids_file(datasets_file)

    assert dataset_ids == ("hc3", "finance_dataset")


def test_materialize_all_datasets_default_mode_skips_unsupported(tmp_path: Path) -> None:
    config_dir = tmp_path / "configs"
    config_dir.mkdir(parents=True)
    _write_dataset_config(config_dir / "hc3.dataset.json", "hc3")
    _write_dataset_config(config_dir / "kaggle.dataset.json", "kaggle")

    called: list[tuple[Path, Path, int, int]] = []

    def fake_hc3_materializer(project_root: Path, config_path: Path, sample_size: int, seed: int) -> tuple[MaterializedSplitOutput, ...]:
        called.append((project_root, config_path, sample_size, seed))
        return (
            MaterializedSplitOutput(
                split="all_train",
                sampled_count=100,
                parquet_path=project_root / "data/interim/datasets/hc3/all_train.parquet",
                metadata_path=project_root / "data/interim/datasets/hc3/all_train.metadata.json",
                raw_snapshot_path=project_root / "data/raw/datasets/hc3/all_train/sampled_records.jsonl",
            ),
        )

    report = materialize_all_datasets(
        project_root=tmp_path,
        config_dir=config_dir,
        sample_size=100,
        seed=42,
        materializers={"hc3": fake_hc3_materializer},
    )

    assert report.discovered_dataset_ids == ("hc3", "kaggle")
    assert report.skipped_unsupported_dataset_ids == ("kaggle",)
    assert len(report.materialized_datasets) == 1
    assert report.materialized_datasets[0].dataset_id == "hc3"
    assert len(called) == 1


def test_materialize_all_datasets_requested_missing_dataset_fails_fast(tmp_path: Path) -> None:
    config_dir = tmp_path / "configs"
    config_dir.mkdir(parents=True)
    _write_dataset_config(config_dir / "hc3.dataset.json", "hc3")

    with pytest.raises(ValueError, match="not found in configs"):
        materialize_all_datasets(
            project_root=tmp_path,
            config_dir=config_dir,
            sample_size=100,
            seed=42,
            datasets=("missing_dataset",),
            materializers={"hc3": lambda *_: ()},
        )


def test_materialize_all_datasets_requested_unsupported_dataset_fails_fast(tmp_path: Path) -> None:
    config_dir = tmp_path / "configs"
    config_dir.mkdir(parents=True)
    _write_dataset_config(config_dir / "kaggle.dataset.json", "kaggle")

    with pytest.raises(ValueError, match="without materializer"):
        materialize_all_datasets(
            project_root=tmp_path,
            config_dir=config_dir,
            sample_size=100,
            seed=42,
            datasets=("kaggle",),
            materializers={"hc3": lambda *_: ()},
        )
