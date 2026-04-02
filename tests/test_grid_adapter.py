"""Tests for GriD adapter/downloader/materialization behavior."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from apm.data.adapters.grid_adapter import GridAdapter, load_grid_config
from apm.data.adapters.grid_download import ensure_grid_sources
from apm.data.adapters.grid_materialize import materialize_grid_samples
from apm.data.dataset_registry import DatasetRegistry
from apm.data.hf_loader import load_dataset
from apm.types import DatasetLoadRequest


def _write_grid_csv(csv_path: Path, rows: list[dict[str, object]]) -> None:
    dataframe = pd.DataFrame(rows)
    dataframe.to_csv(csv_path, index=False)


def _write_grid_config(config_path: Path, filtered_path: Path, unfiltered_path: Path, auto_download_if_missing: bool = True) -> None:
    payload = {
        "dataset_id": "grid",
        "source_type": "github_csv_http",
        "source_uri": "https://github.com/madlab-ucr/GriD",
        "sampling": {"strategy": "random", "seed": 42},
        "splits": {
            "filtered": {"path": str(filtered_path)},
            "unfiltered": {"path": str(unfiltered_path)},
        },
        "download": {
            "output_dir": "data/raw/datasets/grid",
            "files": [
                {
                    "name": "reddit_filtered_dataset.csv",
                    "url": "https://raw.githubusercontent.com/madlab-ucr/GriD/main/reddit_datasets/reddit_filtered_dataset.csv",
                },
                {
                    "name": "reddit_unfiltered_data.csv",
                    "url": "https://raw.githubusercontent.com/madlab-ucr/GriD/main/reddit_datasets/reddit_unfiltered_data.csv",
                },
            ],
            "auto_download_if_missing": auto_download_if_missing,
            "force_download": False,
        },
        "mapping": {
            "id_field": None,
            "text_field": "Data",
            "label_field": "Labels",
            "optional_fields": [],
            "canonical_label_mapping": {"0": "human", "1": "ai"},
            "sample_id_pattern": "{split}:{row_index}",
        },
    }
    config_path.write_text(json.dumps(payload), encoding="utf-8")


def _build_rows() -> list[dict[str, object]]:
    return [
        {"Data": "Human text 0", "Labels": 0},
        {"Data": "AI text 0", "Labels": 1},
        {"Data": "Human text 1", "Labels": 0},
        {"Data": "AI text 1", "Labels": 1},
    ]


def test_load_grid_config_parses_splits_mapping_and_download(tmp_path: Path) -> None:
    filtered_path = tmp_path / "reddit_filtered_dataset.csv"
    unfiltered_path = tmp_path / "reddit_unfiltered_data.csv"
    _write_grid_csv(filtered_path, _build_rows())
    _write_grid_csv(unfiltered_path, _build_rows())
    config_path = tmp_path / "grid.dataset.json"
    _write_grid_config(config_path, filtered_path, unfiltered_path)

    config = load_grid_config(config_path)

    assert config.dataset_id == "grid"
    assert tuple(config.splits.keys()) == ("filtered", "unfiltered")
    assert config.mapping.text_field == "Data"
    assert config.mapping.label_field == "Labels"
    assert config.download is not None
    assert len(config.download.files) == 2


def test_grid_adapter_load_split_maps_human_and_ai(tmp_path: Path) -> None:
    filtered_path = tmp_path / "reddit_filtered_dataset.csv"
    unfiltered_path = tmp_path / "reddit_unfiltered_data.csv"
    _write_grid_csv(filtered_path, _build_rows())
    _write_grid_csv(unfiltered_path, _build_rows())
    config_path = tmp_path / "grid.dataset.json"
    _write_grid_config(config_path, filtered_path, unfiltered_path)

    adapter = GridAdapter.from_config_path(config_path=config_path, project_root=tmp_path)
    records = adapter.load_split("filtered")

    assert len(records) == 4
    labels = [record.label for record in records]
    assert labels.count("human") == 2
    assert labels.count("ai") == 2
    assert records[0].sample_id == "filtered:0"


def test_grid_adapter_sampling_is_deterministic(tmp_path: Path) -> None:
    filtered_path = tmp_path / "reddit_filtered_dataset.csv"
    unfiltered_path = tmp_path / "reddit_unfiltered_data.csv"
    _write_grid_csv(filtered_path, _build_rows())
    _write_grid_csv(unfiltered_path, _build_rows())
    config_path = tmp_path / "grid.dataset.json"
    _write_grid_config(config_path, filtered_path, unfiltered_path)

    adapter = GridAdapter.from_config_path(config_path=config_path, project_root=tmp_path)
    registry = DatasetRegistry()
    registry.register(dataset_id=adapter.dataset_id, adapter=adapter)

    first = load_dataset(
        request=DatasetLoadRequest(
            dataset_id=adapter.dataset_id,
            split="filtered",
            per_label_sample_size=1,
            seed=7,
            sampling_strategy="balanced_random",
        ),
        registry=registry,
    )
    second = load_dataset(
        request=DatasetLoadRequest(
            dataset_id=adapter.dataset_id,
            split="filtered",
            per_label_sample_size=1,
            seed=7,
            sampling_strategy="balanced_random",
        ),
        registry=registry,
    )

    assert [record.sample_id for record in first.records] == [record.sample_id for record in second.records]


def test_materialize_grid_samples_writes_expected_artifacts(tmp_path: Path) -> None:
    filtered_path = tmp_path / "data" / "raw" / "datasets" / "grid" / "reddit_filtered_dataset.csv"
    unfiltered_path = tmp_path / "data" / "raw" / "datasets" / "grid" / "reddit_unfiltered_data.csv"
    filtered_path.parent.mkdir(parents=True)
    _write_grid_csv(filtered_path, _build_rows())
    _write_grid_csv(unfiltered_path, _build_rows())

    config_path = tmp_path / "grid.dataset.json"
    _write_grid_config(
        config_path,
        Path("data/raw/datasets/grid/reddit_filtered_dataset.csv"),
        Path("data/raw/datasets/grid/reddit_unfiltered_data.csv"),
    )

    outputs = materialize_grid_samples(
        project_root=tmp_path,
        config_path=config_path,
        sample_size=2,
        seed=42,
    )

    assert len(outputs) == 2
    for output in outputs:
        assert output.sampled_count == 4
        assert output.parquet_path.exists()
        assert output.metadata_path.exists()
        assert output.raw_snapshot_path.exists()

        metadata = json.loads(output.metadata_path.read_text(encoding="utf-8"))
        assert metadata["requested_per_label_sample_size"] == 2
        assert metadata["sampled_per_label_counts"] == {"human": 2, "ai": 2}

        raw_base = tmp_path / "data" / "raw" / "datasets" / "grid" / output.split
        assert (raw_base / "human" / "sampled_records.jsonl").exists()
        assert (raw_base / "ai" / "sampled_records.jsonl").exists()


def test_grid_adapter_auto_downloads_when_source_is_missing(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    config_path = tmp_path / "grid.dataset.json"
    filtered_rel = Path("data/raw/datasets/grid/reddit_filtered_dataset.csv")
    unfiltered_rel = Path("data/raw/datasets/grid/reddit_unfiltered_data.csv")
    _write_grid_config(config_path, filtered_rel, unfiltered_rel)

    def fake_urlretrieve(url: str, filename: str | Path, reporthook: object = None, data: object = None) -> tuple[str, object]:
        _ = reporthook, data
        target_path = Path(filename)
        target_path.parent.mkdir(parents=True, exist_ok=True)
        if url.endswith("reddit_filtered_dataset.csv"):
            rows = _build_rows()
        else:
            rows = _build_rows()
        _write_grid_csv(target_path, rows)
        return (str(target_path), None)

    monkeypatch.setattr("apm.data.adapters.grid_adapter.urlretrieve", fake_urlretrieve)

    adapter = GridAdapter.from_config_path(config_path=config_path, project_root=tmp_path)
    records = adapter.load_split("filtered")

    assert len(records) == 4
    assert (tmp_path / filtered_rel).exists()
    assert (tmp_path / unfiltered_rel).exists()


def test_grid_adapter_raises_when_auto_download_disabled_and_file_missing(tmp_path: Path) -> None:
    config_path = tmp_path / "grid.dataset.json"
    filtered_rel = Path("data/raw/datasets/grid/reddit_filtered_dataset.csv")
    unfiltered_rel = Path("data/raw/datasets/grid/reddit_unfiltered_data.csv")
    _write_grid_config(config_path, filtered_rel, unfiltered_rel, auto_download_if_missing=False)

    adapter = GridAdapter.from_config_path(config_path=config_path, project_root=tmp_path)
    with pytest.raises(FileNotFoundError, match="Missing required source file"):
        adapter.load_split("filtered")


def test_grid_download_helper_returns_resolved_source_paths(tmp_path: Path) -> None:
    filtered_path = tmp_path / "data" / "raw" / "datasets" / "grid" / "reddit_filtered_dataset.csv"
    unfiltered_path = tmp_path / "data" / "raw" / "datasets" / "grid" / "reddit_unfiltered_data.csv"
    filtered_path.parent.mkdir(parents=True)
    _write_grid_csv(filtered_path, _build_rows())
    _write_grid_csv(unfiltered_path, _build_rows())

    config_path = tmp_path / "grid.dataset.json"
    _write_grid_config(
        config_path,
        Path("data/raw/datasets/grid/reddit_filtered_dataset.csv"),
        Path("data/raw/datasets/grid/reddit_unfiltered_data.csv"),
    )

    resolved = ensure_grid_sources(project_root=tmp_path, config_path=config_path)

    assert resolved["filtered"] == str(filtered_path)
    assert resolved["unfiltered"] == str(unfiltered_path)
