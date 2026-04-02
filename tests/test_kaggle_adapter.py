"""Tests for Kaggle LLM Detect AI Generated Text adapter and materialization."""

from __future__ import annotations

import json
import subprocess
import zipfile
from pathlib import Path

import pandas as pd
import pytest

from apm.data.adapters.kaggle_llm_detect_ai_generated_text_adapter import (
    KaggleLlmDetectAiGeneratedTextAdapter,
    load_kaggle_config,
)
from apm.data.adapters.kaggle_llm_detect_ai_generated_text_download import (
    ensure_kaggle_llm_detect_ai_generated_text_sources,
)
from apm.data.adapters.kaggle_llm_detect_ai_generated_text_materialize import (
    materialize_kaggle_llm_detect_ai_generated_text_samples,
)
from apm.data.dataset_registry import DatasetRegistry
from apm.data.hf_loader import load_dataset
from apm.types import DatasetLoadRequest


def _write_train_csv(csv_path: Path) -> None:
    dataframe = pd.DataFrame(
        [
            {"id": "essay-0", "prompt_id": 10, "text": "Human essay", "generated": 0},
            {"id": "essay-1", "prompt_id": 10, "text": "AI essay A", "generated": 1},
            {"id": "essay-2", "prompt_id": 11, "text": "AI essay B", "generated": 1},
            {"id": "essay-3", "prompt_id": 12, "text": "Human essay B", "generated": 0},
        ]
    )
    dataframe.to_csv(csv_path, index=False)


def _write_kaggle_config(config_path: Path, train_csv_path: Path) -> None:
    payload = {
        "dataset_id": "kaggle_llm_detect_ai_generated_text",
        "source_type": "kaggle_csv_local",
        "source_uri": "https://www.kaggle.com/competitions/llm-detect-ai-generated-text/data",
        "sampling": {"strategy": "random", "seed": 42},
        "splits": {"train": {"path": str(train_csv_path)}},
        "download": {
            "competition": "llm-detect-ai-generated-text",
            "output_dir": "data/raw/datasets/kaggle_llm_detect_ai_generated_text",
            "archive_filename": "llm-detect-ai-generated-text.zip",
            "required_files": ["train_essays.csv"],
            "auto_download_if_missing": True,
            "force_download": False,
        },
        "mapping": {
            "id_field": "id",
            "text_field": "text",
            "label_field": "generated",
            "optional_fields": ["prompt_id"],
            "canonical_label_mapping": {"0": "human", "1": "ai"},
            "sample_id_pattern": "{split}:{id}",
        },
    }
    config_path.write_text(json.dumps(payload), encoding="utf-8")


def test_load_kaggle_config_parses_split_and_mapping(tmp_path: Path) -> None:
    csv_path = tmp_path / "train_essays.csv"
    _write_train_csv(csv_path)
    config_path = tmp_path / "kaggle.dataset.json"
    _write_kaggle_config(config_path, csv_path)

    config = load_kaggle_config(config_path)

    assert config.dataset_id == "kaggle_llm_detect_ai_generated_text"
    assert config.source_uri.startswith("https://www.kaggle.com/")
    assert tuple(config.splits.keys()) == ("train",)
    assert config.mapping.canonical_label_mapping["0"] == "human"
    assert config.mapping.canonical_label_mapping["1"] == "ai"
    assert config.download is not None
    assert config.download.competition == "llm-detect-ai-generated-text"


def test_kaggle_adapter_load_split_maps_labels_and_source_fields(tmp_path: Path) -> None:
    csv_path = tmp_path / "train_essays.csv"
    _write_train_csv(csv_path)
    config_path = tmp_path / "kaggle.dataset.json"
    _write_kaggle_config(config_path, csv_path)

    adapter = KaggleLlmDetectAiGeneratedTextAdapter.from_config_path(config_path=config_path, project_root=tmp_path)
    records = adapter.load_split("train")

    assert len(records) == 4
    labels = [record.label for record in records]
    assert labels.count("human") == 2
    assert labels.count("ai") == 2
    assert records[0].sample_id == "train:essay-0"
    assert records[1].source_fields["prompt_id"] == 10
    assert records[1].source_fields["raw_label"] == "1"


def test_kaggle_adapter_sampling_is_deterministic(tmp_path: Path) -> None:
    csv_path = tmp_path / "train_essays.csv"
    _write_train_csv(csv_path)
    config_path = tmp_path / "kaggle.dataset.json"
    _write_kaggle_config(config_path, csv_path)

    adapter = KaggleLlmDetectAiGeneratedTextAdapter.from_config_path(config_path=config_path, project_root=tmp_path)
    registry = DatasetRegistry()
    registry.register(dataset_id=adapter.dataset_id, adapter=adapter)

    first = load_dataset(
        request=DatasetLoadRequest(dataset_id=adapter.dataset_id, split="train", sample_size=2, seed=7),
        registry=registry,
    )
    second = load_dataset(
        request=DatasetLoadRequest(dataset_id=adapter.dataset_id, split="train", sample_size=2, seed=7),
        registry=registry,
    )

    first_sample_ids = [record.sample_id for record in first.records]
    second_sample_ids = [record.sample_id for record in second.records]
    assert first_sample_ids == second_sample_ids


def test_materialize_kaggle_samples_writes_expected_artifacts(tmp_path: Path) -> None:
    csv_path = tmp_path / "data" / "raw" / "datasets" / "kaggle_llm_detect_ai_generated_text" / "train_essays.csv"
    csv_path.parent.mkdir(parents=True)
    _write_train_csv(csv_path)

    config_path = tmp_path / "kaggle.dataset.json"
    _write_kaggle_config(config_path, Path("data/raw/datasets/kaggle_llm_detect_ai_generated_text/train_essays.csv"))

    outputs = materialize_kaggle_llm_detect_ai_generated_text_samples(
        project_root=tmp_path,
        config_path=config_path,
        sample_size=2,
        seed=42,
    )

    assert len(outputs) == 1
    output = outputs[0]
    assert output.split == "train"
    assert output.sampled_count == 4
    assert output.parquet_path.exists()
    assert output.metadata_path.exists()
    assert output.raw_snapshot_path.exists()
    assert (tmp_path / "data" / "raw" / "datasets" / "kaggle_llm_detect_ai_generated_text" / "train" / "human").exists()
    assert (tmp_path / "data" / "raw" / "datasets" / "kaggle_llm_detect_ai_generated_text" / "train" / "ai").exists()
    assert (
        tmp_path
        / "data"
        / "interim"
        / "datasets"
        / "kaggle_llm_detect_ai_generated_text"
        / "train"
        / "human"
        / "sampled_records.parquet"
    ).exists()
    assert (
        tmp_path
        / "data"
        / "interim"
        / "datasets"
        / "kaggle_llm_detect_ai_generated_text"
        / "train"
        / "ai"
        / "sampled_records.parquet"
    ).exists()

    metadata = json.loads(output.metadata_path.read_text(encoding="utf-8"))
    assert metadata["dataset_id"] == "kaggle_llm_detect_ai_generated_text"
    assert metadata["split"] == "train"
    assert metadata["sample_size"] == 2
    assert metadata["sampled_count"] == 4
    assert metadata["requested_per_label_sample_size"] == 2
    assert metadata["sampled_per_label_counts"] == {"human": 2, "ai": 2}


def test_kaggle_adapter_auto_downloads_when_source_file_is_missing(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    config_path = tmp_path / "kaggle.dataset.json"
    train_csv_path = Path("data/raw/datasets/kaggle_llm_detect_ai_generated_text/train_essays.csv")
    _write_kaggle_config(config_path, train_csv_path)

    output_dir = tmp_path / "data" / "raw" / "datasets" / "kaggle_llm_detect_ai_generated_text"

    def fake_run(command: list[str], check: bool) -> subprocess.CompletedProcess[bytes]:
        assert check is True
        assert command[1:4] == ["competitions", "download", "-c"]
        output_dir.mkdir(parents=True, exist_ok=True)
        archive_path = output_dir / "llm-detect-ai-generated-text.zip"
        csv_payload = "\n".join(
            [
                "id,prompt_id,text,generated",
                "essay-0,10,Human essay,0",
                "essay-1,11,AI essay,1",
            ]
        )
        with zipfile.ZipFile(archive_path, mode="w") as archive:
            archive.writestr("train_essays.csv", csv_payload)
        return subprocess.CompletedProcess(args=command, returncode=0)

    monkeypatch.setattr(
        "apm.data.adapters.kaggle_llm_detect_ai_generated_text_adapter.shutil.which",
        lambda _program: "/usr/bin/kaggle",
    )
    monkeypatch.setattr(
        "apm.data.adapters.kaggle_llm_detect_ai_generated_text_adapter.subprocess.run",
        fake_run,
    )

    adapter = KaggleLlmDetectAiGeneratedTextAdapter.from_config_path(config_path=config_path, project_root=tmp_path)
    records = adapter.load_split("train")

    assert len(records) == 2
    assert (output_dir / "train_essays.csv").exists()
    assert {record.label for record in records} == {"human", "ai"}


def test_kaggle_adapter_raises_when_auto_download_required_but_kaggle_cli_missing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config_path = tmp_path / "kaggle.dataset.json"
    train_csv_path = Path("data/raw/datasets/kaggle_llm_detect_ai_generated_text/train_essays.csv")
    _write_kaggle_config(config_path, train_csv_path)

    monkeypatch.setattr(
        "apm.data.adapters.kaggle_llm_detect_ai_generated_text_adapter.shutil.which",
        lambda _program: None,
    )

    adapter = KaggleLlmDetectAiGeneratedTextAdapter.from_config_path(config_path=config_path, project_root=tmp_path)
    with pytest.raises(RuntimeError, match="kaggle CLI not found"):
        adapter.load_split("train")


def test_kaggle_download_helper_returns_resolved_source_paths(tmp_path: Path) -> None:
    csv_path = tmp_path / "data" / "raw" / "datasets" / "kaggle_llm_detect_ai_generated_text" / "train_essays.csv"
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    _write_train_csv(csv_path)
    config_path = tmp_path / "kaggle.dataset.json"
    _write_kaggle_config(
        config_path,
        Path("data/raw/datasets/kaggle_llm_detect_ai_generated_text/train_essays.csv"),
    )

    resolved_sources = ensure_kaggle_llm_detect_ai_generated_text_sources(
        project_root=tmp_path,
        config_path=config_path,
    )

    assert resolved_sources["train"] == str(csv_path)
