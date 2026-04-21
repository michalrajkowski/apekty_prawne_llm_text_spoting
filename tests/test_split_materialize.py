"""Tests for deterministic train/test split materialization artifacts."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from apm.experiments.matrix import DatasetSpec
from apm.experiments.split_materialize import SplitMaterializeRequest, materialize_train_test_splits


def _write_label_source(
    *,
    root: Path,
    dataset_id: str,
    source_split: str,
    label: str,
    count: int,
) -> None:
    target = root / "data" / "interim" / "datasets" / dataset_id / source_split / label / "sampled_records.parquet"
    target.parent.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(
        {
            "sample_id": [f"{label}-{idx}" for idx in range(count)],
            "text": [f"{label} text {idx}" for idx in range(count)],
            "label": [label for _ in range(count)],
        }
    )
    frame.to_parquet(target, index=False)


def test_materialize_train_test_splits_persists_deterministic_non_overlapping_outputs(tmp_path: Path) -> None:
    _write_label_source(root=tmp_path, dataset_id="toy", source_split="s1", label="human", count=10)
    _write_label_source(root=tmp_path, dataset_id="toy", source_split="s1", label="ai", count=10)

    request = SplitMaterializeRequest(
        project_root=tmp_path,
        dataset_specs=(DatasetSpec(dataset_id="toy", split="s1"),),
        train_ratio=0.6,
        seed=42,
        output_root=tmp_path / "data" / "interim" / "splits",
    )

    first = materialize_train_test_splits(request)
    second = materialize_train_test_splits(request)

    assert len(first) == 1
    output = first[0]

    train_human = pd.read_parquet(output.output_dir / "train" / "human" / "sampled_records.parquet")
    test_human = pd.read_parquet(output.output_dir / "test" / "human" / "sampled_records.parquet")
    train_ai = pd.read_parquet(output.output_dir / "train" / "ai" / "sampled_records.parquet")
    test_ai = pd.read_parquet(output.output_dir / "test" / "ai" / "sampled_records.parquet")

    assert len(train_human.index) == 6
    assert len(test_human.index) == 4
    assert len(train_ai.index) == 6
    assert len(test_ai.index) == 4

    train_human_ids = set(str(item) for item in train_human["sample_id"].tolist())
    test_human_ids = set(str(item) for item in test_human["sample_id"].tolist())
    train_ai_ids = set(str(item) for item in train_ai["sample_id"].tolist())
    test_ai_ids = set(str(item) for item in test_ai["sample_id"].tolist())

    assert train_human_ids.isdisjoint(test_human_ids)
    assert train_ai_ids.isdisjoint(test_ai_ids)

    first_assignments = output.assignments_path.read_text(encoding="utf-8")
    second_assignments = second[0].assignments_path.read_text(encoding="utf-8")
    assert first_assignments == second_assignments

    metadata = json.loads(output.metadata_path.read_text(encoding="utf-8"))
    assert metadata["train_counts_by_label"] == {"human": 6, "ai": 6}
    assert metadata["test_counts_by_label"] == {"human": 4, "ai": 4}
