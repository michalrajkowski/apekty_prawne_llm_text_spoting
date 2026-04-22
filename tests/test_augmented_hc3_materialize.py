"""Tests for augmented HC3 scenario materialization from text folders."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from apm.experiments.augmented_hc3_materialize import (
    AugmentedHc3MaterializeRequest,
    materialize_augmented_hc3_scenarios,
)


def _write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def test_materialize_augmented_hc3_scenarios_builds_baseline_and_mixed_splits(tmp_path: Path) -> None:
    root = tmp_path / "data" / "augmented_data" / "hc3"

    _write_text(root / "all_train" / "human" / "0001__all_train_human_a.txt", "train human A")
    _write_text(root / "all_train" / "human" / "0002__all_train_human_b.txt", "train human B")
    _write_text(root / "all_train" / "ai" / "0001__all_train_ai_a.txt", "train ai A")
    _write_text(root / "all_train" / "ai" / "0002__all_train_ai_b.txt", "train ai B")

    _write_text(root / "test" / "human" / "0001__base_human_a.txt", "base human A")
    _write_text(root / "test" / "human" / "0002__base_human_b.txt", "base human B")
    _write_text(root / "test" / "ai" / "0001__base_ai_a.txt", "base ai A")
    _write_text(root / "test" / "ai" / "0002__base_ai_b.txt", "base ai B")

    _write_text(
        root / "test_back_trans_3langs" / "human" / "0001__0001__base_human_a__backtrans_3langs_en.txt",
        "aug human A",
    )
    _write_text(
        root / "test_back_trans_3langs" / "human" / "0002__0002__base_human_b__backtrans_3langs_en.txt",
        "aug human B",
    )
    _write_text(
        root / "test_back_trans_3langs" / "ai" / "0001__0001__base_ai_a__backtrans_3langs_en.txt",
        "aug ai A",
    )
    _write_text(
        root / "test_back_trans_3langs" / "ai" / "0002__0002__base_ai_b__backtrans_3langs_en.txt",
        "aug ai B",
    )

    request = AugmentedHc3MaterializeRequest(
        project_root=tmp_path,
        input_root=root,
        output_root=tmp_path / "data" / "interim" / "splits" / "hc3",
        train_source="all_train",
        baseline_test_source="test",
        variant_sources=("test_back_trans_3langs",),
        output_split_prefix="aug",
    )

    result = materialize_augmented_hc3_scenarios(request)
    assert result.scenario_splits == (
        "aug_baseline",
        "aug_back_trans_3langs_orig_h_aug_ai",
        "aug_back_trans_3langs_aug_h_orig_ai",
        "aug_back_trans_3langs_aug_both",
    )
    assert result.manifest_path.exists()
    assert result.split_names_path.exists()

    scenario_dir = request.output_root / "aug_back_trans_3langs_orig_h_aug_ai"
    train_human = pd.read_parquet(scenario_dir / "train" / "human" / "sampled_records.parquet")
    test_human = pd.read_parquet(scenario_dir / "test" / "human" / "sampled_records.parquet")
    test_ai = pd.read_parquet(scenario_dir / "test" / "ai" / "sampled_records.parquet")

    assert list(train_human["text"]) == ["train human A", "train human B"]
    assert list(test_human["text"]) == ["base human A", "base human B"]
    assert list(test_ai["text"]) == ["aug ai A", "aug ai B"]

    assignments = (scenario_dir / "split_assignments.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(assignments) == 8
