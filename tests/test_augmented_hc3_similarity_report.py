"""Tests for augmented HC3 similarity report helpers."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from apm.experiments.augmented_hc3_similarity_report import (
    _build_similarity_rows,
    _build_similarity_summary,
    _compute_similarity_frame,
    ScenarioRow,
)


def test_similarity_rows_and_summary_basic_case() -> None:
    scenario_rows = [
        ScenarioRow(source_group="baseline", split_name="aug_baseline", display_label="baseline", augmentation="baseline"),
        ScenarioRow(
            source_group="orig_h_aug_ai",
            split_name="aug_style_orig_h_aug_ai",
            display_label="style | orig_h+aug_ai",
            augmentation="style",
        ),
    ]
    texts = {
        ("aug_baseline", "human", "human:0001"): "human baseline",
        ("aug_baseline", "ai", "ai:0001"): "ai baseline",
        ("aug_style_orig_h_aug_ai", "human", "human:0001"): "human baseline",
        ("aug_style_orig_h_aug_ai", "ai", "ai:0001"): "ai modified",
    }
    similarity_rows = _build_similarity_rows(scenario_rows=scenario_rows, texts=texts)
    assert len(similarity_rows) == 4

    text_to_vector = {
        "ai baseline": np.asarray([1.0, 0.0], dtype=np.float32),
        "ai modified": np.asarray([0.5, np.sqrt(3.0) / 2.0], dtype=np.float32),
    }
    frame = _compute_similarity_frame(
        similarity_rows=similarity_rows,
        text_to_vector=text_to_vector,
    )
    assert len(frame) == 4

    style_ai = frame[(frame["source_split"] == "aug_style_orig_h_aug_ai") & (frame["label"] == "ai")].iloc[0]
    baseline_h = frame[(frame["source_split"] == "aug_baseline") & (frame["label"] == "human")].iloc[0]

    assert abs(float(style_ai["cosine_similarity"]) - 0.5) < 1e-6
    assert float(baseline_h["cosine_similarity"]) == 1.0

    summary = _build_similarity_summary(similarity_frame=frame)
    assert len(summary) == 4
    baseline_summary = summary[
        (summary["source_group"] == "baseline") & (summary["label"] == "human")
    ].iloc[0]
    assert float(baseline_summary["mean_similarity"]) == 1.0
