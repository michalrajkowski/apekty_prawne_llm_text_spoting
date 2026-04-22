"""Tests for augmented HC3 JIWER report helpers."""

from __future__ import annotations

from apm.experiments.augmented_hc3_jiwer_report import (
    _build_summary_by_label,
    _build_summary_overall,
    _compute_metrics_frame,
)


def test_compute_metrics_and_summary() -> None:
    modified_rows = [
        {
            "source_group": "orig_h_aug_ai",
            "source_split": "aug_fewshot_orig_h_aug_ai",
            "display_label": "fewshot (orig_h+aug_ai)",
            "augmentation": "fewshot",
            "label": "ai",
            "sample_id": "ai:1",
            "baseline_text": "the cat is on the mat",
            "scenario_text": "the cat was on mat",
        },
        {
            "source_group": "aug_h_orig_ai",
            "source_split": "aug_fewshot_aug_h_orig_ai",
            "display_label": "fewshot (aug_h+orig_ai)",
            "augmentation": "fewshot",
            "label": "human",
            "sample_id": "human:1",
            "baseline_text": "hello world",
            "scenario_text": "hello brave world",
        },
    ]

    frame = _compute_metrics_frame(modified_rows=modified_rows)
    assert len(frame) == 2
    assert set(frame.columns.tolist()) >= {"wer", "cer", "augmentation", "label"}
    assert float(frame["wer"].min()) >= 0.0
    assert float(frame["cer"].min()) >= 0.0

    by_label = _build_summary_by_label(metrics_frame=frame)
    assert len(by_label) == 2
    assert set(by_label["label"].tolist()) == {"ai", "human"}
    assert all(int(value) == 1 for value in by_label["sample_count"].tolist())

    overall = _build_summary_overall(metrics_frame=frame)
    assert len(overall) == 1
    row = overall.iloc[0]
    assert str(row["augmentation"]) == "fewshot"
    assert int(row["sample_count"]) == 2
