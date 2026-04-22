"""Tests for baseline vs augmented HC3 score-shift analysis."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from apm.experiments.augmented_hc3_score_shift_report import (
    AugmentedHc3ScoreShiftRequest,
    run_augmented_hc3_score_shift_report,
)


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    payload = "\n".join(json.dumps(row, sort_keys=True) for row in rows) + "\n"
    path.write_text(payload, encoding="utf-8")


def test_augmented_hc3_score_shift_report_outputs_and_expected_delta(tmp_path: Path) -> None:
    run_dir = tmp_path / "runs" / "global_local_experiments" / "run_aug"
    run_dir.mkdir(parents=True, exist_ok=True)

    local_rows = [
        {
            "scope_id": "hc3:aug_baseline",
            "detector_run_id": "detector_a",
            "threshold": 0.5,
        }
    ]
    _write_jsonl(run_dir / "local_results.jsonl", local_rows)

    raw_rows: list[dict[str, object]] = []
    for split, ai_scores, human_scores in (
        ("aug_baseline", [0.9, 0.8, 0.7, 0.6], [0.2, 0.1, 0.2, 0.1]),
        ("aug_style_orig_h_aug_ai", [0.7, 0.6, 0.5, 0.4], [0.2, 0.1, 0.2, 0.1]),
        ("aug_style_aug_h_orig_ai", [0.9, 0.8, 0.7, 0.6], [0.4, 0.3, 0.2, 0.1]),
    ):
        for label, scores in (("ai", ai_scores), ("human", human_scores)):
            for partition in ("train", "test"):
                for index, score in enumerate(scores, start=1):
                    raw_rows.append(
                        {
                            "dataset_id": "hc3",
                            "detector_run_id": "detector_a",
                            "label": label,
                            "partition": partition,
                            "sample_id": f"{split}:{partition}:{label}:{index}",
                            "score": score,
                            "source_split": split,
                        }
                    )
    _write_jsonl(run_dir / "raw_scores.jsonl", raw_rows)

    result = run_augmented_hc3_score_shift_report(
        AugmentedHc3ScoreShiftRequest(
            project_root=tmp_path,
            run_dir=run_dir,
            output_dir=run_dir / "analysis_augmented_score_shift",
            dataset_id="hc3",
            baseline_split="aug_baseline",
            scenario_suffix="orig_h_aug_ai",
            augmented_human_suffix="aug_h_orig_ai",
            include_augmented_human=True,
            partition="test",
        )
    )

    assert result.score_summary_csv.exists()
    assert result.ai_shift_csv.exists()
    assert result.human_shift_csv.exists()
    assert result.markdown_summary.exists()
    assert (result.output_dir / "thin_bars_baseline_vs_aug_detector_a.png").exists()
    assert (result.output_dir / "thin_bars_combined_baseline_vs_aug_both_detector_a.png").exists()
    assert result.augmented_human_output_dir is not None
    assert (result.augmented_human_output_dir / "thin_bars_baseline_vs_aug_detector_a.png").exists()

    ai_shift = pd.read_csv(result.ai_shift_csv)
    row = ai_shift[ai_shift["augmentation"] == "style"].iloc[0]
    assert abs(float(row["delta_mean_score"]) - (-0.2)) < 1e-9
