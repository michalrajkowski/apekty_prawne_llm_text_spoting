"""Tests for augmented HC3 run analysis and baseline delta computation."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from apm.experiments.augmented_hc3_analysis import (
    AugmentedHc3AnalysisRequest,
    run_augmented_hc3_analysis,
)


def _metrics(*, accuracy: float, precision: float, recall: float, f1: float, ba: float) -> dict[str, object]:
    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "balanced_accuracy": ba,
        "roc_auc": 0.9,
        "pr_auc": 0.9,
        "mean_score_human": 0.2,
        "mean_score_ai": 0.8,
        "confusion_matrix": {
            "true_positive": 80,
            "false_positive": 20,
            "true_negative": 80,
            "false_negative": 20,
        },
    }


def _write_local_results(path: Path) -> None:
    rows = [
        {
            "scope_id": "hc3:aug_baseline",
            "detector_run_id": "aigc_detector_env3",
            "threshold": 0.5,
            "test_metrics": _metrics(accuracy=0.8, precision=0.8, recall=0.8, f1=0.8, ba=0.8),
        },
        {
            "scope_id": "hc3:aug_back_trans_3langs_aug_both",
            "detector_run_id": "aigc_detector_env3",
            "threshold": 0.5,
            "test_metrics": _metrics(accuracy=0.7, precision=0.7, recall=0.7, f1=0.7, ba=0.7),
        },
    ]
    payload = "\n".join(json.dumps(row, sort_keys=True) for row in rows) + "\n"
    path.write_text(payload, encoding="utf-8")


def test_augmented_hc3_analysis_generates_outputs_and_expected_delta(tmp_path: Path) -> None:
    run_dir = tmp_path / "runs" / "global_local_experiments" / "run_x"
    run_dir.mkdir(parents=True, exist_ok=True)
    _write_local_results(run_dir / "local_results.jsonl")

    request = AugmentedHc3AnalysisRequest(
        project_root=tmp_path,
        run_dir=run_dir,
        output_dir=run_dir / "analysis_augmented",
        baseline_scope_id="hc3:aug_baseline",
    )
    result = run_augmented_hc3_analysis(request)

    assert result.scenario_metrics_csv.exists()
    assert result.scenario_deltas_csv.exists()
    assert result.summary_csv.exists()
    assert result.delta_balanced_accuracy_plot.exists()
    assert result.delta_recall_plot.exists()
    assert result.delta_fp_rate_plot.exists()

    deltas = pd.read_csv(result.scenario_deltas_csv)
    row = deltas[deltas["scenario_name"] == "aug_back_trans_3langs_aug_both"].iloc[0]
    assert abs(float(row["delta_accuracy"]) - (-0.1)) < 1e-9
    assert abs(float(row["delta_balanced_accuracy"]) - (-0.1)) < 1e-9
