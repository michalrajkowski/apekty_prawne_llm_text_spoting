"""Tests for global/local Q1/Q2 report generation."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from apm.experiments.global_local_q1_q2_report import (
    GlobalLocalQ1Q2ReportRequest,
    run_global_local_q1_q2_report,
)


def _build_scope_row(
    *,
    scope_id: str,
    detector_run_id: str,
    source_splits: list[str],
    threshold: float,
    train_ba: float,
    test_ba: float,
    train_conf: dict[str, int],
    test_conf: dict[str, int],
) -> dict[str, object]:
    return {
        "detector_id": "mock_detector",
        "detector_run_id": detector_run_id,
        "scope_id": scope_id,
        "scope_type": "global" if ":" not in scope_id else "local",
        "source_splits": source_splits,
        "threshold": threshold,
        "threshold_objective": "balanced_accuracy",
        "threshold_objective_value": train_ba,
        "nonfinite_scores_dropped_train": 0,
        "nonfinite_scores_dropped_test": 0,
        "train_count": 8,
        "test_count": 8,
        "train_counts_by_label": {"ai": 4, "human": 4},
        "test_counts_by_label": {"ai": 4, "human": 4},
        "train_metrics": {
            "accuracy": train_ba,
            "balanced_accuracy": train_ba,
            "precision": train_ba,
            "recall": train_ba,
            "f1": train_ba,
            "roc_auc": train_ba,
            "pr_auc": train_ba,
            "mean_score_human": 0.2,
            "mean_score_ai": 0.8,
            "std_score_human": 0.1,
            "std_score_ai": 0.1,
            "score_overlap_rate": 0.1,
            "threshold": threshold,
            "confusion_matrix": train_conf,
        },
        "test_metrics": {
            "accuracy": test_ba,
            "balanced_accuracy": test_ba,
            "precision": test_ba,
            "recall": test_ba,
            "f1": test_ba,
            "roc_auc": test_ba,
            "pr_auc": test_ba,
            "mean_score_human": 0.2,
            "mean_score_ai": 0.8,
            "std_score_human": 0.1,
            "std_score_ai": 0.1,
            "score_overlap_rate": 0.1,
            "threshold": threshold,
            "confusion_matrix": test_conf,
        },
        "probability_metrics_test": {
            "brier_score": 0.1,
            "log_loss": 0.2,
        },
    }


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    payload = "\n".join(json.dumps(row, sort_keys=True) for row in rows) + "\n"
    path.write_text(payload, encoding="utf-8")


def test_global_local_q1_q2_report_generates_expected_artifacts(tmp_path: Path) -> None:
    run_dir = tmp_path / "runs" / "global_local_experiments" / "run_test"
    run_dir.mkdir(parents=True, exist_ok=True)

    global_rows = [
        _build_scope_row(
            scope_id="hc3",
            detector_run_id="mock_detector_run",
            source_splits=["split_a", "split_b"],
            threshold=0.5,
            train_ba=0.8,
            test_ba=0.75,
            train_conf={"true_positive": 3, "false_positive": 1, "true_negative": 3, "false_negative": 1},
            test_conf={"true_positive": 3, "false_positive": 1, "true_negative": 3, "false_negative": 1},
        )
    ]
    local_rows = [
        _build_scope_row(
            scope_id="hc3:split_a",
            detector_run_id="mock_detector_run",
            source_splits=["split_a"],
            threshold=0.4,
            train_ba=0.9,
            test_ba=0.85,
            train_conf={"true_positive": 4, "false_positive": 0, "true_negative": 4, "false_negative": 0},
            test_conf={"true_positive": 3, "false_positive": 1, "true_negative": 3, "false_negative": 1},
        ),
        _build_scope_row(
            scope_id="hc3:split_b",
            detector_run_id="mock_detector_run",
            source_splits=["split_b"],
            threshold=0.6,
            train_ba=0.85,
            test_ba=0.8,
            train_conf={"true_positive": 3, "false_positive": 1, "true_negative": 3, "false_negative": 1},
            test_conf={"true_positive": 3, "false_positive": 1, "true_negative": 3, "false_negative": 1},
        ),
    ]

    raw_rows: list[dict[str, object]] = []
    for split in ("split_a", "split_b"):
        for partition in ("train", "test"):
            for label, scores in (("human", [0.1, 0.2, 0.3, 0.4]), ("ai", [0.6, 0.7, 0.8, 0.9])):
                for index, score in enumerate(scores, start=1):
                    raw_rows.append(
                        {
                            "dataset_id": "hc3",
                            "detector_run_id": "mock_detector_run",
                            "label": label,
                            "partition": partition,
                            "sample_id": f"{split}:{partition}:{label}:{index}",
                            "score": score,
                            "source_split": split,
                        }
                    )

    _write_jsonl(run_dir / "global_results.jsonl", global_rows)
    _write_jsonl(run_dir / "local_results.jsonl", local_rows)
    _write_jsonl(run_dir / "raw_scores.jsonl", raw_rows)

    result = run_global_local_q1_q2_report(
        GlobalLocalQ1Q2ReportRequest(
            project_root=tmp_path,
            run_dir=run_dir,
            output_dir=run_dir / "analysis_q1_q2",
            dataset_id="hc3",
            global_scope_id="hc3",
            local_scope_prefix="hc3:",
        )
    )

    assert result.transfer_csv.exists()
    assert result.local_vs_global_csv.exists()
    assert result.train_eval_gap_csv.exists()
    assert result.detector_ranking_csv.exists()
    assert result.q1_markdown.exists()
    assert result.q2_markdown.exists()
    assert result.findings_markdown.exists()

    transfer = pd.read_csv(result.transfer_csv)
    local_vs_global = pd.read_csv(result.local_vs_global_csv)
    assert not transfer.empty
    assert not local_vs_global.empty
    assert set(local_vs_global["target_split"].unique().tolist()) == {"split_a", "split_b"}
