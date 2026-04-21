"""Unit tests for binary classification metric helpers."""

from __future__ import annotations

import math

from apm.metrics.classification import (
    compute_classification_metrics,
    compute_probability_metrics,
    compute_pr_auc,
    compute_roc_auc,
    evaluate_threshold_candidates,
    select_threshold,
)


def test_classification_metrics_perfect_separation() -> None:
    labels = ["human", "human", "ai", "ai"]
    scores = [0.1, 0.2, 0.8, 0.9]

    metrics = compute_classification_metrics(labels=labels, scores=scores, threshold=0.5)

    assert math.isclose(metrics.accuracy, 1.0)
    assert math.isclose(metrics.precision, 1.0)
    assert math.isclose(metrics.recall, 1.0)
    assert math.isclose(metrics.f1, 1.0)
    assert math.isclose(metrics.balanced_accuracy, 1.0)
    assert math.isclose(metrics.roc_auc or 0.0, 1.0)
    assert math.isclose(metrics.pr_auc or 0.0, 1.0)
    assert math.isclose(metrics.score_overlap_rate, 0.0)


def test_select_threshold_maximizes_balanced_accuracy() -> None:
    labels = ["human", "human", "ai", "ai"]
    scores = [0.2, 0.3, 0.7, 0.8]

    selection = select_threshold(labels=labels, scores=scores, objective="balanced_accuracy")

    metrics = compute_classification_metrics(labels=labels, scores=scores, threshold=selection.threshold)
    assert math.isclose(selection.objective_value, 1.0)
    assert selection.candidate_count >= 3
    assert math.isclose(metrics.balanced_accuracy, 1.0)


def test_probability_metrics_are_computed() -> None:
    labels = ["human", "ai", "human", "ai"]
    probabilities = [0.1, 0.9, 0.2, 0.8]

    metrics = compute_probability_metrics(labels=labels, probabilities=probabilities)

    assert metrics.brier_score < 0.05
    assert metrics.log_loss < 0.3


def test_auc_functions_return_none_for_single_class() -> None:
    labels = ["human", "human", "human"]
    scores = [0.1, 0.2, 0.3]

    assert compute_roc_auc(labels=labels, scores=scores) is None
    assert compute_pr_auc(labels=labels, scores=scores) is None


def test_threshold_candidate_evaluations_are_generated() -> None:
    labels = ["human", "human", "ai", "ai"]
    scores = [0.2, 0.3, 0.7, 0.8]

    candidates = evaluate_threshold_candidates(labels=labels, scores=scores)

    assert len(candidates) >= 3
    assert all(candidate.threshold == float(candidate.threshold) for candidate in candidates)
