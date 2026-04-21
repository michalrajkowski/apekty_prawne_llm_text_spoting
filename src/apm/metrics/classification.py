"""Deterministic binary classification metrics and threshold selection helpers."""

from __future__ import annotations

import math
import statistics
from dataclasses import dataclass
from typing import Literal, Sequence


Label = Literal["human", "ai"]
ThresholdObjective = Literal["balanced_accuracy", "accuracy", "f1"]


@dataclass(frozen=True, slots=True)
class ConfusionCounts:
    """Confusion matrix counts for binary `human`/`ai` classification."""

    true_positive: int
    false_positive: int
    true_negative: int
    false_negative: int


@dataclass(frozen=True, slots=True)
class ClassificationMetrics:
    """Threshold-based and ranking metrics for one detector."""

    threshold: float
    accuracy: float
    precision: float
    recall: float
    f1: float
    balanced_accuracy: float
    roc_auc: float | None
    pr_auc: float | None
    mean_score_human: float
    mean_score_ai: float
    std_score_human: float
    std_score_ai: float
    score_overlap_rate: float
    confusion: ConfusionCounts


@dataclass(frozen=True, slots=True)
class ThresholdSelection:
    """Selected threshold and optimization objective outcome."""

    threshold: float
    objective_name: ThresholdObjective
    objective_value: float
    candidate_count: int


@dataclass(frozen=True, slots=True)
class ThresholdCandidateMetrics:
    """One threshold candidate with deterministic objective metric values."""

    threshold: float
    accuracy: float
    precision: float
    recall: float
    f1: float
    balanced_accuracy: float


@dataclass(frozen=True, slots=True)
class ProbabilityMetrics:
    """Probability-calibration metrics for detectors with calibrated probabilities."""

    brier_score: float
    log_loss: float


def predict_labels_from_threshold(scores: Sequence[float], threshold: float) -> list[Label]:
    """Convert numeric scores into labels using `score >= threshold -> ai` rule."""

    predictions: list[Label] = []
    for score in scores:
        predictions.append("ai" if score >= threshold else "human")
    return predictions


def compute_confusion_counts(labels: Sequence[Label], predictions: Sequence[Label]) -> ConfusionCounts:
    """Compute confusion matrix counts for canonical `human`/`ai` labels."""

    if len(labels) != len(predictions):
        raise ValueError("labels and predictions must have the same length.")

    true_positive = 0
    false_positive = 0
    true_negative = 0
    false_negative = 0

    for label, predicted in zip(labels, predictions, strict=True):
        if label == "ai" and predicted == "ai":
            true_positive += 1
        elif label == "human" and predicted == "ai":
            false_positive += 1
        elif label == "human" and predicted == "human":
            true_negative += 1
        elif label == "ai" and predicted == "human":
            false_negative += 1
        else:
            raise ValueError(f"Unsupported label value pair: {label!r}, {predicted!r}")

    return ConfusionCounts(
        true_positive=true_positive,
        false_positive=false_positive,
        true_negative=true_negative,
        false_negative=false_negative,
    )


def compute_classification_metrics(
    labels: Sequence[Label],
    scores: Sequence[float],
    threshold: float,
) -> ClassificationMetrics:
    """Compute threshold metrics and ranking diagnostics for scored records."""

    normalized_labels, normalized_scores = _normalize_binary_inputs(labels=labels, scores=scores)
    predictions = predict_labels_from_threshold(normalized_scores, threshold=threshold)
    confusion = compute_confusion_counts(labels=normalized_labels, predictions=predictions)

    accuracy = _safe_divide(
        confusion.true_positive + confusion.true_negative,
        len(normalized_labels),
    )
    precision = _safe_divide(
        confusion.true_positive,
        confusion.true_positive + confusion.false_positive,
    )
    recall = _safe_divide(
        confusion.true_positive,
        confusion.true_positive + confusion.false_negative,
    )
    f1 = _safe_divide(2.0 * precision * recall, precision + recall)

    specificity = _safe_divide(
        confusion.true_negative,
        confusion.true_negative + confusion.false_positive,
    )
    balanced_accuracy = 0.5 * (recall + specificity)

    roc_auc = compute_roc_auc(labels=normalized_labels, scores=normalized_scores)
    pr_auc = compute_pr_auc(labels=normalized_labels, scores=normalized_scores)

    human_scores = [score for label, score in zip(normalized_labels, normalized_scores, strict=True) if label == "human"]
    ai_scores = [score for label, score in zip(normalized_labels, normalized_scores, strict=True) if label == "ai"]

    return ClassificationMetrics(
        threshold=float(threshold),
        accuracy=accuracy,
        precision=precision,
        recall=recall,
        f1=f1,
        balanced_accuracy=balanced_accuracy,
        roc_auc=roc_auc,
        pr_auc=pr_auc,
        mean_score_human=statistics.fmean(human_scores),
        mean_score_ai=statistics.fmean(ai_scores),
        std_score_human=_population_std(human_scores),
        std_score_ai=_population_std(ai_scores),
        score_overlap_rate=_score_overlap_rate(human_scores=human_scores, ai_scores=ai_scores),
        confusion=confusion,
    )


def select_threshold(
    labels: Sequence[Label],
    scores: Sequence[float],
    objective: ThresholdObjective = "balanced_accuracy",
) -> ThresholdSelection:
    """Select threshold on calibration data via deterministic objective maximization."""

    if objective not in {"balanced_accuracy", "accuracy", "f1"}:
        raise ValueError(f"Unsupported threshold objective: {objective!r}")

    candidate_metrics = evaluate_threshold_candidates(labels=labels, scores=scores)
    if not candidate_metrics:
        raise ValueError("No threshold candidates generated.")

    best_threshold = candidate_metrics[0].threshold
    best_objective = float("-inf")
    best_f1 = float("-inf")

    for candidate in candidate_metrics:
        objective_value = getattr(candidate, objective)
        if objective_value > best_objective:
            best_threshold = candidate.threshold
            best_objective = objective_value
            best_f1 = candidate.f1
            continue
        if objective_value == best_objective and candidate.f1 > best_f1:
            best_threshold = candidate.threshold
            best_f1 = candidate.f1
            continue
        if objective_value == best_objective and candidate.f1 == best_f1 and candidate.threshold < best_threshold:
            best_threshold = candidate.threshold

    return ThresholdSelection(
        threshold=float(best_threshold),
        objective_name=objective,
        objective_value=float(best_objective),
        candidate_count=len(candidate_metrics),
    )


def evaluate_threshold_candidates(
    labels: Sequence[Label],
    scores: Sequence[float],
) -> list[ThresholdCandidateMetrics]:
    """Evaluate all deterministic threshold candidates on provided calibration data."""

    normalized_labels, normalized_scores = _normalize_binary_inputs(labels=labels, scores=scores)
    candidates = _build_threshold_candidates(normalized_scores)

    evaluations: list[ThresholdCandidateMetrics] = []
    for threshold in candidates:
        metrics = compute_classification_metrics(
            labels=normalized_labels,
            scores=normalized_scores,
            threshold=threshold,
        )
        evaluations.append(
            ThresholdCandidateMetrics(
                threshold=threshold,
                accuracy=metrics.accuracy,
                precision=metrics.precision,
                recall=metrics.recall,
                f1=metrics.f1,
                balanced_accuracy=metrics.balanced_accuracy,
            )
        )
    return evaluations


def compute_probability_metrics(labels: Sequence[Label], probabilities: Sequence[float]) -> ProbabilityMetrics:
    """Compute Brier score and log loss for calibrated probability outputs."""

    normalized_labels, normalized_probabilities = _normalize_binary_inputs(labels=labels, scores=probabilities)
    eps = 1e-15

    squared_errors: list[float] = []
    log_losses: list[float] = []
    for label, probability in zip(normalized_labels, normalized_probabilities, strict=True):
        y_true = 1.0 if label == "ai" else 0.0
        clipped = min(max(probability, eps), 1.0 - eps)
        squared_errors.append((clipped - y_true) ** 2)
        log_losses.append(-((y_true * math.log(clipped)) + ((1.0 - y_true) * math.log(1.0 - clipped))))

    return ProbabilityMetrics(
        brier_score=statistics.fmean(squared_errors),
        log_loss=statistics.fmean(log_losses),
    )


def compute_roc_auc(labels: Sequence[Label], scores: Sequence[float]) -> float | None:
    """Compute ROC AUC via rank-based Mann-Whitney formulation."""

    normalized_labels, normalized_scores = _normalize_binary_inputs(labels=labels, scores=scores)
    n = len(normalized_labels)

    positive_indices: list[int] = []
    for index, label in enumerate(normalized_labels):
        if label == "ai":
            positive_indices.append(index)

    n_pos = len(positive_indices)
    n_neg = n - n_pos
    if n_pos == 0 or n_neg == 0:
        return None

    order = sorted(range(n), key=lambda idx: normalized_scores[idx])
    ranks = [0.0] * n

    pointer = 0
    while pointer < n:
        next_pointer = pointer
        current_score = normalized_scores[order[pointer]]
        while next_pointer < n and normalized_scores[order[next_pointer]] == current_score:
            next_pointer += 1
        average_rank = (pointer + 1 + next_pointer) / 2.0
        for ranked_index in order[pointer:next_pointer]:
            ranks[ranked_index] = average_rank
        pointer = next_pointer

    positive_rank_sum = sum(ranks[index] for index in positive_indices)
    auc = (positive_rank_sum - (n_pos * (n_pos + 1) / 2.0)) / float(n_pos * n_neg)
    return float(auc)


def compute_pr_auc(labels: Sequence[Label], scores: Sequence[float]) -> float | None:
    """Compute average precision (area under PR curve) for binary labels."""

    normalized_labels, normalized_scores = _normalize_binary_inputs(labels=labels, scores=scores)
    positives = sum(1 for label in normalized_labels if label == "ai")
    if positives == 0:
        return None

    ranked = sorted(
        zip(normalized_scores, normalized_labels, strict=True),
        key=lambda item: item[0],
        reverse=True,
    )

    true_positives = 0
    false_positives = 0
    precision_sum = 0.0

    for _score, label in ranked:
        if label == "ai":
            true_positives += 1
            precision_sum += _safe_divide(true_positives, true_positives + false_positives)
        else:
            false_positives += 1

    return precision_sum / float(positives)


def _normalize_binary_inputs(labels: Sequence[Label], scores: Sequence[float]) -> tuple[list[Label], list[float]]:
    """Validate and normalize binary classification inputs."""

    if len(labels) != len(scores):
        raise ValueError("labels and scores must have the same length.")
    if not labels:
        raise ValueError("labels cannot be empty.")

    normalized_labels: list[Label] = []
    for label in labels:
        if label not in {"human", "ai"}:
            raise ValueError(f"Unsupported label value: {label!r}")
        normalized_labels.append(label)

    normalized_scores: list[float] = []
    for score in scores:
        normalized_value = float(score)
        if not math.isfinite(normalized_value):
            raise ValueError("scores must contain only finite values.")
        normalized_scores.append(normalized_value)

    return normalized_labels, normalized_scores


def _build_threshold_candidates(scores: Sequence[float]) -> list[float]:
    """Build deterministic threshold candidates from score distribution."""

    unique_scores = sorted(set(float(score) for score in scores))
    if len(unique_scores) == 1:
        score = unique_scores[0]
        return [score - 1e-12, score, score + 1e-12]

    candidates: list[float] = [unique_scores[0] - 1e-12]
    for left, right in zip(unique_scores[:-1], unique_scores[1:], strict=True):
        candidates.append((left + right) / 2.0)
    candidates.append(unique_scores[-1] + 1e-12)
    return candidates


def _safe_divide(numerator: float, denominator: float) -> float:
    """Safely divide numeric values with zero-denominator fallback."""

    if denominator == 0:
        return 0.0
    return numerator / denominator


def _population_std(values: Sequence[float]) -> float:
    """Compute population standard deviation with deterministic singleton fallback."""

    if len(values) <= 1:
        return 0.0
    return statistics.pstdev(values)


def _score_overlap_rate(human_scores: Sequence[float], ai_scores: Sequence[float]) -> float:
    """Estimate overlap intensity between human and ai score distributions."""

    lower = max(min(human_scores), min(ai_scores))
    upper = min(max(human_scores), max(ai_scores))
    if lower > upper:
        return 0.0

    overlap_count = 0
    combined_count = len(human_scores) + len(ai_scores)
    for score in human_scores:
        if lower <= score <= upper:
            overlap_count += 1
    for score in ai_scores:
        if lower <= score <= upper:
            overlap_count += 1

    return overlap_count / float(combined_count)
