"""Immutable detector-scoring experiment runner with train/evaluation splits."""

from __future__ import annotations

import argparse
import importlib
import json
import random
import statistics
import uuid
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import pandas as pd

from apm.experiments.matrix import (
    DEFAULT_MODEL_RUN_IDS,
    DatasetSpec,
    ModelRunSpec,
    default_dataset_tokens,
    discover_model_runs,
    parse_dataset_specs,
)
from apm.metrics.classification import (
    ClassificationMetrics,
    ProbabilityMetrics,
    ThresholdCandidateMetrics,
    ThresholdObjective,
    compute_classification_metrics,
    evaluate_threshold_candidates,
    compute_probability_metrics,
    predict_labels_from_threshold,
    select_threshold,
)


PROBABILITY_DETECTOR_IDS: tuple[str, ...] = ("aigc_detector_env3", "aigc_detector_env3short")


@dataclass(frozen=True, slots=True)
class ExperimentRequest:
    """Full immutable request for one scoring experiment run."""

    project_root: Path
    dataset_specs: tuple[DatasetSpec, ...]
    model_run_ids: tuple[str, ...]
    train_examples_per_label: int
    evaluation_examples_per_label: int
    seed: int
    threshold_objective: ThresholdObjective
    output_root: Path


@dataclass(frozen=True, slots=True)
class ScoredRecord:
    """One source text record with detector score output."""

    dataset_id: str
    dataset_split: str
    sample_id: str
    text: str
    label: str


@dataclass(frozen=True, slots=True)
class SplitRecord:
    """One text record assigned into train or evaluation split."""

    split_name: str
    record: ScoredRecord


@dataclass(frozen=True, slots=True)
class ExperimentResult:
    """Summary of persisted experiment artifacts for one run."""

    run_id: str
    run_dir: Path
    split_assignments_path: Path
    predictions_path: Path
    thresholds_path: Path
    metrics_by_detector_path: Path
    metrics_overall_path: Path
    config_snapshot_path: Path


def run_experiment(request: ExperimentRequest) -> ExperimentResult:
    """Execute one detector scoring experiment and persist immutable artifacts."""

    _validate_request(request)
    model_runs = discover_model_runs(
        project_root=request.project_root,
        selected_run_ids=request.model_run_ids,
    )

    available_records = _load_records(
        project_root=request.project_root,
        dataset_specs=request.dataset_specs,
    )
    split_records = _split_records_for_train_and_evaluation(
        records=available_records,
        train_examples_per_label=request.train_examples_per_label,
        evaluation_examples_per_label=request.evaluation_examples_per_label,
        seed=request.seed,
    )

    run_id = _build_run_id(seed=request.seed)
    run_dir = _make_run_dir(output_root=request.output_root, run_id=run_id)

    split_assignments_path = run_dir / "split_assignments.jsonl"
    predictions_path = run_dir / "raw_predictions.jsonl"
    thresholds_path = run_dir / "thresholds.json"
    metrics_by_detector_path = run_dir / "metrics_by_detector.json"
    metrics_overall_path = run_dir / "metrics_overall.json"
    config_snapshot_path = run_dir / "config_snapshot.json"

    prediction_rows: list[dict[str, Any]] = []
    thresholds_payload: dict[str, Any] = {}
    metrics_by_detector: dict[str, Any] = {}

    train_records = [item.record for item in split_records if item.split_name == "train"]
    evaluation_records = [item.record for item in split_records if item.split_name == "evaluation"]

    train_texts = [record.text for record in train_records]
    evaluation_texts = [record.text for record in evaluation_records]
    train_labels = [record.label for record in train_records]
    evaluation_labels = [record.label for record in evaluation_records]

    for model_run in model_runs:
        scores_train, scores_evaluation = _score_detector(
            model_run=model_run,
            train_texts=train_texts,
            evaluation_texts=evaluation_texts,
        )

        threshold_candidates = evaluate_threshold_candidates(
            labels=train_labels,
            scores=scores_train,
        )
        threshold = select_threshold(
            labels=train_labels,
            scores=scores_train,
            objective=request.threshold_objective,
        )
        train_metrics = compute_classification_metrics(
            labels=train_labels,
            scores=scores_train,
            threshold=threshold.threshold,
        )
        evaluation_metrics = compute_classification_metrics(
            labels=evaluation_labels,
            scores=scores_evaluation,
            threshold=threshold.threshold,
        )

        thresholds_payload[model_run.run_id] = {
            "threshold": threshold.threshold,
            "objective": threshold.objective_name,
            "objective_value": threshold.objective_value,
            "candidate_count": threshold.candidate_count,
            "candidate_metrics": _threshold_candidates_to_rows(threshold_candidates),
        }

        probability_metrics_payload: dict[str, float] | None = None
        if model_run.detector_id in PROBABILITY_DETECTOR_IDS:
            probability_metrics = compute_probability_metrics(
                labels=evaluation_labels,
                probabilities=scores_evaluation,
            )
            probability_metrics_payload = _probability_metrics_to_dict(probability_metrics)

        metrics_by_detector[model_run.run_id] = {
            "detector_id": model_run.detector_id,
            "config_path": str(model_run.config_path),
            "config_overrides": dict(model_run.config_overrides),
            "threshold": threshold.threshold,
            "threshold_objective": threshold.objective_name,
            "threshold_objective_value": threshold.objective_value,
            "train": _classification_metrics_to_dict(train_metrics),
            "evaluation": _classification_metrics_to_dict(evaluation_metrics),
            "probability_metrics": probability_metrics_payload,
        }

        prediction_rows.extend(
            _build_prediction_rows(
                run_id=run_id,
                model_run=model_run,
                split_name="train",
                records=train_records,
                scores=scores_train,
                threshold=threshold.threshold,
            )
        )
        prediction_rows.extend(
            _build_prediction_rows(
                run_id=run_id,
                model_run=model_run,
                split_name="evaluation",
                records=evaluation_records,
                scores=scores_evaluation,
                threshold=threshold.threshold,
            )
        )

    created_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
    config_snapshot = {
        "run_id": run_id,
        "created_at_utc": created_at,
        "project_root": str(request.project_root),
        "seed": request.seed,
        "train_examples_per_label": request.train_examples_per_label,
        "evaluation_examples_per_label": request.evaluation_examples_per_label,
        "threshold_objective": request.threshold_objective,
        "dataset_specs": [asdict(dataset) for dataset in request.dataset_specs],
        "model_run_ids": list(request.model_run_ids),
        "resolved_model_runs": [
            {
                "run_id": run.run_id,
                "detector_id": run.detector_id,
                "adapter_module": run.adapter_module,
                "adapter_class": run.adapter_class,
                "config_path": str(run.config_path),
                "config_overrides": dict(run.config_overrides),
            }
            for run in model_runs
        ],
    }

    metrics_overall = _build_overall_metrics(
        run_id=run_id,
        created_at=created_at,
        split_records=split_records,
        metrics_by_detector=metrics_by_detector,
        model_run_ids=request.model_run_ids,
    )

    split_assignment_rows = [
        {
            "run_id": run_id,
            "split_name": item.split_name,
            "dataset_id": item.record.dataset_id,
            "dataset_split": item.record.dataset_split,
            "sample_id": item.record.sample_id,
            "label": item.record.label,
        }
        for item in split_records
    ]

    _write_jsonl(path=split_assignments_path, rows=split_assignment_rows)
    _write_jsonl(path=predictions_path, rows=prediction_rows)
    _write_json(path=thresholds_path, payload=thresholds_payload)
    _write_json(path=metrics_by_detector_path, payload=metrics_by_detector)
    _write_json(path=metrics_overall_path, payload=metrics_overall)
    _write_json(path=config_snapshot_path, payload=config_snapshot)

    _append_run_index(
        output_root=request.output_root,
        record={
            "run_id": run_id,
            "created_at_utc": created_at,
            "run_dir": str(run_dir),
            "detector_runs": list(request.model_run_ids),
            "train_count": sum(1 for item in split_records if item.split_name == "train"),
            "evaluation_count": sum(1 for item in split_records if item.split_name == "evaluation"),
            "prediction_rows": len(prediction_rows),
            "macro_eval_accuracy": metrics_overall["macro_eval_accuracy"],
            "macro_eval_f1": metrics_overall["macro_eval_f1"],
        },
    )

    return ExperimentResult(
        run_id=run_id,
        run_dir=run_dir,
        split_assignments_path=split_assignments_path,
        predictions_path=predictions_path,
        thresholds_path=thresholds_path,
        metrics_by_detector_path=metrics_by_detector_path,
        metrics_overall_path=metrics_overall_path,
        config_snapshot_path=config_snapshot_path,
    )


def build_request_from_args(args: argparse.Namespace) -> ExperimentRequest:
    """Build typed experiment request from parsed CLI namespace."""

    project_root = args.project_root.resolve()
    dataset_specs = parse_dataset_specs(tuple(args.datasets))
    output_root = (project_root / args.output_root).resolve()

    return ExperimentRequest(
        project_root=project_root,
        dataset_specs=dataset_specs,
        model_run_ids=tuple(args.model_runs),
        train_examples_per_label=args.train_examples_per_label,
        evaluation_examples_per_label=args.evaluation_examples_per_label,
        seed=args.seed,
        threshold_objective=args.threshold_objective,
        output_root=output_root,
    )


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run immutable detector scoring experiment.")
    parser.add_argument("--project-root", type=Path, default=Path.cwd(), help="Repository root directory.")
    parser.add_argument(
        "--datasets",
        nargs="*",
        default=list(default_dataset_tokens()),
        help="Dataset specs in `<dataset_id>:<split>` form.",
    )
    parser.add_argument(
        "--model-runs",
        nargs="+",
        default=list(DEFAULT_MODEL_RUN_IDS),
        help="Detector run ids (for seqxgpt use `seqxgpt:<variant_id>`).",
    )
    parser.add_argument(
        "--train-examples-per-label",
        type=int,
        default=100,
        help="How many train examples to sample per label (`human` and `ai`).",
    )
    parser.add_argument(
        "--evaluation-examples-per-label",
        type=int,
        default=100,
        help="How many evaluation examples to sample per label (`human` and `ai`).",
    )
    parser.add_argument("--seed", type=int, default=42, help="Deterministic seed for sampling and splits.")
    parser.add_argument(
        "--threshold-objective",
        choices=("balanced_accuracy", "accuracy", "f1"),
        default="balanced_accuracy",
        help="Objective used for threshold selection on train split.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("runs/experiments"),
        help="Output root directory for immutable experiment artifacts.",
    )
    return parser


def main() -> int:
    """CLI entrypoint for experiment runner."""

    parser = build_arg_parser()
    args = parser.parse_args()
    request = build_request_from_args(args)
    result = run_experiment(request)

    summary = {
        "run_id": result.run_id,
        "run_dir": str(result.run_dir),
        "split_assignments_path": str(result.split_assignments_path),
        "predictions_path": str(result.predictions_path),
        "thresholds_path": str(result.thresholds_path),
        "metrics_by_detector_path": str(result.metrics_by_detector_path),
        "metrics_overall_path": str(result.metrics_overall_path),
        "config_snapshot_path": str(result.config_snapshot_path),
    }
    print(json.dumps(summary, indent=2))
    return 0


def _validate_request(request: ExperimentRequest) -> None:
    """Validate experiment request invariants before execution."""

    if request.train_examples_per_label <= 0:
        raise ValueError("train_examples_per_label must be > 0")
    if request.evaluation_examples_per_label <= 0:
        raise ValueError("evaluation_examples_per_label must be > 0")
    if not request.dataset_specs:
        raise ValueError("dataset_specs cannot be empty")
    if not request.model_run_ids:
        raise ValueError("model_run_ids cannot be empty")


def _load_records(
    project_root: Path,
    dataset_specs: Sequence[DatasetSpec],
) -> list[ScoredRecord]:
    """Load labeled records from interim parquet files."""

    by_label: dict[str, list[ScoredRecord]] = {"human": [], "ai": []}

    for dataset in dataset_specs:
        for label in ("human", "ai"):
            label_path = (
                project_root
                / "data"
                / "interim"
                / "datasets"
                / dataset.dataset_id
                / dataset.split
                / label
                / "sampled_records.parquet"
            )
            if not label_path.exists():
                raise FileNotFoundError(f"Missing parquet file: {label_path}")

            frame = pd.read_parquet(label_path)
            if "text" not in frame.columns or "sample_id" not in frame.columns:
                raise ValueError(
                    f"Parquet file must contain `text` and `sample_id` columns: {label_path}"
                )

            for row in frame.itertuples(index=False):
                text = getattr(row, "text")
                sample_id = getattr(row, "sample_id")
                if not isinstance(text, str) or not text.strip():
                    raise ValueError(f"Invalid non-text row found in {label_path}")
                if not isinstance(sample_id, str) or not sample_id.strip():
                    raise ValueError(f"Invalid sample_id row found in {label_path}")
                by_label[label].append(
                    ScoredRecord(
                        dataset_id=dataset.dataset_id,
                        dataset_split=dataset.split,
                        sample_id=sample_id,
                        text=text,
                        label=label,
                    )
                )

    merged_records: list[ScoredRecord] = []
    merged_records.extend(by_label["human"])
    merged_records.extend(by_label["ai"])
    return merged_records


def _split_records_for_train_and_evaluation(
    records: Sequence[ScoredRecord],
    train_examples_per_label: int,
    evaluation_examples_per_label: int,
    seed: int,
) -> list[SplitRecord]:
    """Build deterministic train/evaluation splits with explicit per-label sizes."""

    by_label: dict[str, list[ScoredRecord]] = {"human": [], "ai": []}
    for record in records:
        by_label[record.label].append(record)

    split_rows: list[SplitRecord] = []
    for label_offset, label in enumerate(("human", "ai")):
        label_records = list(by_label[label])
        required = train_examples_per_label + evaluation_examples_per_label
        if len(label_records) < required:
            raise ValueError(
                f"Need {required} `{label}` records for train+evaluation "
                f"({train_examples_per_label}+{evaluation_examples_per_label}), got {len(label_records)}."
            )

        label_rng = random.Random(seed + 100 + label_offset)
        label_rng.shuffle(label_records)

        train_records = label_records[:train_examples_per_label]
        evaluation_records = label_records[
            train_examples_per_label : train_examples_per_label + evaluation_examples_per_label
        ]

        split_rows.extend(
            SplitRecord(split_name="train", record=record)
            for record in train_records
        )
        split_rows.extend(
            SplitRecord(split_name="evaluation", record=record)
            for record in evaluation_records
        )

    split_rng = random.Random(seed + 999)
    split_rng.shuffle(split_rows)
    return split_rows


def _score_detector(
    model_run: ModelRunSpec,
    train_texts: Sequence[str],
    evaluation_texts: Sequence[str],
) -> tuple[list[float], list[float]]:
    """Initialize one detector and score train/evaluation text batches."""

    config_payload = json.loads(model_run.config_path.read_text(encoding="utf-8"))
    if not isinstance(config_payload, dict):
        raise ValueError(f"Detector config must be JSON object: {model_run.config_path}")

    merged_config = dict(config_payload)
    merged_config.update(model_run.config_overrides)

    detector_cls = _load_detector_class(
        module_name=model_run.adapter_module,
        class_name=model_run.adapter_class,
    )
    detector = detector_cls.initialize(config=merged_config)

    train_scores = detector.predict_batch(list(train_texts))
    evaluation_scores = detector.predict_batch(list(evaluation_texts))
    detector.delete()

    if len(train_scores) != len(train_texts):
        raise ValueError(
            f"Detector '{model_run.run_id}' returned {len(train_scores)} train scores "
            f"for {len(train_texts)} texts."
        )
    if len(evaluation_scores) != len(evaluation_texts):
        raise ValueError(
            f"Detector '{model_run.run_id}' returned {len(evaluation_scores)} evaluation scores "
            f"for {len(evaluation_texts)} texts."
        )

    normalized_train = [float(score) for score in train_scores]
    normalized_evaluation = [float(score) for score in evaluation_scores]
    _validate_finite_scores(normalized_train)
    _validate_finite_scores(normalized_evaluation)
    return normalized_train, normalized_evaluation


def _load_detector_class(module_name: str, class_name: str) -> type[Any]:
    """Resolve detector class dynamically from module and class names."""

    module = importlib.import_module(module_name)
    detector_cls = getattr(module, class_name)
    return detector_cls


def _validate_finite_scores(scores: Sequence[float]) -> None:
    """Validate scored values are finite floats."""

    for score in scores:
        if not isinstance(score, float):
            raise ValueError("Detector scores must be float values.")
        if not (float("-inf") < score < float("inf")):
            raise ValueError("Detector scores must be finite values.")


def _build_prediction_rows(
    run_id: str,
    model_run: ModelRunSpec,
    split_name: str,
    records: Sequence[ScoredRecord],
    scores: Sequence[float],
    threshold: float,
) -> list[dict[str, Any]]:
    """Build JSONL prediction rows for one detector and one split."""

    predicted_labels = predict_labels_from_threshold(scores=scores, threshold=threshold)
    rows: list[dict[str, Any]] = []
    for record, score, predicted in zip(records, scores, predicted_labels, strict=True):
        rows.append(
            {
                "run_id": run_id,
                "detector_run_id": model_run.run_id,
                "detector_id": model_run.detector_id,
                "split_name": split_name,
                "dataset_id": record.dataset_id,
                "dataset_split": record.dataset_split,
                "sample_id": record.sample_id,
                "label": record.label,
                "score": score,
                "threshold": threshold,
                "predicted_label": predicted,
            }
        )
    return rows


def _classification_metrics_to_dict(metrics: ClassificationMetrics) -> dict[str, Any]:
    """Serialize classification metrics dataclass into JSON-friendly dict."""

    return {
        "threshold": metrics.threshold,
        "accuracy": metrics.accuracy,
        "precision": metrics.precision,
        "recall": metrics.recall,
        "f1": metrics.f1,
        "balanced_accuracy": metrics.balanced_accuracy,
        "roc_auc": metrics.roc_auc,
        "pr_auc": metrics.pr_auc,
        "mean_score_human": metrics.mean_score_human,
        "mean_score_ai": metrics.mean_score_ai,
        "std_score_human": metrics.std_score_human,
        "std_score_ai": metrics.std_score_ai,
        "score_overlap_rate": metrics.score_overlap_rate,
        "confusion_matrix": {
            "true_positive": metrics.confusion.true_positive,
            "false_positive": metrics.confusion.false_positive,
            "true_negative": metrics.confusion.true_negative,
            "false_negative": metrics.confusion.false_negative,
        },
    }


def _probability_metrics_to_dict(metrics: ProbabilityMetrics) -> dict[str, float]:
    """Serialize probability metrics dataclass into JSON-friendly dict."""

    return {
        "brier_score": metrics.brier_score,
        "log_loss": metrics.log_loss,
    }


def _threshold_candidates_to_rows(candidates: Sequence[ThresholdCandidateMetrics]) -> list[dict[str, float]]:
    """Serialize threshold candidate diagnostics for run artifacts."""

    rows: list[dict[str, float]] = []
    for candidate in candidates:
        rows.append(
            {
                "threshold": candidate.threshold,
                "accuracy": candidate.accuracy,
                "precision": candidate.precision,
                "recall": candidate.recall,
                "f1": candidate.f1,
                "balanced_accuracy": candidate.balanced_accuracy,
            }
        )
    return rows


def _build_overall_metrics(
    run_id: str,
    created_at: str,
    split_records: Sequence[SplitRecord],
    metrics_by_detector: Mapping[str, Any],
    model_run_ids: Sequence[str],
) -> dict[str, Any]:
    """Build top-level run summary metrics."""

    evaluation_metrics: list[Mapping[str, Any]] = []
    for run_id_key in model_run_ids:
        detector_payload = metrics_by_detector.get(run_id_key)
        if not isinstance(detector_payload, Mapping):
            continue
        evaluation = detector_payload.get("evaluation")
        if isinstance(evaluation, Mapping):
            evaluation_metrics.append(evaluation)

    macro_eval_accuracy = statistics.fmean(float(item["accuracy"]) for item in evaluation_metrics)
    macro_eval_f1 = statistics.fmean(float(item["f1"]) for item in evaluation_metrics)

    return {
        "run_id": run_id,
        "created_at_utc": created_at,
        "detector_runs": list(model_run_ids),
        "train_count": sum(1 for item in split_records if item.split_name == "train"),
        "evaluation_count": sum(1 for item in split_records if item.split_name == "evaluation"),
        "macro_eval_accuracy": macro_eval_accuracy,
        "macro_eval_f1": macro_eval_f1,
    }


def _build_run_id(seed: int) -> str:
    """Build unique run id with UTC timestamp and random suffix."""

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    suffix = uuid.uuid4().hex[:8]
    return f"{timestamp}_seed{seed}_{suffix}"


def _make_run_dir(output_root: Path, run_id: str) -> Path:
    """Create immutable per-run output directory."""

    output_root.mkdir(parents=True, exist_ok=True)
    run_dir = output_root / run_id
    if run_dir.exists():
        raise FileExistsError(f"Run directory already exists: {run_dir}")
    run_dir.mkdir(parents=False, exist_ok=False)
    return run_dir


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    """Write deterministic pretty JSON file."""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(payload), indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    """Write JSONL artifact with one object per line."""

    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [json.dumps(dict(row), ensure_ascii=False, sort_keys=True) for row in rows]
    path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def _append_run_index(output_root: Path, record: Mapping[str, Any]) -> None:
    """Append one run-summary row to global experiments index JSONL."""

    output_root.mkdir(parents=True, exist_ok=True)
    index_path = output_root / "index.jsonl"
    with index_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(dict(record), ensure_ascii=False, sort_keys=True) + "\n")


if __name__ == "__main__":
    raise SystemExit(main())
