"""Run detector scoring once and evaluate global/local threshold strategies."""

from __future__ import annotations

import argparse
import importlib
import json
import math
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Literal, Mapping, Sequence

import pandas as pd

from apm.experiments.matrix import DEFAULT_MODEL_RUN_IDS, ModelRunSpec, discover_model_runs
from apm.metrics.classification import (
    ClassificationMetrics,
    ProbabilityMetrics,
    ThresholdObjective,
    compute_classification_metrics,
    compute_probability_metrics,
    select_threshold,
)

try:
    from tqdm.auto import tqdm as _tqdm
except ModuleNotFoundError:  # pragma: no cover - fallback path if tqdm is unavailable
    _tqdm = None

PROBABILITY_DETECTOR_IDS: tuple[str, ...] = ("aigc_detector_env3", "aigc_detector_env3short")
DEFAULT_HC3_SPLITS: tuple[str, ...] = (
    "all_train",
    "finance_train",
    "medicine_train",
    "open_qa_train",
    "reddit_eli5_train",
    "wiki_csai_train",
)
DEFAULT_GRID_SPLITS: tuple[str, ...] = ("filtered", "unfiltered")


@dataclass(frozen=True, slots=True)
class SplitSelection:
    """One dataset split selected for experiment scoring."""

    dataset_id: str
    source_split: str


@dataclass(frozen=True, slots=True)
class ScoredTextRecord:
    """One text row loaded from materialized train/test split artifacts."""

    dataset_id: str
    source_split: str
    partition: Literal["train", "test"]
    label: Literal["human", "ai"]
    sample_id: str
    text: str


@dataclass(frozen=True, slots=True)
class GlobalLocalExperimentRequest:
    """Config for running pooled-global and per-split-local threshold evaluations."""

    project_root: Path
    model_run_ids: tuple[str, ...]
    hc3_splits: tuple[str, ...]
    grid_splits: tuple[str, ...]
    threshold_objective: ThresholdObjective
    batch_size: int
    output_root: Path


@dataclass(frozen=True, slots=True)
class GlobalLocalExperimentResult:
    """Paths for persisted experiment artifacts for one immutable run."""

    run_id: str
    run_dir: Path
    config_snapshot_path: Path
    raw_scores_path: Path
    global_results_path: Path
    local_results_path: Path
    summary_path: Path


@dataclass(frozen=True, slots=True)
class ScopeScores:
    """Collected labels/scores for one scope partition with non-finite diagnostics."""

    labels: list[str]
    scores: list[float]
    dropped_nonfinite_count: int


def run_global_local_experiment(request: GlobalLocalExperimentRequest) -> GlobalLocalExperimentResult:
    """Run one-pass scoring and evaluate global/local threshold strategies."""

    _validate_request(request)

    model_runs = discover_model_runs(
        project_root=request.project_root,
        selected_run_ids=request.model_run_ids,
    )

    split_selections = _build_split_selections(
        hc3_splits=request.hc3_splits,
        grid_splits=request.grid_splits,
    )
    records = _load_split_records(
        project_root=request.project_root,
        split_selections=split_selections,
    )

    run_id = _build_run_id()
    run_dir = _make_run_dir(output_root=request.output_root, run_id=run_id)

    config_snapshot_path = run_dir / "config_snapshot.json"
    raw_scores_path = run_dir / "raw_scores.jsonl"
    global_results_path = run_dir / "global_results.jsonl"
    local_results_path = run_dir / "local_results.jsonl"
    summary_path = run_dir / "summary.json"

    scores_by_detector = _score_records_once(
        records=records,
        model_runs=model_runs,
        batch_size=request.batch_size,
    )

    raw_score_rows = _build_raw_score_rows(records=records, scores_by_detector=scores_by_detector)
    _write_jsonl(path=raw_scores_path, rows=raw_score_rows)

    global_scope_rows = _evaluate_global_scopes(
        records=records,
        model_runs=model_runs,
        scores_by_detector=scores_by_detector,
        threshold_objective=request.threshold_objective,
        hc3_splits=request.hc3_splits,
        grid_splits=request.grid_splits,
        progress_desc="Global scopes",
    )
    local_scope_rows = _evaluate_local_scopes(
        records=records,
        model_runs=model_runs,
        scores_by_detector=scores_by_detector,
        threshold_objective=request.threshold_objective,
        split_selections=split_selections,
        progress_desc="Local scopes",
    )

    _write_jsonl(path=global_results_path, rows=global_scope_rows)
    _write_jsonl(path=local_results_path, rows=local_scope_rows)

    summary_payload = {
        "run_id": run_id,
        "created_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "record_count": len(records),
        "detector_count": len(model_runs),
        "global_result_rows": len(global_scope_rows),
        "local_result_rows": len(local_scope_rows),
        "global_scopes": sorted({str(row["scope_id"]) for row in global_scope_rows}),
        "local_scopes": sorted({str(row["scope_id"]) for row in local_scope_rows}),
    }
    _write_json(path=summary_path, payload=summary_payload)

    _write_json(
        path=config_snapshot_path,
        payload={
            "run_id": run_id,
            "created_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "project_root": str(request.project_root),
            "model_run_ids": list(request.model_run_ids),
            "threshold_objective": request.threshold_objective,
            "batch_size": request.batch_size,
            "hc3_splits": list(request.hc3_splits),
            "grid_splits": list(request.grid_splits),
            "resolved_model_runs": [
                {
                    "run_id": model_run.run_id,
                    "detector_id": model_run.detector_id,
                    "adapter_module": model_run.adapter_module,
                    "adapter_class": model_run.adapter_class,
                    "config_path": str(model_run.config_path),
                    "config_overrides": dict(model_run.config_overrides),
                }
                for model_run in model_runs
            ],
        },
    )

    return GlobalLocalExperimentResult(
        run_id=run_id,
        run_dir=run_dir,
        config_snapshot_path=config_snapshot_path,
        raw_scores_path=raw_scores_path,
        global_results_path=global_results_path,
        local_results_path=local_results_path,
        summary_path=summary_path,
    )


def build_request_from_args(args: argparse.Namespace) -> GlobalLocalExperimentRequest:
    """Build typed request from CLI args."""

    project_root = args.project_root.resolve()
    output_root = (project_root / args.output_root).resolve()

    return GlobalLocalExperimentRequest(
        project_root=project_root,
        model_run_ids=tuple(args.model_runs),
        hc3_splits=tuple(args.hc3_splits),
        grid_splits=tuple(args.grid_splits),
        threshold_objective=args.threshold_objective,
        batch_size=args.batch_size,
        output_root=output_root,
    )


def build_arg_parser() -> argparse.ArgumentParser:
    """Build CLI parser for global/local threshold experiments."""

    parser = argparse.ArgumentParser(description="Run global/local threshold experiments from persisted train/test splits.")
    parser.add_argument("--project-root", type=Path, default=Path.cwd(), help="Repository root directory.")
    parser.add_argument(
        "--model-runs",
        nargs="+",
        default=list(DEFAULT_MODEL_RUN_IDS),
        help="Detector run ids (for seqxgpt use `seqxgpt:<variant_id>`).",
    )
    parser.add_argument(
        "--hc3-splits",
        nargs="+",
        default=list(DEFAULT_HC3_SPLITS),
        help="HC3 source splits to include in global/local evaluation.",
    )
    parser.add_argument(
        "--grid-splits",
        nargs="+",
        default=list(DEFAULT_GRID_SPLITS),
        help="GriD source splits to include in global/local evaluation.",
    )
    parser.add_argument(
        "--threshold-objective",
        choices=("balanced_accuracy", "accuracy", "f1"),
        default="balanced_accuracy",
        help="Objective used to select thresholds on train data.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Prediction batch size used for detector scoring.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("runs/global_local_experiments"),
        help="Output root directory for immutable experiment runs.",
    )
    return parser


def main() -> int:
    """CLI entrypoint."""

    parser = build_arg_parser()
    args = parser.parse_args()
    request = build_request_from_args(args)
    result = run_global_local_experiment(request)

    print(
        json.dumps(
            {
                "run_id": result.run_id,
                "run_dir": str(result.run_dir),
                "config_snapshot_path": str(result.config_snapshot_path),
                "raw_scores_path": str(result.raw_scores_path),
                "global_results_path": str(result.global_results_path),
                "local_results_path": str(result.local_results_path),
                "summary_path": str(result.summary_path),
            },
            indent=2,
        )
    )
    return 0


def _validate_request(request: GlobalLocalExperimentRequest) -> None:
    """Validate request invariants."""

    if not request.model_run_ids:
        raise ValueError("model_run_ids cannot be empty")
    if not request.hc3_splits:
        raise ValueError("hc3_splits cannot be empty")
    if not request.grid_splits:
        raise ValueError("grid_splits cannot be empty")
    if request.batch_size <= 0:
        raise ValueError("batch_size must be > 0")


def _build_split_selections(
    *,
    hc3_splits: Sequence[str],
    grid_splits: Sequence[str],
) -> tuple[SplitSelection, ...]:
    """Build deterministic ordered split-selection list."""

    selections: list[SplitSelection] = []
    for source_split in hc3_splits:
        selections.append(SplitSelection(dataset_id="hc3", source_split=source_split))
    for source_split in grid_splits:
        selections.append(SplitSelection(dataset_id="grid", source_split=source_split))
    return tuple(selections)


def _load_split_records(
    *,
    project_root: Path,
    split_selections: Sequence[SplitSelection],
) -> list[ScoredTextRecord]:
    """Load train/test records from persisted split artifacts."""

    rows: list[ScoredTextRecord] = []

    for selection in split_selections:
        for partition in ("train", "test"):
            for label in ("human", "ai"):
                parquet_path = (
                    project_root
                    / "data"
                    / "interim"
                    / "splits"
                    / selection.dataset_id
                    / selection.source_split
                    / partition
                    / label
                    / "sampled_records.parquet"
                )
                if not parquet_path.exists():
                    raise FileNotFoundError(f"Missing split parquet: {parquet_path}")

                frame = pd.read_parquet(parquet_path)
                if "sample_id" not in frame.columns or "text" not in frame.columns:
                    raise ValueError(f"Parquet file must contain `sample_id` and `text`: {parquet_path}")

                for row in frame.itertuples(index=False):
                    sample_id = getattr(row, "sample_id")
                    text = getattr(row, "text")
                    if not isinstance(sample_id, str) or not sample_id.strip():
                        raise ValueError(f"Invalid `sample_id` value in {parquet_path}")
                    if not isinstance(text, str) or not text.strip():
                        raise ValueError(f"Invalid `text` value in {parquet_path}")
                    rows.append(
                        ScoredTextRecord(
                            dataset_id=selection.dataset_id,
                            source_split=selection.source_split,
                            partition=partition,
                            label=label,
                            sample_id=sample_id,
                            text=text,
                        )
                    )

    rows.sort(
        key=lambda item: (
            item.dataset_id,
            item.source_split,
            item.partition,
            item.label,
            item.sample_id,
        )
    )
    return rows


def _score_records_once(
    *,
    records: Sequence[ScoredTextRecord],
    model_runs: Sequence[ModelRunSpec],
    batch_size: int,
) -> dict[str, list[float]]:
    """Score each detector once across all records and cache scores in-memory."""

    texts = [record.text for record in records]
    outputs: dict[str, list[float]] = {}

    for model_run in model_runs:
        detector_cls = _load_detector_class(
            module_name=model_run.adapter_module,
            class_name=model_run.adapter_class,
        )
        detector = detector_cls.initialize(config=_build_detector_config(model_run))
        try:
            scores = _predict_in_chunks(detector=detector, texts=texts, batch_size=batch_size)
        finally:
            detector.delete()

        if len(scores) != len(records):
            raise ValueError(
                f"Detector '{model_run.run_id}' returned {len(scores)} scores for {len(records)} records."
            )

        outputs[model_run.run_id] = scores

    return outputs


def _predict_in_chunks(*, detector: Any, texts: Sequence[str], batch_size: int) -> list[float]:
    """Predict scores in fixed-size chunks to limit peak runtime memory."""

    all_scores: list[float] = []
    for offset in range(0, len(texts), batch_size):
        chunk = list(texts[offset : offset + batch_size])
        chunk_scores = detector.predict_batch(chunk)
        if len(chunk_scores) != len(chunk):
            raise ValueError("Detector returned score count that does not match input chunk size.")
        all_scores.extend(float(score) for score in chunk_scores)
    return all_scores


def _build_detector_config(model_run: ModelRunSpec) -> dict[str, Any]:
    """Load base detector config and apply run-level overrides."""

    payload = json.loads(model_run.config_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Detector config must be JSON object: {model_run.config_path}")
    merged = dict(payload)
    merged.update(model_run.config_overrides)
    return merged


def _load_detector_class(module_name: str, class_name: str) -> type[Any]:
    """Load detector class from fully qualified module path."""

    module = importlib.import_module(module_name)
    detector_cls = getattr(module, class_name)
    return detector_cls


def _build_raw_score_rows(
    *,
    records: Sequence[ScoredTextRecord],
    scores_by_detector: Mapping[str, Sequence[float]],
) -> list[dict[str, Any]]:
    """Build row-wise score export for every detector and sample."""

    rows: list[dict[str, Any]] = []
    for detector_run_id, scores in scores_by_detector.items():
        for record, score in zip(records, scores, strict=True):
            rows.append(
                {
                    "detector_run_id": detector_run_id,
                    "dataset_id": record.dataset_id,
                    "source_split": record.source_split,
                    "partition": record.partition,
                    "label": record.label,
                    "sample_id": record.sample_id,
                    "score": float(score),
                }
            )
    return rows


def _evaluate_global_scopes(
    *,
    records: Sequence[ScoredTextRecord],
    model_runs: Sequence[ModelRunSpec],
    scores_by_detector: Mapping[str, Sequence[float]],
    threshold_objective: ThresholdObjective,
    hc3_splits: Sequence[str],
    grid_splits: Sequence[str],
    progress_desc: str | None = None,
) -> list[dict[str, Any]]:
    """Evaluate thresholds tuned globally per dataset family (HC3/GriD)."""

    scope_specs = (
        ("hc3", tuple(hc3_splits)),
        ("grid", tuple(grid_splits)),
    )

    tasks: list[tuple[str, Sequence[str], ModelRunSpec]] = []
    for dataset_id, source_splits in scope_specs:
        for model_run in model_runs:
            tasks.append((dataset_id, source_splits, model_run))

    rows: list[dict[str, Any]] = []
    for dataset_id, source_splits, model_run in _with_tqdm(
        tasks,
        desc=progress_desc,
        unit="scope-detector",
    ):
        train_scope = _collect_labels_and_scores(
            records=records,
            detector_scores=scores_by_detector[model_run.run_id],
            dataset_id=dataset_id,
            source_splits=source_splits,
            partition="train",
        )
        test_scope = _collect_labels_and_scores(
            records=records,
            detector_scores=scores_by_detector[model_run.run_id],
            dataset_id=dataset_id,
            source_splits=source_splits,
            partition="test",
        )

        threshold = select_threshold(
            labels=train_scope.labels,
            scores=train_scope.scores,
            objective=threshold_objective,
        )
        train_metrics = compute_classification_metrics(
            labels=train_scope.labels,
            scores=train_scope.scores,
            threshold=threshold.threshold,
        )
        test_metrics = compute_classification_metrics(
            labels=test_scope.labels,
            scores=test_scope.scores,
            threshold=threshold.threshold,
        )
        probability_metrics = _maybe_probability_metrics(
            detector_id=model_run.detector_id,
            labels=test_scope.labels,
            scores=test_scope.scores,
        )

        rows.append(
            _build_result_row(
                scope_type="global",
                scope_id=dataset_id,
                model_run=model_run,
                threshold=threshold.threshold,
                threshold_objective=threshold.objective_name,
                threshold_objective_value=threshold.objective_value,
                train_labels=train_scope.labels,
                test_labels=test_scope.labels,
                train_metrics=train_metrics,
                test_metrics=test_metrics,
                probability_metrics=probability_metrics,
                source_splits=source_splits,
                nonfinite_dropped_train=train_scope.dropped_nonfinite_count,
                nonfinite_dropped_test=test_scope.dropped_nonfinite_count,
            )
        )

    return rows


def _evaluate_local_scopes(
    *,
    records: Sequence[ScoredTextRecord],
    model_runs: Sequence[ModelRunSpec],
    scores_by_detector: Mapping[str, Sequence[float]],
    threshold_objective: ThresholdObjective,
    split_selections: Sequence[SplitSelection],
    progress_desc: str | None = None,
) -> list[dict[str, Any]]:
    """Evaluate thresholds tuned locally per concrete source split."""

    tasks: list[tuple[SplitSelection, ModelRunSpec]] = []
    for selection in split_selections:
        for model_run in model_runs:
            tasks.append((selection, model_run))

    rows: list[dict[str, Any]] = []
    for selection, model_run in _with_tqdm(
        tasks,
        desc=progress_desc,
        unit="scope-detector",
    ):
        scope_id = f"{selection.dataset_id}:{selection.source_split}"
        train_scope = _collect_labels_and_scores(
            records=records,
            detector_scores=scores_by_detector[model_run.run_id],
            dataset_id=selection.dataset_id,
            source_splits=(selection.source_split,),
            partition="train",
        )
        test_scope = _collect_labels_and_scores(
            records=records,
            detector_scores=scores_by_detector[model_run.run_id],
            dataset_id=selection.dataset_id,
            source_splits=(selection.source_split,),
            partition="test",
        )

        threshold = select_threshold(
            labels=train_scope.labels,
            scores=train_scope.scores,
            objective=threshold_objective,
        )
        train_metrics = compute_classification_metrics(
            labels=train_scope.labels,
            scores=train_scope.scores,
            threshold=threshold.threshold,
        )
        test_metrics = compute_classification_metrics(
            labels=test_scope.labels,
            scores=test_scope.scores,
            threshold=threshold.threshold,
        )
        probability_metrics = _maybe_probability_metrics(
            detector_id=model_run.detector_id,
            labels=test_scope.labels,
            scores=test_scope.scores,
        )

        rows.append(
            _build_result_row(
                scope_type="local",
                scope_id=scope_id,
                model_run=model_run,
                threshold=threshold.threshold,
                threshold_objective=threshold.objective_name,
                threshold_objective_value=threshold.objective_value,
                train_labels=train_scope.labels,
                test_labels=test_scope.labels,
                train_metrics=train_metrics,
                test_metrics=test_metrics,
                probability_metrics=probability_metrics,
                source_splits=(selection.source_split,),
                nonfinite_dropped_train=train_scope.dropped_nonfinite_count,
                nonfinite_dropped_test=test_scope.dropped_nonfinite_count,
            )
        )

    return rows


def _with_tqdm(
    items: Sequence[Any],
    *,
    desc: str | None,
    unit: str,
) -> Iterable[Any]:
    """Wrap sequence with tqdm progress bar when available and requested."""

    if desc is None or _tqdm is None:
        return items
    return _tqdm(items, desc=desc, unit=unit)


def _collect_labels_and_scores(
    *,
    records: Sequence[ScoredTextRecord],
    detector_scores: Sequence[float],
    dataset_id: str,
    source_splits: Sequence[str],
    partition: Literal["train", "test"],
) -> ScopeScores:
    """Collect labels and scores for one filtered subset."""

    source_split_set = set(source_splits)
    labels: list[str] = []
    scores: list[float] = []
    dropped_nonfinite_count = 0

    for record, score in zip(records, detector_scores, strict=True):
        if record.dataset_id != dataset_id:
            continue
        if record.source_split not in source_split_set:
            continue
        if record.partition != partition:
            continue
        normalized_score = float(score)
        if not math.isfinite(normalized_score):
            dropped_nonfinite_count += 1
            continue
        labels.append(record.label)
        scores.append(normalized_score)

    if not labels:
        if dropped_nonfinite_count > 0:
            raise ValueError(
                f"All selected scores were non-finite for dataset_id={dataset_id!r}, "
                f"source_splits={tuple(source_splits)!r}, partition={partition!r}. "
                f"non_finite_count={dropped_nonfinite_count}."
            )
        raise ValueError(
            f"No records selected for dataset_id={dataset_id!r}, source_splits={tuple(source_splits)!r}, "
            f"partition={partition!r}."
        )

    return ScopeScores(
        labels=labels,
        scores=scores,
        dropped_nonfinite_count=dropped_nonfinite_count,
    )


def _maybe_probability_metrics(
    *,
    detector_id: str,
    labels: Sequence[str],
    scores: Sequence[float],
) -> ProbabilityMetrics | None:
    """Compute probability metrics only for detectors with probability-like outputs."""

    if detector_id not in PROBABILITY_DETECTOR_IDS:
        return None
    return compute_probability_metrics(labels=labels, probabilities=scores)


def _build_result_row(
    *,
    scope_type: Literal["global", "local"],
    scope_id: str,
    model_run: ModelRunSpec,
    threshold: float,
    threshold_objective: str,
    threshold_objective_value: float,
    train_labels: Sequence[str],
    test_labels: Sequence[str],
    train_metrics: ClassificationMetrics,
    test_metrics: ClassificationMetrics,
    probability_metrics: ProbabilityMetrics | None,
    source_splits: Sequence[str],
    nonfinite_dropped_train: int,
    nonfinite_dropped_test: int,
) -> dict[str, Any]:
    """Serialize one experiment evaluation row."""

    return {
        "scope_type": scope_type,
        "scope_id": scope_id,
        "source_splits": list(source_splits),
        "detector_run_id": model_run.run_id,
        "detector_id": model_run.detector_id,
        "threshold": threshold,
        "threshold_objective": threshold_objective,
        "threshold_objective_value": threshold_objective_value,
        "train_count": len(train_labels),
        "test_count": len(test_labels),
        "train_counts_by_label": _count_by_label(train_labels),
        "test_counts_by_label": _count_by_label(test_labels),
        "nonfinite_scores_dropped_train": nonfinite_dropped_train,
        "nonfinite_scores_dropped_test": nonfinite_dropped_test,
        "train_metrics": _classification_to_dict(train_metrics),
        "test_metrics": _classification_to_dict(test_metrics),
        "probability_metrics_test": _probability_to_dict(probability_metrics),
    }


def _count_by_label(labels: Sequence[str]) -> dict[str, int]:
    """Count `human` and `ai` labels for one subset."""

    counts = {"human": 0, "ai": 0}
    for label in labels:
        if label not in counts:
            raise ValueError(f"Unsupported label value: {label!r}")
        counts[label] += 1
    return counts


def _classification_to_dict(metrics: ClassificationMetrics) -> dict[str, Any]:
    """Convert classification metrics dataclass to plain dictionary."""

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


def _probability_to_dict(metrics: ProbabilityMetrics | None) -> dict[str, float] | None:
    """Convert probability metrics dataclass to plain dictionary."""

    if metrics is None:
        return None
    return {
        "brier_score": metrics.brier_score,
        "log_loss": metrics.log_loss,
    }


def _build_run_id() -> str:
    """Build unique immutable run id."""

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    suffix = uuid.uuid4().hex[:8]
    return f"{timestamp}_{suffix}"


def _make_run_dir(*, output_root: Path, run_id: str) -> Path:
    """Create immutable run directory under output root."""

    run_dir = output_root / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    """Write deterministic JSON payload."""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    """Write JSONL rows with deterministic formatting."""

    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [json.dumps(dict(row), sort_keys=True) for row in rows]
    payload = "\n".join(lines)
    if payload:
        payload += "\n"
    path.write_text(payload, encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
