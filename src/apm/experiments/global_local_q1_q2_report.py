"""Generate Q1/Q2 analysis artifacts for global/local detector experiment runs."""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import matplotlib
import numpy as np
import pandas as pd
import seaborn as sns

from apm.metrics.classification import compute_classification_metrics

matplotlib.use("Agg")
import matplotlib.pyplot as plt


@dataclass(frozen=True, slots=True)
class GlobalLocalQ1Q2ReportRequest:
    """Request payload for Q1/Q2 report generation."""

    project_root: Path
    run_dir: Path
    output_dir: Path
    dataset_id: str
    global_scope_id: str
    local_scope_prefix: str


@dataclass(frozen=True, slots=True)
class GlobalLocalQ1Q2ReportResult:
    """Output paths generated for Q1/Q2 analysis."""

    output_dir: Path
    transfer_csv: Path
    local_vs_global_csv: Path
    train_eval_gap_csv: Path
    detector_ranking_csv: Path
    q1_markdown: Path
    q2_markdown: Path
    findings_markdown: Path


def run_global_local_q1_q2_report(request: GlobalLocalQ1Q2ReportRequest) -> GlobalLocalQ1Q2ReportResult:
    """Generate Q1/Q2 plots and markdown tables from one finished run."""

    _validate_request(request)
    request.output_dir.mkdir(parents=True, exist_ok=True)

    global_rows = _read_jsonl(request.run_dir / "global_results.jsonl")
    local_rows = _read_jsonl(request.run_dir / "local_results.jsonl")
    raw_rows = _read_jsonl(request.run_dir / "raw_scores.jsonl")

    global_frame = _flatten_scope_rows(rows=global_rows, strategy="global")
    local_frame = _flatten_scope_rows(rows=local_rows, strategy="local")

    selected_global = global_frame[global_frame["scope_id"] == request.global_scope_id].copy()
    selected_local = local_frame[local_frame["scope_id"].str.startswith(request.local_scope_prefix)].copy()
    if selected_global.empty:
        raise ValueError(f"No global rows found for scope_id={request.global_scope_id!r}.")
    if selected_local.empty:
        raise ValueError(f"No local rows found for prefix={request.local_scope_prefix!r}.")

    detector_ids = sorted(
        set(str(value) for value in selected_global["detector_run_id"])
        & set(str(value) for value in selected_local["detector_run_id"])
    )
    if not detector_ids:
        raise ValueError("No detector overlap between selected global/local rows.")

    local_splits = sorted(set(str(value) for value in selected_local["source_split"]))
    if not local_splits:
        raise ValueError("No local source splits resolved.")

    grouped_scores = _group_raw_scores(
        raw_rows=raw_rows,
        dataset_id=request.dataset_id,
        detector_ids=detector_ids,
        source_splits=local_splits,
    )

    transfer_frame = _build_threshold_transfer_frame(
        detector_ids=detector_ids,
        local_splits=local_splits,
        selected_global=selected_global,
        selected_local=selected_local,
        grouped_scores=grouped_scores,
    )
    local_vs_global = _build_local_vs_global_corresponding_delta(transfer_frame=transfer_frame)
    corresponding_vs_others = _build_corresponding_vs_others(transfer_frame=transfer_frame)
    train_eval_gap = _build_train_eval_gap_table(
        selected_global=selected_global,
        selected_local=selected_local,
    )

    corresponding = transfer_frame[
        (transfer_frame["source_type"] == "local")
        & (transfer_frame["source_split"] == transfer_frame["target_split"])
    ].copy()
    global_local_mean = _build_global_local_mean_comparison(local_vs_global=local_vs_global)
    threshold_profile = _build_threshold_profile_summary(
        local_vs_global=local_vs_global,
        transfer_frame=transfer_frame,
    )
    detector_ranking = _build_detector_ranking(corresponding=corresponding)
    detector_error_profile = _build_detector_error_profile(corresponding=corresponding)
    global_error_by_split = _build_global_threshold_error_by_split(transfer_frame=transfer_frame)
    global_error_all_data = _build_global_threshold_error_all_data(global_error_by_split=global_error_by_split)
    split_difficulty = _build_split_difficulty(corresponding=corresponding)

    transfer_csv = request.output_dir / "q1_threshold_transfer_test_metrics.csv"
    local_vs_global_csv = request.output_dir / "q1_local_vs_global_corresponding_delta.csv"
    train_eval_gap_csv = request.output_dir / "q1_train_to_eval_gap.csv"
    corresponding_vs_others_csv = request.output_dir / "q1_corresponding_vs_other_splits.csv"
    detector_ranking_csv = request.output_dir / "q2_detector_ranking.csv"
    detector_error_profile_csv = request.output_dir / "q2_detector_error_profile.csv"
    global_error_by_split_csv = request.output_dir / "q2_global_threshold_error_by_split.csv"
    global_error_all_data_csv = request.output_dir / "q2_global_threshold_error_all_data.csv"
    split_difficulty_csv = request.output_dir / "q2_split_difficulty.csv"
    global_local_mean_csv = request.output_dir / "q1_global_vs_local_mean_test_ba.csv"
    threshold_profile_csv = request.output_dir / "q1_threshold_profile_by_detector.csv"

    transfer_frame.to_csv(transfer_csv, index=False)
    local_vs_global.to_csv(local_vs_global_csv, index=False)
    train_eval_gap.to_csv(train_eval_gap_csv, index=False)
    corresponding_vs_others.to_csv(corresponding_vs_others_csv, index=False)
    detector_ranking.to_csv(detector_ranking_csv, index=False)
    detector_error_profile.to_csv(detector_error_profile_csv, index=False)
    global_error_by_split.to_csv(global_error_by_split_csv, index=False)
    global_error_all_data.to_csv(global_error_all_data_csv, index=False)
    split_difficulty.to_csv(split_difficulty_csv, index=False)
    global_local_mean.to_csv(global_local_mean_csv, index=False)
    threshold_profile.to_csv(threshold_profile_csv, index=False)

    _plot_local_vs_global_delta_heatmap(
        local_vs_global=local_vs_global,
        output_path=request.output_dir / "q1_local_minus_global_ba_heatmap.png",
    )
    _plot_transfer_heatmaps_by_detector(
        transfer_frame=transfer_frame,
        output_dir=request.output_dir,
    )
    _plot_detector_split_balanced_accuracy(
        corresponding=corresponding,
        output_path=request.output_dir / "q2_detector_balanced_accuracy_by_split.png",
    )
    _plot_global_threshold_error_dumbbell_all_data(
        error_all_data=global_error_all_data,
        output_path=request.output_dir / "q2_fp_fn_rate_by_detector.png",
    )
    _plot_global_threshold_error_dumbbell_all_data(
        error_all_data=global_error_all_data,
        output_path=request.output_dir / "06_error_profile_fp_fn_rates.png",
    )
    _plot_global_threshold_error_dumbbell_by_split(
        error_by_split=global_error_by_split,
        output_path=request.output_dir / "q2_global_threshold_error_dumbbell_by_split.png",
    )
    _plot_global_local_mean_bars(
        global_local_mean=global_local_mean,
        output_path=request.output_dir / "q1_global_vs_local_mean_test_ba_by_detector.png",
    )
    _plot_relative_delta_matrices_by_detector(
        transfer_frame=transfer_frame,
        output_dir=request.output_dir,
    )
    _plot_threshold_profile_bars(
        threshold_profile=threshold_profile,
        output_path=request.output_dir / "q1_threshold_profile_by_detector.png",
    )

    q1_markdown = request.output_dir / "Q1_threshold_tuning_analysis.md"
    q2_markdown = request.output_dir / "Q2_detector_comparison_analysis.md"
    findings_markdown = request.output_dir / "KEY_FINDINGS.md"

    q1_markdown.write_text(
        _build_q1_markdown(
            train_eval_gap=train_eval_gap,
            local_vs_global=local_vs_global,
            corresponding_vs_others=corresponding_vs_others,
        ),
        encoding="utf-8",
    )
    q2_markdown.write_text(
        _build_q2_markdown(
            detector_ranking=detector_ranking,
            detector_error_profile=detector_error_profile,
            split_difficulty=split_difficulty,
        ),
        encoding="utf-8",
    )
    findings_markdown.write_text(
        _build_findings_markdown(
            local_vs_global=local_vs_global,
            detector_ranking=detector_ranking,
            split_difficulty=split_difficulty,
            detector_error_profile=detector_error_profile,
        ),
        encoding="utf-8",
    )

    return GlobalLocalQ1Q2ReportResult(
        output_dir=request.output_dir,
        transfer_csv=transfer_csv,
        local_vs_global_csv=local_vs_global_csv,
        train_eval_gap_csv=train_eval_gap_csv,
        detector_ranking_csv=detector_ranking_csv,
        q1_markdown=q1_markdown,
        q2_markdown=q2_markdown,
        findings_markdown=findings_markdown,
    )


def build_request_from_args(args: argparse.Namespace) -> GlobalLocalQ1Q2ReportRequest:
    """Build typed request from CLI args."""

    project_root = args.project_root.resolve()
    runs_root = (project_root / args.runs_root).resolve()

    if args.run_dir is not None:
        run_dir_candidate = Path(args.run_dir)
        run_dir = run_dir_candidate if run_dir_candidate.is_absolute() else (project_root / run_dir_candidate)
        run_dir = run_dir.resolve()
    elif args.run_id:
        run_dir = (runs_root / args.run_id).resolve()
    else:
        run_dir = _resolve_latest_run_dir(runs_root=runs_root)

    output_dir_candidate = Path(args.output_dir) if args.output_dir else (run_dir / "analysis_q1_q2")
    output_dir = output_dir_candidate if output_dir_candidate.is_absolute() else (project_root / output_dir_candidate)
    output_dir = output_dir.resolve()

    return GlobalLocalQ1Q2ReportRequest(
        project_root=project_root,
        run_dir=run_dir,
        output_dir=output_dir,
        dataset_id=str(args.dataset_id),
        global_scope_id=str(args.global_scope_id),
        local_scope_prefix=str(args.local_scope_prefix),
    )


def build_arg_parser() -> argparse.ArgumentParser:
    """Build CLI parser for Q1/Q2 report generation."""

    parser = argparse.ArgumentParser(description="Generate Q1/Q2 analysis plots and markdown tables.")
    parser.add_argument("--project-root", type=Path, default=Path.cwd(), help="Repository root directory.")
    parser.add_argument(
        "--runs-root",
        type=Path,
        default=Path("runs/global_local_experiments"),
        help="Runs root used when --run-id / latest resolution is requested.",
    )
    parser.add_argument("--run-id", type=str, default="", help="Run id under runs/global_local_experiments.")
    parser.add_argument("--run-dir", type=Path, default=None, help="Explicit run directory path.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory. Defaults to <run_dir>/analysis_q1_q2.",
    )
    parser.add_argument("--dataset-id", type=str, default="hc3", help="Dataset id to analyze from raw scores.")
    parser.add_argument("--global-scope-id", type=str, default="hc3", help="Global scope id baseline for Q1 deltas.")
    parser.add_argument(
        "--local-scope-prefix",
        type=str,
        default="hc3:",
        help="Local scope prefix used as split-specific tuning scopes.",
    )
    return parser


def main() -> int:
    """CLI entrypoint."""

    parser = build_arg_parser()
    args = parser.parse_args()
    request = build_request_from_args(args)
    result = run_global_local_q1_q2_report(request)
    print(
        json.dumps(
            {
                "run_dir": str(request.run_dir),
                "output_dir": str(result.output_dir),
                "transfer_csv": str(result.transfer_csv),
                "local_vs_global_csv": str(result.local_vs_global_csv),
                "train_eval_gap_csv": str(result.train_eval_gap_csv),
                "detector_ranking_csv": str(result.detector_ranking_csv),
                "q1_markdown": str(result.q1_markdown),
                "q2_markdown": str(result.q2_markdown),
                "findings_markdown": str(result.findings_markdown),
            },
            indent=2,
        )
    )
    return 0


def _validate_request(request: GlobalLocalQ1Q2ReportRequest) -> None:
    """Validate request invariants."""

    if not request.dataset_id.strip():
        raise ValueError("dataset_id must be non-empty.")
    if not request.global_scope_id.strip():
        raise ValueError("global_scope_id must be non-empty.")
    if not request.local_scope_prefix.strip():
        raise ValueError("local_scope_prefix must be non-empty.")
    if not request.run_dir.exists():
        raise FileNotFoundError(f"Missing run directory: {request.run_dir}")
    for required in ("global_results.jsonl", "local_results.jsonl", "raw_scores.jsonl"):
        path = request.run_dir / required
        if not path.exists():
            raise FileNotFoundError(f"Missing required file: {path}")


def _resolve_latest_run_dir(runs_root: Path) -> Path:
    """Resolve most recently modified run directory containing required result files."""

    if not runs_root.exists():
        raise FileNotFoundError(f"Missing runs root: {runs_root}")
    candidates: list[Path] = []
    for child in runs_root.iterdir():
        if not child.is_dir():
            continue
        if all((child / filename).exists() for filename in ("global_results.jsonl", "local_results.jsonl", "raw_scores.jsonl")):
            candidates.append(child)
    if not candidates:
        raise ValueError(f"No completed run directories found under: {runs_root}")
    candidates.sort(key=lambda path: path.stat().st_mtime, reverse=True)
    return candidates[0]


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    """Read JSONL into dictionaries."""

    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        payload = json.loads(line)
        if not isinstance(payload, dict):
            raise ValueError(f"Expected object row in {path}, got: {payload!r}")
        rows.append(payload)
    return rows


def _flatten_scope_rows(*, rows: Sequence[Mapping[str, Any]], strategy: str) -> pd.DataFrame:
    """Flatten nested run rows into one record per detector scope."""

    flattened: list[dict[str, Any]] = []
    for row in rows:
        scope_id = _read_str(row, "scope_id")
        detector_id = _read_str(row, "detector_id")
        detector_run_id = _read_str(row, "detector_run_id")
        source_splits = row.get("source_splits")
        if not isinstance(source_splits, Sequence):
            raise ValueError(f"source_splits must be a sequence in row: {row!r}")
        normalized_splits = [str(value) for value in source_splits]
        source_split = normalized_splits[0] if len(normalized_splits) == 1 else ""

        train_metrics = _read_mapping(row, "train_metrics")
        test_metrics = _read_mapping(row, "test_metrics")

        train_conf = _read_mapping(train_metrics, "confusion_matrix")
        test_conf = _read_mapping(test_metrics, "confusion_matrix")

        flattened.append(
            {
                "strategy": strategy,
                "scope_id": scope_id,
                "detector_id": detector_id,
                "detector_run_id": detector_run_id,
                "source_splits": normalized_splits,
                "source_split": source_split,
                "threshold": _read_float(row, "threshold"),
                "train_balanced_accuracy": _read_float(train_metrics, "balanced_accuracy"),
                "test_balanced_accuracy": _read_float(test_metrics, "balanced_accuracy"),
                "train_f1": _read_float(train_metrics, "f1"),
                "test_f1": _read_float(test_metrics, "f1"),
                "test_precision": _read_float(test_metrics, "precision"),
                "test_recall": _read_float(test_metrics, "recall"),
                "test_accuracy": _read_float(test_metrics, "accuracy"),
                "test_fp_rate": _compute_fp_rate(confusion=test_conf),
                "test_fn_rate": _compute_fn_rate(confusion=test_conf),
                "train_fp_rate": _compute_fp_rate(confusion=train_conf),
                "train_fn_rate": _compute_fn_rate(confusion=train_conf),
                "train_count": int(_read_float(row, "train_count")),
                "test_count": int(_read_float(row, "test_count")),
            }
        )

    frame = pd.DataFrame(flattened)
    if frame.empty:
        raise ValueError(f"No rows to flatten for strategy={strategy!r}.")
    return frame


def _group_raw_scores(
    *,
    raw_rows: Sequence[Mapping[str, Any]],
    dataset_id: str,
    detector_ids: Sequence[str],
    source_splits: Sequence[str],
) -> dict[tuple[str, str, str], tuple[list[str], list[float]]]:
    """Group raw scores by detector/split/partition."""

    detector_set = set(detector_ids)
    split_set = set(source_splits)
    labels_by_key: dict[tuple[str, str, str], list[str]] = {}
    scores_by_key: dict[tuple[str, str, str], list[float]] = {}

    for row in raw_rows:
        row_dataset = _read_str(row, "dataset_id")
        if row_dataset != dataset_id:
            continue
        detector_run_id = _read_str(row, "detector_run_id")
        if detector_run_id not in detector_set:
            continue
        source_split = _read_str(row, "source_split")
        if source_split not in split_set:
            continue
        partition = _read_str(row, "partition")
        if partition not in {"train", "test"}:
            continue
        label = _read_str(row, "label")
        if label not in {"human", "ai"}:
            continue
        score = _read_float(row, "score")
        if not math.isfinite(score):
            continue
        key = (detector_run_id, source_split, partition)
        labels_by_key.setdefault(key, []).append(label)
        scores_by_key.setdefault(key, []).append(score)

    grouped: dict[tuple[str, str, str], tuple[list[str], list[float]]] = {}
    for detector_id in detector_ids:
        for split in source_splits:
            for partition in ("train", "test"):
                key = (detector_id, split, partition)
                labels = labels_by_key.get(key, [])
                scores = scores_by_key.get(key, [])
                if not labels or not scores:
                    raise ValueError(f"Missing raw scores for key={key!r}")
                grouped[key] = (labels, scores)
    return grouped


def _build_threshold_transfer_frame(
    *,
    detector_ids: Sequence[str],
    local_splits: Sequence[str],
    selected_global: pd.DataFrame,
    selected_local: pd.DataFrame,
    grouped_scores: Mapping[tuple[str, str, str], tuple[list[str], list[float]]],
) -> pd.DataFrame:
    """Evaluate global/local thresholds on every target split test partition."""

    rows: list[dict[str, Any]] = []
    for detector_id in detector_ids:
        global_threshold_series = selected_global[selected_global["detector_run_id"] == detector_id]["threshold"]
        if global_threshold_series.empty:
            raise ValueError(f"Missing global threshold for detector={detector_id!r}")
        global_threshold = float(global_threshold_series.iloc[0])

        local_thresholds = selected_local[selected_local["detector_run_id"] == detector_id][["source_split", "threshold"]]
        local_threshold_map = {str(split): float(threshold) for split, threshold in local_thresholds.itertuples(index=False)}
        missing_local_splits = set(local_splits) - set(local_threshold_map.keys())
        if missing_local_splits:
            raise ValueError(f"Missing local thresholds for detector={detector_id!r}, splits={sorted(missing_local_splits)!r}")

        source_thresholds: list[tuple[str, str, float]] = [("global", "__global__", global_threshold)]
        for split in local_splits:
            source_thresholds.append(("local", split, local_threshold_map[split]))

        for source_type, source_split, threshold in source_thresholds:
            for target_split in local_splits:
                labels, scores = grouped_scores[(detector_id, target_split, "test")]
                metrics = compute_classification_metrics(labels=labels, scores=scores, threshold=threshold)
                rows.append(
                    {
                        "detector_run_id": detector_id,
                        "source_type": source_type,
                        "source_split": source_split,
                        "threshold": threshold,
                        "target_split": target_split,
                        "target_count": len(labels),
                        "accuracy": metrics.accuracy,
                        "balanced_accuracy": metrics.balanced_accuracy,
                        "precision": metrics.precision,
                        "recall": metrics.recall,
                        "f1": metrics.f1,
                        "fp_rate": _safe_divide(metrics.confusion.false_positive, metrics.confusion.false_positive + metrics.confusion.true_negative),
                        "fn_rate": _safe_divide(metrics.confusion.false_negative, metrics.confusion.false_negative + metrics.confusion.true_positive),
                        "true_positive": metrics.confusion.true_positive,
                        "false_positive": metrics.confusion.false_positive,
                        "true_negative": metrics.confusion.true_negative,
                        "false_negative": metrics.confusion.false_negative,
                    }
                )
    return pd.DataFrame(rows)


def _build_local_vs_global_corresponding_delta(*, transfer_frame: pd.DataFrame) -> pd.DataFrame:
    """Compute BA delta for local-threshold vs global-threshold on each corresponding split."""

    local_matching = transfer_frame[
        (transfer_frame["source_type"] == "local")
        & (transfer_frame["source_split"] == transfer_frame["target_split"])
    ][["detector_run_id", "target_split", "threshold", "balanced_accuracy", "f1", "recall", "precision", "fp_rate"]].copy()
    local_matching = local_matching.rename(
        columns={
            "threshold": "local_threshold",
            "balanced_accuracy": "local_balanced_accuracy",
            "f1": "local_f1",
            "recall": "local_recall",
            "precision": "local_precision",
            "fp_rate": "local_fp_rate",
        }
    )

    global_rows = transfer_frame[transfer_frame["source_type"] == "global"][
        ["detector_run_id", "target_split", "threshold", "balanced_accuracy", "f1", "recall", "precision", "fp_rate"]
    ].copy()
    global_rows = global_rows.rename(
        columns={
            "threshold": "global_threshold",
            "balanced_accuracy": "global_balanced_accuracy",
            "f1": "global_f1",
            "recall": "global_recall",
            "precision": "global_precision",
            "fp_rate": "global_fp_rate",
        }
    )

    merged = local_matching.merge(
        global_rows,
        on=["detector_run_id", "target_split"],
        how="inner",
        validate="one_to_one",
    )
    merged["delta_local_minus_global_ba"] = merged["local_balanced_accuracy"] - merged["global_balanced_accuracy"]
    merged["delta_local_minus_global_f1"] = merged["local_f1"] - merged["global_f1"]
    merged["delta_local_minus_global_recall"] = merged["local_recall"] - merged["global_recall"]
    merged["delta_local_minus_global_fp_rate"] = merged["local_fp_rate"] - merged["global_fp_rate"]
    return merged.sort_values(by=["detector_run_id", "target_split"], kind="mergesort").reset_index(drop=True)


def _build_corresponding_vs_others(*, transfer_frame: pd.DataFrame) -> pd.DataFrame:
    """Compare split-tuned threshold on corresponding split vs average of other splits."""

    local_rows = transfer_frame[transfer_frame["source_type"] == "local"].copy()
    records: list[dict[str, Any]] = []
    unique_pairs = local_rows[["detector_run_id", "source_split"]].drop_duplicates()
    for detector_id, source_split in unique_pairs.itertuples(index=False):
        subset = local_rows[
            (local_rows["detector_run_id"] == detector_id)
            & (local_rows["source_split"] == source_split)
        ].copy()
        corresponding = subset[subset["target_split"] == source_split]
        others = subset[subset["target_split"] != source_split]
        if corresponding.empty or others.empty:
            raise ValueError(f"Insufficient transfer rows for detector={detector_id!r}, source_split={source_split!r}")
        corresponding_ba = float(corresponding["balanced_accuracy"].iloc[0])
        others_mean_ba = float(others["balanced_accuracy"].mean())
        threshold = float(corresponding["threshold"].iloc[0])
        records.append(
            {
                "detector_run_id": detector_id,
                "source_split": source_split,
                "threshold": threshold,
                "corresponding_balanced_accuracy": corresponding_ba,
                "other_splits_mean_balanced_accuracy": others_mean_ba,
                "delta_corresponding_minus_others": corresponding_ba - others_mean_ba,
            }
        )
    return pd.DataFrame(records).sort_values(by=["detector_run_id", "source_split"], kind="mergesort").reset_index(drop=True)


def _build_train_eval_gap_table(*, selected_global: pd.DataFrame, selected_local: pd.DataFrame) -> pd.DataFrame:
    """Build train-to-evaluation gap table for selected global and local scopes."""

    table = pd.concat([selected_global, selected_local], axis=0, ignore_index=True)
    table["train_to_test_gap_ba"] = table["test_balanced_accuracy"] - table["train_balanced_accuracy"]
    return table[
        [
            "strategy",
            "scope_id",
            "detector_run_id",
            "source_split",
            "threshold",
            "train_balanced_accuracy",
            "test_balanced_accuracy",
            "train_to_test_gap_ba",
        ]
    ].sort_values(by=["detector_run_id", "strategy", "scope_id"], kind="mergesort")


def _build_detector_ranking(*, corresponding: pd.DataFrame) -> pd.DataFrame:
    """Rank detectors by average corresponding-split performance."""

    ranking = (
        corresponding.groupby("detector_run_id", as_index=False)
        .agg(
            mean_balanced_accuracy=("balanced_accuracy", "mean"),
            std_balanced_accuracy=("balanced_accuracy", "std"),
            min_balanced_accuracy=("balanced_accuracy", "min"),
            max_balanced_accuracy=("balanced_accuracy", "max"),
            mean_f1=("f1", "mean"),
            mean_recall=("recall", "mean"),
            mean_precision=("precision", "mean"),
            mean_fp_rate=("fp_rate", "mean"),
            mean_fn_rate=("fn_rate", "mean"),
        )
        .sort_values(by=["mean_balanced_accuracy", "mean_f1"], ascending=False, kind="mergesort")
        .reset_index(drop=True)
    )
    ranking["rank"] = np.arange(1, len(ranking) + 1, dtype=int)
    return ranking[
        [
            "rank",
            "detector_run_id",
            "mean_balanced_accuracy",
            "std_balanced_accuracy",
            "min_balanced_accuracy",
            "max_balanced_accuracy",
            "mean_f1",
            "mean_recall",
            "mean_precision",
            "mean_fp_rate",
            "mean_fn_rate",
        ]
    ]


def _build_global_local_mean_comparison(*, local_vs_global: pd.DataFrame) -> pd.DataFrame:
    """Compute per-detector mean BA for global and corresponding local thresholds."""

    summary = (
        local_vs_global.groupby("detector_run_id", as_index=False)
        .agg(
            mean_global_test_ba=("global_balanced_accuracy", "mean"),
            mean_local_test_ba=("local_balanced_accuracy", "mean"),
            split_count=("target_split", "size"),
        )
        .sort_values(by="detector_run_id", kind="mergesort")
        .reset_index(drop=True)
    )
    summary["mean_delta_local_minus_global"] = summary["mean_local_test_ba"] - summary["mean_global_test_ba"]
    return summary


def _build_threshold_profile_summary(
    *,
    local_vs_global: pd.DataFrame,
    transfer_frame: pd.DataFrame,
) -> pd.DataFrame:
    """Build per-detector score profile summary for visualization."""

    mean_scores = (
        local_vs_global.groupby("detector_run_id", as_index=False)
        .agg(
            mean_global_test_ba=("global_balanced_accuracy", "mean"),
            mean_local_test_ba=("local_balanced_accuracy", "mean"),
        )
        .sort_values(by="detector_run_id", kind="mergesort")
        .reset_index(drop=True)
    )

    local_transfer_rows = transfer_frame[transfer_frame["source_type"] == "local"].copy()
    local_transfer_rows = local_transfer_rows[
        ["detector_run_id", "source_split", "target_split", "balanced_accuracy"]
    ].sort_values(by=["detector_run_id", "source_split", "target_split"], kind="mergesort")

    extrema_rows: list[dict[str, Any]] = []
    for detector_id in sorted(set(str(value) for value in mean_scores["detector_run_id"].tolist())):
        subset = local_transfer_rows[local_transfer_rows["detector_run_id"] == detector_id].copy()
        if subset.empty:
            raise ValueError(f"No local threshold-source scores for detector={detector_id!r}.")
        worst_idx = int(subset["balanced_accuracy"].idxmin())
        best_idx = int(subset["balanced_accuracy"].idxmax())
        worst_row = subset.loc[worst_idx]
        best_row = subset.loc[best_idx]
        extrema_rows.append(
            {
                "detector_run_id": detector_id,
                "worst_threshold_source_split": str(worst_row["source_split"]),
                "worst_threshold_target_split": str(worst_row["target_split"]),
                "worst_threshold_score_ba": float(worst_row["balanced_accuracy"]),
                "best_threshold_source_split": str(best_row["source_split"]),
                "best_threshold_target_split": str(best_row["target_split"]),
                "best_threshold_score_ba": float(best_row["balanced_accuracy"]),
            }
        )
    extrema = pd.DataFrame(extrema_rows)
    if extrema.empty:
        raise ValueError("No threshold extrema rows built.")

    output = mean_scores.merge(extrema, on="detector_run_id", how="inner", validate="one_to_one")
    return output.sort_values(by="detector_run_id", kind="mergesort").reset_index(drop=True)


def _build_detector_error_profile(*, corresponding: pd.DataFrame) -> pd.DataFrame:
    """Aggregate detector FP/FN rates on corresponding local-split evaluations."""

    profile = (
        corresponding.groupby("detector_run_id", as_index=False)
        .agg(
            mean_fp_rate=("fp_rate", "mean"),
            mean_fn_rate=("fn_rate", "mean"),
            mean_precision=("precision", "mean"),
            mean_recall=("recall", "mean"),
        )
        .sort_values(by="detector_run_id", kind="mergesort")
        .reset_index(drop=True)
    )
    profile["fp_minus_fn_rate"] = profile["mean_fp_rate"] - profile["mean_fn_rate"]
    return profile


def _build_global_threshold_error_by_split(*, transfer_frame: pd.DataFrame) -> pd.DataFrame:
    """Build per-split FP/FN error profile using only global threshold rows."""

    global_rows = transfer_frame[transfer_frame["source_type"] == "global"].copy()
    if global_rows.empty:
        raise ValueError("No global-threshold transfer rows available.")

    frame = global_rows[
        [
            "detector_run_id",
            "target_split",
            "true_positive",
            "false_positive",
            "true_negative",
            "false_negative",
        ]
    ].copy()
    frame["detector_run_id"] = frame["detector_run_id"].astype(str)
    frame["target_split"] = frame["target_split"].astype(str)
    for column in ("true_positive", "false_positive", "true_negative", "false_negative"):
        frame[column] = frame[column].astype(int)

    frame["human_count"] = frame["true_negative"] + frame["false_positive"]
    frame["ai_count"] = frame["true_positive"] + frame["false_negative"]
    frame["fp_rate"] = frame.apply(
        lambda row: _safe_divide(float(row["false_positive"]), float(row["human_count"])),
        axis=1,
    )
    frame["fn_rate"] = frame.apply(
        lambda row: _safe_divide(float(row["false_negative"]), float(row["ai_count"])),
        axis=1,
    )
    frame["fp_minus_fn_rate"] = frame["fp_rate"] - frame["fn_rate"]
    return frame.sort_values(by=["target_split", "detector_run_id"], kind="mergesort").reset_index(drop=True)


def _build_global_threshold_error_all_data(*, global_error_by_split: pd.DataFrame) -> pd.DataFrame:
    """Aggregate global-threshold confusion counts over all target splits per detector."""

    grouped = (
        global_error_by_split.groupby("detector_run_id", as_index=False)
        .agg(
            true_positive=("true_positive", "sum"),
            false_positive=("false_positive", "sum"),
            true_negative=("true_negative", "sum"),
            false_negative=("false_negative", "sum"),
            human_count=("human_count", "sum"),
            ai_count=("ai_count", "sum"),
        )
        .sort_values(by="detector_run_id", kind="mergesort")
        .reset_index(drop=True)
    )
    grouped["fp_rate"] = grouped.apply(
        lambda row: _safe_divide(float(row["false_positive"]), float(row["human_count"])),
        axis=1,
    )
    grouped["fn_rate"] = grouped.apply(
        lambda row: _safe_divide(float(row["false_negative"]), float(row["ai_count"])),
        axis=1,
    )
    grouped["fp_minus_fn_rate"] = grouped["fp_rate"] - grouped["fn_rate"]
    return grouped


def _build_split_difficulty(*, corresponding: pd.DataFrame) -> pd.DataFrame:
    """Estimate split-level difficulty by averaging detector metrics."""

    difficulty = (
        corresponding.groupby("target_split", as_index=False)
        .agg(
            mean_balanced_accuracy=("balanced_accuracy", "mean"),
            std_balanced_accuracy=("balanced_accuracy", "std"),
            mean_fp_rate=("fp_rate", "mean"),
            mean_fn_rate=("fn_rate", "mean"),
        )
        .sort_values(by="mean_balanced_accuracy", ascending=True, kind="mergesort")
        .reset_index(drop=True)
    )
    return difficulty


def _plot_local_vs_global_delta_heatmap(*, local_vs_global: pd.DataFrame, output_path: Path) -> None:
    """Plot detector-by-split heatmap of local minus global BA deltas."""

    pivot = local_vs_global.pivot_table(
        index="target_split",
        columns="detector_run_id",
        values="delta_local_minus_global_ba",
        aggfunc="mean",
    )
    _plot_heatmap(
        matrix=pivot.to_numpy(dtype=float),
        x_labels=[str(value) for value in pivot.columns.tolist()],
        y_labels=[str(value) for value in pivot.index.tolist()],
        title="Q1: Local - Global Balanced Accuracy (Corresponding Splits)",
        output_path=output_path,
        center_zero=True,
    )


def _plot_transfer_heatmaps_by_detector(*, transfer_frame: pd.DataFrame, output_dir: Path) -> None:
    """Plot source-threshold to target-split BA transfer heatmap per detector."""

    for detector_id in sorted(set(str(value) for value in transfer_frame["detector_run_id"])):
        subset = transfer_frame[transfer_frame["detector_run_id"] == detector_id].copy()
        subset["source_label"] = subset["source_split"].map(lambda value: "global_hc3" if value == "__global__" else str(value))
        pivot = subset.pivot_table(
            index="source_label",
            columns="target_split",
            values="balanced_accuracy",
            aggfunc="mean",
        )
        source_order = ["global_hc3"] + sorted(label for label in pivot.index.tolist() if label != "global_hc3")
        pivot = pivot.reindex(index=source_order)
        _plot_heatmap(
            matrix=pivot.to_numpy(dtype=float),
            x_labels=[str(value) for value in pivot.columns.tolist()],
            y_labels=[str(value) for value in pivot.index.tolist()],
            title=f"Q1 Transfer Matrix (BA): {detector_id}",
            output_path=output_dir / f"q1_transfer_matrix_{_slugify(detector_id)}.png",
            center_zero=False,
        )


def _display_split_name(split_name: str) -> str:
    """Return compact split label for figure axes."""

    value = str(split_name)
    return value[:-6] if value.endswith("_train") else value


def _to_train_axis_label(split_name: str) -> str:
    """Render split label with explicit _train suffix for threshold-source axes."""

    value = str(split_name)
    if value.endswith("_train"):
        return value
    if value.endswith("_test"):
        return f"{value[:-5]}_train"
    return f"{value}_train"


def _to_test_axis_label(split_name: str) -> str:
    """Render split label with explicit _test suffix for evaluated-set axes."""

    value = str(split_name)
    if value.endswith("_test"):
        return value
    if value.endswith("_train"):
        return f"{value[:-6]}_test"
    return f"{value}_test"


def _plot_detector_split_balanced_accuracy(*, corresponding: pd.DataFrame, output_path: Path) -> None:
    """Plot local corresponding balanced accuracy per split and detector (styled)."""

    plot_frame = corresponding.sort_values(by=["target_split", "detector_run_id"], kind="mergesort").copy()
    plot_frame["target_split"] = plot_frame["target_split"].astype(str)
    plot_frame["detector_run_id"] = plot_frame["detector_run_id"].astype(str)

    split_order = sorted(set(plot_frame["target_split"].tolist()))
    split_labels = [_display_split_name(value) for value in split_order]
    detector_order = sorted(set(plot_frame["detector_run_id"].tolist()))
    palette = dict(
        zip(
            detector_order,
            sns.color_palette("colorblind", n_colors=max(len(detector_order), 3))[: len(detector_order)],
            strict=True,
        )
    )

    fig, ax = plt.subplots(figsize=(max(9.2, 1.55 * len(split_order)), 5.4))
    sns.barplot(
        data=plot_frame,
        x="target_split",
        y="balanced_accuracy",
        hue="detector_run_id",
        order=split_order,
        hue_order=detector_order,
        palette=palette,
        edgecolor="#1A1A1A",
        linewidth=0.5,
        ax=ax,
    )

    for patch in ax.patches:
        value = patch.get_height()
        if not math.isfinite(value):
            continue
        ax.text(
            patch.get_x() + patch.get_width() / 2.0,
            value + 0.006,
            f"{value:.3f}",
            ha="center",
            va="bottom",
            fontsize=8.8,
            color="#1A1A1A",
            rotation=0,
        )

    max_score = float(plot_frame["balanced_accuracy"].max())
    upper_ylim = min(1.08, max(1.0, max_score + 0.08))
    ax.set_xticks(np.arange(len(split_order)))
    ax.set_xticklabels(split_labels, rotation=0)
    ax.set_ylabel("Balanced Accuracy")
    ax.set_xlabel("Evaluated Split")
    ax.set_title("Detector Balanced Accuracy by Split")
    ax.set_ylim(0.0, upper_ylim)
    ax.grid(axis="y", alpha=0.22)
    ax.legend(title="", frameon=True, loc="lower right", fontsize=10)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _plot_detector_error_bars(*, detector_error_profile: pd.DataFrame, output_path: Path) -> None:
    """Plot mean FP and FN rates by detector (styled)."""

    frame = detector_error_profile.copy()
    frame["detector_run_id"] = frame["detector_run_id"].astype(str)
    long_frame = pd.concat(
        [
            frame[["detector_run_id", "mean_fp_rate"]].rename(columns={"mean_fp_rate": "rate"}).assign(metric="FP"),
            frame[["detector_run_id", "mean_fn_rate"]].rename(columns={"mean_fn_rate": "rate"}).assign(metric="FN"),
        ],
        axis=0,
        ignore_index=True,
    )
    detector_order = frame["detector_run_id"].tolist()
    metric_order = ["FP", "FN"]
    palette = {"FP": "#E15759", "FN": "#59A14F"}

    fig, ax = plt.subplots(figsize=(max(8.0, 2.8 * len(detector_order)), 5.2))
    sns.barplot(
        data=long_frame,
        x="detector_run_id",
        y="rate",
        hue="metric",
        order=detector_order,
        hue_order=metric_order,
        palette=palette,
        edgecolor="#1A1A1A",
        linewidth=0.5,
        ax=ax,
    )

    for patch in ax.patches:
        value = patch.get_height()
        if not math.isfinite(value):
            continue
        ax.text(
            patch.get_x() + patch.get_width() / 2.0,
            value + 0.008,
            f"{value:.3f}",
            ha="center",
            va="bottom",
            fontsize=9.2,
            color="#1A1A1A",
        )

    ax.set_xticks(np.arange(len(detector_order)))
    ax.set_xticklabels(detector_order, rotation=0)
    ax.set_ylim(0.0, 1.0)
    ax.set_ylabel("Rate")
    ax.set_xlabel("Detector")
    ax.set_title("FP / FN Error Profile by Detector")
    ax.grid(axis="y", alpha=0.22)
    ax.legend(title="", frameon=True, loc="upper right")
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _plot_global_threshold_error_dumbbell_all_data(*, error_all_data: pd.DataFrame, output_path: Path) -> None:
    """Plot FP/FN dumbbell chart per detector on all test data (global threshold)."""

    frame = error_all_data.sort_values(by="detector_run_id", kind="mergesort").copy()
    detectors = frame["detector_run_id"].astype(str).tolist()
    y_positions = np.arange(len(detectors), dtype=float)
    fp_values = frame["fp_rate"].to_numpy(dtype=float)
    fn_values = frame["fn_rate"].to_numpy(dtype=float)

    fig, ax = plt.subplots(figsize=(11.4, max(3.6, 1.45 * len(detectors))))
    for idx, y_value in enumerate(y_positions):
        left = min(fp_values[idx], fn_values[idx])
        right = max(fp_values[idx], fn_values[idx])
        ax.hlines(y=y_value, xmin=left, xmax=right, color="#9A9A9A", linewidth=2.2, zorder=1)

    ax.scatter(fp_values, y_positions, s=90, c="#E15759", marker="o", edgecolors="#1A1A1A", linewidths=0.55, label="FP rate", zorder=3)
    ax.scatter(fn_values, y_positions, s=90, c="#4E79A7", marker="D", edgecolors="#1A1A1A", linewidths=0.55, label="FN rate", zorder=3)

    for idx, y_value in enumerate(y_positions):
        gap = float(fp_values[idx] - fn_values[idx])
        right = max(fp_values[idx], fn_values[idx])
        ax.text(
            right + 0.015,
            y_value,
            f"gap {gap:+.3f}",
            va="center",
            ha="left",
            fontsize=9.2,
            color="#2A2A2A",
        )

    x_max = float(max(np.max(fp_values), np.max(fn_values)))
    x_right = min(1.06, max(0.42, x_max + 0.16))
    ax.set_xlim(0.0, x_right)
    ax.set_ylim(-0.7, float(len(detectors) - 1) + 0.7)
    ax.set_yticks(y_positions)
    ax.set_yticklabels(detectors, fontsize=10)
    ax.set_xlabel("Error rate")
    ax.set_ylabel("Detector")
    ax.set_title("Global-threshold FP/FN profile on all test data")
    ax.grid(axis="x", alpha=0.24)
    ax.legend(frameon=True, loc="lower right")
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _plot_global_threshold_error_dumbbell_by_split(*, error_by_split: pd.DataFrame, output_path: Path) -> None:
    """Plot split-wise FP/FN grouped bars for all detectors using global threshold."""

    frame = error_by_split.sort_values(by=["target_split", "detector_run_id"], kind="mergesort").copy()
    splits = sorted(set(frame["target_split"].astype(str).tolist()))
    detectors = sorted(set(frame["detector_run_id"].astype(str).tolist()))
    n_cols = 3
    n_rows = int(math.ceil(len(splits) / n_cols))
    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(max(12.0, 4.4 * n_cols), max(3.2 * n_rows, 4.0)),
        sharey=True,
    )
    axes_array = np.atleast_1d(axes).reshape(n_rows, n_cols)
    x_positions = np.arange(len(detectors), dtype=float)
    bar_width = 0.38

    for panel_index, split_name in enumerate(splits):
        axis = axes_array.flat[panel_index]
        subset = frame[frame["target_split"] == split_name].copy()

        fp_values: list[float] = []
        fn_values: list[float] = []
        for detector_id in detectors:
            row_subset = subset[subset["detector_run_id"] == detector_id]
            if row_subset.empty:
                fp_values.append(float("nan"))
                fn_values.append(float("nan"))
                continue
            row = row_subset.iloc[0]
            fp_values.append(float(row["fp_rate"]))
            fn_values.append(float(row["fn_rate"]))

        bars_fp = axis.bar(
            x_positions - bar_width / 2.0,
            fp_values,
            width=bar_width,
            color="#E15759",
            edgecolor="#1A1A1A",
            linewidth=0.45,
            label="FP rate",
            zorder=3,
        )
        bars_fn = axis.bar(
            x_positions + bar_width / 2.0,
            fn_values,
            width=bar_width,
            color="#4E79A7",
            edgecolor="#1A1A1A",
            linewidth=0.45,
            label="FN rate",
            zorder=3,
        )
        for bar in list(bars_fp) + list(bars_fn):
            value = float(bar.get_height())
            if not math.isfinite(value):
                continue
            axis.text(
                bar.get_x() + bar.get_width() / 2.0,
                value + 0.015,
                f"{value:.2f}",
                ha="center",
                va="bottom",
                fontsize=7.9,
                color="#1A1A1A",
            )

        sample_human = int(subset["human_count"].iloc[0])
        sample_ai = int(subset["ai_count"].iloc[0])
        axis.set_title(f"{_display_split_name(split_name)}  (H={sample_human}, AI={sample_ai})", fontsize=10.5)
        axis.grid(axis="y", alpha=0.22)
        axis.set_ylim(0.0, 1.06)
        axis.set_xticks(x_positions)
        axis.set_xticklabels(detectors, rotation=15, ha="right", fontsize=8.2)
        if panel_index % n_cols == 0:
            axis.set_ylabel("Error rate")

    for panel_index in range(len(splits), n_rows * n_cols):
        axes_array.flat[panel_index].axis("off")

    legend_handles = [
        plt.Rectangle((0, 0), 1, 1, fc="#E15759", ec="#1A1A1A"),
        plt.Rectangle((0, 0), 1, 1, fc="#4E79A7", ec="#1A1A1A"),
    ]
    fig.legend(handles=legend_handles, labels=["FP rate", "FN rate"], loc="upper right", frameon=True)
    fig.suptitle("Global-threshold FP/FN by evaluated split", fontsize=14)
    fig.supxlabel("Detector")
    fig.tight_layout(rect=[0.0, 0.0, 0.98, 0.95])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _plot_global_local_mean_bars(*, global_local_mean: pd.DataFrame, output_path: Path) -> None:
    """Plot grouped bars per detector: global mean BA vs local mean BA."""

    plot_frame = global_local_mean.copy()
    detectors = [str(value) for value in plot_frame["detector_run_id"].tolist()]
    global_values = plot_frame["mean_global_test_ba"].to_numpy(dtype=float)
    local_values = plot_frame["mean_local_test_ba"].to_numpy(dtype=float)
    x = np.arange(len(detectors), dtype=float)
    width = 0.36

    fig, ax = plt.subplots(figsize=(max(8.0, 2.8 * len(detectors)), 5.6))
    bars_global = ax.bar(
        x - width / 2.0,
        global_values,
        width=width,
        color="#8C8C8C",
        edgecolor="#1A1A1A",
        linewidth=0.6,
        label="global",
    )
    bars_local = ax.bar(
        x + width / 2.0,
        local_values,
        width=width,
        color="#4E79A7",
        edgecolor="#1A1A1A",
        linewidth=0.6,
        label="local",
    )

    for bar in bars_global:
        value = float(bar.get_height())
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            value + 0.008,
            f"{value:.3f}",
            ha="center",
            va="bottom",
            fontsize=10,
            color="#1A1A1A",
        )
    for bar in bars_local:
        value = float(bar.get_height())
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            value + 0.008,
            f"{value:.3f}",
            ha="center",
            va="bottom",
            fontsize=10,
            color="#1A1A1A",
        )

    ax.set_xticks(x)
    ax.set_xticklabels(detectors, rotation=0)
    ax.set_ylim(0.0, min(1.0, float(max(np.max(global_values), np.max(local_values)) + 0.09)))
    ax.set_ylabel("mean test BA")
    ax.set_title("Global vs Local (mean over evaluated splits)")
    ax.legend(frameon=True, loc="lower right")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _plot_threshold_profile_bars(*, threshold_profile: pd.DataFrame, output_path: Path) -> None:
    """Plot per-detector 4-bar score profile summary."""

    plot_frame = threshold_profile.copy()
    detectors = [str(value) for value in plot_frame["detector_run_id"].tolist()]
    x = np.arange(len(detectors), dtype=float)
    width = 0.19

    global_scores = plot_frame["mean_global_test_ba"].to_numpy(dtype=float)
    local_scores = plot_frame["mean_local_test_ba"].to_numpy(dtype=float)
    worst_scores = plot_frame["worst_threshold_score_ba"].to_numpy(dtype=float)
    best_scores = plot_frame["best_threshold_score_ba"].to_numpy(dtype=float)

    fig, ax = plt.subplots(figsize=(max(8.0, 3.0 * len(detectors)), 6.0))
    bars_global = ax.bar(
        x - 1.5 * width,
        global_scores,
        width=width,
        color="#8C8C8C",
        edgecolor="#1A1A1A",
        linewidth=0.6,
        label="mean test BA (global)",
    )
    bars_local = ax.bar(
        x - 0.5 * width,
        local_scores,
        width=width,
        color="#4E79A7",
        edgecolor="#1A1A1A",
        linewidth=0.6,
        label="mean test BA (local)",
    )
    bars_worst = ax.bar(
        x + 0.5 * width,
        worst_scores,
        width=width,
        color="#E15759",
        edgecolor="#1A1A1A",
        linewidth=0.6,
        label="worst threshold score",
    )
    bars_best = ax.bar(
        x + 1.5 * width,
        best_scores,
        width=width,
        color="#59A14F",
        edgecolor="#1A1A1A",
        linewidth=0.6,
        label="best threshold score",
    )

    for bar in bars_global:
        value = float(bar.get_height())
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            value + 0.010,
            f"{value:.3f}",
            ha="center",
            va="bottom",
            fontsize=9.5,
            color="#1A1A1A",
        )
    for bar in bars_local:
        value = float(bar.get_height())
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            value + 0.010,
            f"{value:.3f}",
            ha="center",
            va="bottom",
            fontsize=9.5,
            color="#1A1A1A",
        )
    for bar, source, target in zip(
        bars_worst,
        plot_frame["worst_threshold_source_split"].tolist(),
        plot_frame["worst_threshold_target_split"].tolist(),
        strict=True,
    ):
        value = float(bar.get_height())
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            value / 2.0,
            f"{source}\n→ {target}",
            rotation=90,
            ha="center",
            va="center",
            fontsize=9.5,
            color="white",
            fontweight="bold",
        )
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            value + 0.010,
            f"{value:.3f}",
            ha="center",
            va="bottom",
            fontsize=9.5,
            color="#1A1A1A",
        )
    for bar, source, target in zip(
        bars_best,
        plot_frame["best_threshold_source_split"].tolist(),
        plot_frame["best_threshold_target_split"].tolist(),
        strict=True,
    ):
        value = float(bar.get_height())
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            value / 2.0,
            f"{source}\n→ {target}",
            rotation=90,
            ha="center",
            va="center",
            fontsize=9.5,
            color="white",
            fontweight="bold",
        )
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            value + 0.010,
            f"{value:.3f}",
            ha="center",
            va="bottom",
            fontsize=9.5,
            color="#1A1A1A",
        )

    ax.set_xticks(x)
    ax.set_xticklabels(detectors, rotation=0)
    ax.set_ylim(0.0, 1.0)
    ax.set_ylabel("balanced accuracy")
    ax.set_title("Threshold Profile by Detector")
    ax.legend(frameon=True, loc="lower right")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _plot_relative_delta_matrices_by_detector(*, transfer_frame: pd.DataFrame, output_dir: Path) -> None:
    """Plot per-detector matrix of BA delta vs global baseline per evaluated split."""

    detectors = sorted(set(str(value) for value in transfer_frame["detector_run_id"]))
    for detector_id in detectors:
        subset = transfer_frame[transfer_frame["detector_run_id"] == detector_id].copy()
        global_baseline = subset[subset["source_type"] == "global"][["target_split", "balanced_accuracy"]].copy()
        global_baseline = global_baseline.rename(columns={"balanced_accuracy": "global_balanced_accuracy"})

        merged = subset.merge(global_baseline, on="target_split", how="inner", validate="many_to_one")
        merged["delta_vs_global"] = merged["balanced_accuracy"] - merged["global_balanced_accuracy"]
        merged = merged[merged["source_type"] == "local"].copy()
        merged["source_label"] = merged["source_split"].map(lambda value: _to_train_axis_label(str(value)))
        merged["target_label"] = merged["target_split"].map(lambda value: _to_test_axis_label(str(value)))

        pivot = merged.pivot_table(
            index="source_label",
            columns="target_label",
            values="delta_vs_global",
            aggfunc="mean",
        )
        pivot = pivot.reindex(index=sorted(pivot.index.tolist()), columns=sorted(pivot.columns.tolist()))

        matrix = pivot.to_numpy(dtype=float)
        finite_values = matrix[np.isfinite(matrix)]
        if finite_values.size == 0:
            finite_values = np.array([0.0], dtype=float)
        vmax = float(np.max(np.abs(finite_values)))
        if not math.isfinite(vmax) or vmax <= 0.0:
            vmax = 1e-9

        fig, ax = plt.subplots(
            figsize=(max(8.4, len(pivot.columns) * 1.05), max(4.8, len(pivot.index) * 0.52 + 1.2))
        )
        image = ax.imshow(matrix, aspect="auto", cmap="RdYlGn", vmin=-vmax, vmax=vmax)
        ax.set_xticks(np.arange(len(pivot.columns)))
        ax.set_xticklabels([str(value) for value in pivot.columns.tolist()], rotation=20, ha="right")
        ax.set_yticks(np.arange(len(pivot.index)))
        ax.set_yticklabels([str(value) for value in pivot.index.tolist()])
        ax.set_title("Threshold Source vs Evaluated Split (ΔBA vs global)")

        for row_idx in range(matrix.shape[0]):
            for col_idx in range(matrix.shape[1]):
                value = matrix[row_idx, col_idx]
                if not math.isfinite(value):
                    label = "nan"
                else:
                    label = f"{value:+.3f}"
                ax.text(col_idx, row_idx, label, ha="center", va="center", fontsize=8.8, color="#1A1A1A")

        fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
        fig.tight_layout()
        output_path = output_dir / f"q1_relative_delta_matrix_{_slugify(detector_id)}.png"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=180, bbox_inches="tight")
        plt.close(fig)


def _plot_heatmap(
    *,
    matrix: np.ndarray,
    x_labels: Sequence[str],
    y_labels: Sequence[str],
    title: str,
    output_path: Path,
    center_zero: bool,
) -> None:
    """Plot annotated heatmap matrix."""

    if matrix.size == 0:
        raise ValueError("Cannot plot empty matrix.")
    finite_values = matrix[np.isfinite(matrix)]
    if finite_values.size == 0:
        finite_values = np.array([0.0], dtype=float)

    if center_zero:
        vmax = float(np.max(np.abs(finite_values)))
        if not math.isfinite(vmax) or vmax <= 0.0:
            vmax = 1e-9
        vmin = -vmax
        cmap = "coolwarm"
    else:
        vmin = float(np.min(finite_values))
        vmax = float(np.max(finite_values))
        if not math.isfinite(vmin):
            vmin = 0.0
        if not math.isfinite(vmax) or vmax <= vmin:
            vmax = vmin + 1e-9
        cmap = "viridis"

    fig, ax = plt.subplots(figsize=(max(7.0, len(x_labels) * 1.0), max(4.0, len(y_labels) * 0.6 + 1.2)))
    image = ax.imshow(matrix, aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_xticks(np.arange(len(x_labels)))
    ax.set_xticklabels(list(x_labels), rotation=25, ha="right")
    ax.set_yticks(np.arange(len(y_labels)))
    ax.set_yticklabels(list(y_labels))
    ax.set_title(title)

    for row_idx in range(matrix.shape[0]):
        for col_idx in range(matrix.shape[1]):
            value = matrix[row_idx, col_idx]
            label = "nan" if not math.isfinite(value) else f"{value:.3f}"
            ax.text(col_idx, row_idx, label, ha="center", va="center", fontsize=8, color="black")

    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _build_q1_markdown(
    *,
    train_eval_gap: pd.DataFrame,
    local_vs_global: pd.DataFrame,
    corresponding_vs_others: pd.DataFrame,
) -> str:
    """Build markdown report content for Question 1."""

    lines: list[str] = []
    lines.append("# Pytanie 1: Wplyw Tuningu Thresholdu")
    lines.append("")
    lines.append("## Train -> Eval Gap")
    lines.append("")
    lines.append(_dataframe_to_markdown(train_eval_gap))
    lines.append("")
    lines.append("## Local Threshold vs Global Threshold (Corresponding Split)")
    lines.append("")
    lines.append(_dataframe_to_markdown(local_vs_global))
    lines.append("")
    lines.append("## Corresponding Split vs Other Splits")
    lines.append("")
    lines.append(_dataframe_to_markdown(corresponding_vs_others))
    lines.append("")
    return "\n".join(lines)


def _build_q2_markdown(
    *,
    detector_ranking: pd.DataFrame,
    detector_error_profile: pd.DataFrame,
    split_difficulty: pd.DataFrame,
) -> str:
    """Build markdown report content for Question 2."""

    lines: list[str] = []
    lines.append("# Pytanie 2: Jakosc Detektorow")
    lines.append("")
    lines.append("## Ranking Detektorow")
    lines.append("")
    lines.append(_dataframe_to_markdown(detector_ranking))
    lines.append("")
    lines.append("## Profil Bledow (FP/FN)")
    lines.append("")
    lines.append(_dataframe_to_markdown(detector_error_profile))
    lines.append("")
    lines.append("## Trudnosc Splitow")
    lines.append("")
    lines.append(_dataframe_to_markdown(split_difficulty))
    lines.append("")
    return "\n".join(lines)


def _build_findings_markdown(
    *,
    local_vs_global: pd.DataFrame,
    detector_ranking: pd.DataFrame,
    split_difficulty: pd.DataFrame,
    detector_error_profile: pd.DataFrame,
) -> str:
    """Build concise key findings markdown."""

    best_change = local_vs_global.loc[local_vs_global["delta_local_minus_global_ba"].idxmax()]
    worst_change = local_vs_global.loc[local_vs_global["delta_local_minus_global_ba"].idxmin()]
    best_detector = detector_ranking.iloc[0]
    hardest_split = split_difficulty.iloc[0]
    easiest_split = split_difficulty.iloc[-1]
    highest_fp = detector_error_profile.loc[detector_error_profile["mean_fp_rate"].idxmax()]
    highest_fn = detector_error_profile.loc[detector_error_profile["mean_fn_rate"].idxmax()]

    lines: list[str] = []
    lines.append("# Key Findings")
    lines.append("")
    lines.append(
        f"- Largest improvement local-vs-global BA: `{best_change['detector_run_id']}` on `{best_change['target_split']}` "
        f"with delta `{float(best_change['delta_local_minus_global_ba']):.4f}`."
    )
    lines.append(
        f"- Largest degradation local-vs-global BA: `{worst_change['detector_run_id']}` on `{worst_change['target_split']}` "
        f"with delta `{float(worst_change['delta_local_minus_global_ba']):.4f}`."
    )
    lines.append(
        f"- Best detector by mean corresponding BA: `{best_detector['detector_run_id']}` "
        f"(`{float(best_detector['mean_balanced_accuracy']):.4f}`)."
    )
    lines.append(
        f"- Hardest split (lowest mean BA): `{hardest_split['target_split']}` "
        f"(`{float(hardest_split['mean_balanced_accuracy']):.4f}`)."
    )
    lines.append(
        f"- Easiest split (highest mean BA): `{easiest_split['target_split']}` "
        f"(`{float(easiest_split['mean_balanced_accuracy']):.4f}`)."
    )
    lines.append(
        f"- Highest FP rate detector: `{highest_fp['detector_run_id']}` "
        f"(`{float(highest_fp['mean_fp_rate']):.4f}`)."
    )
    lines.append(
        f"- Highest FN rate detector: `{highest_fn['detector_run_id']}` "
        f"(`{float(highest_fn['mean_fn_rate']):.4f}`)."
    )
    lines.append("")
    return "\n".join(lines)


def _dataframe_to_markdown(frame: pd.DataFrame, *, float_precision: int = 4) -> str:
    """Render dataframe as markdown table without external tabulate dependency."""

    if frame.empty:
        return "_No rows available._"

    display_frame = frame.copy()
    for column in display_frame.columns:
        if pd.api.types.is_float_dtype(display_frame[column]):
            display_frame[column] = display_frame[column].map(lambda value: f"{float(value):.{float_precision}f}")

    headers = [str(column) for column in display_frame.columns.tolist()]
    rows = [[str(value) for value in row] for row in display_frame.to_numpy(dtype=object).tolist()]

    lines: list[str] = []
    lines.append("| " + " | ".join(headers) + " |")
    lines.append("| " + " | ".join("---" for _ in headers) + " |")
    for row in rows:
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


def _compute_fp_rate(*, confusion: Mapping[str, Any]) -> float:
    """Compute false-positive rate from confusion payload."""

    false_positive = _read_float(confusion, "false_positive")
    true_negative = _read_float(confusion, "true_negative")
    return _safe_divide(false_positive, false_positive + true_negative)


def _compute_fn_rate(*, confusion: Mapping[str, Any]) -> float:
    """Compute false-negative rate from confusion payload."""

    false_negative = _read_float(confusion, "false_negative")
    true_positive = _read_float(confusion, "true_positive")
    return _safe_divide(false_negative, false_negative + true_positive)


def _safe_divide(numerator: float, denominator: float) -> float:
    """Safely divide values with zero-denominator fallback."""

    if denominator == 0:
        return 0.0
    return float(numerator) / float(denominator)


def _read_mapping(payload: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    """Read required mapping from payload."""

    value = payload.get(key)
    if not isinstance(value, Mapping):
        raise ValueError(f"Expected mapping for key={key!r} in payload={payload!r}")
    return value


def _read_str(payload: Mapping[str, Any], key: str) -> str:
    """Read required string value from payload."""

    value = payload.get(key)
    if not isinstance(value, str):
        raise ValueError(f"Expected string for key={key!r} in payload={payload!r}")
    return value


def _read_float(payload: Mapping[str, Any], key: str) -> float:
    """Read required numeric value from payload."""

    value = payload.get(key)
    if not isinstance(value, (int, float)):
        raise ValueError(f"Expected numeric value for key={key!r} in payload={payload!r}")
    return float(value)


def _slugify(value: str) -> str:
    """Convert arbitrary identifier into safe filename token."""

    normalized = value.strip().lower().replace(":", "_").replace("/", "_")
    normalized = normalized.replace(" ", "_")
    return "".join(character for character in normalized if character.isalnum() or character in {"_", "-"})


if __name__ == "__main__":
    raise SystemExit(main())
sns.set_theme(style="whitegrid", context="talk")
