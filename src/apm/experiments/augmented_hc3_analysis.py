"""Analyze augmented HC3 global/local results and compute deltas vs baseline."""

from __future__ import annotations

import argparse
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt

CONDITION_ORDER: tuple[str, ...] = ("orig_h_aug_ai", "aug_h_orig_ai", "aug_both")
BASELINE_SCENARIO = "aug_baseline"
BASELINE_SCOPE_ID = "hc3:aug_baseline"


@dataclass(frozen=True, slots=True)
class AugmentedHc3AnalysisRequest:
    """Request payload for augmented HC3 run analysis."""

    project_root: Path
    run_dir: Path
    output_dir: Path
    baseline_scope_id: str


@dataclass(frozen=True, slots=True)
class AugmentedHc3AnalysisResult:
    """Generated output paths for augmented HC3 analysis."""

    scenario_metrics_csv: Path
    scenario_deltas_csv: Path
    summary_csv: Path
    delta_balanced_accuracy_plot: Path
    delta_recall_plot: Path
    delta_fp_rate_plot: Path


def run_augmented_hc3_analysis(request: AugmentedHc3AnalysisRequest) -> AugmentedHc3AnalysisResult:
    """Run augmented HC3 analysis from local/global runner outputs."""

    _validate_request(request)

    local_rows = _read_jsonl(request.run_dir / "local_results.jsonl")
    rows = _extract_augmented_rows(local_rows=local_rows)
    frame = pd.DataFrame(rows)
    if frame.empty:
        raise ValueError("No `hc3:aug_*` local rows found in local_results.jsonl.")

    frame_with_deltas = _compute_deltas(
        frame=frame,
        baseline_scope_id=request.baseline_scope_id,
    )

    request.output_dir.mkdir(parents=True, exist_ok=True)

    scenario_metrics_csv = request.output_dir / "scenario_metrics.csv"
    scenario_deltas_csv = request.output_dir / "scenario_deltas_vs_baseline.csv"
    summary_csv = request.output_dir / "summary_by_augmentation_condition.csv"

    frame.sort_values(
        by=["detector_run_id", "scenario_name"],
        ascending=True,
        kind="mergesort",
    ).to_csv(scenario_metrics_csv, index=False)

    delta_columns = [column for column in frame_with_deltas.columns if column.startswith("delta_")]
    frame_with_deltas.sort_values(
        by=["detector_run_id", "scenario_name"],
        ascending=True,
        kind="mergesort",
    ).to_csv(
        scenario_deltas_csv,
        index=False,
        columns=[
            "detector_run_id",
            "scope_id",
            "scenario_name",
            "augmentation",
            "condition",
            *delta_columns,
        ],
    )

    summary = (
        frame_with_deltas[frame_with_deltas["condition"] != "baseline"]
        .groupby(["detector_run_id", "augmentation", "condition"], as_index=False)
        .agg(
            delta_balanced_accuracy_mean=("delta_balanced_accuracy", "mean"),
            delta_recall_mean=("delta_recall", "mean"),
            delta_fp_rate_mean=("delta_fp_rate", "mean"),
        )
    )
    summary.to_csv(summary_csv, index=False)

    delta_balanced_accuracy_plot = request.output_dir / "delta_balanced_accuracy_heatmap.png"
    delta_recall_plot = request.output_dir / "delta_recall_heatmap.png"
    delta_fp_rate_plot = request.output_dir / "delta_fp_rate_heatmap.png"

    _plot_delta_heatmaps(
        frame=frame_with_deltas,
        metric_column="delta_balanced_accuracy",
        title="Delta Balanced Accuracy vs Baseline",
        output_path=delta_balanced_accuracy_plot,
    )
    _plot_delta_heatmaps(
        frame=frame_with_deltas,
        metric_column="delta_recall",
        title="Delta Recall (AI) vs Baseline",
        output_path=delta_recall_plot,
    )
    _plot_delta_heatmaps(
        frame=frame_with_deltas,
        metric_column="delta_fp_rate",
        title="Delta False Positive Rate vs Baseline",
        output_path=delta_fp_rate_plot,
    )

    return AugmentedHc3AnalysisResult(
        scenario_metrics_csv=scenario_metrics_csv,
        scenario_deltas_csv=scenario_deltas_csv,
        summary_csv=summary_csv,
        delta_balanced_accuracy_plot=delta_balanced_accuracy_plot,
        delta_recall_plot=delta_recall_plot,
        delta_fp_rate_plot=delta_fp_rate_plot,
    )


def build_request_from_args(args: argparse.Namespace) -> AugmentedHc3AnalysisRequest:
    """Build typed request from CLI args."""

    project_root = args.project_root.resolve()
    runs_root = (project_root / args.runs_root).resolve()

    if args.run_dir is not None:
        run_dir_input = Path(args.run_dir)
        run_dir = run_dir_input if run_dir_input.is_absolute() else (project_root / run_dir_input)
        run_dir = run_dir.resolve()
    elif args.run_id:
        run_dir = (runs_root / args.run_id).resolve()
    else:
        run_dir = _resolve_latest_run_dir(runs_root)

    output_dir_input = Path(args.output_dir) if args.output_dir else (run_dir / "analysis_augmented")
    output_dir = output_dir_input if output_dir_input.is_absolute() else (project_root / output_dir_input)
    output_dir = output_dir.resolve()

    return AugmentedHc3AnalysisRequest(
        project_root=project_root,
        run_dir=run_dir,
        output_dir=output_dir,
        baseline_scope_id=args.baseline_scope_id,
    )


def build_arg_parser() -> argparse.ArgumentParser:
    """Build CLI parser for augmented HC3 analysis."""

    parser = argparse.ArgumentParser(description="Analyze augmented HC3 local results and compute deltas vs baseline.")
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
        help="Output directory for analysis artifacts. Defaults to <run_dir>/analysis_augmented.",
    )
    parser.add_argument(
        "--baseline-scope-id",
        type=str,
        default=BASELINE_SCOPE_ID,
        help="Local scope id used as baseline for delta calculation.",
    )
    return parser


def main() -> int:
    """CLI entrypoint."""

    parser = build_arg_parser()
    args = parser.parse_args()
    request = build_request_from_args(args)
    result = run_augmented_hc3_analysis(request)
    print(
        json.dumps(
            {
                "run_dir": str(request.run_dir),
                "output_dir": str(request.output_dir),
                "scenario_metrics_csv": str(result.scenario_metrics_csv),
                "scenario_deltas_csv": str(result.scenario_deltas_csv),
                "summary_csv": str(result.summary_csv),
                "delta_balanced_accuracy_plot": str(result.delta_balanced_accuracy_plot),
                "delta_recall_plot": str(result.delta_recall_plot),
                "delta_fp_rate_plot": str(result.delta_fp_rate_plot),
            },
            indent=2,
        )
    )
    return 0


def _validate_request(request: AugmentedHc3AnalysisRequest) -> None:
    """Validate request invariants."""

    if not request.baseline_scope_id.strip():
        raise ValueError("baseline_scope_id must be non-empty.")
    if not request.run_dir.exists():
        raise FileNotFoundError(f"Missing run directory: {request.run_dir}")
    local_path = request.run_dir / "local_results.jsonl"
    if not local_path.exists():
        raise FileNotFoundError(f"Missing local results file: {local_path}")


def _resolve_latest_run_dir(runs_root: Path) -> Path:
    """Resolve most recently modified run directory containing local_results.jsonl."""

    if not runs_root.exists():
        raise FileNotFoundError(f"Missing runs root: {runs_root}")
    candidates: list[Path] = []
    for child in runs_root.iterdir():
        if not child.is_dir():
            continue
        if (child / "local_results.jsonl").exists():
            candidates.append(child)
    if not candidates:
        raise ValueError(f"No run directories with local_results.jsonl found under: {runs_root}")
    candidates.sort(key=lambda path: path.stat().st_mtime, reverse=True)
    return candidates[0]


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    """Read JSONL file into list of dictionaries."""

    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        payload = json.loads(line)
        if not isinstance(payload, dict):
            raise ValueError(f"Expected object row in {path}, got: {payload!r}")
        rows.append(payload)
    return rows


def _extract_augmented_rows(*, local_rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Extract augmented HC3 local rows into normalized analysis records."""

    records: list[dict[str, Any]] = []
    for row in local_rows:
        scope_id = row.get("scope_id")
        detector_run_id = row.get("detector_run_id")
        threshold = row.get("threshold")
        test_metrics = row.get("test_metrics")
        if not isinstance(scope_id, str):
            continue
        if not scope_id.startswith("hc3:aug_"):
            continue
        if not isinstance(detector_run_id, str):
            raise ValueError(f"Invalid detector_run_id in row: {row!r}")
        if not isinstance(threshold, (int, float)):
            raise ValueError(f"Invalid threshold in row: {row!r}")
        if not isinstance(test_metrics, Mapping):
            raise ValueError(f"Missing test_metrics in row: {row!r}")

        scenario_name = scope_id.split(":", 1)[1]
        augmentation, condition = _parse_scenario_name(scenario_name)
        confusion = test_metrics.get("confusion_matrix")
        if not isinstance(confusion, Mapping):
            raise ValueError(f"Missing confusion matrix in row: {row!r}")

        true_positive = _read_int(confusion, "true_positive")
        false_positive = _read_int(confusion, "false_positive")
        true_negative = _read_int(confusion, "true_negative")
        false_negative = _read_int(confusion, "false_negative")

        fp_rate = _safe_divide(false_positive, false_positive + true_negative)
        fn_rate = _safe_divide(false_negative, false_negative + true_positive)

        records.append(
            {
                "detector_run_id": detector_run_id,
                "scope_id": scope_id,
                "scenario_name": scenario_name,
                "augmentation": augmentation,
                "condition": condition,
                "is_baseline": scenario_name == BASELINE_SCENARIO,
                "threshold": float(threshold),
                "accuracy": _read_float(test_metrics, "accuracy"),
                "precision": _read_float(test_metrics, "precision"),
                "recall": _read_float(test_metrics, "recall"),
                "f1": _read_float(test_metrics, "f1"),
                "balanced_accuracy": _read_float(test_metrics, "balanced_accuracy"),
                "roc_auc": _read_optional_float(test_metrics, "roc_auc"),
                "pr_auc": _read_optional_float(test_metrics, "pr_auc"),
                "mean_score_human": _read_float(test_metrics, "mean_score_human"),
                "mean_score_ai": _read_float(test_metrics, "mean_score_ai"),
                "true_positive": true_positive,
                "false_positive": false_positive,
                "true_negative": true_negative,
                "false_negative": false_negative,
                "fp_rate": fp_rate,
                "fn_rate": fn_rate,
            }
        )
    return records


def _parse_scenario_name(scenario_name: str) -> tuple[str, str]:
    """Parse scenario name into augmentation family and condition type."""

    if scenario_name == BASELINE_SCENARIO:
        return ("baseline", "baseline")
    match = re.match(r"^aug_(.+)_(orig_h_aug_ai|aug_h_orig_ai|aug_both)$", scenario_name)
    if match is None:
        raise ValueError(f"Unsupported scenario naming pattern: {scenario_name!r}")
    augmentation = str(match.group(1))
    condition = str(match.group(2))
    return (augmentation, condition)


def _compute_deltas(
    *,
    frame: pd.DataFrame,
    baseline_scope_id: str,
) -> pd.DataFrame:
    """Compute per-detector metric deltas against baseline scope."""

    metric_columns = [
        "accuracy",
        "precision",
        "recall",
        "f1",
        "balanced_accuracy",
        "roc_auc",
        "pr_auc",
        "mean_score_human",
        "mean_score_ai",
        "fp_rate",
        "fn_rate",
    ]

    baseline_rows = frame[frame["scope_id"] == baseline_scope_id].copy()
    if baseline_rows.empty:
        raise ValueError(f"Baseline scope_id not found in local rows: {baseline_scope_id!r}")

    baseline_rows = baseline_rows[["detector_run_id", *metric_columns]].rename(
        columns={column: f"baseline_{column}" for column in metric_columns}
    )
    merged = frame.merge(baseline_rows, on="detector_run_id", how="left", validate="many_to_one")

    for column in metric_columns:
        merged[f"delta_{column}"] = merged[column] - merged[f"baseline_{column}"]

    return merged


def _plot_delta_heatmaps(
    *,
    frame: pd.DataFrame,
    metric_column: str,
    title: str,
    output_path: Path,
) -> None:
    """Render per-detector heatmap for one delta metric across augmentations/conditions."""

    filtered = frame[frame["condition"] != "baseline"].copy()
    if filtered.empty:
        raise ValueError("No non-baseline scenarios found for plotting.")

    detector_ids = sorted(str(value) for value in filtered["detector_run_id"].dropna().unique().tolist())
    augmentations = sorted(str(value) for value in filtered["augmentation"].dropna().unique().tolist())

    fig, axes = plt.subplots(
        nrows=1,
        ncols=len(detector_ids),
        figsize=(4.5 * len(detector_ids), max(4.0, 0.45 * len(augmentations) + 1.5)),
        squeeze=False,
    )

    vmax = float(np.nanmax(np.abs(filtered[metric_column].to_numpy(dtype=float))))
    if not math.isfinite(vmax) or vmax <= 0.0:
        vmax = 1e-9

    for axis_index, detector_id in enumerate(detector_ids):
        ax = axes[0, axis_index]
        subset = filtered[filtered["detector_run_id"] == detector_id]
        pivot = subset.pivot_table(
            index="augmentation",
            columns="condition",
            values=metric_column,
            aggfunc="mean",
        )
        pivot = pivot.reindex(index=augmentations, columns=list(CONDITION_ORDER))
        matrix = pivot.to_numpy(dtype=float)
        image = ax.imshow(matrix, aspect="auto", cmap="coolwarm", vmin=-vmax, vmax=vmax)
        ax.set_title(detector_id)
        ax.set_xticks(np.arange(len(CONDITION_ORDER)))
        ax.set_xticklabels(list(CONDITION_ORDER), rotation=25, ha="right")
        ax.set_yticks(np.arange(len(augmentations)))
        ax.set_yticklabels(augmentations)

        for row_index in range(matrix.shape[0]):
            for col_index in range(matrix.shape[1]):
                value = matrix[row_index, col_index]
                label = "nan" if not math.isfinite(value) else f"{value:.3f}"
                ax.text(col_index, row_index, label, ha="center", va="center", fontsize=8)

        fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)

    fig.suptitle(title)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _read_int(payload: Mapping[str, Any], key: str) -> int:
    """Read required integer-like value from mapping."""

    value = payload.get(key)
    if not isinstance(value, (int, float)):
        raise ValueError(f"Missing numeric value for key={key!r} in payload={payload!r}")
    return int(value)


def _read_float(payload: Mapping[str, Any], key: str) -> float:
    """Read required float-like value from mapping."""

    value = payload.get(key)
    if not isinstance(value, (int, float)):
        raise ValueError(f"Missing numeric value for key={key!r} in payload={payload!r}")
    return float(value)


def _read_optional_float(payload: Mapping[str, Any], key: str) -> float:
    """Read optional float-like value from mapping; NaN when absent."""

    value = payload.get(key)
    if value is None:
        return float("nan")
    if not isinstance(value, (int, float)):
        raise ValueError(f"Invalid optional numeric value for key={key!r} in payload={payload!r}")
    return float(value)


def _safe_divide(numerator: int, denominator: int) -> float:
    """Compute safe ratio returning 0.0 when denominator is zero."""

    if denominator <= 0:
        return 0.0
    return float(numerator) / float(denominator)


if __name__ == "__main__":
    raise SystemExit(main())
