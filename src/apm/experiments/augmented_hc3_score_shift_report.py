"""Analyze baseline vs orig_h_aug_ai score shifts for augmented HC3 runs."""

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
import seaborn as sns
from matplotlib import font_manager
from matplotlib.font_manager import FontProperties

from apm.experiments.augmentation_palette import augmentation_color, ordered_augmentations

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

sns.set_theme(style="whitegrid", context="talk")


@dataclass(frozen=True, slots=True)
class AugmentedHc3ScoreShiftRequest:
    """Request payload for score-shift analysis on augmented HC3 runs."""

    project_root: Path
    run_dir: Path
    output_dir: Path
    dataset_id: str
    baseline_split: str
    scenario_suffix: str
    augmented_human_suffix: str
    include_augmented_human: bool
    partition: str


@dataclass(frozen=True, slots=True)
class AugmentedHc3ScoreShiftResult:
    """Generated output paths for score-shift analysis."""

    output_dir: Path
    score_summary_csv: Path
    ai_shift_csv: Path
    human_shift_csv: Path
    markdown_summary: Path
    augmented_human_output_dir: Path | None


def run_augmented_hc3_score_shift_report(request: AugmentedHc3ScoreShiftRequest) -> AugmentedHc3ScoreShiftResult:
    """Run score-shift analysis for baseline and orig_h_aug_ai scenarios."""

    _validate_request(request)
    request.output_dir.mkdir(parents=True, exist_ok=True)

    raw_rows = _read_jsonl(request.run_dir / "raw_scores.jsonl")
    local_rows = _read_jsonl(request.run_dir / "local_results.jsonl")

    baseline_thresholds = _extract_baseline_thresholds(
        local_rows=local_rows,
        dataset_id=request.dataset_id,
        baseline_split=request.baseline_split,
    )
    primary_score_frame = _build_score_frame(
        raw_rows=raw_rows,
        dataset_id=request.dataset_id,
        baseline_split=request.baseline_split,
        scenario_suffix=request.scenario_suffix,
        partition=request.partition,
        baseline_thresholds=baseline_thresholds,
    )
    primary_outputs = _generate_scenario_outputs(
        raw_rows=raw_rows,
        dataset_id=request.dataset_id,
        baseline_split=request.baseline_split,
        scenario_suffix=request.scenario_suffix,
        partition=request.partition,
        baseline_thresholds=baseline_thresholds,
        output_dir=request.output_dir,
        focus_label="ai",
    )

    augmented_human_output_dir: Path | None = None
    if request.include_augmented_human:
        augmented_human_score_frame = _build_score_frame(
            raw_rows=raw_rows,
            dataset_id=request.dataset_id,
            baseline_split=request.baseline_split,
            scenario_suffix=request.augmented_human_suffix,
            partition=request.partition,
            baseline_thresholds=baseline_thresholds,
        )
        augmented_human_output_dir = request.output_dir / "augmented_human"
        _generate_scenario_outputs(
            raw_rows=raw_rows,
            dataset_id=request.dataset_id,
            baseline_split=request.baseline_split,
            scenario_suffix=request.augmented_human_suffix,
            partition=request.partition,
            baseline_thresholds=baseline_thresholds,
            output_dir=augmented_human_output_dir,
            focus_label="human",
        )
        _plot_per_detector_rowwise_thin_bars_combined(
            primary_score_frame=primary_score_frame,
            augmented_human_score_frame=augmented_human_score_frame,
            baseline_split=request.baseline_split,
            baseline_thresholds=baseline_thresholds,
            output_dir=request.output_dir,
            max_samples_per_label=100,
        )

    return AugmentedHc3ScoreShiftResult(
        output_dir=request.output_dir,
        score_summary_csv=primary_outputs["score_summary_csv"],
        ai_shift_csv=primary_outputs["ai_shift_csv"],
        human_shift_csv=primary_outputs["human_shift_csv"],
        markdown_summary=primary_outputs["markdown_summary"],
        augmented_human_output_dir=augmented_human_output_dir,
    )


def _generate_scenario_outputs(
    *,
    raw_rows: Sequence[Mapping[str, Any]],
    dataset_id: str,
    baseline_split: str,
    scenario_suffix: str,
    partition: str,
    baseline_thresholds: Mapping[str, float],
    output_dir: Path,
    focus_label: str,
) -> dict[str, Path]:
    """Generate one scenario output package and return key file paths."""

    output_dir.mkdir(parents=True, exist_ok=True)

    score_frame = _build_score_frame(
        raw_rows=raw_rows,
        dataset_id=dataset_id,
        baseline_split=baseline_split,
        scenario_suffix=scenario_suffix,
        partition=partition,
        baseline_thresholds=baseline_thresholds,
    )
    score_summary = _build_score_summary(score_frame=score_frame)
    ai_shift, human_shift = _build_shift_tables(score_summary=score_summary)

    score_summary_csv = output_dir / "score_summary_baseline_vs_orig_h_aug_ai.csv"
    ai_shift_csv = output_dir / "ai_shift_vs_baseline_orig_h_aug_ai.csv"
    human_shift_csv = output_dir / "human_shift_vs_baseline_orig_h_aug_ai.csv"
    markdown_summary = output_dir / "SCORE_SHIFT_SUMMARY.md"

    score_summary.to_csv(score_summary_csv, index=False)
    ai_shift.to_csv(ai_shift_csv, index=False)
    human_shift.to_csv(human_shift_csv, index=False)

    if focus_label == "ai":
        _plot_ai_delta_heatmap(
            ai_shift=ai_shift,
            output_path=output_dir / "ai_mean_score_delta_heatmap.png",
        )
        _plot_ai_detection_rate_delta_heatmap(
            ai_shift=ai_shift,
            output_path=output_dir / "ai_detection_rate_delta_heatmap.png",
        )
        _plot_per_detector_ai_bars(
            ai_shift=ai_shift,
            output_dir=output_dir,
        )
        _plot_per_detector_ai_histograms(
            score_frame=score_frame,
            baseline_thresholds=baseline_thresholds,
            output_dir=output_dir,
        )
    else:
        _plot_human_delta_heatmap(
            human_shift=human_shift,
            output_path=output_dir / "human_mean_score_delta_heatmap.png",
        )
        _plot_human_detection_rate_delta_heatmap(
            human_shift=human_shift,
            output_path=output_dir / "human_detection_rate_delta_heatmap.png",
        )
        _plot_per_detector_human_bars(
            human_shift=human_shift,
            output_dir=output_dir,
        )
        _plot_per_detector_human_histograms(
            score_frame=score_frame,
            baseline_thresholds=baseline_thresholds,
            output_dir=output_dir,
        )

    _plot_per_detector_rowwise_thin_bars(
        score_frame=score_frame,
        baseline_split=baseline_split,
        baseline_thresholds=baseline_thresholds,
        output_dir=output_dir,
        max_samples_per_label=100,
    )

    markdown_summary.write_text(
        _build_markdown_summary(ai_shift=ai_shift, human_shift=human_shift),
        encoding="utf-8",
    )

    return {
        "score_summary_csv": score_summary_csv,
        "ai_shift_csv": ai_shift_csv,
        "human_shift_csv": human_shift_csv,
        "markdown_summary": markdown_summary,
    }


def build_request_from_args(args: argparse.Namespace) -> AugmentedHc3ScoreShiftRequest:
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
        run_dir = _resolve_latest_run_dir(runs_root)

    output_dir_candidate = Path(args.output_dir) if args.output_dir else (run_dir / "analysis_augmented_score_shift")
    output_dir = output_dir_candidate if output_dir_candidate.is_absolute() else (project_root / output_dir_candidate)
    output_dir = output_dir.resolve()

    return AugmentedHc3ScoreShiftRequest(
        project_root=project_root,
        run_dir=run_dir,
        output_dir=output_dir,
        dataset_id=str(args.dataset_id),
        baseline_split=str(args.baseline_split),
        scenario_suffix=str(args.scenario_suffix),
        augmented_human_suffix=str(args.augmented_human_suffix),
        include_augmented_human=bool(args.include_augmented_human),
        partition=str(args.partition),
    )


def build_arg_parser() -> argparse.ArgumentParser:
    """Build CLI parser for score-shift report."""

    parser = argparse.ArgumentParser(
        description="Analyze baseline vs orig_h_aug_ai score shifts for augmented HC3 run outputs."
    )
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
        help="Output directory. Defaults to <run_dir>/analysis_augmented_score_shift.",
    )
    parser.add_argument("--dataset-id", type=str, default="hc3", help="Dataset id in raw_scores.jsonl.")
    parser.add_argument("--baseline-split", type=str, default="aug_baseline", help="Baseline split name.")
    parser.add_argument(
        "--scenario-suffix",
        type=str,
        default="orig_h_aug_ai",
        help="Augmented split suffix to compare against baseline.",
    )
    parser.add_argument(
        "--augmented-human-suffix",
        type=str,
        default="aug_h_orig_ai",
        help="Suffix used for augmented-human scenario outputs.",
    )
    parser.add_argument(
        "--include-augmented-human",
        dest="include_augmented_human",
        action="store_true",
        help="Also generate a second output set for augmented-human condition (enabled by default).",
    )
    parser.add_argument(
        "--no-augmented-human",
        dest="include_augmented_human",
        action="store_false",
        help="Disable augmented-human output generation.",
    )
    parser.set_defaults(include_augmented_human=True)
    parser.add_argument("--partition", type=str, default="test", choices=("train", "test"), help="Partition to analyze.")
    return parser


def main() -> int:
    """CLI entrypoint."""

    parser = build_arg_parser()
    args = parser.parse_args()
    request = build_request_from_args(args)
    result = run_augmented_hc3_score_shift_report(request)
    print(
        json.dumps(
            {
                "run_dir": str(request.run_dir),
                "output_dir": str(result.output_dir),
                "score_summary_csv": str(result.score_summary_csv),
                "ai_shift_csv": str(result.ai_shift_csv),
                "human_shift_csv": str(result.human_shift_csv),
                "markdown_summary": str(result.markdown_summary),
                "augmented_human_output_dir": (
                    str(result.augmented_human_output_dir) if result.augmented_human_output_dir is not None else ""
                ),
            },
            indent=2,
        )
    )
    return 0


def _validate_request(request: AugmentedHc3ScoreShiftRequest) -> None:
    """Validate request invariants."""

    if not request.dataset_id.strip():
        raise ValueError("dataset_id must be non-empty.")
    if not request.baseline_split.strip():
        raise ValueError("baseline_split must be non-empty.")
    if not request.scenario_suffix.strip():
        raise ValueError("scenario_suffix must be non-empty.")
    if not request.augmented_human_suffix.strip():
        raise ValueError("augmented_human_suffix must be non-empty.")
    if request.partition not in {"train", "test"}:
        raise ValueError(f"Unsupported partition: {request.partition!r}")
    if not request.run_dir.exists():
        raise FileNotFoundError(f"Missing run directory: {request.run_dir}")
    for required in ("raw_scores.jsonl", "local_results.jsonl"):
        path = request.run_dir / required
        if not path.exists():
            raise FileNotFoundError(f"Missing required file: {path}")


def _resolve_latest_run_dir(runs_root: Path) -> Path:
    """Resolve most recently modified complete run directory."""

    if not runs_root.exists():
        raise FileNotFoundError(f"Missing runs root: {runs_root}")
    candidates: list[Path] = []
    for child in runs_root.iterdir():
        if not child.is_dir():
            continue
        if (child / "raw_scores.jsonl").exists() and (child / "local_results.jsonl").exists():
            candidates.append(child)
    if not candidates:
        raise ValueError(f"No run directories with required files found under: {runs_root}")
    candidates.sort(key=lambda path: path.stat().st_mtime, reverse=True)
    return candidates[0]


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    """Read JSONL file into dictionaries."""

    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        payload = json.loads(line)
        if not isinstance(payload, dict):
            raise ValueError(f"Expected JSON object rows in {path}, got: {payload!r}")
        rows.append(payload)
    return rows


def _extract_baseline_thresholds(
    *,
    local_rows: Sequence[Mapping[str, Any]],
    dataset_id: str,
    baseline_split: str,
) -> dict[str, float]:
    """Extract baseline threshold per detector from local_results rows."""

    baseline_scope_id = f"{dataset_id}:{baseline_split}"
    thresholds: dict[str, float] = {}
    for row in local_rows:
        scope_id = row.get("scope_id")
        detector_run_id = row.get("detector_run_id")
        threshold = row.get("threshold")
        if scope_id != baseline_scope_id:
            continue
        if not isinstance(detector_run_id, str):
            raise ValueError(f"Invalid detector_run_id in row: {row!r}")
        if not isinstance(threshold, (int, float)):
            raise ValueError(f"Invalid threshold in row: {row!r}")
        thresholds[detector_run_id] = float(threshold)
    if not thresholds:
        raise ValueError(f"No baseline thresholds found for scope_id={baseline_scope_id!r}.")
    return thresholds


def _build_score_frame(
    *,
    raw_rows: Sequence[Mapping[str, Any]],
    dataset_id: str,
    baseline_split: str,
    scenario_suffix: str,
    partition: str,
    baseline_thresholds: Mapping[str, float],
) -> pd.DataFrame:
    """Build normalized frame of baseline + target augmented scores."""

    records: list[dict[str, Any]] = []
    pattern = re.compile(rf"^aug_(.+)_{re.escape(scenario_suffix)}$")

    for row in raw_rows:
        row_dataset = row.get("dataset_id")
        detector_run_id = row.get("detector_run_id")
        source_split = row.get("source_split")
        row_partition = row.get("partition")
        label = row.get("label")
        score = row.get("score")
        sample_id = row.get("sample_id")

        if row_dataset != dataset_id:
            continue
        if row_partition != partition:
            continue
        if detector_run_id not in baseline_thresholds:
            continue
        if label not in {"human", "ai"}:
            continue
        if not isinstance(source_split, str):
            continue
        if not isinstance(score, (int, float)):
            continue
        if not isinstance(sample_id, str):
            continue
        score_value = float(score)
        if not math.isfinite(score_value):
            continue

        augmentation = ""
        split_kind = ""
        if source_split == baseline_split:
            augmentation = "baseline"
            split_kind = "baseline"
        else:
            match = pattern.match(source_split)
            if match is None:
                continue
            augmentation = str(match.group(1))
            split_kind = "orig_h_aug_ai"

        threshold = baseline_thresholds[str(detector_run_id)]
        records.append(
            {
                "detector_run_id": str(detector_run_id),
                "source_split": source_split,
                "augmentation": augmentation,
                "split_kind": split_kind,
                "label": str(label),
                "sample_id": sample_id,
                "score": score_value,
                "above_baseline_threshold": score_value >= threshold,
                "baseline_threshold": threshold,
            }
        )

    frame = pd.DataFrame(records)
    if frame.empty:
        raise ValueError("No score rows found for baseline/orig_h_aug_ai filter.")

    if frame[frame["split_kind"] == "baseline"].empty:
        raise ValueError("Missing baseline rows in filtered score frame.")
    if frame[frame["split_kind"] == "orig_h_aug_ai"].empty:
        raise ValueError("Missing orig_h_aug_ai rows in filtered score frame.")
    return frame


def _build_score_summary(*, score_frame: pd.DataFrame) -> pd.DataFrame:
    """Aggregate score statistics per detector/split/label."""

    summary = (
        score_frame.groupby(
            ["detector_run_id", "source_split", "augmentation", "split_kind", "label"],
            as_index=False,
        )
        .agg(
            sample_count=("score", "size"),
            mean_score=("score", "mean"),
            std_score=("score", "std"),
            median_score=("score", "median"),
            q25_score=("score", lambda values: float(np.quantile(values.to_numpy(dtype=float), 0.25))),
            q75_score=("score", lambda values: float(np.quantile(values.to_numpy(dtype=float), 0.75))),
            above_baseline_threshold_rate=("above_baseline_threshold", "mean"),
        )
    )
    return summary.sort_values(
        by=["detector_run_id", "label", "split_kind", "augmentation"],
        kind="mergesort",
    ).reset_index(drop=True)


def _build_shift_tables(*, score_summary: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build delta tables against baseline for AI and human labels."""

    ai_shift = _build_shift_table_for_label(score_summary=score_summary, label="ai")
    human_shift = _build_shift_table_for_label(score_summary=score_summary, label="human")
    return ai_shift, human_shift


def _build_shift_table_for_label(*, score_summary: pd.DataFrame, label: str) -> pd.DataFrame:
    """Build one label-specific delta table."""

    label_frame = score_summary[score_summary["label"] == label].copy()
    baseline = label_frame[label_frame["split_kind"] == "baseline"][
        [
            "detector_run_id",
            "mean_score",
            "median_score",
            "above_baseline_threshold_rate",
        ]
    ].rename(
        columns={
            "mean_score": "baseline_mean_score",
            "median_score": "baseline_median_score",
            "above_baseline_threshold_rate": "baseline_above_threshold_rate",
        }
    )
    augmented = label_frame[label_frame["split_kind"] == "orig_h_aug_ai"][
        [
            "detector_run_id",
            "source_split",
            "augmentation",
            "sample_count",
            "mean_score",
            "median_score",
            "std_score",
            "q25_score",
            "q75_score",
            "above_baseline_threshold_rate",
        ]
    ].rename(
        columns={
            "mean_score": "augmented_mean_score",
            "median_score": "augmented_median_score",
            "above_baseline_threshold_rate": "augmented_above_threshold_rate",
        }
    )

    merged = augmented.merge(
        baseline,
        on=["detector_run_id"],
        how="left",
        validate="many_to_one",
    )
    merged["delta_mean_score"] = merged["augmented_mean_score"] - merged["baseline_mean_score"]
    merged["delta_median_score"] = merged["augmented_median_score"] - merged["baseline_median_score"]
    merged["delta_above_threshold_rate"] = (
        merged["augmented_above_threshold_rate"] - merged["baseline_above_threshold_rate"]
    )
    merged["label"] = label
    return merged.sort_values(by=["detector_run_id", "augmentation"], kind="mergesort").reset_index(drop=True)


def _plot_ai_delta_heatmap(*, ai_shift: pd.DataFrame, output_path: Path) -> None:
    """Plot augmentation x detector heatmap for AI mean score deltas."""

    pivot = ai_shift.pivot_table(
        index="augmentation",
        columns="detector_run_id",
        values="delta_mean_score",
        aggfunc="mean",
    )
    _plot_centered_heatmap(
        matrix=pivot.to_numpy(dtype=float),
        x_labels=[str(value) for value in pivot.columns.tolist()],
        y_labels=[str(value) for value in pivot.index.tolist()],
        title="AI Mean Score Delta vs Baseline (orig_h + aug_ai)",
        output_path=output_path,
    )


def _plot_ai_detection_rate_delta_heatmap(*, ai_shift: pd.DataFrame, output_path: Path) -> None:
    """Plot augmentation x detector heatmap for AI above-threshold rate deltas."""

    pivot = ai_shift.pivot_table(
        index="augmentation",
        columns="detector_run_id",
        values="delta_above_threshold_rate",
        aggfunc="mean",
    )
    _plot_centered_heatmap(
        matrix=pivot.to_numpy(dtype=float),
        x_labels=[str(value) for value in pivot.columns.tolist()],
        y_labels=[str(value) for value in pivot.index.tolist()],
        title="AI Detection-Rate Delta vs Baseline Threshold (orig_h + aug_ai)",
        output_path=output_path,
    )


def _plot_human_delta_heatmap(*, human_shift: pd.DataFrame, output_path: Path) -> None:
    """Plot augmentation x detector heatmap for human mean score deltas."""

    pivot = human_shift.pivot_table(
        index="augmentation",
        columns="detector_run_id",
        values="delta_mean_score",
        aggfunc="mean",
    )
    _plot_centered_heatmap(
        matrix=pivot.to_numpy(dtype=float),
        x_labels=[str(value) for value in pivot.columns.tolist()],
        y_labels=[str(value) for value in pivot.index.tolist()],
        title="Human Mean Score Delta vs Baseline (aug_h + orig_ai)",
        output_path=output_path,
    )


def _plot_human_detection_rate_delta_heatmap(*, human_shift: pd.DataFrame, output_path: Path) -> None:
    """Plot augmentation x detector heatmap for human above-threshold rate deltas."""

    pivot = human_shift.pivot_table(
        index="augmentation",
        columns="detector_run_id",
        values="delta_above_threshold_rate",
        aggfunc="mean",
    )
    _plot_centered_heatmap(
        matrix=pivot.to_numpy(dtype=float),
        x_labels=[str(value) for value in pivot.columns.tolist()],
        y_labels=[str(value) for value in pivot.index.tolist()],
        title="Human Above-Threshold Delta vs Baseline Threshold (aug_h + orig_ai)",
        output_path=output_path,
    )


def _plot_per_detector_ai_bars(*, ai_shift: pd.DataFrame, output_dir: Path) -> None:
    """Plot per-detector bars for baseline and augmented AI mean scores."""

    detectors = sorted(set(str(value) for value in ai_shift["detector_run_id"]))
    for detector in detectors:
        subset = ai_shift[ai_shift["detector_run_id"] == detector].copy()
        subset = subset.sort_values(by="augmentation", kind="mergesort")
        baseline_mean = float(subset["baseline_mean_score"].iloc[0])

        labels = ["baseline"] + ordered_augmentations([str(value) for value in subset["augmentation"].tolist()])
        values_by_augmentation = {
            str(augmentation): float(value)
            for augmentation, value in subset[["augmentation", "augmented_mean_score"]].itertuples(index=False)
        }
        values = [baseline_mean] + [float(value) for value in subset["augmented_mean_score"].tolist()]
        values = [baseline_mean] + [values_by_augmentation[label] for label in labels[1:]]
        colors = ["#9D9D9D"] + [augmentation_color(label) for label in labels[1:]]

        fig, ax = plt.subplots(figsize=(max(8.0, len(labels) * 1.0), 4.8))
        x = np.arange(len(labels), dtype=float)
        ax.bar(x, values, color=colors)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=20, ha="right")
        ax.set_ylim(0.0, 1.0)
        ax.set_ylabel("Mean AI score")
        ax.set_title(f"Mean AI confidence by augmentation: {detector}")
        fig.tight_layout()

        output_path = output_dir / f"ai_mean_score_bars_{_slugify(detector)}.png"
        fig.savefig(output_path, dpi=180, bbox_inches="tight")
        plt.close(fig)


def _plot_per_detector_human_bars(*, human_shift: pd.DataFrame, output_dir: Path) -> None:
    """Plot per-detector bars for baseline and augmented human mean scores."""

    detectors = sorted(set(str(value) for value in human_shift["detector_run_id"]))
    for detector in detectors:
        subset = human_shift[human_shift["detector_run_id"] == detector].copy()
        subset = subset.sort_values(by="augmentation", kind="mergesort")
        baseline_mean = float(subset["baseline_mean_score"].iloc[0])

        labels = ["baseline"] + ordered_augmentations([str(value) for value in subset["augmentation"].tolist()])
        values_by_augmentation = {
            str(augmentation): float(value)
            for augmentation, value in subset[["augmentation", "augmented_mean_score"]].itertuples(index=False)
        }
        values = [baseline_mean] + [values_by_augmentation[label] for label in labels[1:]]
        colors = ["#9D9D9D"] + [augmentation_color(label) for label in labels[1:]]

        fig, ax = plt.subplots(figsize=(max(8.0, len(labels) * 1.0), 4.8))
        x = np.arange(len(labels), dtype=float)
        ax.bar(x, values, color=colors)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=20, ha="right")
        ax.set_ylim(0.0, 1.0)
        ax.set_ylabel("Mean human score")
        ax.set_title(f"Mean human-as-AI score by augmentation: {detector}")
        fig.tight_layout()

        output_path = output_dir / f"human_mean_score_bars_{_slugify(detector)}.png"
        fig.savefig(output_path, dpi=180, bbox_inches="tight")
        plt.close(fig)


def _plot_per_detector_ai_histograms(
    *,
    score_frame: pd.DataFrame,
    baseline_thresholds: Mapping[str, float],
    output_dir: Path,
) -> None:
    """Plot baseline vs augmented AI-score histograms per detector."""

    detectors = sorted(set(str(value) for value in score_frame["detector_run_id"]))
    for detector in detectors:
        subset = score_frame[
            (score_frame["detector_run_id"] == detector)
            & (score_frame["label"] == "ai")
        ].copy()
        baseline_scores = subset[subset["split_kind"] == "baseline"]["score"].to_numpy(dtype=float)
        augments = ordered_augmentations(
            [str(value) for value in subset[subset["split_kind"] == "orig_h_aug_ai"]["augmentation"]]
        )
        if baseline_scores.size == 0 or not augments:
            continue

        ncols = 3
        nrows = int(math.ceil(len(augments) / float(ncols)))
        fig, axes = plt.subplots(
            nrows=nrows,
            ncols=ncols,
            figsize=(ncols * 4.5, max(3.0, nrows * 3.2)),
            squeeze=False,
        )
        threshold = float(baseline_thresholds[detector])

        for index, augmentation in enumerate(augments):
            row_idx = index // ncols
            col_idx = index % ncols
            ax = axes[row_idx, col_idx]
            augmentation_scores = subset[
                (subset["split_kind"] == "orig_h_aug_ai")
                & (subset["augmentation"] == augmentation)
            ]["score"].to_numpy(dtype=float)

            ax.hist(
                baseline_scores,
                bins=20,
                density=True,
                histtype="step",
                linewidth=1.6,
                label="baseline",
                color="#9D9D9D",
            )
            ax.hist(
                augmentation_scores,
                bins=20,
                density=True,
                histtype="step",
                linewidth=1.6,
                label=augmentation,
                color=augmentation_color(augmentation),
            )
            ax.axvline(threshold, linestyle="--", linewidth=1.0, label="baseline threshold")
            ax.set_title(augmentation)
            ax.set_xlim(0.0, 1.0)
            ax.set_xlabel("AI score")
            ax.set_ylabel("Density")
            ax.legend(fontsize=7)

        total_plots = nrows * ncols
        for index in range(len(augments), total_plots):
            row_idx = index // ncols
            col_idx = index % ncols
            axes[row_idx, col_idx].axis("off")

        fig.suptitle(f"AI score distributions: baseline vs orig_h+aug_ai ({detector})")
        fig.tight_layout(rect=(0, 0, 1, 0.96))
        output_path = output_dir / f"ai_score_hist_baseline_vs_aug_{_slugify(detector)}.png"
        fig.savefig(output_path, dpi=180, bbox_inches="tight")
        plt.close(fig)


def _plot_per_detector_human_histograms(
    *,
    score_frame: pd.DataFrame,
    baseline_thresholds: Mapping[str, float],
    output_dir: Path,
) -> None:
    """Plot baseline vs augmented human-score histograms per detector."""

    detectors = sorted(set(str(value) for value in score_frame["detector_run_id"]))
    for detector in detectors:
        subset = score_frame[
            (score_frame["detector_run_id"] == detector)
            & (score_frame["label"] == "human")
        ].copy()
        baseline_scores = subset[subset["split_kind"] == "baseline"]["score"].to_numpy(dtype=float)
        augments = ordered_augmentations(
            [str(value) for value in subset[subset["split_kind"] == "orig_h_aug_ai"]["augmentation"]]
        )
        if baseline_scores.size == 0 or not augments:
            continue

        ncols = 3
        nrows = int(math.ceil(len(augments) / float(ncols)))
        fig, axes = plt.subplots(
            nrows=nrows,
            ncols=ncols,
            figsize=(ncols * 4.5, max(3.0, nrows * 3.2)),
            squeeze=False,
        )
        threshold = float(baseline_thresholds[detector])

        for index, augmentation in enumerate(augments):
            row_idx = index // ncols
            col_idx = index % ncols
            ax = axes[row_idx, col_idx]
            augmentation_scores = subset[
                (subset["split_kind"] == "orig_h_aug_ai")
                & (subset["augmentation"] == augmentation)
            ]["score"].to_numpy(dtype=float)

            ax.hist(
                baseline_scores,
                bins=20,
                density=True,
                histtype="step",
                linewidth=1.6,
                label="baseline",
                color="#9D9D9D",
            )
            ax.hist(
                augmentation_scores,
                bins=20,
                density=True,
                histtype="step",
                linewidth=1.6,
                label=augmentation,
                color=augmentation_color(augmentation),
            )
            ax.axvline(threshold, linestyle="--", linewidth=1.0, label="baseline threshold")
            ax.set_title(augmentation)
            ax.set_xlim(0.0, 1.0)
            ax.set_xlabel("Human score")
            ax.set_ylabel("Density")
            ax.legend(fontsize=7)

        total_plots = nrows * ncols
        for index in range(len(augments), total_plots):
            row_idx = index // ncols
            col_idx = index % ncols
            axes[row_idx, col_idx].axis("off")

        fig.suptitle(f"Human score distributions: baseline vs aug_h+orig_ai ({detector})")
        fig.tight_layout(rect=(0, 0, 1, 0.96))
        output_path = output_dir / f"human_score_hist_baseline_vs_aug_{_slugify(detector)}.png"
        fig.savefig(output_path, dpi=180, bbox_inches="tight")
        plt.close(fig)


def _plot_per_detector_rowwise_thin_bars(
    *,
    score_frame: pd.DataFrame,
    baseline_split: str,
    baseline_thresholds: Mapping[str, float],
    output_dir: Path,
    max_samples_per_label: int,
) -> None:
    """Plot per-detector row panels with thin bars for baseline and each augmentation."""

    color_map = {
        "TN": "#4c78a8",
        "TP": "#54a24b",
        "FP": "#f58518",
        "FN": "#e45756",
    }
    detectors = sorted(set(str(value) for value in score_frame["detector_run_id"]))
    for detector in detectors:
        detector_scores = score_frame[score_frame["detector_run_id"] == detector].copy()
        augmented_rows = detector_scores[detector_scores["split_kind"] == "orig_h_aug_ai"][
            ["source_split", "augmentation"]
        ].drop_duplicates()
        augmented_rows = augmented_rows.sort_values(by=["augmentation", "source_split"], kind="mergesort")
        row_specs: list[tuple[str, str]] = [(baseline_split, "baseline")]
        for source_split, augmentation in augmented_rows.itertuples(index=False):
            row_specs.append((str(source_split), str(augmentation)))

        fig, axes = plt.subplots(
            nrows=len(row_specs),
            ncols=1,
            figsize=(18.0, max(4.0, 2.3 * len(row_specs))),
            sharex=True,
            squeeze=False,
        )
        threshold = float(baseline_thresholds[detector])

        used_count = 0
        for row_index, (source_split, row_label) in enumerate(row_specs):
            ax = axes[row_index, 0]
            row_frame = detector_scores[detector_scores["source_split"] == source_split].copy()
            human_scores = row_frame[row_frame["label"] == "human"][["sample_id", "score"]].copy()
            ai_scores = row_frame[row_frame["label"] == "ai"][["sample_id", "score"]].copy()
            if human_scores.empty or ai_scores.empty:
                ax.axis("off")
                continue

            human_scores = human_scores.sort_values(by=["score", "sample_id"], kind="mergesort")
            ai_scores = ai_scores.sort_values(by=["score", "sample_id"], kind="mergesort")
            selected = min(max_samples_per_label, len(human_scores), len(ai_scores))
            if selected <= 0:
                ax.axis("off")
                continue
            used_count = selected
            human_scores = human_scores.head(selected)
            ai_scores = ai_scores.head(selected)

            combined = pd.concat(
                [
                    human_scores.assign(label="human"),
                    ai_scores.assign(label="ai"),
                ],
                axis=0,
                ignore_index=True,
            )

            x_values = np.arange(len(combined), dtype=float)
            scores = combined["score"].to_numpy(dtype=float)
            labels = combined["label"].tolist()

            colors: list[str] = []
            tp = fp = tn = fn = 0
            for label_value, score_value in zip(labels, scores, strict=True):
                predicted_ai = bool(score_value >= threshold)
                if label_value == "ai" and predicted_ai:
                    colors.append(color_map["TP"])
                    tp += 1
                elif label_value == "ai" and not predicted_ai:
                    colors.append(color_map["FN"])
                    fn += 1
                elif label_value == "human" and predicted_ai:
                    colors.append(color_map["FP"])
                    fp += 1
                else:
                    colors.append(color_map["TN"])
                    tn += 1

            ax.bar(x_values, scores, width=0.85, color=colors, linewidth=0)
            ax.axhline(threshold, linestyle="--", linewidth=1.0, color="black")
            ax.axvline(selected - 0.5, linestyle="-", linewidth=0.8, color="#555555")
            ax.axvspan(-0.5, selected - 0.5, color="#f3f3f3", alpha=0.35)
            ax.axvspan(selected - 0.5, (2 * selected) - 0.5, color="#eaf4ff", alpha=0.20)

            # Draw thicker separators between correct/incorrect regions per label block:
            # human block: TN (below threshold) | FP (above threshold)
            # ai block: FN (below threshold) | TP (above threshold)
            human_values = human_scores["score"].to_numpy(dtype=float)
            ai_values = ai_scores["score"].to_numpy(dtype=float)

            human_predicted_ai = np.where(human_values >= threshold)[0]
            if human_predicted_ai.size > 0:
                first_fp_idx = int(human_predicted_ai[0])
                if 0 < first_fp_idx < selected:
                    ax.axvline(
                        first_fp_idx - 0.5,
                        linestyle="-",
                        linewidth=2.2,
                        color="#1f1f1f",
                        alpha=0.95,
                    )

            ai_predicted_ai = np.where(ai_values >= threshold)[0]
            if ai_predicted_ai.size > 0:
                first_tp_idx = int(ai_predicted_ai[0])
                if 0 < first_tp_idx < selected:
                    ax.axvline(
                        selected + first_tp_idx - 0.5,
                        linestyle="-",
                        linewidth=2.2,
                        color="#1f1f1f",
                        alpha=0.95,
                    )

            ax.set_ylim(0.0, 1.0)
            ax.set_ylabel("score")
            ax.set_title(
                f"{row_label} | split={source_split} | n_h={selected} n_ai={selected} | TP={tp} FP={fp} TN={tn} FN={fn}",
                fontsize=9,
            )

        if used_count > 0:
            axes[-1, 0].set_xlim(-0.5, (2 * used_count) - 0.5)
            axes[-1, 0].set_xticks([used_count / 2.0 - 0.5, used_count + used_count / 2.0 - 0.5])
            axes[-1, 0].set_xticklabels([f"human ({used_count})", f"ai ({used_count})"])

        legend_handles = [
            Patch(facecolor=color_map["TN"], label="TN"),
            Patch(facecolor=color_map["FP"], label="FP"),
            Patch(facecolor=color_map["FN"], label="FN"),
            Patch(facecolor=color_map["TP"], label="TP"),
            Line2D([0], [0], color="black", linestyle="--", linewidth=1.0, label="baseline threshold"),
        ]
        fig.legend(handles=legend_handles, loc="upper right")
        fig.suptitle(
            f"Baseline vs orig_h+aug_ai thin bars ({detector})",
            fontsize=12,
        )
        fig.tight_layout(rect=(0, 0, 1, 0.97))
        output_path = output_dir / f"thin_bars_baseline_vs_aug_{_slugify(detector)}.png"
        fig.savefig(output_path, dpi=180, bbox_inches="tight")
        plt.close(fig)


def _plot_per_detector_rowwise_thin_bars_combined(
    *,
    primary_score_frame: pd.DataFrame,
    augmented_human_score_frame: pd.DataFrame,
    baseline_split: str,
    baseline_thresholds: Mapping[str, float],
    output_dir: Path,
    max_samples_per_label: int,
) -> None:
    """Plot one row per method with separate human/ai panels and baseline ghost curves."""

    detectors = sorted(
        set(str(value) for value in primary_score_frame["detector_run_id"])
        & set(str(value) for value in augmented_human_score_frame["detector_run_id"])
    )
    for detector in detectors:
        primary_scores = primary_score_frame[primary_score_frame["detector_run_id"] == detector].copy()
        human_scores_frame = augmented_human_score_frame[
            augmented_human_score_frame["detector_run_id"] == detector
        ].copy()

        aug_ai_rows = primary_scores[primary_scores["split_kind"] == "orig_h_aug_ai"][
            ["source_split", "augmentation"]
        ].drop_duplicates()
        aug_ai_rows = aug_ai_rows.sort_values(by=["augmentation", "source_split"], kind="mergesort")

        aug_h_rows = human_scores_frame[human_scores_frame["split_kind"] == "orig_h_aug_ai"][
            ["source_split", "augmentation"]
        ].drop_duplicates()
        aug_h_rows = aug_h_rows.sort_values(by=["augmentation", "source_split"], kind="mergesort")

        ai_split_by_augmentation = {
            str(augmentation): str(source_split)
            for source_split, augmentation in aug_ai_rows.itertuples(index=False)
        }
        human_split_by_augmentation = {
            str(augmentation): str(source_split)
            for source_split, augmentation in aug_h_rows.itertuples(index=False)
        }
        augmentation_names = ordered_augmentations(
            [name for name in ai_split_by_augmentation.keys() if name in human_split_by_augmentation]
        )
        row_specs: list[tuple[str, str]] = [("baseline", baseline_split)]
        for augmentation in augmentation_names:
            row_specs.append((augmentation, augmentation))

        n_panels = len(row_specs)
        ncols = 2
        nrows = n_panels
        fig, axes = plt.subplots(
            nrows=nrows,
            ncols=ncols,
            figsize=(16.5, max(9.5, 2.65 * nrows)),
            sharey=True,
            squeeze=False,
        )
        threshold = float(baseline_thresholds[detector])

        baseline_frame = primary_scores[primary_scores["source_split"] == baseline_split].copy()
        baseline_human_all = baseline_frame[baseline_frame["label"] == "human"]["score"].to_numpy(dtype=float)
        baseline_ai_all = baseline_frame[baseline_frame["label"] == "ai"]["score"].to_numpy(dtype=float)
        for row_i, (row_name, row_key) in enumerate(row_specs):
            if row_name == "baseline":
                row_human_all = baseline_human_all
                row_ai_all = baseline_ai_all
                method_label = "BASELINE"
                base_color = augmentation_color("baseline")
                show_ghost = False
            else:
                augmentation = row_key
                ai_split = ai_split_by_augmentation[augmentation]
                human_split = human_split_by_augmentation[augmentation]
                row_ai_all = primary_scores[
                    (primary_scores["source_split"] == ai_split) & (primary_scores["label"] == "ai")
                ]["score"].to_numpy(dtype=float)
                row_human_all = human_scores_frame[
                    (human_scores_frame["source_split"] == human_split) & (human_scores_frame["label"] == "human")
                ]["score"].to_numpy(dtype=float)
                method_label = augmentation.upper()
                base_color = augmentation_color(augmentation)
                show_ghost = True

            selected = min(
                max_samples_per_label,
                len(row_human_all),
                len(row_ai_all),
                len(baseline_human_all),
                len(baseline_ai_all),
            )
            ax_human = axes[row_i, 0]
            ax_ai = axes[row_i, 1]
            if selected <= 0:
                ax_human.axis("off")
                ax_ai.axis("off")
                continue

            row_human = np.sort(row_human_all)[:selected]
            row_ai = np.sort(row_ai_all)[::-1][:selected]
            baseline_human = np.sort(baseline_human_all)[:selected]
            baseline_ai = np.sort(baseline_ai_all)[::-1][:selected]
            percentile_axis = np.linspace(0.0, 100.0, num=selected, dtype=float)

            _plot_single_label_curve_panel(
                ax=ax_human,
                values=row_human,
                baseline_values=baseline_human,
                threshold=threshold,
                positive_when_above=False,
                color=base_color,
                show_ghost=show_ghost,
                show_xlabel=(row_i == nrows - 1),
                class_count_text=("FP", "TN"),
            )
            _plot_single_label_curve_panel(
                ax=ax_ai,
                values=row_ai,
                baseline_values=baseline_ai,
                threshold=threshold,
                positive_when_above=True,
                color=base_color,
                show_ghost=show_ghost,
                show_xlabel=(row_i == nrows - 1),
                class_count_text=("FN", "TP"),
            )
            ax_human.set_title(method_label, fontsize=15, fontweight="bold", color=base_color, pad=8, loc="left")
            ax_ai.set_title("", fontsize=15, pad=8)
            ax_human.set_xlim(0.0, 100.0)
            ax_ai.set_xlim(0.0, 100.0)

        emoji_font = _resolve_emoji_font_properties()
        fig.text(0.27, 0.962, "HUMAN", ha="center", va="center", fontsize=18, fontweight="bold")
        fig.text(0.74, 0.962, "AI", ha="center", va="center", fontsize=18, fontweight="bold")
        if emoji_font is not None:
            fig.text(0.35, 0.962, "👨", ha="center", va="center", fontsize=18, fontproperties=emoji_font)
            fig.text(0.80, 0.962, "🤖", ha="center", va="center", fontsize=18, fontproperties=emoji_font)
        fig.suptitle(detector, fontsize=24, fontweight="bold", y=0.995)
        fig.tight_layout(rect=(0, 0, 1, 0.955))
        output_path = output_dir / f"thin_bars_combined_baseline_vs_aug_both_{_slugify(detector)}.png"
        fig.savefig(output_path, dpi=180, bbox_inches="tight")
        plt.close(fig)


def _plot_single_label_curve_panel(
    *,
    ax: Any,
    values: np.ndarray,
    baseline_values: np.ndarray,
    threshold: float,
    positive_when_above: bool,
    color: str,
    show_ghost: bool,
    show_xlabel: bool,
    class_count_text: tuple[str, str],
) -> None:
    """Plot one label panel with color shift at prediction transition and optional baseline ghost."""

    display_floor = 0.010
    y_values = np.maximum(values, display_floor)
    baseline_display = np.maximum(baseline_values, display_floor)
    percentile_axis = np.linspace(0.0, 100.0, num=len(values), dtype=float)

    if show_ghost:
        ax.plot(
            percentile_axis,
            baseline_display,
            linestyle="--",
            linewidth=1.9,
            color="#9A9A9A",
            alpha=0.45,
            label="baseline ghost",
            zorder=1,
        )

    correct_mask = (values >= threshold) if positive_when_above else (values < threshold)
    transition_index = _first_boolean_change_index(correct_mask)
    correct_color = "#2E9F4D"
    wrong_color = "#D94A3A"

    if transition_index is None:
        segment_color = correct_color if bool(correct_mask[0]) else wrong_color
        ax.plot(percentile_axis, y_values, color=segment_color, linewidth=2.8, zorder=3)
    else:
        left_slice = slice(0, transition_index)
        right_slice = slice(transition_index, len(values))
        left_color = correct_color if bool(correct_mask[0]) else wrong_color
        right_color = correct_color if bool(correct_mask[transition_index]) else wrong_color
        ax.plot(percentile_axis[left_slice], y_values[left_slice], color=left_color, linewidth=2.8, zorder=3)
        ax.plot(percentile_axis[right_slice], y_values[right_slice], color=right_color, linewidth=2.8, zorder=3)

        shift_x = float(percentile_axis[transition_index])
        shift_y = float(y_values[transition_index])
        ax.axvline(shift_x, linestyle="-", linewidth=2.2, color="#111111", alpha=0.98, zorder=2)
        ax.scatter([shift_x], [shift_y], s=28, color="#111111", zorder=4)

    ax.fill_between(percentile_axis, display_floor, y_values, color=_lighten_hex(color, fraction=0.55), alpha=0.20, zorder=0)

    wrong_count = int(np.sum(~correct_mask))
    correct_count = int(np.sum(correct_mask))
    correct_pct = 100.0 * (float(correct_count) / float(len(values)))
    wrong_label, correct_label = class_count_text

    ax.axhline(threshold, linestyle="-", linewidth=2.8, color="#161616", alpha=0.98)
    ax.text(
        0.50,
        1.02,
        f"{correct_pct:.1f}% Correct",
        fontsize=16.5,
        fontweight="bold",
        color=correct_color,
        transform=ax.transAxes,
        ha="center",
        va="bottom",
        bbox={"facecolor": "white", "alpha": 0.90, "edgecolor": "none", "pad": 2.2},
        clip_on=False,
    )
    ax.text(
        0.985,
        0.88,
        f"{wrong_label}={wrong_count}  {correct_label}={correct_count}",
        fontsize=14.0,
        fontweight="bold",
        color="#1A1A1A",
        transform=ax.transAxes,
        ha="right",
        va="top",
        bbox={"facecolor": "white", "alpha": 0.86, "edgecolor": "#D7D7D7", "pad": 2.3},
    )

    ax.set_ylim(0.0, 1.0)
    ax.grid(alpha=0.20)
    if show_xlabel:
        ax.set_xlabel("sorted percentile")
    else:
        ax.set_xlabel("")
    ax.set_ylabel("AI score")


def _first_boolean_change_index(mask: np.ndarray) -> int | None:
    """Return first index where boolean array changes value, else None."""

    if mask.size <= 1:
        return None
    initial = bool(mask[0])
    differing = np.where(mask != initial)[0]
    if differing.size == 0:
        return None
    return int(differing[0])


def _resolve_emoji_font_properties() -> FontProperties | None:
    """Resolve emoji-capable font properties if available on the system."""

    cached = getattr(_resolve_emoji_font_properties, "_cached", None)
    if cached is not None:
        return cached if isinstance(cached, FontProperties) else None

    local_candidates = (
        Path.home() / ".local" / "share" / "fonts" / "NotoColorEmoji.ttf",
        Path("/usr/share/fonts/truetype/noto/NotoColorEmoji.ttf"),
    )
    for font_path in local_candidates:
        if font_path.exists():
            try:
                font_manager.fontManager.addfont(str(font_path))
            except (RuntimeError, ValueError):
                continue
            props = FontProperties(fname=str(font_path))
            setattr(_resolve_emoji_font_properties, "_cached", props)
            return props

    preferred_names = ("Noto Color Emoji", "Noto Emoji", "Symbola", "Segoe UI Emoji")
    for entry in font_manager.fontManager.ttflist:
        if entry.name in preferred_names:
            props = FontProperties(fname=entry.fname)
            setattr(_resolve_emoji_font_properties, "_cached", props)
            return props

    setattr(_resolve_emoji_font_properties, "_cached", False)
    return None


def _lighten_hex(color_hex: str, *, fraction: float) -> str:
    """Lighten HEX color toward white by `fraction` in [0, 1]."""

    value = color_hex.lstrip("#")
    if len(value) != 6:
        return color_hex
    red = int(value[0:2], 16)
    green = int(value[2:4], 16)
    blue = int(value[4:6], 16)
    ratio = min(max(fraction, 0.0), 1.0)
    red = int(round(red + (255 - red) * ratio))
    green = int(round(green + (255 - green) * ratio))
    blue = int(round(blue + (255 - blue) * ratio))
    return f"#{red:02x}{green:02x}{blue:02x}"


def _plot_centered_heatmap(
    *,
    matrix: np.ndarray,
    x_labels: Sequence[str],
    y_labels: Sequence[str],
    title: str,
    output_path: Path,
) -> None:
    """Plot annotated zero-centered heatmap."""

    finite_values = matrix[np.isfinite(matrix)]
    if finite_values.size == 0:
        finite_values = np.array([0.0], dtype=float)
    vmax = float(np.max(np.abs(finite_values)))
    if not math.isfinite(vmax) or vmax <= 0.0:
        vmax = 1e-9

    fig, ax = plt.subplots(figsize=(max(7.0, len(x_labels) * 1.0), max(4.0, len(y_labels) * 0.6 + 1.2)))
    image = ax.imshow(matrix, aspect="auto", cmap="coolwarm", vmin=-vmax, vmax=vmax)
    ax.set_xticks(np.arange(len(x_labels)))
    ax.set_xticklabels(list(x_labels), rotation=20, ha="right")
    ax.set_yticks(np.arange(len(y_labels)))
    ax.set_yticklabels(list(y_labels))
    ax.set_title(title)

    for row_idx in range(matrix.shape[0]):
        for col_idx in range(matrix.shape[1]):
            value = matrix[row_idx, col_idx]
            label = "nan" if not math.isfinite(value) else f"{value:.3f}"
            ax.text(col_idx, row_idx, label, ha="center", va="center", fontsize=8)

    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _build_markdown_summary(*, ai_shift: pd.DataFrame, human_shift: pd.DataFrame) -> str:
    """Build markdown summary with key shift tables."""

    lines: list[str] = []
    lines.append("# Score Shift Summary")
    lines.append("")

    lines.append("## AI Score Shift (orig_h + aug_ai vs baseline)")
    lines.append("")
    lines.append(_dataframe_to_markdown(ai_shift))
    lines.append("")

    lines.append("## Human Score Shift (orig_h + aug_ai vs baseline)")
    lines.append("")
    lines.append(_dataframe_to_markdown(human_shift))
    lines.append("")

    lines.append("## Per-Detector Strongest AI Confidence Drop")
    lines.append("")
    strongest = (
        ai_shift.sort_values(by=["detector_run_id", "delta_mean_score"], ascending=[True, True], kind="mergesort")
        .groupby("detector_run_id", as_index=False)
        .first()
    )
    lines.append(_dataframe_to_markdown(strongest))
    lines.append("")
    return "\n".join(lines)


def _dataframe_to_markdown(frame: pd.DataFrame, *, float_precision: int = 4) -> str:
    """Render dataframe as markdown without optional dependencies."""

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


def _slugify(value: str) -> str:
    """Create safe filename token from arbitrary string."""

    normalized = value.strip().lower().replace(":", "_").replace("/", "_").replace(" ", "_")
    return "".join(character for character in normalized if character.isalnum() or character in {"_", "-"})


if __name__ == "__main__":
    raise SystemExit(main())
