"""Compute WER/CER text-change summaries for augmented HC3 scenarios."""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import matplotlib
import numpy as np
import pandas as pd
import seaborn as sns
from jiwer import cer, wer
from matplotlib.colors import to_rgba

from apm.experiments.augmentation_palette import augmentation_color, ordered_augmentations

matplotlib.use("Agg")
import matplotlib.pyplot as plt

sns.set_theme(style="whitegrid", context="talk")


@dataclass(frozen=True, slots=True)
class AugmentedHc3JiwerReportRequest:
    """Configuration for augmented HC3 WER/CER analysis."""

    project_root: Path
    run_dir: Path
    splits_root: Path
    output_dir: Path
    dataset_id: str
    baseline_split: str
    orig_h_aug_ai_suffix: str
    aug_h_orig_ai_suffix: str
    partition: str


@dataclass(frozen=True, slots=True)
class AugmentedHc3JiwerReportResult:
    """Output file pointers for WER/CER analysis."""

    output_dir: Path
    sample_metrics_csv: Path
    summary_by_label_csv: Path
    summary_overall_csv: Path
    markdown_summary: Path
    wer_plot: Path
    cer_plot: Path


@dataclass(frozen=True, slots=True)
class ScenarioRow:
    """One scenario split definition."""

    source_group: str
    split_name: str
    display_label: str
    augmentation: str


def run_augmented_hc3_jiwer_report(
    request: AugmentedHc3JiwerReportRequest,
) -> AugmentedHc3JiwerReportResult:
    """Run WER/CER computation and plot generation."""

    _validate_request(request)
    request.output_dir.mkdir(parents=True, exist_ok=True)

    local_rows = _read_jsonl(request.run_dir / "local_results.jsonl")
    scenario_rows = _build_scenario_rows(
        local_rows=local_rows,
        dataset_id=request.dataset_id,
        baseline_split=request.baseline_split,
        orig_h_aug_ai_suffix=request.orig_h_aug_ai_suffix,
        aug_h_orig_ai_suffix=request.aug_h_orig_ai_suffix,
    )
    texts = _load_text_tables(
        splits_root=request.splits_root,
        scenario_rows=scenario_rows,
        partition=request.partition,
    )
    modified_rows = _build_modified_text_pairs(
        scenario_rows=scenario_rows,
        texts=texts,
    )
    metrics_frame = _compute_metrics_frame(modified_rows=modified_rows)
    summary_by_label = _build_summary_by_label(metrics_frame=metrics_frame)
    summary_overall = _build_summary_overall(metrics_frame=metrics_frame)

    sample_metrics_csv = request.output_dir / "sample_edit_metrics.csv"
    summary_by_label_csv = request.output_dir / "summary_by_label.csv"
    summary_overall_csv = request.output_dir / "summary_overall.csv"
    markdown_summary = request.output_dir / "JIWER_SUMMARY.md"
    wer_plot = request.output_dir / "wer_bars.png"
    cer_plot = request.output_dir / "cer_bars.png"

    _remove_legacy_plots(output_dir=request.output_dir)

    metrics_frame.to_csv(sample_metrics_csv, index=False)
    summary_by_label.to_csv(summary_by_label_csv, index=False)
    summary_overall.to_csv(summary_overall_csv, index=False)
    markdown_summary.write_text(
        _build_markdown_summary(
            summary_by_label=summary_by_label,
            summary_overall=summary_overall,
        ),
        encoding="utf-8",
    )

    _plot_metric_grouped_bars(
        summary_by_label=summary_by_label,
        metric_prefix="wer",
        title="WER by Augmentation (AI vs Human modifications)",
        output_path=wer_plot,
    )
    _plot_metric_grouped_bars(
        summary_by_label=summary_by_label,
        metric_prefix="cer",
        title="CER by Augmentation (AI vs Human modifications)",
        output_path=cer_plot,
    )

    return AugmentedHc3JiwerReportResult(
        output_dir=request.output_dir,
        sample_metrics_csv=sample_metrics_csv,
        summary_by_label_csv=summary_by_label_csv,
        summary_overall_csv=summary_overall_csv,
        markdown_summary=markdown_summary,
        wer_plot=wer_plot,
        cer_plot=cer_plot,
    )


def build_request_from_args(args: argparse.Namespace) -> AugmentedHc3JiwerReportRequest:
    """Build typed request from CLI args."""

    project_root = args.project_root.resolve()
    runs_root = (project_root / args.runs_root).resolve()
    splits_root = (project_root / args.splits_root).resolve()

    if args.run_dir is not None:
        run_dir_candidate = Path(args.run_dir)
        run_dir = run_dir_candidate if run_dir_candidate.is_absolute() else (project_root / run_dir_candidate)
        run_dir = run_dir.resolve()
    elif args.run_id:
        run_dir = (runs_root / args.run_id).resolve()
    else:
        run_dir = _resolve_latest_run_dir(runs_root)

    output_dir_candidate = (
        Path(args.output_dir) if args.output_dir else (run_dir / "analysis_augmented_jiwer")
    )
    output_dir = output_dir_candidate if output_dir_candidate.is_absolute() else (project_root / output_dir_candidate)
    output_dir = output_dir.resolve()

    return AugmentedHc3JiwerReportRequest(
        project_root=project_root,
        run_dir=run_dir,
        splits_root=splits_root,
        output_dir=output_dir,
        dataset_id=str(args.dataset_id),
        baseline_split=str(args.baseline_split),
        orig_h_aug_ai_suffix=str(args.orig_h_aug_ai_suffix),
        aug_h_orig_ai_suffix=str(args.aug_h_orig_ai_suffix),
        partition=str(args.partition),
    )


def build_arg_parser() -> argparse.ArgumentParser:
    """Build CLI parser."""

    parser = argparse.ArgumentParser(
        description="Compute WER/CER between baseline and augmented HC3 samples.",
    )
    parser.add_argument("--project-root", type=Path, default=Path.cwd(), help="Repository root directory.")
    parser.add_argument(
        "--runs-root",
        type=Path,
        default=Path("runs/global_local_experiments"),
        help="Runs root when --run-id is used.",
    )
    parser.add_argument(
        "--splits-root",
        type=Path,
        default=Path("data/interim/splits/hc3"),
        help="Root with split parquet artifacts.",
    )
    parser.add_argument("--run-id", type=str, default="", help="Run id under runs/global_local_experiments.")
    parser.add_argument("--run-dir", type=Path, default=None, help="Explicit run directory.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory. Defaults to <run_dir>/analysis_augmented_jiwer.",
    )
    parser.add_argument("--dataset-id", type=str, default="hc3", help="Dataset id.")
    parser.add_argument("--baseline-split", type=str, default="aug_baseline", help="Baseline split.")
    parser.add_argument("--orig-h-aug-ai-suffix", type=str, default="orig_h_aug_ai", help="Suffix for augmented-ai rows.")
    parser.add_argument("--aug-h-orig-ai-suffix", type=str, default="aug_h_orig_ai", help="Suffix for augmented-human rows.")
    parser.add_argument("--partition", type=str, default="test", choices=("train", "test"), help="Partition to analyze.")
    return parser


def main() -> int:
    """CLI entrypoint."""

    parser = build_arg_parser()
    args = parser.parse_args()
    request = build_request_from_args(args)
    result = run_augmented_hc3_jiwer_report(request)
    print(
        json.dumps(
            {
                "run_dir": str(request.run_dir),
                "output_dir": str(result.output_dir),
                "sample_metrics_csv": str(result.sample_metrics_csv),
                "summary_by_label_csv": str(result.summary_by_label_csv),
                "summary_overall_csv": str(result.summary_overall_csv),
                "markdown_summary": str(result.markdown_summary),
                "wer_plot": str(result.wer_plot),
                "cer_plot": str(result.cer_plot),
            },
            indent=2,
        )
    )
    return 0


def _validate_request(request: AugmentedHc3JiwerReportRequest) -> None:
    """Validate request invariants."""

    if not request.dataset_id.strip():
        raise ValueError("dataset_id must be non-empty.")
    if not request.run_dir.exists():
        raise FileNotFoundError(f"Missing run directory: {request.run_dir}")
    if not request.splits_root.exists():
        raise FileNotFoundError(f"Missing splits root: {request.splits_root}")
    required = request.run_dir / "local_results.jsonl"
    if not required.exists():
        raise FileNotFoundError(f"Missing required file: {required}")


def _resolve_latest_run_dir(runs_root: Path) -> Path:
    """Resolve latest run with required files."""

    if not runs_root.exists():
        raise FileNotFoundError(f"Missing runs root: {runs_root}")
    candidates: list[Path] = []
    for child in runs_root.iterdir():
        if not child.is_dir():
            continue
        if (child / "local_results.jsonl").exists():
            candidates.append(child)
    if not candidates:
        raise ValueError(f"No completed runs found under {runs_root}")
    candidates.sort(key=lambda path: path.stat().st_mtime, reverse=True)
    return candidates[0]


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    """Read JSONL rows as dict objects."""

    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        payload = json.loads(line)
        if not isinstance(payload, dict):
            raise ValueError(f"Expected object row in {path}, got: {payload!r}")
        rows.append(payload)
    return rows


def _build_scenario_rows(
    *,
    local_rows: Sequence[Mapping[str, Any]],
    dataset_id: str,
    baseline_split: str,
    orig_h_aug_ai_suffix: str,
    aug_h_orig_ai_suffix: str,
) -> list[ScenarioRow]:
    """Build baseline + augmentation scenario definitions."""

    prefix = f"{dataset_id}:"
    split_names: set[str] = set()
    for row in local_rows:
        scope_id = row.get("scope_id")
        if not isinstance(scope_id, str) or not scope_id.startswith(prefix):
            continue
        split_names.add(scope_id.split(":", 1)[1])

    orig_h_aug_ai_pattern = re.compile(rf"^aug_(.+)_{re.escape(orig_h_aug_ai_suffix)}$")
    aug_h_orig_ai_pattern = re.compile(rf"^aug_(.+)_{re.escape(aug_h_orig_ai_suffix)}$")
    orig_rows: list[tuple[str, str]] = []
    aug_h_rows: list[tuple[str, str]] = []
    for split in sorted(split_names):
        if split == baseline_split:
            continue
        match_orig = orig_h_aug_ai_pattern.match(split)
        if match_orig is not None:
            orig_rows.append((split, str(match_orig.group(1))))
        match_aug_h = aug_h_orig_ai_pattern.match(split)
        if match_aug_h is not None:
            aug_h_rows.append((split, str(match_aug_h.group(1))))

    rows: list[ScenarioRow] = [ScenarioRow("baseline", baseline_split, "baseline", "baseline")]
    for split, augmentation in sorted(orig_rows, key=lambda item: item[1]):
        rows.append(ScenarioRow("orig_h_aug_ai", split, f"{augmentation} (orig_h+aug_ai)", augmentation))
    for split, augmentation in sorted(aug_h_rows, key=lambda item: item[1]):
        rows.append(ScenarioRow("aug_h_orig_ai", split, f"{augmentation} (aug_h+orig_ai)", augmentation))
    if len(rows) <= 1:
        raise ValueError("No augmented scenario rows resolved.")
    return rows


def _load_text_tables(
    *,
    splits_root: Path,
    scenario_rows: Sequence[ScenarioRow],
    partition: str,
) -> dict[tuple[str, str, str], str]:
    """Load split text mapping: (split,label,sample_id) -> text."""

    needed_splits = sorted(set(row.split_name for row in scenario_rows))
    mapping: dict[tuple[str, str, str], str] = {}
    for split in needed_splits:
        for label in ("human", "ai"):
            parquet_path = splits_root / split / partition / label / "sampled_records.parquet"
            if not parquet_path.exists():
                raise FileNotFoundError(f"Missing parquet: {parquet_path}")
            frame = pd.read_parquet(parquet_path)
            if "sample_id" not in frame.columns or "text" not in frame.columns:
                raise ValueError(f"Missing sample_id/text columns in {parquet_path}")
            for sample_id, text in frame[["sample_id", "text"]].itertuples(index=False):
                if not isinstance(sample_id, str) or not isinstance(text, str):
                    raise ValueError(f"Invalid sample row in {parquet_path}: {sample_id!r}, {type(text)!r}")
                mapping[(split, label, sample_id)] = text
    return mapping


def _build_modified_text_pairs(
    *,
    scenario_rows: Sequence[ScenarioRow],
    texts: Mapping[tuple[str, str, str], str],
) -> list[dict[str, str]]:
    """Build baseline-vs-augmented rows only for modified sides."""

    baseline_split = scenario_rows[0].split_name
    rows: list[dict[str, str]] = []
    for scenario in scenario_rows:
        if scenario.source_group == "baseline":
            continue
        for label in ("human", "ai"):
            is_modified = (
                (scenario.source_group == "orig_h_aug_ai" and label == "ai")
                or (scenario.source_group == "aug_h_orig_ai" and label == "human")
            )
            if not is_modified:
                continue
            sample_ids = sorted(
                sample_id
                for split_name, label_value, sample_id in texts.keys()
                if split_name == scenario.split_name and label_value == label
            )
            for sample_id in sample_ids:
                baseline_key = (baseline_split, label, sample_id)
                scenario_key = (scenario.split_name, label, sample_id)
                if baseline_key not in texts:
                    raise ValueError(f"Missing baseline text for key={baseline_key!r}")
                if scenario_key not in texts:
                    raise ValueError(f"Missing scenario text for key={scenario_key!r}")
                rows.append(
                    {
                        "source_group": scenario.source_group,
                        "source_split": scenario.split_name,
                        "display_label": scenario.display_label,
                        "augmentation": scenario.augmentation,
                        "label": label,
                        "sample_id": sample_id,
                        "baseline_text": texts[baseline_key],
                        "scenario_text": texts[scenario_key],
                    }
                )
    if not rows:
        raise ValueError("No modified baseline-vs-augmentation rows were built.")
    return rows


def _compute_metrics_frame(*, modified_rows: Sequence[Mapping[str, str]]) -> pd.DataFrame:
    """Compute per-sample WER/CER."""

    rows: list[dict[str, Any]] = []
    for row in modified_rows:
        reference = row["baseline_text"]
        hypothesis = row["scenario_text"]
        rows.append(
            {
                "source_group": row["source_group"],
                "source_split": row["source_split"],
                "display_label": row["display_label"],
                "augmentation": row["augmentation"],
                "label": row["label"],
                "sample_id": row["sample_id"],
                "wer": float(wer(reference, hypothesis)),
                "cer": float(cer(reference, hypothesis)),
            }
        )
    frame = pd.DataFrame(rows)
    if frame.empty:
        raise ValueError("Metrics frame is empty.")
    return frame


def _build_summary_by_label(*, metrics_frame: pd.DataFrame) -> pd.DataFrame:
    """Build per-augmentation, per-label metric summary."""

    summary = (
        metrics_frame.groupby(["source_group", "augmentation", "label"], as_index=False)
        .agg(
            sample_count=("sample_id", "size"),
            wer_mean=("wer", "mean"),
            wer_std=("wer", "std"),
            cer_mean=("cer", "mean"),
            cer_std=("cer", "std"),
        )
        .reset_index(drop=True)
    )
    summary["augmentation"] = pd.Categorical(
        summary["augmentation"],
        categories=ordered_augmentations(summary["augmentation"].astype(str).tolist()),
        ordered=True,
    )
    return summary.sort_values(by=["augmentation", "label"], kind="mergesort").reset_index(drop=True)


def _build_summary_overall(*, metrics_frame: pd.DataFrame) -> pd.DataFrame:
    """Build per-augmentation summary across all modified samples."""

    summary = (
        metrics_frame.groupby(["augmentation"], as_index=False)
        .agg(
            sample_count=("sample_id", "size"),
            wer_mean=("wer", "mean"),
            wer_std=("wer", "std"),
            cer_mean=("cer", "mean"),
            cer_std=("cer", "std"),
        )
        .reset_index(drop=True)
    )
    summary["augmentation"] = pd.Categorical(
        summary["augmentation"],
        categories=ordered_augmentations(summary["augmentation"].astype(str).tolist()),
        ordered=True,
    )
    return summary.sort_values(by=["augmentation"], kind="mergesort").reset_index(drop=True)


def _remove_legacy_plots(*, output_dir: Path) -> None:
    """Remove legacy plot filenames to avoid stale files after refactors."""

    legacy_paths = (
        output_dir / "wer_bars_ai.png",
        output_dir / "wer_bars_human.png",
        output_dir / "cer_bars_ai.png",
        output_dir / "cer_bars_human.png",
    )
    for path in legacy_paths:
        if path.exists():
            path.unlink()


def _plot_metric_grouped_bars(
    *,
    summary_by_label: pd.DataFrame,
    metric_prefix: str,
    title: str,
    output_path: Path,
) -> None:
    """Plot grouped bars by augmentation: AI and Human side by side."""

    mean_column = f"{metric_prefix}_mean"
    std_column = f"{metric_prefix}_std"
    frame = summary_by_label.copy()
    frame["augmentation"] = frame["augmentation"].astype(str)

    augmentations = ordered_augmentations(frame["augmentation"].tolist())
    frame = frame[frame["augmentation"].isin(augmentations)].copy()
    if frame.empty:
        raise ValueError("No rows to plot.")

    ai_rows = frame[frame["label"] == "ai"].set_index("augmentation")
    human_rows = frame[frame["label"] == "human"].set_index("augmentation")

    ai_means: list[float] = []
    ai_stds: list[float] = []
    human_means: list[float] = []
    human_stds: list[float] = []

    for augmentation in augmentations:
        if augmentation not in ai_rows.index or augmentation not in human_rows.index:
            continue
        ai_means.append(float(ai_rows.loc[augmentation, mean_column]))
        ai_stds.append(float(ai_rows.loc[augmentation, std_column]) if pd.notna(ai_rows.loc[augmentation, std_column]) else 0.0)
        human_means.append(float(human_rows.loc[augmentation, mean_column]))
        human_stds.append(
            float(human_rows.loc[augmentation, std_column]) if pd.notna(human_rows.loc[augmentation, std_column]) else 0.0
        )

    valid_augmentations = [
        augmentation
        for augmentation in augmentations
        if augmentation in ai_rows.index and augmentation in human_rows.index
    ]
    if not valid_augmentations:
        raise ValueError("No augmentations with both AI and Human rows.")

    x_positions = np.arange(len(valid_augmentations), dtype=float)
    width = 0.38

    ai_colors = [augmentation_color(augmentation) for augmentation in valid_augmentations]
    human_colors = [_lighten_color(augmentation_color(augmentation), blend=0.40) for augmentation in valid_augmentations]

    fig, ax = plt.subplots(figsize=(max(10.0, 1.8 * len(valid_augmentations)), 6.6), constrained_layout=True)
    bars_ai = ax.bar(
        x_positions - width / 2.0,
        ai_means,
        width=width,
        color=ai_colors,
        edgecolor="#1A1A1A",
        linewidth=0.6,
        label="AI -> Augmented AI",
    )
    bars_human = ax.bar(
        x_positions + width / 2.0,
        human_means,
        width=width,
        color=human_colors,
        edgecolor="#1A1A1A",
        linewidth=0.6,
        hatch="//",
        label="Human -> Augmented Human",
    )

    ax.errorbar(
        x=x_positions - width / 2.0,
        y=ai_means,
        yerr=ai_stds,
        fmt="none",
        ecolor="#2C2C2C",
        elinewidth=1.0,
        capsize=3.5,
        capthick=1.0,
    )
    ax.errorbar(
        x=x_positions + width / 2.0,
        y=human_means,
        yerr=human_stds,
        fmt="none",
        ecolor="#2C2C2C",
        elinewidth=1.0,
        capsize=3.5,
        capthick=1.0,
    )

    y_max = max(
        max(ai_mean + ai_std for ai_mean, ai_std in zip(ai_means, ai_stds, strict=True)),
        max(human_mean + human_std for human_mean, human_std in zip(human_means, human_stds, strict=True)),
    )
    offset = max(0.02, y_max * 0.035)
    ax.set_ylim(0.0, y_max + offset * 4.6)

    for bar, mean_value, std_value in zip(bars_ai, ai_means, ai_stds, strict=True):
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            bar.get_height() + std_value + offset,
            f"μ={mean_value:.3f}\nσ={std_value:.3f}",
            ha="center",
            va="bottom",
            fontsize=9.5,
            fontweight="bold",
            color="#1A1A1A",
        )
    for bar, mean_value, std_value in zip(bars_human, human_means, human_stds, strict=True):
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            bar.get_height() + std_value + offset,
            f"μ={mean_value:.3f}\nσ={std_value:.3f}",
            ha="center",
            va="bottom",
            fontsize=9.5,
            fontweight="bold",
            color="#1A1A1A",
        )

    ax.set_xticks(x_positions)
    ax.set_xticklabels(valid_augmentations, rotation=24, ha="right")
    ax.set_xlabel("Augmentation Method")
    ax.set_ylabel(metric_prefix.upper())
    ax.set_title(title)
    ax.grid(axis="y", alpha=0.22)
    ax.legend(loc="upper left", frameon=True)

    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _lighten_color(color: str, *, blend: float) -> tuple[float, float, float, float]:
    """Blend input color towards white for label-distinct paired bars."""

    red, green, blue, alpha = to_rgba(color)
    ratio = min(max(blend, 0.0), 1.0)
    return (
        red + (1.0 - red) * ratio,
        green + (1.0 - green) * ratio,
        blue + (1.0 - blue) * ratio,
        alpha,
    )


def _build_markdown_summary(*, summary_by_label: pd.DataFrame, summary_overall: pd.DataFrame) -> str:
    """Render markdown summary with both tables."""

    lines: list[str] = []
    lines.append("# JIWER Summary")
    lines.append("")
    lines.append("## Overall (all modified samples)")
    lines.append("")
    lines.append(_dataframe_to_markdown(summary_overall))
    lines.append("")
    lines.append("## By Label")
    lines.append("")
    lines.append(_dataframe_to_markdown(summary_by_label))
    lines.append("")
    return "\n".join(lines)


def _dataframe_to_markdown(frame: pd.DataFrame, *, float_precision: int = 4) -> str:
    """Render DataFrame to markdown without extra deps."""

    if frame.empty:
        return "_No rows available._"
    display = frame.copy()
    for column in display.columns:
        if pd.api.types.is_float_dtype(display[column]):
            display[column] = display[column].map(lambda value: f"{float(value):.{float_precision}f}")
    headers = [str(column) for column in display.columns.tolist()]
    rows = [[str(value) for value in row] for row in display.to_numpy(dtype=object).tolist()]
    lines: list[str] = []
    lines.append("| " + " | ".join(headers) + " |")
    lines.append("| " + " | ".join("---" for _ in headers) + " |")
    for row in rows:
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


if __name__ == "__main__":
    raise SystemExit(main())
