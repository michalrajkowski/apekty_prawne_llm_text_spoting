"""Compute embedding similarity for augmented HC3 samples and plot similarity curves."""

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
import torch
from transformers import AutoModel, AutoTokenizer

from apm.experiments.augmentation_palette import augmentation_color, ordered_augmentations

matplotlib.use("Agg")
import matplotlib.pyplot as plt


@dataclass(frozen=True, slots=True)
class AugmentedHc3SimilarityReportRequest:
    """Configuration for augmented HC3 similarity analysis."""

    project_root: Path
    run_dir: Path
    splits_root: Path
    output_dir: Path
    dataset_id: str
    baseline_split: str
    orig_h_aug_ai_suffix: str
    aug_h_orig_ai_suffix: str
    partition: str
    model_id: str
    batch_size: int
    max_length: int
    max_samples_per_label: int
    device: str


@dataclass(frozen=True, slots=True)
class AugmentedHc3SimilarityReportResult:
    """Output paths for similarity analysis."""

    output_dir: Path
    sample_similarity_csv: Path
    summary_csv: Path
    markdown_summary: Path


@dataclass(frozen=True, slots=True)
class ScenarioRow:
    """One row definition in combined baseline/augmentation panel."""

    source_group: str
    split_name: str
    display_label: str
    augmentation: str


def run_augmented_hc3_similarity_report(
    request: AugmentedHc3SimilarityReportRequest,
) -> AugmentedHc3SimilarityReportResult:
    """Run similarity computation and visualization workflow."""

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
    similarity_rows = _build_similarity_rows(
        scenario_rows=scenario_rows,
        texts=texts,
    )
    text_to_vector = _embed_unique_texts(
        texts=_collect_unique_texts(similarity_rows),
        model_id=request.model_id,
        batch_size=request.batch_size,
        max_length=request.max_length,
        device=request.device,
    )
    similarity_frame = _compute_similarity_frame(
        similarity_rows=similarity_rows,
        text_to_vector=text_to_vector,
    )
    summary_frame = _build_similarity_summary(similarity_frame=similarity_frame)

    sample_similarity_csv = request.output_dir / "sample_similarity.csv"
    summary_csv = request.output_dir / "similarity_summary.csv"
    markdown_summary = request.output_dir / "SIMILARITY_SUMMARY.md"

    similarity_frame.to_csv(sample_similarity_csv, index=False)
    summary_frame.to_csv(summary_csv, index=False)
    markdown_summary.write_text(_build_markdown_summary(summary_frame), encoding="utf-8")

    legacy_bar_plot = request.output_dir / "similarity_bars_model_agnostic_combined.png"
    if legacy_bar_plot.exists():
        legacy_bar_plot.unlink()

    _plot_similarity_heatmaps(
        summary_frame=summary_frame,
        output_dir=request.output_dir,
    )
    _plot_similarity_grouped_bars(
        summary_frame=summary_frame,
        output_dir=request.output_dir,
    )
    _plot_similarity_curves_model_agnostic(
        similarity_frame=similarity_frame,
        output_dir=request.output_dir,
    )

    return AugmentedHc3SimilarityReportResult(
        output_dir=request.output_dir,
        sample_similarity_csv=sample_similarity_csv,
        summary_csv=summary_csv,
        markdown_summary=markdown_summary,
    )


def build_request_from_args(args: argparse.Namespace) -> AugmentedHc3SimilarityReportRequest:
    """Build typed request from command-line args."""

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
        Path(args.output_dir) if args.output_dir else (run_dir / "analysis_augmented_similarity")
    )
    output_dir = output_dir_candidate if output_dir_candidate.is_absolute() else (project_root / output_dir_candidate)
    output_dir = output_dir.resolve()

    return AugmentedHc3SimilarityReportRequest(
        project_root=project_root,
        run_dir=run_dir,
        splits_root=splits_root,
        output_dir=output_dir,
        dataset_id=str(args.dataset_id),
        baseline_split=str(args.baseline_split),
        orig_h_aug_ai_suffix=str(args.orig_h_aug_ai_suffix),
        aug_h_orig_ai_suffix=str(args.aug_h_orig_ai_suffix),
        partition=str(args.partition),
        model_id=str(args.model_id),
        batch_size=int(args.batch_size),
        max_length=int(args.max_length),
        max_samples_per_label=int(args.max_samples_per_label),
        device=str(args.device),
    )


def build_arg_parser() -> argparse.ArgumentParser:
    """Build CLI parser."""

    parser = argparse.ArgumentParser(
        description="Compute embedding similarities for augmented HC3 samples and create plots.",
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
        help="Output directory. Defaults to <run_dir>/analysis_augmented_similarity.",
    )
    parser.add_argument("--dataset-id", type=str, default="hc3", help="Dataset id.")
    parser.add_argument("--baseline-split", type=str, default="aug_baseline", help="Baseline split.")
    parser.add_argument("--orig-h-aug-ai-suffix", type=str, default="orig_h_aug_ai", help="Suffix for augmented-ai rows.")
    parser.add_argument("--aug-h-orig-ai-suffix", type=str, default="aug_h_orig_ai", help="Suffix for augmented-human rows.")
    parser.add_argument("--partition", type=str, default="test", choices=("train", "test"), help="Partition to analyze.")
    parser.add_argument("--model-id", type=str, default="BAAI/bge-m3", help="HF embedding model id.")
    parser.add_argument("--batch-size", type=int, default=128, help="Embedding batch size.")
    parser.add_argument("--max-length", type=int, default=1024, help="Max tokenizer length.")
    parser.add_argument("--max-samples-per-label", type=int, default=100, help="Max plotted bars per label.")
    parser.add_argument("--device", type=str, default="cuda", help="Torch device, e.g. cuda, cuda:0, cpu.")
    return parser


def main() -> int:
    """CLI entrypoint."""

    parser = build_arg_parser()
    args = parser.parse_args()
    request = build_request_from_args(args)
    result = run_augmented_hc3_similarity_report(request)
    print(
        json.dumps(
            {
                "run_dir": str(request.run_dir),
                "output_dir": str(result.output_dir),
                "sample_similarity_csv": str(result.sample_similarity_csv),
                "summary_csv": str(result.summary_csv),
                "markdown_summary": str(result.markdown_summary),
            },
            indent=2,
        )
    )
    return 0


def _validate_request(request: AugmentedHc3SimilarityReportRequest) -> None:
    """Validate request invariants."""

    if request.batch_size <= 0:
        raise ValueError("batch_size must be > 0.")
    if request.max_length <= 0:
        raise ValueError("max_length must be > 0.")
    if request.max_samples_per_label <= 0:
        raise ValueError("max_samples_per_label must be > 0.")
    if not request.run_dir.exists():
        raise FileNotFoundError(f"Missing run directory: {request.run_dir}")
    if not request.splits_root.exists():
        raise FileNotFoundError(f"Missing splits root: {request.splits_root}")
    for required in ("local_results.jsonl",):
        path = request.run_dir / required
        if not path.exists():
            raise FileNotFoundError(f"Missing required file: {path}")


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
    """Read jsonl rows."""

    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        payload = json.loads(line)
        if not isinstance(payload, dict):
            raise ValueError(f"Expected object row in {path}, got: {payload!r}")
        rows.append(payload)
    return rows


def _extract_baseline_thresholds(
    *,
    local_rows: Sequence[Mapping[str, Any]],
    dataset_id: str,
    baseline_split: str,
) -> dict[str, float]:
    """Get detector thresholds from baseline local scope."""

    baseline_scope = f"{dataset_id}:{baseline_split}"
    thresholds: dict[str, float] = {}
    for row in local_rows:
        if row.get("scope_id") != baseline_scope:
            continue
        detector_run_id = row.get("detector_run_id")
        threshold = row.get("threshold")
        if not isinstance(detector_run_id, str):
            raise ValueError(f"Invalid detector id row: {row!r}")
        if not isinstance(threshold, (int, float)):
            raise ValueError(f"Invalid threshold row: {row!r}")
        thresholds[detector_run_id] = float(threshold)
    if not thresholds:
        raise ValueError(f"No thresholds found for scope {baseline_scope!r}")
    return thresholds


def _build_scenario_rows(
    *,
    local_rows: Sequence[Mapping[str, Any]],
    dataset_id: str,
    baseline_split: str,
    orig_h_aug_ai_suffix: str,
    aug_h_orig_ai_suffix: str,
) -> list[ScenarioRow]:
    """Build combined row order baseline + augmented-ai + augmented-human."""

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
    """Load split text data as dict[(split,label,sample_id)] -> text."""

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


def _build_score_frame(
    *,
    raw_rows: Sequence[Mapping[str, Any]],
    dataset_id: str,
    thresholds: Mapping[str, float],
    scenario_rows: Sequence[ScenarioRow],
    partition: str,
) -> pd.DataFrame:
    """Build score frame filtered to selected scenarios."""

    split_set = {row.split_name for row in scenario_rows}
    records: list[dict[str, Any]] = []
    for row in raw_rows:
        if row.get("dataset_id") != dataset_id:
            continue
        if row.get("partition") != partition:
            continue
        detector_run_id = row.get("detector_run_id")
        source_split = row.get("source_split")
        label = row.get("label")
        sample_id = row.get("sample_id")
        score = row.get("score")
        if not isinstance(detector_run_id, str) or detector_run_id not in thresholds:
            continue
        if not isinstance(source_split, str) or source_split not in split_set:
            continue
        if label not in {"human", "ai"}:
            continue
        if not isinstance(sample_id, str):
            continue
        if not isinstance(score, (int, float)):
            continue
        score_value = float(score)
        if not math.isfinite(score_value):
            continue
        records.append(
            {
                "detector_run_id": detector_run_id,
                "source_split": source_split,
                "label": str(label),
                "sample_id": sample_id,
                "score": score_value,
                "threshold": float(thresholds[detector_run_id]),
            }
        )
    frame = pd.DataFrame(records)
    if frame.empty:
        raise ValueError("No score rows after filtering.")
    return frame


def _build_similarity_rows(
    *,
    scenario_rows: Sequence[ScenarioRow],
    texts: Mapping[tuple[str, str, str], str],
) -> list[dict[str, Any]]:
    """Construct baseline-vs-augmented similarity row records."""

    baseline_split = scenario_rows[0].split_name
    sample_rows: list[dict[str, Any]] = []

    for scenario in scenario_rows:
        for label in ("human", "ai"):
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

                baseline_text = texts[baseline_key]
                scenario_text = texts[scenario_key]
                is_modified = (
                    (scenario.source_group == "orig_h_aug_ai" and label == "ai")
                    or (scenario.source_group == "aug_h_orig_ai" and label == "human")
                )
                sample_rows.append(
                    {
                        "source_group": scenario.source_group,
                        "source_split": scenario.split_name,
                        "display_label": scenario.display_label,
                        "augmentation": scenario.augmentation,
                        "label": label,
                        "sample_id": sample_id,
                        "baseline_text": baseline_text,
                        "scenario_text": scenario_text,
                        "is_modified": is_modified,
                    }
                )
    return sample_rows


def _collect_unique_texts(similarity_rows: Sequence[Mapping[str, Any]]) -> list[str]:
    """Collect unique texts that need embedding."""

    unique: dict[str, None] = {}
    for row in similarity_rows:
        if bool(row["is_modified"]):
            baseline_text = row["baseline_text"]
            scenario_text = row["scenario_text"]
            if not isinstance(baseline_text, str) or not isinstance(scenario_text, str):
                raise ValueError("Invalid text row for embedding.")
            unique[baseline_text] = None
            unique[scenario_text] = None
    return list(unique.keys())


def _resolve_device(device: str) -> torch.device:
    """Resolve torch device robustly."""

    if device.startswith("cuda") and not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device(device)


def _embed_unique_texts(
    *,
    texts: Sequence[str],
    model_id: str,
    batch_size: int,
    max_length: int,
    device: str,
) -> dict[str, np.ndarray]:
    """Embed texts with HF encoder and return normalized vectors."""

    if not texts:
        return {}

    resolved_device = _resolve_device(device)
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModel.from_pretrained(model_id)
    model.to(resolved_device)
    model.eval()

    vectors: dict[str, np.ndarray] = {}
    for start in range(0, len(texts), batch_size):
        batch_texts = list(texts[start : start + batch_size])
        encoded = tokenizer(
            batch_texts,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        encoded = {key: value.to(resolved_device) for key, value in encoded.items()}
        with torch.no_grad():
            outputs = model(**encoded)
            hidden = outputs.last_hidden_state
            mask = encoded["attention_mask"].unsqueeze(-1).to(hidden.dtype)
            pooled = (hidden * mask).sum(dim=1) / torch.clamp(mask.sum(dim=1), min=1.0)
            normalized = torch.nn.functional.normalize(pooled, p=2, dim=1)
        array = normalized.detach().cpu().numpy()
        for text, vector in zip(batch_texts, array, strict=True):
            vectors[text] = vector.astype(np.float32, copy=False)
    return vectors


def _cosine_similarity(left: np.ndarray, right: np.ndarray) -> float:
    """Cosine similarity for already-normalized vectors."""

    return float(np.dot(left, right))


def _compute_similarity_frame(
    *,
    similarity_rows: Sequence[Mapping[str, Any]],
    text_to_vector: Mapping[str, np.ndarray],
) -> pd.DataFrame:
    """Compute per-sample similarity frame."""

    rows: list[dict[str, Any]] = []
    for row in similarity_rows:
        is_modified = bool(row["is_modified"])
        baseline_text = str(row["baseline_text"])
        scenario_text = str(row["scenario_text"])
        if is_modified:
            if baseline_text not in text_to_vector or scenario_text not in text_to_vector:
                raise ValueError("Missing embedding vector for modified sample.")
            similarity = _cosine_similarity(text_to_vector[baseline_text], text_to_vector[scenario_text])
        else:
            similarity = 1.0
        rows.append(
            {
                "source_group": str(row["source_group"]),
                "source_split": str(row["source_split"]),
                "display_label": str(row["display_label"]),
                "augmentation": str(row["augmentation"]),
                "label": str(row["label"]),
                "sample_id": str(row["sample_id"]),
                "is_modified": is_modified,
                "cosine_similarity": float(similarity),
            }
        )
    frame = pd.DataFrame(rows)
    if frame.empty:
        raise ValueError("Similarity frame is empty.")
    return frame


def _build_similarity_summary(*, similarity_frame: pd.DataFrame) -> pd.DataFrame:
    """Aggregate similarity statistics by split/label."""

    summary = (
        similarity_frame.groupby(
            ["source_group", "source_split", "display_label", "augmentation", "label", "is_modified"],
            as_index=False,
        )
        .agg(
            sample_count=("cosine_similarity", "size"),
            mean_similarity=("cosine_similarity", "mean"),
            median_similarity=("cosine_similarity", "median"),
            std_similarity=("cosine_similarity", "std"),
            min_similarity=("cosine_similarity", "min"),
            max_similarity=("cosine_similarity", "max"),
        )
    )
    return summary.sort_values(
        by=["source_group", "augmentation", "label"],
        kind="mergesort",
    ).reset_index(drop=True)


def _build_markdown_summary(summary_frame: pd.DataFrame) -> str:
    """Render markdown summary."""

    lines: list[str] = []
    lines.append("# Similarity Summary")
    lines.append("")
    lines.append(_dataframe_to_markdown(summary_frame))
    lines.append("")
    return "\n".join(lines)


def _plot_similarity_heatmaps(*, summary_frame: pd.DataFrame, output_dir: Path) -> None:
    """Plot detector-independent similarity heatmaps by augmentation and label."""

    for label in ("human", "ai"):
        subset = summary_frame[
            (summary_frame["source_group"] != "baseline") & (summary_frame["label"] == label)
        ]
        if subset.empty:
            continue
        pivot = subset.pivot_table(
            index="augmentation",
            columns="source_group",
            values="mean_similarity",
            aggfunc="mean",
        )
        matrix = pivot.to_numpy(dtype=float)
        fig, ax = plt.subplots(figsize=(6.0, max(4.0, len(pivot.index) * 0.6 + 1.0)))
        image = ax.imshow(matrix, aspect="auto", cmap="viridis", vmin=0.0, vmax=1.0)
        ax.set_xticks(np.arange(len(pivot.columns)))
        ax.set_xticklabels([str(value) for value in pivot.columns.tolist()], rotation=20, ha="right")
        ax.set_yticks(np.arange(len(pivot.index)))
        ax.set_yticklabels([str(value) for value in pivot.index.tolist()])
        ax.set_title(f"Mean cosine similarity ({label})")
        for row_idx in range(matrix.shape[0]):
            for col_idx in range(matrix.shape[1]):
                value = matrix[row_idx, col_idx]
                ax.text(col_idx, row_idx, f"{value:.3f}", ha="center", va="center", fontsize=8)
        fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
        fig.tight_layout()
        fig.savefig(output_dir / f"similarity_heatmap_{label}.png", dpi=180, bbox_inches="tight")
        plt.close(fig)


def _plot_similarity_grouped_bars(*, summary_frame: pd.DataFrame, output_dir: Path) -> None:
    """Plot one grouped bar chart: AI and Human mean similarity per augmentation."""

    ai_rows = summary_frame[
        (summary_frame["source_group"] == "orig_h_aug_ai")
        & (summary_frame["label"] == "ai")
        & (summary_frame["is_modified"])
    ].copy()
    human_rows = summary_frame[
        (summary_frame["source_group"] == "aug_h_orig_ai")
        & (summary_frame["label"] == "human")
        & (summary_frame["is_modified"])
    ].copy()
    if ai_rows.empty or human_rows.empty:
        return

    ai_rows = ai_rows.set_index("augmentation")
    human_rows = human_rows.set_index("augmentation")
    augmentations = ordered_augmentations(
        [name for name in ai_rows.index.astype(str).tolist() if name in set(human_rows.index.astype(str).tolist())]
    )
    if not augmentations:
        return

    ai_means: list[float] = []
    ai_stds: list[float] = []
    human_means: list[float] = []
    human_stds: list[float] = []
    for augmentation in augmentations:
        ai_means.append(float(ai_rows.loc[augmentation, "mean_similarity"]))
        ai_std_value = ai_rows.loc[augmentation, "std_similarity"]
        ai_stds.append(float(ai_std_value) if pd.notna(ai_std_value) else 0.0)
        human_means.append(float(human_rows.loc[augmentation, "mean_similarity"]))
        human_std_value = human_rows.loc[augmentation, "std_similarity"]
        human_stds.append(float(human_std_value) if pd.notna(human_std_value) else 0.0)

    x_positions = np.arange(len(augmentations), dtype=float)
    width = 0.38

    ai_colors = [augmentation_color(augmentation) for augmentation in augmentations]
    human_colors = [_lighten_hex(augmentation_color(augmentation), fraction=0.40) for augmentation in augmentations]

    fig, ax = plt.subplots(figsize=(max(10.0, 1.8 * len(augmentations)), 6.4), constrained_layout=True)
    ai_bars = ax.bar(
        x_positions - width / 2.0,
        ai_means,
        width=width,
        color=ai_colors,
        edgecolor="#1A1A1A",
        linewidth=0.6,
        label="AI -> transformed AI",
    )
    human_bars = ax.bar(
        x_positions + width / 2.0,
        human_means,
        width=width,
        color=human_colors,
        edgecolor="#1A1A1A",
        linewidth=0.6,
        hatch="//",
        label="Human -> transformed Human",
    )
    ax.errorbar(
        x_positions - width / 2.0,
        ai_means,
        yerr=ai_stds,
        fmt="none",
        ecolor="#2C2C2C",
        elinewidth=1.0,
        capsize=3.5,
        capthick=1.0,
    )
    ax.errorbar(
        x_positions + width / 2.0,
        human_means,
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
    offset = max(0.01, y_max * 0.03)
    ax.set_ylim(0.0, min(1.02, y_max + offset * 5.0))

    for bar, mean_value, std_value in zip(ai_bars, ai_means, ai_stds, strict=True):
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            bar.get_height() + std_value + offset,
            f"μ={mean_value:.3f}\nσ={std_value:.3f}",
            ha="center",
            va="bottom",
            fontsize=9.4,
            fontweight="bold",
            color="#1A1A1A",
        )
    for bar, mean_value, std_value in zip(human_bars, human_means, human_stds, strict=True):
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            bar.get_height() + std_value + offset,
            f"μ={mean_value:.3f}\nσ={std_value:.3f}",
            ha="center",
            va="bottom",
            fontsize=9.4,
            fontweight="bold",
            color="#1A1A1A",
        )

    ax.set_xticks(x_positions)
    ax.set_xticklabels(augmentations, rotation=24, ha="right")
    ax.set_title("Cosine Similarity by Augmentation (AI vs Human transformed)")
    ax.set_xlabel("Augmentation Method")
    ax.set_ylabel("Cosine Similarity to Original")
    ax.grid(axis="y", alpha=0.22)
    ax.legend(loc="lower right", frameon=True)

    fig.savefig(output_dir / "similarity_bars_grouped.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def _plot_similarity_curves_model_agnostic(
    *,
    similarity_frame: pd.DataFrame,
    output_dir: Path,
) -> None:
    """Plot model-agnostic similarity curves for AI and human modified sets."""

    _plot_similarity_curve_for_group(
        similarity_frame=similarity_frame,
        source_group="orig_h_aug_ai",
        label="ai",
        title="AI Similarity Curves (orig_h + aug_ai vs baseline)",
        output_path=output_dir / "similarity_curves_ai.png",
    )
    _plot_similarity_curve_for_group(
        similarity_frame=similarity_frame,
        source_group="aug_h_orig_ai",
        label="human",
        title="Human Similarity Curves (aug_h + orig_ai vs baseline)",
        output_path=output_dir / "similarity_curves_human.png",
    )


def _plot_similarity_curve_for_group(
    *,
    similarity_frame: pd.DataFrame,
    source_group: str,
    label: str,
    title: str,
    output_path: Path,
) -> None:
    """Plot percentile curves and side-panel key stats for one group."""

    subset = similarity_frame[
        (similarity_frame["source_group"] == source_group)
        & (similarity_frame["label"] == label)
        & (similarity_frame["is_modified"])
    ].copy()
    if subset.empty:
        return

    augmentations = ordered_augmentations([str(value) for value in subset["augmentation"]])

    fig = plt.figure(
        figsize=(13.5, max(5.0, 1.0 * len(augmentations) + 2.2)),
        constrained_layout=True,
    )
    grid = fig.add_gridspec(1, 2, width_ratios=[3.8, 1.8], wspace=0.05)
    ax_curve = fig.add_subplot(grid[0, 0])
    ax_stats = fig.add_subplot(grid[0, 1])

    ax_curve.grid(alpha=0.25)
    ax_curve.set_title(title)
    ax_curve.set_xlabel("Percentile of Samples")
    ax_curve.set_ylabel("Cosine Similarity to Original")
    ax_curve.set_xlim(0.0, 100.0)
    ax_curve.set_ylim(0.0, 1.01)
    ax_curve.axhline(1.0, linestyle="--", linewidth=1.1, color="#777777", alpha=0.85)

    ax_stats.axis("off")
    ax_stats.set_title("Key Stats", fontsize=14, pad=10)

    stats_lines: list[str] = []
    stats_colors: list[Any] = []
    for index, augmentation in enumerate(augmentations):
        values = subset[subset["augmentation"] == augmentation]["cosine_similarity"].to_numpy(dtype=float)
        if values.size == 0:
            continue
        values_sorted = np.sort(values)
        percentile_axis = np.linspace(0.0, 100.0, num=values_sorted.size)
        color = augmentation_color(augmentation)
        ax_curve.plot(percentile_axis, values_sorted, linewidth=2.4, color=color, label=augmentation)

        mean_value = float(np.mean(values_sorted))
        std_value = float(np.std(values_sorted))
        median_value = float(np.median(values_sorted))
        stats_lines.append(f"{augmentation}\nμ={mean_value:.3f}  σ={std_value:.3f}  med={median_value:.3f}")
        stats_colors.append(color)

    ax_curve.legend(loc="lower right", fontsize=9)

    y = 0.98
    spacing = 0.90 / max(len(stats_lines), 1)
    for line, color in zip(stats_lines, stats_colors, strict=True):
        ax_stats.text(
            0.01,
            y,
            line,
            fontsize=12.5,
            color=color,
            va="top",
            ha="left",
            linespacing=1.2,
            transform=ax_stats.transAxes,
        )
        y -= spacing

    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


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


def _slugify(value: str) -> str:
    """Create safe filename token."""

    normalized = value.strip().lower().replace(":", "_").replace("/", "_").replace(" ", "_")
    return "".join(character for character in normalized if character.isalnum() or character in {"_", "-"})


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


if __name__ == "__main__":
    raise SystemExit(main())
