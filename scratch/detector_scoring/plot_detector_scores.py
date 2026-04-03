"""Scratch helper: create separate per-model score bar charts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Plot one detector score chart per model.")
    parser.add_argument(
        "--summary-json",
        type=Path,
        default=Path("scratch/detector_scoring/results/summary_scores.json"),
        help="Input summary JSON produced by summarize_scores.py.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("scratch/detector_scoring/results/by_model"),
        help="Output directory for per-model PNG charts.",
    )
    parser.add_argument("--project-root", type=Path, default=Path.cwd(), help="Repository root directory.")
    parser.add_argument("--human-samples", type=int, default=30, help="How many human samples to plot.")
    parser.add_argument("--ai-samples", type=int, default=30, help="How many ai samples to plot.")
    parser.add_argument("--dpi", type=int, default=180, help="Plot DPI.")
    return parser


def _load_json(path: Path) -> dict[str, Any]:
    parsed = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(parsed, dict):
        raise ValueError(f"Expected root JSON object in {path}.")
    return parsed


def _sanitize_model_id(model_id: str) -> str:
    safe = model_id.replace("/", "_").replace(":", "_").replace(" ", "_")
    return safe


def _select_plot_rows(
    *,
    scores: list[float],
    labels: list[str],
    human_samples: int,
    ai_samples: int,
) -> tuple[list[str], list[float], list[str], int]:
    human_values: list[float] = []
    ai_values: list[float] = []

    for score, label in zip(scores, labels, strict=True):
        if label == "human" and len(human_values) < human_samples:
            human_values.append(float(score))
        if label == "ai" and len(ai_values) < ai_samples:
            ai_values.append(float(score))
        if len(human_values) == human_samples and len(ai_values) == ai_samples:
            break

    if len(human_values) < human_samples:
        raise ValueError(
            f"Requested {human_samples} human samples but only {len(human_values)} available in summary order."
        )
    if len(ai_values) < ai_samples:
        raise ValueError(f"Requested {ai_samples} ai samples but only {len(ai_values)} available in summary order.")

    x_labels = [f"H{i + 1}" for i in range(human_samples)] + [f"A{i + 1}" for i in range(ai_samples)]
    ordered_values = human_values + ai_values
    colors = ["#3A86FF"] * human_samples + ["#FF006E"] * ai_samples
    return x_labels, ordered_values, colors, human_samples


def _plot_one_model(
    *,
    model_id: str,
    scores: list[float],
    labels: list[str],
    output_path: Path,
    human_samples: int,
    ai_samples: int,
    dpi: int,
) -> None:
    x_labels, ordered_values, colors, human_count = _select_plot_rows(
        scores=scores,
        labels=labels,
        human_samples=human_samples,
        ai_samples=ai_samples,
    )

    x_positions = list(range(len(ordered_values)))
    fig_width = max(14.0, len(ordered_values) * 0.26)
    fig, ax = plt.subplots(figsize=(fig_width, 6.2))

    ax.bar(x_positions, ordered_values, color=colors, width=0.85)
    ax.axvline(x=human_count - 0.5, color="#444444", linestyle="--", linewidth=1.0)
    ax.text(human_count / 2.0 - 0.5, 1.02, "HUMAN", ha="center", va="bottom", fontsize=10, transform=ax.get_xaxis_transform())
    ax.text(
        human_count + (len(ordered_values) - human_count) / 2.0 - 0.5,
        1.02,
        "AI",
        ha="center",
        va="bottom",
        fontsize=10,
        transform=ax.get_xaxis_transform(),
    )

    ax.set_title(f"{model_id} scores: first {human_samples} human, then {ai_samples} ai")
    ax.set_xlabel("Ordered samples")
    ax.set_ylabel("Score")
    ax.set_xticks(x_positions)
    ax.set_xticklabels(x_labels, rotation=90)
    ax.set_ylim(0.0, 1.0)
    ax.grid(axis="y", linestyle="--", alpha=0.35)

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi)
    plt.close(fig)


def _plot(summary: dict[str, Any], output_dir: Path, human_samples: int, ai_samples: int, dpi: int) -> list[Path]:
    model_ids = summary.get("model_ids")
    scores_matrix_by_model = summary.get("scores_matrix_by_model")
    example_labels = summary.get("example_labels")

    if not isinstance(model_ids, list):
        raise ValueError("Summary field 'model_ids' must be a list.")
    if not isinstance(scores_matrix_by_model, list):
        raise ValueError("Summary field 'scores_matrix_by_model' must be a list.")
    if not isinstance(example_labels, list):
        raise ValueError("Summary field 'example_labels' must be a list.")
    if len(model_ids) != len(scores_matrix_by_model):
        raise ValueError("Summary 'model_ids' and 'scores_matrix_by_model' length mismatch.")

    created_paths: list[Path] = []
    for model_id, model_scores_raw in zip(model_ids, scores_matrix_by_model, strict=True):
        if not isinstance(model_id, str):
            raise ValueError("Each model id must be a string.")
        if not isinstance(model_scores_raw, list):
            raise ValueError(f"Model '{model_id}' scores must be a list.")
        if len(model_scores_raw) != len(example_labels):
            raise ValueError(
                f"Model '{model_id}' has {len(model_scores_raw)} scores but there are {len(example_labels)} labels."
            )

        model_scores = [float(value) for value in model_scores_raw]
        output_path = output_dir / f"{_sanitize_model_id(model_id)}.png"
        _plot_one_model(
            model_id=model_id,
            scores=model_scores,
            labels=[str(label) for label in example_labels],
            output_path=output_path,
            human_samples=human_samples,
            ai_samples=ai_samples,
            dpi=dpi,
        )
        created_paths.append(output_path)

    return created_paths


def main() -> int:
    parser = _build_arg_parser()
    args = parser.parse_args()

    summary_path = (args.project_root / args.summary_json).resolve()
    output_dir = (args.project_root / args.output_dir).resolve()

    summary = _load_json(summary_path)
    created_paths = _plot(
        summary=summary,
        output_dir=output_dir,
        human_samples=args.human_samples,
        ai_samples=args.ai_samples,
        dpi=args.dpi,
    )

    print(f"Saved {len(created_paths)} per-model bar plots to: {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
