"""Scratch helper: plot grouped detector scores for all sampled examples."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Plot detector score comparison bar chart.")
    parser.add_argument(
        "--summary-json",
        type=Path,
        default=Path("scratch/detector_scoring/results/summary_scores.json"),
        help="Input summary JSON produced by summarize_scores.py.",
    )
    parser.add_argument(
        "--output-png",
        type=Path,
        default=Path("scratch/detector_scoring/results/detector_scores_barplot.png"),
        help="Output PNG path for bar chart.",
    )
    parser.add_argument("--project-root", type=Path, default=Path.cwd(), help="Repository root directory.")
    parser.add_argument("--dpi", type=int, default=180, help="Plot DPI.")
    return parser


def _load_json(path: Path) -> dict[str, Any]:
    parsed = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(parsed, dict):
        raise ValueError(f"Expected root JSON object in {path}.")
    return parsed


def _plot(summary: dict[str, Any], output_path: Path, dpi: int) -> None:
    example_ids = summary.get("example_ids")
    model_ids = summary.get("model_ids")
    scores_matrix_by_model = summary.get("scores_matrix_by_model")

    if not isinstance(example_ids, list):
        raise ValueError("Summary field 'example_ids' must be a list.")
    if not isinstance(model_ids, list):
        raise ValueError("Summary field 'model_ids' must be a list.")
    if not isinstance(scores_matrix_by_model, list):
        raise ValueError("Summary field 'scores_matrix_by_model' must be a list.")

    num_examples = len(example_ids)
    num_models = len(model_ids)
    if num_examples == 0 or num_models == 0:
        raise ValueError("Summary must contain at least one example and one model.")

    x_positions = list(range(num_examples))
    bar_group_width = 0.86
    bar_width = bar_group_width / num_models
    start_offset = -bar_group_width / 2 + bar_width / 2

    figure_width = max(14.0, num_examples * 0.8)
    fig, ax = plt.subplots(figsize=(figure_width, 8.0))

    cmap = plt.get_cmap("tab20")
    for model_index, model_id in enumerate(model_ids):
        model_scores = scores_matrix_by_model[model_index]
        if not isinstance(model_scores, list):
            raise ValueError("Each score matrix row must be a list.")
        if len(model_scores) != num_examples:
            raise ValueError(
                f"Model '{model_id}' has {len(model_scores)} scores, expected {num_examples}."
            )

        offsets = [x + start_offset + model_index * bar_width for x in x_positions]
        color = cmap(model_index % 20)
        ax.bar(offsets, model_scores, width=bar_width, label=str(model_id), color=color)

    ax.set_title("Detector Scores Per Example")
    ax.set_xlabel("Example ID")
    ax.set_ylabel("Score")
    ax.set_xticks(x_positions)
    ax.set_xticklabels([str(example_id) for example_id in example_ids], rotation=45, ha="right")
    ax.set_ylim(0.0, 1.0)
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0), fontsize=8)

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi)
    plt.close(fig)


def main() -> int:
    parser = _build_arg_parser()
    args = parser.parse_args()

    summary_path = (args.project_root / args.summary_json).resolve()
    output_path = (args.project_root / args.output_png).resolve()

    summary = _load_json(summary_path)
    _plot(summary=summary, output_path=output_path, dpi=args.dpi)

    print(f"Saved bar plot to: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
