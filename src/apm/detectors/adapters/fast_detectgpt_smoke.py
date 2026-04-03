"""Smoke-run entry point for Fast-DetectGPT detector scoring and HC3 validation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from apm.detectors.adapters.fast_detectgpt import FastDetectGptDetector
from apm.detectors.adapters.smoke_validation import (
    build_hc3_separation_summary,
    load_hc3_samples,
    load_texts_from_file,
)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run smoke inference for Fast-DetectGPT detector.")
    parser.add_argument("--project-root", type=Path, default=Path.cwd(), help="Repository root path.")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/detectors/fast_detectgpt.detector.json"),
        help="Path to detector config JSON.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("runs/detectors/fast_detectgpt_smoke.json"),
        help="Output JSON path for scored records and validation summary.",
    )
    parser.add_argument("--input", type=Path, default=None, help="Optional .txt/.jsonl/.parquet file with texts.")
    parser.add_argument("--text-field", type=str, default="text", help="Text field name for JSONL/parquet input.")
    parser.add_argument("--limit", type=int, default=10, help="Maximum number of optional input texts to score.")
    parser.add_argument(
        "--hc3-split",
        type=str,
        default="all_train",
        help="HC3 split directory under data/interim/datasets/hc3.",
    )
    parser.add_argument(
        "--samples-per-label",
        type=int,
        default=10,
        help="How many HC3 human/ai samples to score for separation check.",
    )
    parser.add_argument(
        "--separation-margin",
        type=float,
        default=0.05,
        help="Required minimum (ai_mean - human_mean) margin.",
    )
    return parser


def main() -> int:
    parser = _build_parser()
    args = parser.parse_args()

    config = json.loads(args.config.read_text(encoding="utf-8"))
    detector = FastDetectGptDetector.initialize(config=config)

    try:
        hc3_batch = load_hc3_samples(
            project_root=args.project_root,
            split=args.hc3_split,
            samples_per_label=args.samples_per_label,
        )
        human_scores = detector.predict_batch(hc3_batch.human_texts)
        ai_scores = detector.predict_batch(hc3_batch.ai_texts)
        separation_summary = build_hc3_separation_summary(
            human_scores=human_scores,
            ai_scores=ai_scores,
            margin=args.separation_margin,
        )

        optional_records: list[dict[str, object]] = []
        if args.input is not None:
            texts = load_texts_from_file(args.input, args.text_field)
            selected_texts = texts[: args.limit]
            if selected_texts:
                scores = detector.predict_batch(selected_texts)
                optional_records = [
                    {"index": index, "text": text, "score": score}
                    for index, (text, score) in enumerate(zip(selected_texts, scores, strict=True))
                ]
    finally:
        detector.delete()

    output_payload: dict[str, object] = {
        "detector_id": "fast_detectgpt",
        "hc3_split": args.hc3_split,
        "samples_per_label": args.samples_per_label,
        "separation": separation_summary,
        "optional_input_scores": optional_records,
    }

    output_path = args.project_root / args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output_payload, indent=2, ensure_ascii=False), encoding="utf-8")

    if separation_summary["passed"] is not True:
        raise ValueError(
            "HC3 separation check failed for Fast-DetectGPT: "
            f"expected ai_mean - human_mean >= {args.separation_margin}."
        )

    print(json.dumps({"output_path": str(output_path), "separation": separation_summary}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
