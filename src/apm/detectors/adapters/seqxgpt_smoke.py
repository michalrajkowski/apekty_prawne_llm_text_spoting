"""Smoke-run entry point for SeqXGPT detector scoring."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from apm.detectors.adapters.seqxgpt import SeqXGPTDetector


def _load_texts(input_path: Path, text_field: str) -> list[str]:
    if input_path.suffix == ".jsonl":
        texts: list[str] = []
        for raw_line in input_path.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if not line:
                continue
            parsed = json.loads(line)
            text_value = parsed.get(text_field)
            if not isinstance(text_value, str):
                raise ValueError(f"Missing string field '{text_field}' in JSONL input row.")
            texts.append(text_value)
        return texts

    if input_path.suffix == ".txt":
        lines = input_path.read_text(encoding="utf-8").splitlines()
        return [line.strip() for line in lines if line.strip()]

    raise ValueError("Only .txt and .jsonl input formats are supported.")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run smoke inference for SeqXGPT detector.")
    parser.add_argument("--project-root", type=Path, default=Path.cwd(), help="Repository root path.")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/detectors/seqxgpt.detector.json"),
        help="Path to detector config JSON.",
    )
    parser.add_argument("--input", type=Path, required=True, help="Input .txt or .jsonl file with texts.")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("runs/detectors/seqxgpt_smoke.jsonl"),
        help="Output JSONL path for scored records.",
    )
    parser.add_argument("--text-field", type=str, default="text", help="Text field name for JSONL input.")
    parser.add_argument("--limit", type=int, default=10, help="Maximum number of texts to score.")
    parser.add_argument(
        "--all-variants",
        action="store_true",
        help="Score all config-supported variants under configured VRAM ceiling.",
    )
    return parser


def _score_variant(config: dict[str, Any], texts: list[str], variant_id: str) -> list[dict[str, Any]]:
    variant_config = dict(config)
    variant_config["variant_id"] = variant_id

    detector = SeqXGPTDetector.initialize(config=variant_config)
    try:
        scores = detector.predict_batch(texts)
    finally:
        detector.delete()

    return [
        {
            "variant_id": variant_id,
            "index": index,
            "text": text,
            "score": score,
        }
        for index, (text, score) in enumerate(zip(texts, scores, strict=True))
    ]


def main() -> int:
    parser = _build_parser()
    args = parser.parse_args()

    config = json.loads(args.config.read_text(encoding="utf-8"))
    texts = _load_texts(args.input, args.text_field)
    selected_texts = texts[: args.limit]
    if not selected_texts:
        raise ValueError("No input texts found for smoke inference.")

    if args.all_variants:
        variant_ids = SeqXGPTDetector.list_supported_variants(config=config)
    else:
        variant_ids = [str(config.get("variant_id", "gpt2_small"))]

    records: list[dict[str, Any]] = []
    for variant_id in variant_ids:
        records.extend(_score_variant(config=config, texts=selected_texts, variant_id=variant_id))

    output_path = args.project_root / args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as output_handle:
        for record in records:
            output_handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    summary = {
        "scored_count": len(records),
        "variant_count": len(variant_ids),
        "output_path": str(output_path),
    }
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
