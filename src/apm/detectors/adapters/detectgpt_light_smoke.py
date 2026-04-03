"""Smoke-run entry point for DetectGPT light detector scoring."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from apm.detectors.adapters.detectgpt_light import DetectGptLightDetector


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
    parser = argparse.ArgumentParser(description="Run smoke inference for DetectGPT light detector.")
    parser.add_argument("--project-root", type=Path, default=Path.cwd(), help="Repository root path.")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/detectors/detectgpt_light.detector.json"),
        help="Path to detector config JSON.",
    )
    parser.add_argument("--input", type=Path, required=True, help="Input .txt or .jsonl file with texts.")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("runs/detectors/detectgpt_light_smoke.jsonl"),
        help="Output JSONL path for scored records.",
    )
    parser.add_argument("--text-field", type=str, default="text", help="Text field name for JSONL input.")
    parser.add_argument("--limit", type=int, default=10, help="Maximum number of texts to score.")
    return parser


def main() -> int:
    parser = _build_parser()
    args = parser.parse_args()

    config = json.loads(args.config.read_text(encoding="utf-8"))
    texts = _load_texts(args.input, args.text_field)
    selected_texts = texts[: args.limit]
    if not selected_texts:
        raise ValueError("No input texts found for smoke inference.")

    detector = DetectGptLightDetector.initialize(config=config)
    try:
        scores = detector.predict_batch(selected_texts)
    finally:
        detector.delete()

    output_path = args.project_root / args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    records = [
        {"index": index, "text": text, "score": score}
        for index, (text, score) in enumerate(zip(selected_texts, scores, strict=True))
    ]
    with output_path.open("w", encoding="utf-8") as output_handle:
        for record in records:
            output_handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(json.dumps({"scored_count": len(records), "output_path": str(output_path)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
