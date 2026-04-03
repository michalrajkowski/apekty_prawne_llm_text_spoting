"""Shared smoke-validation helpers for detector adapters."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import pandas as pd


@dataclass(frozen=True)
class Hc3SampleBatch:
    """Fixed-size HC3 sample batch for detector smoke checks."""

    human_texts: tuple[str, ...]
    ai_texts: tuple[str, ...]


def load_texts_from_file(input_path: Path, text_field: str) -> list[str]:
    """Load free-form texts from txt/jsonl/parquet files."""
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

    if input_path.suffix == ".parquet":
        frame = pd.read_parquet(input_path)
        if text_field not in frame.columns:
            raise ValueError(f"Missing column '{text_field}' in parquet input: {input_path}")
        values = frame[text_field].tolist()
        if any(not isinstance(value, str) for value in values):
            raise ValueError(f"Column '{text_field}' in {input_path} must contain only strings.")
        return [str(value) for value in values]

    raise ValueError("Only .txt, .jsonl, and .parquet input formats are supported.")


def load_hc3_samples(*, project_root: Path, split: str, samples_per_label: int) -> Hc3SampleBatch:
    """Load fixed-size human/ai samples from HC3 interim parquet artifacts."""
    human_path = (
        project_root
        / "data"
        / "interim"
        / "datasets"
        / "hc3"
        / split
        / "human"
        / "sampled_records.parquet"
    )
    ai_path = (
        project_root
        / "data"
        / "interim"
        / "datasets"
        / "hc3"
        / split
        / "ai"
        / "sampled_records.parquet"
    )

    if not human_path.exists():
        raise FileNotFoundError(f"Missing HC3 human parquet: {human_path}")
    if not ai_path.exists():
        raise FileNotFoundError(f"Missing HC3 ai parquet: {ai_path}")

    human_texts = load_texts_from_file(human_path, text_field="text")
    ai_texts = load_texts_from_file(ai_path, text_field="text")

    if len(human_texts) < samples_per_label:
        raise ValueError(
            f"Need at least {samples_per_label} HC3 human samples in {human_path}, got {len(human_texts)}."
        )
    if len(ai_texts) < samples_per_label:
        raise ValueError(f"Need at least {samples_per_label} HC3 ai samples in {ai_path}, got {len(ai_texts)}.")

    return Hc3SampleBatch(
        human_texts=tuple(human_texts[:samples_per_label]),
        ai_texts=tuple(ai_texts[:samples_per_label]),
    )


def build_hc3_separation_summary(
    *,
    human_scores: Sequence[float],
    ai_scores: Sequence[float],
    margin: float,
) -> dict[str, float | bool]:
    """Compute and validate score separation for HC3 human vs ai batches."""
    if not human_scores:
        raise ValueError("human_scores must be non-empty.")
    if not ai_scores:
        raise ValueError("ai_scores must be non-empty.")

    human_mean = sum(human_scores) / float(len(human_scores))
    ai_mean = sum(ai_scores) / float(len(ai_scores))
    separation = ai_mean - human_mean
    passed = separation >= margin

    return {
        "human_mean": human_mean,
        "ai_mean": ai_mean,
        "separation": separation,
        "required_margin": margin,
        "passed": passed,
    }
