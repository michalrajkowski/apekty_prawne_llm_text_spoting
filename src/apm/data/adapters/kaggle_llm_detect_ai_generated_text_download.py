"""Download/bootstrap local Kaggle source files for LLM Detect AI Generated Text."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from apm.data.adapters.kaggle_llm_detect_ai_generated_text_adapter import (
    KaggleLlmDetectAiGeneratedTextAdapter,
)


def ensure_kaggle_llm_detect_ai_generated_text_sources(
    project_root: Path,
    config_path: Path,
) -> dict[str, str]:
    """Ensure source files exist and return resolved split -> source path mapping."""

    adapter = KaggleLlmDetectAiGeneratedTextAdapter.from_config_path(
        config_path=config_path,
        project_root=project_root,
    )
    adapter.ensure_sources_available()

    resolved: dict[str, str] = {}
    for split in adapter.list_splits():
        split_path = adapter.resolve_split_path(split)
        resolved[split] = str(split_path)
    return resolved


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Ensure Kaggle source files are downloaded and available locally.")
    parser.add_argument("--project-root", type=Path, default=Path.cwd(), help="Repository root path.")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/datasets/kaggle_llm_detect_ai_generated_text.dataset.json"),
        help="Path to Kaggle dataset config JSON.",
    )
    return parser


def main() -> int:
    parser = _build_parser()
    args = parser.parse_args()

    resolved_sources = ensure_kaggle_llm_detect_ai_generated_text_sources(
        project_root=args.project_root,
        config_path=args.config,
    )
    print(json.dumps({"resolved_sources": resolved_sources}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
