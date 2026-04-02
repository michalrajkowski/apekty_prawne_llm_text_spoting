"""Materialize Kaggle LLM Detect AI Generated Text split outputs."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from apm.data.adapters.hc3_materialize import MaterializedSplitOutput
from apm.data.adapters.kaggle_llm_detect_ai_generated_text_adapter import (
    KaggleLlmDetectAiGeneratedTextAdapter,
)
from apm.data.dataset_registry import DatasetRegistry
from apm.data.hf_loader import load_dataset
from apm.data.storage import (
    build_metadata_payload,
    ensure_artifact_parent_dirs,
    resolve_dataset_artifact_paths,
    write_metadata_json,
    write_normalized_parquet,
    write_normalized_parquet_by_label,
    write_raw_snapshot_jsonl,
    write_raw_snapshot_jsonl_by_label,
)
from apm.types import DatasetLoadRequest, SamplingStrategy


def materialize_kaggle_llm_detect_ai_generated_text_samples(
    project_root: Path,
    config_path: Path,
    sample_size: int,
    seed: int,
    sampling_strategy: SamplingStrategy = "balanced_random",
) -> tuple[MaterializedSplitOutput, ...]:
    """Load Kaggle CSV split(s) and persist sampled normalized outputs."""

    adapter = KaggleLlmDetectAiGeneratedTextAdapter.from_config_path(
        config_path=config_path,
        project_root=project_root,
    )
    adapter.ensure_sources_available()
    registry = DatasetRegistry()
    registry.register(dataset_id=adapter.dataset_id, adapter=adapter)

    outputs: list[MaterializedSplitOutput] = []
    for split in adapter.list_splits():
        result = load_dataset(
            request=DatasetLoadRequest(
                dataset_id=adapter.dataset_id,
                split=split,
                sample_size=sample_size if sampling_strategy == "random" else None,
                per_label_sample_size=sample_size if sampling_strategy == "balanced_random" else None,
                seed=seed,
                sampling_strategy=sampling_strategy,
            ),
            registry=registry,
        )

        paths = resolve_dataset_artifact_paths(
            project_root=project_root,
            dataset_id=result.dataset_id,
            split=result.split,
        )
        ensure_artifact_parent_dirs(paths)
        parquet_path = write_normalized_parquet(paths.normalized_parquet_path, list(result.records))
        raw_snapshot_path = write_raw_snapshot_jsonl(paths.raw_snapshot_dir, list(result.records))
        split_output_dir = paths.normalized_parquet_path.parent / result.split
        raw_by_label_paths = write_raw_snapshot_jsonl_by_label(paths.raw_snapshot_dir, list(result.records))
        parquet_by_label_paths = write_normalized_parquet_by_label(split_output_dir, list(result.records))
        sampled_per_label = Counter(record.label for record in result.records)

        metadata = build_metadata_payload(
            dataset_id=result.dataset_id,
            split=result.split,
            seed=result.seed,
            sampling_strategy=result.sampling_strategy,
            source_uri=adapter.source_uri,
            source_revision=None,
        )
        metadata["sample_size"] = sample_size
        metadata["sampled_count"] = result.sampled_count
        metadata["total_available"] = result.total_available
        metadata["sampling_strategy"] = sampling_strategy
        metadata["requested_per_label_sample_size"] = sample_size if sampling_strategy == "balanced_random" else None
        metadata["sampled_per_label_counts"] = {
            "human": sampled_per_label.get("human", 0),
            "ai": sampled_per_label.get("ai", 0),
        }
        metadata["raw_snapshot_paths_by_label"] = {label: str(path) for label, path in raw_by_label_paths.items()}
        metadata["parquet_paths_by_label"] = {label: str(path) for label, path in parquet_by_label_paths.items()}
        write_metadata_json(paths.metadata_json_path, metadata)

        outputs.append(
            MaterializedSplitOutput(
                split=split,
                sampled_count=result.sampled_count,
                parquet_path=parquet_path,
                metadata_path=paths.metadata_json_path,
                raw_snapshot_path=raw_snapshot_path,
            )
        )

    return tuple(outputs)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Materialize Kaggle sampled split outputs.")
    parser.add_argument("--project-root", type=Path, default=Path.cwd(), help="Repository root path.")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/datasets/kaggle_llm_detect_ai_generated_text.dataset.json"),
        help="Path to Kaggle dataset config JSON.",
    )
    parser.add_argument("--sample-size", type=int, default=100, help="Number of sampled records per split.")
    parser.add_argument(
        "--sampling-strategy",
        choices=("random", "balanced_random"),
        default="balanced_random",
        help="Sampling strategy: random total sample or balanced per-label sample.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed for deterministic sampling.")
    return parser


def main() -> int:
    parser = _build_parser()
    args = parser.parse_args()

    outputs = materialize_kaggle_llm_detect_ai_generated_text_samples(
        project_root=args.project_root,
        config_path=args.config,
        sample_size=args.sample_size,
        seed=args.seed,
        sampling_strategy=args.sampling_strategy,
    )
    rendered = [
        {
            "split": output.split,
            "sampled_count": output.sampled_count,
            "parquet_path": str(output.parquet_path),
            "metadata_path": str(output.metadata_path),
            "raw_snapshot_path": str(output.raw_snapshot_path),
        }
        for output in outputs
    ]
    print(json.dumps(rendered, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
