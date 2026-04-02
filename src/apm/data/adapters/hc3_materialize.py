"""Materialize sampled HC3 selector outputs into raw/interim dataset artifacts."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

from apm.data.adapters.hc3_adapter import HC3Adapter
from apm.data.dataset_registry import DatasetRegistry
from apm.data.hf_loader import load_dataset
from apm.data.storage import (
    build_metadata_payload,
    ensure_artifact_parent_dirs,
    resolve_dataset_artifact_paths,
    write_metadata_json,
    write_normalized_parquet,
    write_raw_snapshot_jsonl,
)
from apm.types import DatasetLoadRequest


@dataclass(frozen=True, slots=True)
class MaterializedSplitOutput:
    """Persisted file paths for one selector materialization output."""

    split: str
    sampled_count: int
    parquet_path: Path
    metadata_path: Path
    raw_snapshot_path: Path


def materialize_hc3_samples(
    project_root: Path,
    config_path: Path,
    sample_size: int,
    seed: int,
) -> tuple[MaterializedSplitOutput, ...]:
    """Load HC3 selectors and persist sampled normalized outputs for each selector."""

    adapter = HC3Adapter.from_config_path(config_path)
    registry = DatasetRegistry()
    registry.register(dataset_id=adapter.dataset_id, adapter=adapter)

    outputs: list[MaterializedSplitOutput] = []
    for split in adapter.list_splits():
        result = load_dataset(
            request=DatasetLoadRequest(
                dataset_id=adapter.dataset_id,
                split=split,
                sample_size=sample_size,
                seed=seed,
            ),
            registry=registry,
        )

        paths = resolve_dataset_artifact_paths(project_root=project_root, dataset_id=result.dataset_id, split=result.split)
        ensure_artifact_parent_dirs(paths)

        parquet_path = write_normalized_parquet(paths.normalized_parquet_path, list(result.records))
        raw_snapshot_path = write_raw_snapshot_jsonl(paths.raw_snapshot_dir, list(result.records))

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
    parser = argparse.ArgumentParser(description="Materialize HC3 sampled selector outputs.")
    parser.add_argument("--project-root", type=Path, default=Path.cwd(), help="Repository root path.")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/datasets/hc3.dataset.json"),
        help="Path to HC3 dataset config JSON.",
    )
    parser.add_argument("--sample-size", type=int, default=100, help="Number of sampled records per selector.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for deterministic sampling.")
    return parser


def main() -> int:
    parser = _build_parser()
    args = parser.parse_args()

    outputs = materialize_hc3_samples(
        project_root=args.project_root,
        config_path=args.config,
        sample_size=args.sample_size,
        seed=args.seed,
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
