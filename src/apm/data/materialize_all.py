"""Bulk dataset initialization/materialization runner driven by dataset configs."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from apm.data.adapters.hc3_materialize import MaterializedSplitOutput, materialize_hc3_samples

MaterializerFn = Callable[[Path, Path, int, int], tuple[MaterializedSplitOutput, ...]]


@dataclass(frozen=True, slots=True)
class MaterializedDatasetReport:
    """Materialization outcome for one dataset config."""

    dataset_id: str
    config_path: Path
    split_outputs: tuple[MaterializedSplitOutput, ...]


@dataclass(frozen=True, slots=True)
class MaterializeAllReport:
    """Overall bulk materialization report for selected/discovered datasets."""

    discovered_dataset_ids: tuple[str, ...]
    requested_dataset_ids: tuple[str, ...]
    skipped_unsupported_dataset_ids: tuple[str, ...]
    materialized_datasets: tuple[MaterializedDatasetReport, ...]


def default_materializer_registry() -> dict[str, MaterializerFn]:
    """Return dataset_id -> materializer function mapping."""

    return {
        "hc3": materialize_hc3_samples,
    }


def read_dataset_id_from_config(config_path: Path) -> str:
    """Read dataset id from one dataset config JSON file."""

    payload = json.loads(config_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Config must be an object: {config_path}")
    dataset_id = payload.get("dataset_id")
    if not isinstance(dataset_id, str) or not dataset_id.strip():
        raise ValueError(f"dataset_id missing or invalid in config: {config_path}")
    return dataset_id


def discover_dataset_configs(config_dir: Path, pattern: str = "*.dataset.json") -> dict[str, Path]:
    """Discover dataset config files and map them by dataset id."""

    config_paths = sorted(config_dir.glob(pattern))
    dataset_to_config: dict[str, Path] = {}
    for config_path in config_paths:
        dataset_id = read_dataset_id_from_config(config_path)
        if dataset_id in dataset_to_config:
            raise ValueError(
                f"Duplicate dataset_id {dataset_id!r} in configs {dataset_to_config[dataset_id]} and {config_path}"
            )
        dataset_to_config[dataset_id] = config_path
    return dataset_to_config


def read_dataset_ids_file(datasets_file: Path) -> tuple[str, ...]:
    """Read dataset ids from text file (one id per line, '#' comments allowed)."""

    dataset_ids: list[str] = []
    for raw_line in datasets_file.read_text(encoding="utf-8").splitlines():
        stripped = raw_line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        dataset_ids.append(stripped)
    return tuple(dataset_ids)


def resolve_requested_dataset_ids(
    datasets: tuple[str, ...],
    datasets_file: Path | None,
) -> tuple[str, ...]:
    """Resolve explicit dataset selection from args and optional text file."""

    requested: list[str] = list(datasets)
    if datasets_file is not None:
        requested.extend(read_dataset_ids_file(datasets_file))

    unique: dict[str, None] = {}
    for dataset_id in requested:
        normalized = dataset_id.strip()
        if normalized:
            unique[normalized] = None
    return tuple(unique.keys())


def materialize_all_datasets(
    project_root: Path,
    config_dir: Path,
    sample_size: int,
    seed: int,
    datasets: tuple[str, ...] = (),
    datasets_file: Path | None = None,
    materializers: dict[str, MaterializerFn] | None = None,
) -> MaterializeAllReport:
    """Materialize selected datasets from discovered dataset configs."""

    if sample_size <= 0:
        raise ValueError("sample_size must be > 0")

    dataset_to_config = discover_dataset_configs(config_dir)
    discovered_dataset_ids = tuple(sorted(dataset_to_config.keys()))
    materializer_registry = materializers if materializers is not None else default_materializer_registry()

    requested_dataset_ids = resolve_requested_dataset_ids(datasets=datasets, datasets_file=datasets_file)
    if requested_dataset_ids:
        missing = sorted(dataset_id for dataset_id in requested_dataset_ids if dataset_id not in dataset_to_config)
        if missing:
            raise ValueError(f"Requested datasets not found in configs: {missing}")
        unsupported = sorted(dataset_id for dataset_id in requested_dataset_ids if dataset_id not in materializer_registry)
        if unsupported:
            raise ValueError(f"Requested datasets without materializer: {unsupported}")
        selected_dataset_ids = tuple(dataset_id for dataset_id in discovered_dataset_ids if dataset_id in requested_dataset_ids)
    else:
        selected_dataset_ids = discovered_dataset_ids

    skipped_unsupported_dataset_ids: list[str] = []
    materialized_datasets: list[MaterializedDatasetReport] = []

    for dataset_id in selected_dataset_ids:
        if dataset_id not in materializer_registry:
            skipped_unsupported_dataset_ids.append(dataset_id)
            continue

        config_path = dataset_to_config[dataset_id]
        materializer = materializer_registry[dataset_id]
        split_outputs = materializer(project_root, config_path, sample_size, seed)
        materialized_datasets.append(
            MaterializedDatasetReport(
                dataset_id=dataset_id,
                config_path=config_path,
                split_outputs=split_outputs,
            )
        )

    return MaterializeAllReport(
        discovered_dataset_ids=discovered_dataset_ids,
        requested_dataset_ids=requested_dataset_ids,
        skipped_unsupported_dataset_ids=tuple(skipped_unsupported_dataset_ids),
        materialized_datasets=tuple(materialized_datasets),
    )


def _report_to_json(report: MaterializeAllReport) -> str:
    """Render bulk report as pretty-printed JSON."""

    payload = {
        "discovered_dataset_ids": list(report.discovered_dataset_ids),
        "requested_dataset_ids": list(report.requested_dataset_ids),
        "skipped_unsupported_dataset_ids": list(report.skipped_unsupported_dataset_ids),
        "materialized_datasets": [
            {
                "dataset_id": dataset_report.dataset_id,
                "config_path": str(dataset_report.config_path),
                "splits": [
                    {
                        "split": split_output.split,
                        "sampled_count": split_output.sampled_count,
                        "parquet_path": str(split_output.parquet_path),
                        "metadata_path": str(split_output.metadata_path),
                        "raw_snapshot_path": str(split_output.raw_snapshot_path),
                    }
                    for split_output in dataset_report.split_outputs
                ],
            }
            for dataset_report in report.materialized_datasets
        ],
    }
    return json.dumps(payload, indent=2)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Initialize/materialize datasets from dataset config files.")
    parser.add_argument("--project-root", type=Path, default=Path.cwd(), help="Repository root path.")
    parser.add_argument(
        "--config-dir",
        type=Path,
        default=Path("configs/datasets"),
        help="Directory with *.dataset.json configs.",
    )
    parser.add_argument(
        "--datasets",
        nargs="*",
        default=(),
        help="Optional dataset ids to materialize (space separated).",
    )
    parser.add_argument(
        "--datasets-file",
        type=Path,
        default=None,
        help="Optional text file with dataset ids (one per line, '#' comments allowed).",
    )
    parser.add_argument("--sample-size", type=int, default=100, help="Sample size for each materialized split/selector.")
    parser.add_argument("--seed", type=int, default=42, help="Deterministic sampling seed.")
    return parser


def main() -> int:
    parser = _build_parser()
    args = parser.parse_args()

    datasets_tuple = tuple(args.datasets)
    report = materialize_all_datasets(
        project_root=args.project_root,
        config_dir=args.config_dir,
        datasets=datasets_tuple,
        datasets_file=args.datasets_file,
        sample_size=args.sample_size,
        seed=args.seed,
    )
    print(_report_to_json(report))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
