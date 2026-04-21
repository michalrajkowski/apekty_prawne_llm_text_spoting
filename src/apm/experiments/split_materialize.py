"""Deterministic train/test split materialization for interim dataset records."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import pandas as pd

from apm.experiments.matrix import DatasetSpec, default_dataset_tokens, parse_dataset_specs


@dataclass(frozen=True, slots=True)
class SplitMaterializeRequest:
    """Request for deterministic train/test split materialization."""

    project_root: Path
    dataset_specs: tuple[DatasetSpec, ...]
    train_ratio: float
    seed: int
    output_root: Path


@dataclass(frozen=True, slots=True)
class DatasetSplitOutput:
    """Output artifact pointers and counts for one dataset split materialization."""

    dataset_id: str
    source_split: str
    output_dir: Path
    metadata_path: Path
    assignments_path: Path
    train_counts_by_label: Mapping[str, int]
    test_counts_by_label: Mapping[str, int]


def materialize_train_test_splits(request: SplitMaterializeRequest) -> tuple[DatasetSplitOutput, ...]:
    """Materialize deterministic train/test splits per dataset spec and per label."""

    _validate_request(request)
    outputs: list[DatasetSplitOutput] = []

    for dataset_index, dataset_spec in enumerate(request.dataset_specs):
        output = _materialize_one_dataset(
            project_root=request.project_root,
            output_root=request.output_root,
            dataset_spec=dataset_spec,
            train_ratio=request.train_ratio,
            seed=request.seed,
            dataset_index=dataset_index,
        )
        outputs.append(output)

    return tuple(outputs)


def build_request_from_args(args: argparse.Namespace) -> SplitMaterializeRequest:
    """Build typed split-materialization request from CLI args."""

    project_root = args.project_root.resolve()
    dataset_specs = parse_dataset_specs(tuple(args.datasets))
    output_root = (project_root / args.output_root).resolve()

    return SplitMaterializeRequest(
        project_root=project_root,
        dataset_specs=dataset_specs,
        train_ratio=float(args.train_ratio),
        seed=int(args.seed),
        output_root=output_root,
    )


def build_arg_parser() -> argparse.ArgumentParser:
    """Build CLI parser for split materialization."""

    parser = argparse.ArgumentParser(description="Materialize deterministic train/test dataset splits.")
    parser.add_argument("--project-root", type=Path, default=Path.cwd(), help="Repository root directory.")
    parser.add_argument(
        "--datasets",
        nargs="*",
        default=list(default_dataset_tokens()),
        help="Dataset specs in `<dataset_id>:<split>` form.",
    )
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.7,
        help="Fraction of each label assigned to train split (remaining records go to test).",
    )
    parser.add_argument("--seed", type=int, default=42, help="Deterministic seed for split assignment.")
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("data/interim/splits"),
        help="Output root directory for train/test split artifacts.",
    )
    return parser


def main() -> int:
    """CLI entrypoint for train/test split materialization."""

    parser = build_arg_parser()
    args = parser.parse_args()
    request = build_request_from_args(args)
    outputs = materialize_train_test_splits(request)

    rows = [
        {
            "dataset_id": output.dataset_id,
            "source_split": output.source_split,
            "output_dir": str(output.output_dir),
            "metadata_path": str(output.metadata_path),
            "assignments_path": str(output.assignments_path),
            "train_counts_by_label": dict(output.train_counts_by_label),
            "test_counts_by_label": dict(output.test_counts_by_label),
        }
        for output in outputs
    ]
    print(json.dumps(rows, indent=2))
    return 0


def _validate_request(request: SplitMaterializeRequest) -> None:
    """Validate split request invariants."""

    if not request.dataset_specs:
        raise ValueError("dataset_specs cannot be empty")
    if request.train_ratio <= 0.0 or request.train_ratio >= 1.0:
        raise ValueError("train_ratio must be in open interval (0, 1)")


def _materialize_one_dataset(
    *,
    project_root: Path,
    output_root: Path,
    dataset_spec: DatasetSpec,
    train_ratio: float,
    seed: int,
    dataset_index: int,
) -> DatasetSplitOutput:
    """Materialize train/test split artifacts for one dataset spec."""

    label_frames: dict[str, pd.DataFrame] = {}
    train_frames_by_label: dict[str, pd.DataFrame] = {}
    test_frames_by_label: dict[str, pd.DataFrame] = {}

    for label_offset, label in enumerate(("human", "ai")):
        frame = _load_source_label_frame(
            project_root=project_root,
            dataset_spec=dataset_spec,
            label=label,
        )
        if len(frame.index) < 2:
            raise ValueError(
                f"Need at least 2 `{label}` records to create train/test splits for "
                f"{dataset_spec.dataset_id}:{dataset_spec.split}."
            )

        shuffled = frame.sample(
            frac=1.0,
            random_state=seed + 1000 * (dataset_index + 1) + 100 * (label_offset + 1),
        ).reset_index(drop=True)

        train_count = int(len(shuffled.index) * train_ratio)
        if train_count <= 0 or train_count >= len(shuffled.index):
            raise ValueError(
                f"Invalid train_count={train_count} for `{label}` records in "
                f"{dataset_spec.dataset_id}:{dataset_spec.split}; adjust train_ratio."
            )

        train_frames_by_label[label] = shuffled.iloc[:train_count].copy()
        test_frames_by_label[label] = shuffled.iloc[train_count:].copy()
        label_frames[label] = frame

    dataset_output_dir = output_root / dataset_spec.dataset_id / dataset_spec.split
    assignments_path = dataset_output_dir / "split_assignments.jsonl"
    metadata_path = dataset_output_dir / "split_metadata.json"

    train_output_dir = dataset_output_dir / "train"
    test_output_dir = dataset_output_dir / "test"
    _write_split_frames(train_output_dir, train_frames_by_label)
    _write_split_frames(test_output_dir, test_frames_by_label)

    train_full = pd.concat(
        [train_frames_by_label["human"], train_frames_by_label["ai"]],
        ignore_index=True,
    )
    test_full = pd.concat(
        [test_frames_by_label["human"], test_frames_by_label["ai"]],
        ignore_index=True,
    )
    train_full.to_parquet(dataset_output_dir / "train.parquet", index=False)
    test_full.to_parquet(dataset_output_dir / "test.parquet", index=False)

    assignment_rows = _build_assignments_rows(
        dataset_spec=dataset_spec,
        train_frames_by_label=train_frames_by_label,
        test_frames_by_label=test_frames_by_label,
    )
    _write_jsonl(assignments_path, assignment_rows)

    metadata_payload = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "dataset_id": dataset_spec.dataset_id,
        "source_split": dataset_spec.split,
        "seed": seed,
        "train_ratio": train_ratio,
        "source_counts_by_label": {
            "human": int(len(label_frames["human"].index)),
            "ai": int(len(label_frames["ai"].index)),
        },
        "train_counts_by_label": {
            "human": int(len(train_frames_by_label["human"].index)),
            "ai": int(len(train_frames_by_label["ai"].index)),
        },
        "test_counts_by_label": {
            "human": int(len(test_frames_by_label["human"].index)),
            "ai": int(len(test_frames_by_label["ai"].index)),
        },
        "paths": {
            "assignments": str(assignments_path),
            "train_parquet": str(dataset_output_dir / "train.parquet"),
            "test_parquet": str(dataset_output_dir / "test.parquet"),
            "train_by_label": {
                "human": str(train_output_dir / "human" / "sampled_records.parquet"),
                "ai": str(train_output_dir / "ai" / "sampled_records.parquet"),
            },
            "test_by_label": {
                "human": str(test_output_dir / "human" / "sampled_records.parquet"),
                "ai": str(test_output_dir / "ai" / "sampled_records.parquet"),
            },
        },
    }
    _write_json(metadata_path, metadata_payload)

    return DatasetSplitOutput(
        dataset_id=dataset_spec.dataset_id,
        source_split=dataset_spec.split,
        output_dir=dataset_output_dir,
        metadata_path=metadata_path,
        assignments_path=assignments_path,
        train_counts_by_label=metadata_payload["train_counts_by_label"],
        test_counts_by_label=metadata_payload["test_counts_by_label"],
    )


def _load_source_label_frame(*, project_root: Path, dataset_spec: DatasetSpec, label: str) -> pd.DataFrame:
    """Load one source label parquet from interim dataset materialization outputs."""

    source_path = (
        project_root
        / "data"
        / "interim"
        / "datasets"
        / dataset_spec.dataset_id
        / dataset_spec.split
        / label
        / "sampled_records.parquet"
    )
    if not source_path.exists():
        raise FileNotFoundError(f"Missing source label parquet: {source_path}")

    frame = pd.read_parquet(source_path)
    required_columns = {"sample_id", "text"}
    missing_columns = sorted(column for column in required_columns if column not in frame.columns)
    if missing_columns:
        raise ValueError(f"Missing columns {missing_columns!r} in source parquet: {source_path}")

    normalized = frame.copy()
    normalized["dataset_id"] = dataset_spec.dataset_id
    normalized["dataset_split"] = dataset_spec.split
    normalized["label"] = label
    return normalized


def _write_split_frames(split_output_dir: Path, frames_by_label: Mapping[str, pd.DataFrame]) -> None:
    """Write label-partitioned parquet files for one split name (`train` or `test`)."""

    for label, frame in frames_by_label.items():
        label_dir = split_output_dir / label
        label_dir.mkdir(parents=True, exist_ok=True)
        frame.to_parquet(label_dir / "sampled_records.parquet", index=False)


def _build_assignments_rows(
    *,
    dataset_spec: DatasetSpec,
    train_frames_by_label: Mapping[str, pd.DataFrame],
    test_frames_by_label: Mapping[str, pd.DataFrame],
) -> list[dict[str, Any]]:
    """Build row-wise split assignment diagnostics for reproducibility."""

    rows: list[dict[str, Any]] = []
    for split_name, frames_by_label in (("train", train_frames_by_label), ("test", test_frames_by_label)):
        for label, frame in frames_by_label.items():
            for row in frame.itertuples(index=False):
                sample_id = getattr(row, "sample_id")
                rows.append(
                    {
                        "dataset_id": dataset_spec.dataset_id,
                        "source_split": dataset_spec.split,
                        "sample_id": sample_id,
                        "label": label,
                        "assigned_split": split_name,
                    }
                )
    rows.sort(key=lambda item: (str(item["sample_id"]), str(item["assigned_split"])))
    return rows


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    """Write deterministic JSON payload."""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    """Write JSONL rows with deterministic newline termination."""

    path.parent.mkdir(parents=True, exist_ok=True)
    serialized = "\n".join(json.dumps(dict(row), sort_keys=True) for row in rows)
    if serialized:
        serialized += "\n"
    path.write_text(serialized, encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
