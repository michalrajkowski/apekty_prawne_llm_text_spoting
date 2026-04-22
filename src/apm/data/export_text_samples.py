"""Export fixed-size HC3 text samples to plain-text files for downstream editing workflows."""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd


_LABELS: tuple[str, ...] = ("human", "ai")


@dataclass(frozen=True, slots=True)
class ExportTextSamplesRequest:
    """Request payload for deterministic HC3 text export."""

    project_root: Path
    dataset_id: str
    source_split: str
    per_label_sample_size: int
    seed: int
    output_root: Path


@dataclass(frozen=True, slots=True)
class ExportSubsetSummary:
    """Summary of exported examples for one subset (`all_train` or `test`)."""

    subset_name: str
    counts_by_label: dict[str, int]


@dataclass(frozen=True, slots=True)
class ExportTextSamplesResult:
    """Result metadata for one export execution."""

    export_root: Path
    manifest_path: Path
    metadata_path: Path
    subsets: tuple[ExportSubsetSummary, ...]


def export_hc3_text_samples(request: ExportTextSamplesRequest) -> ExportTextSamplesResult:
    """Sample and export deterministic HC3 plain-text files for `all_train` and `test` subsets."""

    _validate_request(request)
    export_root = (request.project_root / request.output_root / request.dataset_id).resolve()
    manifest_path = export_root / "manifest.jsonl"
    metadata_path = export_root / "metadata.json"

    manifests: list[dict[str, Any]] = []
    subset_summaries: list[ExportSubsetSummary] = []
    subset_rows = (
        (
            "all_train",
            request.project_root
            / "data"
            / "interim"
            / "datasets"
            / request.dataset_id
            / request.source_split,
            request.seed,
        ),
        (
            "test",
            request.project_root
            / "data"
            / "interim"
            / "splits"
            / request.dataset_id
            / request.source_split
            / "test",
            request.seed + 10_000,
        ),
    )

    for subset_name, source_root, subset_seed in subset_rows:
        counts_by_label: dict[str, int] = {}
        for label_offset, label in enumerate(_LABELS):
            source_path = source_root / label / "sampled_records.parquet"
            source_frame = _read_source_frame(source_path=source_path)
            sampled_frame = _sample_label_frame(
                source_frame=source_frame,
                sample_size=request.per_label_sample_size,
                seed=subset_seed + label_offset,
            )
            exported_count = _write_label_texts(
                sampled_frame=sampled_frame,
                destination_dir=export_root / subset_name / label,
                manifests=manifests,
                dataset_id=request.dataset_id,
                source_split=request.source_split,
                subset_name=subset_name,
                label=label,
                source_path=source_path,
                seed=subset_seed + label_offset,
            )
            counts_by_label[label] = exported_count
        subset_summaries.append(ExportSubsetSummary(subset_name=subset_name, counts_by_label=counts_by_label))

    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    _write_jsonl(manifest_path, manifests)
    metadata_payload = {
        "dataset_id": request.dataset_id,
        "source_split": request.source_split,
        "seed": request.seed,
        "per_label_sample_size": request.per_label_sample_size,
        "output_root": str(export_root),
        "manifest_path": str(manifest_path),
        "subsets": [
            {
                "subset_name": subset_summary.subset_name,
                "counts_by_label": dict(subset_summary.counts_by_label),
            }
            for subset_summary in subset_summaries
        ],
    }
    metadata_path.write_text(json.dumps(metadata_payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    return ExportTextSamplesResult(
        export_root=export_root,
        manifest_path=manifest_path,
        metadata_path=metadata_path,
        subsets=tuple(subset_summaries),
    )


def _validate_request(request: ExportTextSamplesRequest) -> None:
    """Validate export request values."""

    if not request.dataset_id.strip():
        raise ValueError("dataset_id must be a non-empty string.")
    if not request.source_split.strip():
        raise ValueError("source_split must be a non-empty string.")
    if request.per_label_sample_size <= 0:
        raise ValueError("per_label_sample_size must be > 0.")


def _read_source_frame(source_path: Path) -> pd.DataFrame:
    """Load one label-partitioned source parquet and validate required columns."""

    if not source_path.exists():
        raise FileNotFoundError(f"Missing source parquet: {source_path}")
    frame = pd.read_parquet(source_path)
    required_columns = {"sample_id", "text"}
    missing = sorted(column for column in required_columns if column not in frame.columns)
    if missing:
        raise ValueError(f"Source parquet missing required columns {missing!r}: {source_path}")
    return frame


def _sample_label_frame(source_frame: pd.DataFrame, sample_size: int, seed: int) -> pd.DataFrame:
    """Draw a deterministic random sample from one label frame."""

    if len(source_frame.index) < sample_size:
        raise ValueError(
            f"Requested {sample_size} examples but only {len(source_frame.index)} are available "
            "in source frame."
        )
    return source_frame.sample(n=sample_size, random_state=seed).reset_index(drop=True)


def _write_label_texts(
    *,
    sampled_frame: pd.DataFrame,
    destination_dir: Path,
    manifests: list[dict[str, Any]],
    dataset_id: str,
    source_split: str,
    subset_name: str,
    label: str,
    source_path: Path,
    seed: int,
) -> int:
    """Persist one sampled label frame as `*.txt` files and append manifest rows."""

    destination_dir.mkdir(parents=True, exist_ok=True)
    for existing_path in destination_dir.glob("*.txt"):
        existing_path.unlink()

    used_stems: set[str] = set()
    for row_index, row in enumerate(sampled_frame.itertuples(index=False), start=1):
        sample_id = str(getattr(row, "sample_id"))
        text = str(getattr(row, "text"))
        filename_stem = _make_unique_filename_stem(
            sample_id=sample_id,
            row_index=row_index,
            used_stems=used_stems,
        )
        destination_path = destination_dir / f"{filename_stem}.txt"
        destination_path.write_text(text, encoding="utf-8")
        manifests.append(
            {
                "dataset_id": dataset_id,
                "source_split": source_split,
                "subset_name": subset_name,
                "label": label,
                "sample_id": sample_id,
                "seed": seed,
                "source_path": str(source_path),
                "export_path": str(destination_path),
                "text_length_chars": len(text),
            }
        )

    return len(sampled_frame.index)


def _make_unique_filename_stem(sample_id: str, row_index: int, used_stems: set[str]) -> str:
    """Build deterministic collision-free filename stem from sample id and row index."""

    normalized_sample_id = re.sub(r"[^A-Za-z0-9._-]+", "_", sample_id).strip("_")
    if not normalized_sample_id:
        normalized_sample_id = "sample"
    truncated = normalized_sample_id[:80]
    candidate = f"{row_index:04d}__{truncated}"
    suffix = 2
    while candidate in used_stems:
        candidate = f"{row_index:04d}__{truncated}__{suffix}"
        suffix += 1
    used_stems.add(candidate)
    return candidate


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    """Write JSONL file for deterministic exported-sample manifest."""

    serialized_rows = [json.dumps(row, ensure_ascii=False, sort_keys=True) for row in rows]
    path.write_text("\n".join(serialized_rows) + ("\n" if serialized_rows else ""), encoding="utf-8")


def _build_arg_parser() -> argparse.ArgumentParser:
    """Build command-line parser for export requests."""

    parser = argparse.ArgumentParser(description="Export HC3 random samples to text files.")
    parser.add_argument("--project-root", type=Path, default=Path.cwd(), help="Repository root directory.")
    parser.add_argument("--dataset-id", type=str, default="hc3", help="Dataset id to export.")
    parser.add_argument("--source-split", type=str, default="all_train", help="Source HC3 selector name.")
    parser.add_argument(
        "--per-label-sample-size",
        type=int,
        default=100,
        help="Random sample size for each label (`human`, `ai`) inside each subset.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Deterministic random seed.")
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("data/to_export"),
        help="Output root relative to project root.",
    )
    return parser


def main() -> int:
    """CLI entrypoint for deterministic text export."""

    parser = _build_arg_parser()
    args = parser.parse_args()

    request = ExportTextSamplesRequest(
        project_root=args.project_root.resolve(),
        dataset_id=args.dataset_id,
        source_split=args.source_split,
        per_label_sample_size=args.per_label_sample_size,
        seed=args.seed,
        output_root=args.output_root,
    )
    result = export_hc3_text_samples(request)
    payload = {
        "export_root": str(result.export_root),
        "manifest_path": str(result.manifest_path),
        "metadata_path": str(result.metadata_path),
        "subsets": [
            {
                "subset_name": subset.subset_name,
                "counts_by_label": subset.counts_by_label,
            }
            for subset in result.subsets
        ],
    }
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
