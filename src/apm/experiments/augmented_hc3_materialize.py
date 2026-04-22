"""Materialize augmented HC3 text folders into train/test split parquet scenarios."""

from __future__ import annotations

import argparse
import json
import re
import shutil
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal, Mapping, Sequence

import pandas as pd


Label = Literal["human", "ai"]


@dataclass(frozen=True, slots=True)
class ParsedTextFilename:
    """Parsed metadata from one `*.txt` sample filename."""

    ordinal: int
    source_sample_id: str
    filename: str


@dataclass(frozen=True, slots=True)
class ScenarioDefinition:
    """One evaluation scenario with selected test-source variants by label."""

    split_name: str
    test_human_source: str
    test_ai_source: str


@dataclass(frozen=True, slots=True)
class AugmentedHc3MaterializeRequest:
    """Configuration for materializing augmented HC3 scenarios."""

    project_root: Path
    input_root: Path
    output_root: Path
    train_source: str
    baseline_test_source: str
    variant_sources: tuple[str, ...]
    output_split_prefix: str


@dataclass(frozen=True, slots=True)
class AugmentedHc3MaterializeResult:
    """Output summary for augmented HC3 scenario materialization."""

    output_root: Path
    scenario_splits: tuple[str, ...]
    manifest_path: Path
    split_names_path: Path


def materialize_augmented_hc3_scenarios(request: AugmentedHc3MaterializeRequest) -> AugmentedHc3MaterializeResult:
    """Build split artifacts for baseline and augmentation-composition scenarios."""

    _validate_request(request)

    train_frames = {
        "human": _load_split_label_frame(
            split_dir=request.input_root / request.train_source,
            label="human",
            sample_id_mode="source_sample_id",
            source_variant=request.train_source,
        ),
        "ai": _load_split_label_frame(
            split_dir=request.input_root / request.train_source,
            label="ai",
            sample_id_mode="source_sample_id",
            source_variant=request.train_source,
        ),
    }
    baseline_frames = {
        "human": _load_split_label_frame(
            split_dir=request.input_root / request.baseline_test_source,
            label="human",
            sample_id_mode="ordinal_by_label",
            source_variant=request.baseline_test_source,
        ),
        "ai": _load_split_label_frame(
            split_dir=request.input_root / request.baseline_test_source,
            label="ai",
            sample_id_mode="ordinal_by_label",
            source_variant=request.baseline_test_source,
        ),
    }

    variant_frames: dict[str, dict[Label, pd.DataFrame]] = {}
    for variant_source in request.variant_sources:
        variant_human = _load_split_label_frame(
            split_dir=request.input_root / variant_source,
            label="human",
            sample_id_mode="ordinal_by_label",
            source_variant=variant_source,
        )
        variant_ai = _load_split_label_frame(
            split_dir=request.input_root / variant_source,
            label="ai",
            sample_id_mode="ordinal_by_label",
            source_variant=variant_source,
        )
        _validate_matching_ordinals(
            baseline_frame=baseline_frames["human"],
            variant_frame=variant_human,
            variant_source=variant_source,
            label="human",
        )
        _validate_matching_ordinals(
            baseline_frame=baseline_frames["ai"],
            variant_frame=variant_ai,
            variant_source=variant_source,
            label="ai",
        )
        variant_frames[variant_source] = {
            "human": variant_human,
            "ai": variant_ai,
        }

    scenarios = _build_scenarios(
        baseline_test_source=request.baseline_test_source,
        variant_sources=request.variant_sources,
        output_split_prefix=request.output_split_prefix,
    )

    for scenario in scenarios:
        test_human_frame = (
            baseline_frames["human"]
            if scenario.test_human_source == request.baseline_test_source
            else variant_frames[scenario.test_human_source]["human"]
        )
        test_ai_frame = (
            baseline_frames["ai"]
            if scenario.test_ai_source == request.baseline_test_source
            else variant_frames[scenario.test_ai_source]["ai"]
        )
        _write_split_artifacts(
            output_root=request.output_root,
            scenario=scenario,
            train_human=train_frames["human"],
            train_ai=train_frames["ai"],
            test_human=test_human_frame,
            test_ai=test_ai_frame,
        )

    manifest_path = request.output_root / "augmented_hc3_scenarios_manifest.json"
    split_names_path = request.output_root / "augmented_hc3_split_names.txt"

    manifest_payload = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "project_root": str(request.project_root),
        "input_root": str(request.input_root),
        "output_root": str(request.output_root),
        "train_source": request.train_source,
        "baseline_test_source": request.baseline_test_source,
        "variant_sources": list(request.variant_sources),
        "scenario_splits": [
            {
                "split_name": scenario.split_name,
                "test_human_source": scenario.test_human_source,
                "test_ai_source": scenario.test_ai_source,
            }
            for scenario in scenarios
        ],
    }
    request.output_root.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest_payload, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")
    split_names_path.write_text("\n".join(scenario.split_name for scenario in scenarios) + "\n", encoding="utf-8")

    return AugmentedHc3MaterializeResult(
        output_root=request.output_root,
        scenario_splits=tuple(scenario.split_name for scenario in scenarios),
        manifest_path=manifest_path,
        split_names_path=split_names_path,
    )


def build_request_from_args(args: argparse.Namespace) -> AugmentedHc3MaterializeRequest:
    """Build typed request from CLI args."""

    project_root = args.project_root.resolve()
    input_root = (project_root / args.input_root).resolve()
    output_root = (project_root / args.output_root).resolve()

    if args.variant_sources:
        variant_sources = tuple(args.variant_sources)
    else:
        variant_sources = discover_variant_sources(
            input_root=input_root,
            train_source=args.train_source,
            baseline_test_source=args.baseline_test_source,
        )

    return AugmentedHc3MaterializeRequest(
        project_root=project_root,
        input_root=input_root,
        output_root=output_root,
        train_source=args.train_source,
        baseline_test_source=args.baseline_test_source,
        variant_sources=variant_sources,
        output_split_prefix=args.output_split_prefix,
    )


def build_arg_parser() -> argparse.ArgumentParser:
    """Build CLI parser for augmented HC3 scenario materialization."""

    parser = argparse.ArgumentParser(description="Materialize augmented HC3 scenarios into interim split artifacts.")
    parser.add_argument("--project-root", type=Path, default=Path.cwd(), help="Repository root directory.")
    parser.add_argument(
        "--input-root",
        type=Path,
        default=Path("data/augmented_data/hc3"),
        help="Input root containing `<split>/<label>/*.txt` folders.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("data/interim/splits/hc3"),
        help="Output root where scenario split artifacts are written.",
    )
    parser.add_argument("--train-source", type=str, default="all_train", help="Input folder used as train source.")
    parser.add_argument("--baseline-test-source", type=str, default="test", help="Input folder used as baseline test source.")
    parser.add_argument(
        "--variant-sources",
        nargs="*",
        default=(),
        help="Optional explicit variant folders; when omitted, all non-log folders except train/baseline are used.",
    )
    parser.add_argument("--output-split-prefix", type=str, default="aug", help="Prefix for generated split names.")
    return parser


def main() -> int:
    """CLI entrypoint."""

    parser = build_arg_parser()
    args = parser.parse_args()
    request = build_request_from_args(args)
    result = materialize_augmented_hc3_scenarios(request)

    print(
        json.dumps(
            {
                "output_root": str(result.output_root),
                "manifest_path": str(result.manifest_path),
                "split_names_path": str(result.split_names_path),
                "scenario_splits": list(result.scenario_splits),
            },
            indent=2,
        )
    )
    return 0


def discover_variant_sources(
    *,
    input_root: Path,
    train_source: str,
    baseline_test_source: str,
) -> tuple[str, ...]:
    """Discover variant folders under input root."""

    if not input_root.exists():
        raise FileNotFoundError(f"Missing input root: {input_root}")
    if not input_root.is_dir():
        raise ValueError(f"input_root must be a directory: {input_root}")

    excluded = {train_source, baseline_test_source, "logs"}
    discovered = [
        path.name
        for path in sorted(input_root.iterdir())
        if path.is_dir() and path.name not in excluded
    ]
    if not discovered:
        raise ValueError(
            f"No variant sources discovered in {input_root}. "
            f"Expected folders other than {sorted(excluded)!r}."
        )
    return tuple(discovered)


def _validate_request(request: AugmentedHc3MaterializeRequest) -> None:
    """Validate request invariants."""

    if not request.train_source.strip():
        raise ValueError("train_source must be non-empty.")
    if not request.baseline_test_source.strip():
        raise ValueError("baseline_test_source must be non-empty.")
    if not request.output_split_prefix.strip():
        raise ValueError("output_split_prefix must be non-empty.")
    if not request.variant_sources:
        raise ValueError("variant_sources cannot be empty.")

    for source_name in (request.train_source, request.baseline_test_source, *request.variant_sources):
        source_dir = request.input_root / source_name
        if not source_dir.exists():
            raise FileNotFoundError(f"Missing source directory: {source_dir}")
        if not source_dir.is_dir():
            raise ValueError(f"Source path must be directory: {source_dir}")
        for label in ("human", "ai"):
            label_dir = source_dir / label
            if not label_dir.exists():
                raise FileNotFoundError(f"Missing label directory: {label_dir}")


def _build_scenarios(
    *,
    baseline_test_source: str,
    variant_sources: Sequence[str],
    output_split_prefix: str,
) -> tuple[ScenarioDefinition, ...]:
    """Build baseline + three composition scenarios per variant source."""

    scenarios: list[ScenarioDefinition] = [
        ScenarioDefinition(
            split_name=f"{_normalize_token(output_split_prefix)}_baseline",
            test_human_source=baseline_test_source,
            test_ai_source=baseline_test_source,
        )
    ]

    for variant_source in variant_sources:
        variant_suffix = _variant_suffix(
            variant_source=variant_source,
            baseline_test_source=baseline_test_source,
        )
        scenario_prefix = f"{_normalize_token(output_split_prefix)}_{_normalize_token(variant_suffix)}"
        scenarios.extend(
            [
                ScenarioDefinition(
                    split_name=f"{scenario_prefix}_orig_h_aug_ai",
                    test_human_source=baseline_test_source,
                    test_ai_source=variant_source,
                ),
                ScenarioDefinition(
                    split_name=f"{scenario_prefix}_aug_h_orig_ai",
                    test_human_source=variant_source,
                    test_ai_source=baseline_test_source,
                ),
                ScenarioDefinition(
                    split_name=f"{scenario_prefix}_aug_both",
                    test_human_source=variant_source,
                    test_ai_source=variant_source,
                ),
            ]
        )

    split_names = [scenario.split_name for scenario in scenarios]
    if len(split_names) != len(set(split_names)):
        raise ValueError(f"Duplicate scenario split names resolved: {split_names!r}")

    return tuple(scenarios)


def _variant_suffix(*, variant_source: str, baseline_test_source: str) -> str:
    """Resolve human-readable variant suffix for scenario split names."""

    prefixed = f"{baseline_test_source}_"
    if variant_source.startswith(prefixed):
        return variant_source[len(prefixed) :]
    return variant_source


def _normalize_token(token: str) -> str:
    """Normalize arbitrary token into split-name-safe lowercase text."""

    collapsed = re.sub(r"[^A-Za-z0-9]+", "_", token).strip("_").lower()
    if not collapsed:
        raise ValueError(f"Cannot normalize empty token from: {token!r}")
    return collapsed


def _load_split_label_frame(
    *,
    split_dir: Path,
    label: Label,
    sample_id_mode: Literal["source_sample_id", "ordinal_by_label"],
    source_variant: str,
) -> pd.DataFrame:
    """Load one `<split>/<label>/*.txt` folder into deterministic row frame."""

    label_dir = split_dir / label
    file_paths = sorted(label_dir.glob("*.txt"))
    if not file_paths:
        raise ValueError(f"No txt files found in label directory: {label_dir}")

    rows: list[dict[str, Any]] = []
    seen_ordinals: set[int] = set()
    seen_sample_ids: set[str] = set()

    for file_path in file_paths:
        parsed = _parse_text_filename(file_path.name)
        if parsed.ordinal in seen_ordinals:
            raise ValueError(f"Duplicate ordinal={parsed.ordinal} in {label_dir}")
        seen_ordinals.add(parsed.ordinal)

        text = file_path.read_text(encoding="utf-8")
        if not text.strip():
            raise ValueError(f"Empty text content is not allowed: {file_path}")

        if sample_id_mode == "source_sample_id":
            sample_id = parsed.source_sample_id
        else:
            sample_id = f"{label}:{parsed.ordinal:04d}"

        if sample_id in seen_sample_ids:
            raise ValueError(f"Duplicate sample_id={sample_id!r} resolved for {label_dir}")
        seen_sample_ids.add(sample_id)

        rows.append(
            {
                "sample_id": sample_id,
                "text": text,
                "ordinal": parsed.ordinal,
                "source_sample_id": parsed.source_sample_id,
                "source_file": parsed.filename,
                "source_variant": source_variant,
            }
        )

    frame = pd.DataFrame(rows)
    frame = frame.sort_values(by=["ordinal", "sample_id"], ascending=True, kind="mergesort").reset_index(drop=True)
    return frame


def _parse_text_filename(filename: str) -> ParsedTextFilename:
    """Parse ordinal and source sample id from one `*.txt` filename."""

    stem = Path(filename).stem
    tokens = stem.split("__")
    if len(tokens) < 2:
        raise ValueError(f"Filename must contain ordinal and payload separated by '__': {filename!r}")
    if not re.fullmatch(r"\d{4}", tokens[0]):
        raise ValueError(f"Filename must start with 4-digit ordinal prefix: {filename!r}")

    ordinal = int(tokens[0])
    payload_tokens = tokens[1:]
    if payload_tokens and re.fullmatch(r"\d{4}", payload_tokens[0]):
        payload_tokens = payload_tokens[1:]
    if not payload_tokens:
        raise ValueError(f"Filename payload is empty after removing ordinal prefixes: {filename!r}")

    source_sample_id = "__".join(payload_tokens)
    return ParsedTextFilename(
        ordinal=ordinal,
        source_sample_id=source_sample_id,
        filename=filename,
    )


def _validate_matching_ordinals(
    *,
    baseline_frame: pd.DataFrame,
    variant_frame: pd.DataFrame,
    variant_source: str,
    label: Label,
) -> None:
    """Ensure variant samples align 1:1 to baseline by ordinal index."""

    baseline_ordinals = set(int(value) for value in baseline_frame["ordinal"].tolist())
    variant_ordinals = set(int(value) for value in variant_frame["ordinal"].tolist())
    if baseline_ordinals != variant_ordinals:
        missing = sorted(baseline_ordinals - variant_ordinals)
        extra = sorted(variant_ordinals - baseline_ordinals)
        raise ValueError(
            f"Variant ordinals mismatch for source={variant_source!r}, label={label!r}. "
            f"missing={missing}, extra={extra}"
        )


def _write_split_artifacts(
    *,
    output_root: Path,
    scenario: ScenarioDefinition,
    train_human: pd.DataFrame,
    train_ai: pd.DataFrame,
    test_human: pd.DataFrame,
    test_ai: pd.DataFrame,
) -> None:
    """Write one scenario into split-materialize-compatible parquet/json artifacts."""

    scenario_dir = output_root / scenario.split_name
    if scenario_dir.exists():
        shutil.rmtree(scenario_dir)
    scenario_dir.mkdir(parents=True, exist_ok=False)

    train_human_export = _decorate_frame(
        base=train_human,
        dataset_id="hc3",
        dataset_split=scenario.split_name,
        label="human",
    )
    train_ai_export = _decorate_frame(
        base=train_ai,
        dataset_id="hc3",
        dataset_split=scenario.split_name,
        label="ai",
    )
    test_human_export = _decorate_frame(
        base=test_human,
        dataset_id="hc3",
        dataset_split=scenario.split_name,
        label="human",
    )
    test_ai_export = _decorate_frame(
        base=test_ai,
        dataset_id="hc3",
        dataset_split=scenario.split_name,
        label="ai",
    )

    _write_partition_label_frame(scenario_dir=scenario_dir, partition="train", label="human", frame=train_human_export)
    _write_partition_label_frame(scenario_dir=scenario_dir, partition="train", label="ai", frame=train_ai_export)
    _write_partition_label_frame(scenario_dir=scenario_dir, partition="test", label="human", frame=test_human_export)
    _write_partition_label_frame(scenario_dir=scenario_dir, partition="test", label="ai", frame=test_ai_export)

    pd.concat([train_human_export, train_ai_export], ignore_index=True).to_parquet(
        scenario_dir / "train.parquet",
        index=False,
    )
    pd.concat([test_human_export, test_ai_export], ignore_index=True).to_parquet(
        scenario_dir / "test.parquet",
        index=False,
    )

    assignment_rows = _build_assignment_rows(
        split_name=scenario.split_name,
        train_human=train_human_export,
        train_ai=train_ai_export,
        test_human=test_human_export,
        test_ai=test_ai_export,
    )
    (scenario_dir / "split_assignments.jsonl").write_text(
        "\n".join(json.dumps(row, sort_keys=True) for row in assignment_rows) + "\n",
        encoding="utf-8",
    )

    metadata_payload = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "dataset_id": "hc3",
        "source_split": scenario.split_name,
        "scenario": {
            "split_name": scenario.split_name,
            "test_human_source": scenario.test_human_source,
            "test_ai_source": scenario.test_ai_source,
        },
        "train_counts_by_label": {
            "human": int(len(train_human_export.index)),
            "ai": int(len(train_ai_export.index)),
        },
        "test_counts_by_label": {
            "human": int(len(test_human_export.index)),
            "ai": int(len(test_ai_export.index)),
        },
        "paths": {
            "assignments": str(scenario_dir / "split_assignments.jsonl"),
            "train_parquet": str(scenario_dir / "train.parquet"),
            "test_parquet": str(scenario_dir / "test.parquet"),
            "train_by_label": {
                "human": str(scenario_dir / "train" / "human" / "sampled_records.parquet"),
                "ai": str(scenario_dir / "train" / "ai" / "sampled_records.parquet"),
            },
            "test_by_label": {
                "human": str(scenario_dir / "test" / "human" / "sampled_records.parquet"),
                "ai": str(scenario_dir / "test" / "ai" / "sampled_records.parquet"),
            },
        },
    }
    (scenario_dir / "split_metadata.json").write_text(
        json.dumps(metadata_payload, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _decorate_frame(
    *,
    base: pd.DataFrame,
    dataset_id: str,
    dataset_split: str,
    label: Label,
) -> pd.DataFrame:
    """Attach canonical split columns to one base frame."""

    frame = base.copy()
    frame["dataset_id"] = dataset_id
    frame["dataset_split"] = dataset_split
    frame["label"] = label
    ordered_columns = [
        "sample_id",
        "text",
        "dataset_id",
        "dataset_split",
        "label",
        "ordinal",
        "source_sample_id",
        "source_file",
        "source_variant",
    ]
    return frame[ordered_columns]


def _write_partition_label_frame(
    *,
    scenario_dir: Path,
    partition: Literal["train", "test"],
    label: Label,
    frame: pd.DataFrame,
) -> None:
    """Write one partition/label parquet frame."""

    target = scenario_dir / partition / label / "sampled_records.parquet"
    target.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(target, index=False)


def _build_assignment_rows(
    *,
    split_name: str,
    train_human: pd.DataFrame,
    train_ai: pd.DataFrame,
    test_human: pd.DataFrame,
    test_ai: pd.DataFrame,
) -> list[dict[str, Any]]:
    """Build deterministic split assignment rows."""

    rows: list[dict[str, Any]] = []
    rows.extend(_frame_to_assignment_rows(frame=train_human, split_name=split_name, assigned_split="train", label="human"))
    rows.extend(_frame_to_assignment_rows(frame=train_ai, split_name=split_name, assigned_split="train", label="ai"))
    rows.extend(_frame_to_assignment_rows(frame=test_human, split_name=split_name, assigned_split="test", label="human"))
    rows.extend(_frame_to_assignment_rows(frame=test_ai, split_name=split_name, assigned_split="test", label="ai"))
    rows.sort(key=lambda row: (str(row["assigned_split"]), str(row["label"]), str(row["sample_id"])))
    return rows


def _frame_to_assignment_rows(
    *,
    frame: pd.DataFrame,
    split_name: str,
    assigned_split: Literal["train", "test"],
    label: Label,
) -> list[dict[str, Any]]:
    """Convert one frame into assignment rows."""

    rows: list[dict[str, Any]] = []
    for row in frame.itertuples(index=False):
        rows.append(
            {
                "dataset_id": "hc3",
                "source_split": split_name,
                "sample_id": str(getattr(row, "sample_id")),
                "label": label,
                "assigned_split": assigned_split,
                "source_variant": str(getattr(row, "source_variant")),
                "source_file": str(getattr(row, "source_file")),
            }
        )
    return rows


if __name__ == "__main__":
    raise SystemExit(main())
