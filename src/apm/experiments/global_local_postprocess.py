"""Post-process global/local experiment outputs from precomputed raw detector scores."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from apm.experiments import global_local_runner as runner
from apm.experiments.matrix import ModelRunSpec, discover_model_runs
from apm.metrics.classification import ThresholdObjective

try:
    from tqdm.auto import tqdm as _tqdm
except ModuleNotFoundError:  # pragma: no cover - fallback path if tqdm is unavailable
    _tqdm = None


ScoreKey = tuple[str, str, str, str, str]


@dataclass(frozen=True, slots=True)
class GlobalLocalPostprocessRequest:
    """Configuration for post-processing saved raw detector scores."""

    project_root: Path
    raw_scores_path: Path
    model_run_ids: tuple[str, ...]
    hc3_splits: tuple[str, ...]
    grid_splits: tuple[str, ...]
    threshold_objective: ThresholdObjective
    output_dir: Path
    run_id: str


@dataclass(frozen=True, slots=True)
class GlobalLocalPostprocessResult:
    """Materialized post-processing artifact paths."""

    run_id: str
    output_dir: Path
    global_results_path: Path
    local_results_path: Path
    summary_path: Path
    config_snapshot_path: Path


def run_postprocess(request: GlobalLocalPostprocessRequest) -> GlobalLocalPostprocessResult:
    """Materialize missing global/local artifacts from raw score JSONL."""

    _validate_request(request)

    model_runs = discover_model_runs(
        project_root=request.project_root,
        selected_run_ids=request.model_run_ids,
    )
    split_selections = runner._build_split_selections(
        hc3_splits=request.hc3_splits,
        grid_splits=request.grid_splits,
    )
    records = runner._load_split_records(
        project_root=request.project_root,
        split_selections=split_selections,
    )
    scores_by_detector = _load_scores_by_detector_from_raw_scores(
        raw_scores_path=request.raw_scores_path,
        records=records,
        selected_run_ids=request.model_run_ids,
    )

    global_scope_rows = runner._evaluate_global_scopes(
        records=records,
        model_runs=model_runs,
        scores_by_detector=scores_by_detector,
        threshold_objective=request.threshold_objective,
        hc3_splits=request.hc3_splits,
        grid_splits=request.grid_splits,
        progress_desc="Postprocess global scopes",
    )
    local_scope_rows = runner._evaluate_local_scopes(
        records=records,
        model_runs=model_runs,
        scores_by_detector=scores_by_detector,
        threshold_objective=request.threshold_objective,
        split_selections=split_selections,
        progress_desc="Postprocess local scopes",
    )

    request.output_dir.mkdir(parents=True, exist_ok=True)

    global_results_path = request.output_dir / "global_results.jsonl"
    local_results_path = request.output_dir / "local_results.jsonl"
    summary_path = request.output_dir / "summary.json"
    config_snapshot_path = request.output_dir / "config_snapshot.json"

    runner._write_jsonl(path=global_results_path, rows=global_scope_rows)
    runner._write_jsonl(path=local_results_path, rows=local_scope_rows)

    combined_rows = [*global_scope_rows, *local_scope_rows]
    dropped_train_total = sum(int(row["nonfinite_scores_dropped_train"]) for row in combined_rows)
    dropped_test_total = sum(int(row["nonfinite_scores_dropped_test"]) for row in combined_rows)
    summary_payload: dict[str, Any] = {
        "run_id": request.run_id,
        "created_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "raw_scores_path": str(request.raw_scores_path),
        "record_count": len(records),
        "detector_count": len(model_runs),
        "global_result_rows": len(global_scope_rows),
        "local_result_rows": len(local_scope_rows),
        "global_scopes": sorted({str(row["scope_id"]) for row in global_scope_rows}),
        "local_scopes": sorted({str(row["scope_id"]) for row in local_scope_rows}),
        "dropped_nonfinite_scores_train_total": dropped_train_total,
        "dropped_nonfinite_scores_test_total": dropped_test_total,
    }
    runner._write_json(path=summary_path, payload=summary_payload)

    config_payload = {
        "run_id": request.run_id,
        "created_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "project_root": str(request.project_root),
        "raw_scores_path": str(request.raw_scores_path),
        "model_run_ids": list(request.model_run_ids),
        "threshold_objective": request.threshold_objective,
        "hc3_splits": list(request.hc3_splits),
        "grid_splits": list(request.grid_splits),
        "resolved_model_runs": [_model_run_spec_to_dict(model_run) for model_run in model_runs],
    }
    runner._write_json(path=config_snapshot_path, payload=config_payload)

    return GlobalLocalPostprocessResult(
        run_id=request.run_id,
        output_dir=request.output_dir,
        global_results_path=global_results_path,
        local_results_path=local_results_path,
        summary_path=summary_path,
        config_snapshot_path=config_snapshot_path,
    )


def build_request_from_args(args: argparse.Namespace) -> GlobalLocalPostprocessRequest:
    """Build typed post-process request from CLI args."""

    project_root = args.project_root.resolve()
    raw_scores_input = Path(args.raw_scores_path)
    raw_scores_path = raw_scores_input if raw_scores_input.is_absolute() else (project_root / raw_scores_input)
    raw_scores_path = raw_scores_path.resolve()

    output_dir_input = Path(args.output_dir) if args.output_dir is not None else raw_scores_path.parent
    output_dir = output_dir_input if output_dir_input.is_absolute() else (project_root / output_dir_input)
    output_dir = output_dir.resolve()

    if args.model_runs:
        model_run_ids = tuple(args.model_runs)
    else:
        model_run_ids = _infer_model_run_ids(raw_scores_path)

    run_id = args.run_id if args.run_id else output_dir.name

    return GlobalLocalPostprocessRequest(
        project_root=project_root,
        raw_scores_path=raw_scores_path,
        model_run_ids=model_run_ids,
        hc3_splits=tuple(args.hc3_splits),
        grid_splits=tuple(args.grid_splits),
        threshold_objective=args.threshold_objective,
        output_dir=output_dir,
        run_id=run_id,
    )


def build_arg_parser() -> argparse.ArgumentParser:
    """Build CLI parser for post-processing from raw scores."""

    parser = argparse.ArgumentParser(description="Post-process global/local results from raw_scores.jsonl.")
    parser.add_argument("--project-root", type=Path, default=Path.cwd(), help="Repository root directory.")
    parser.add_argument(
        "--raw-scores-path",
        type=Path,
        required=True,
        help="Path to raw_scores.jsonl produced by global_local_runner.",
    )
    parser.add_argument(
        "--model-runs",
        nargs="*",
        default=None,
        help="Optional detector run ids; if omitted, detector runs are inferred from raw scores.",
    )
    parser.add_argument(
        "--hc3-splits",
        nargs="+",
        default=list(runner.DEFAULT_HC3_SPLITS),
        help="HC3 source splits included in this experiment.",
    )
    parser.add_argument(
        "--grid-splits",
        nargs="+",
        default=list(runner.DEFAULT_GRID_SPLITS),
        help="GriD source splits included in this experiment.",
    )
    parser.add_argument(
        "--threshold-objective",
        choices=("balanced_accuracy", "accuracy", "f1"),
        default="balanced_accuracy",
        help="Objective used to select thresholds on train data.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory to write post-processed files. Defaults to raw-scores parent directory.",
    )
    parser.add_argument(
        "--run-id",
        type=str,
        default="",
        help="Optional run id for metadata; defaults to output directory name.",
    )
    return parser


def main() -> int:
    """CLI entrypoint."""

    parser = build_arg_parser()
    args = parser.parse_args()
    request = build_request_from_args(args)
    result = run_postprocess(request)
    print(
        json.dumps(
            {
                "run_id": result.run_id,
                "output_dir": str(result.output_dir),
                "global_results_path": str(result.global_results_path),
                "local_results_path": str(result.local_results_path),
                "summary_path": str(result.summary_path),
                "config_snapshot_path": str(result.config_snapshot_path),
            },
            indent=2,
        )
    )
    return 0


def _validate_request(request: GlobalLocalPostprocessRequest) -> None:
    """Validate post-processing request invariants."""

    if not request.raw_scores_path.exists():
        raise FileNotFoundError(f"Missing raw scores file: {request.raw_scores_path}")
    if request.raw_scores_path.is_dir():
        raise ValueError(f"raw_scores_path must be a file, got directory: {request.raw_scores_path}")
    if not request.model_run_ids:
        raise ValueError("model_run_ids cannot be empty")
    if not request.hc3_splits:
        raise ValueError("hc3_splits cannot be empty")
    if not request.grid_splits:
        raise ValueError("grid_splits cannot be empty")


def _infer_model_run_ids(raw_scores_path: Path) -> tuple[str, ...]:
    """Infer detector run IDs from first-seen order in raw score JSONL."""

    run_ids: list[str] = []
    seen: set[str] = set()
    with raw_scores_path.open("r", encoding="utf-8") as handle:
        lines: Iterable[str] = _with_tqdm(handle, desc="Infer detector ids", unit="row")
        for line in lines:
            if not line.strip():
                continue
            payload = json.loads(line)
            detector_run_id = payload.get("detector_run_id")
            if not isinstance(detector_run_id, str):
                raise ValueError(f"Invalid or missing detector_run_id in raw score row: {payload!r}")
            if detector_run_id in seen:
                continue
            seen.add(detector_run_id)
            run_ids.append(detector_run_id)
    if not run_ids:
        raise ValueError(f"No detector_run_id values found in raw scores file: {raw_scores_path}")
    return tuple(run_ids)


def _load_scores_by_detector_from_raw_scores(
    *,
    raw_scores_path: Path,
    records: Sequence[runner.ScoredTextRecord],
    selected_run_ids: Sequence[str],
) -> dict[str, list[float]]:
    """Load detector scores from JSONL and align them with expected row ordering."""

    selected_set = set(selected_run_ids)
    rows_by_detector: dict[str, list[dict[str, Any]]] = {}
    scores_by_detector: dict[str, list[float]] = {}

    with raw_scores_path.open("r", encoding="utf-8") as handle:
        lines: Iterable[str] = _with_tqdm(handle, desc="Load raw scores", unit="row")
        for line in lines:
            if not line.strip():
                continue
            payload = json.loads(line)
            detector_run_id = payload.get("detector_run_id")
            if not isinstance(detector_run_id, str):
                raise ValueError(f"Invalid or missing detector_run_id in raw score row: {payload!r}")
            if detector_run_id not in selected_set:
                continue

            rows_by_detector.setdefault(detector_run_id, []).append(payload)
            scores_by_detector.setdefault(detector_run_id, []).append(float(payload["score"]))

    outputs: dict[str, list[float]] = {}
    run_id_iterable: Iterable[str] = _with_tqdm(selected_run_ids, desc="Align detector rows", unit="detector")
    for run_id in run_id_iterable:
        detector_rows = rows_by_detector.get(run_id)
        detector_scores = scores_by_detector.get(run_id)
        if detector_rows is None or detector_scores is None:
            raise ValueError(f"Missing raw scores for detector_run_id={run_id!r} in {raw_scores_path}")
        if len(detector_scores) != len(records):
            raise ValueError(
                f"Unexpected score row count for detector_run_id={run_id!r}: "
                f"expected={len(records)}, got={len(detector_scores)}"
            )

        for index, (payload_row, record) in enumerate(zip(detector_rows, records, strict=True)):
            payload_key = _payload_key(payload_row)
            record_key = _record_key(record)
            if payload_key != record_key:
                raise ValueError(
                    f"Raw score row order mismatch for detector_run_id={run_id!r} at position={index}: "
                    f"expected={record_key!r}, got={payload_key!r}"
                )

        outputs[run_id] = detector_scores

    return outputs


def _record_key(record: runner.ScoredTextRecord) -> ScoreKey:
    """Create stable score-key tuple from one split record."""

    return (
        record.dataset_id,
        record.source_split,
        record.partition,
        record.label,
        record.sample_id,
    )


def _payload_key(payload: Mapping[str, Any]) -> ScoreKey:
    """Create stable score-key tuple from one raw-score payload row."""

    dataset_id = payload.get("dataset_id")
    source_split = payload.get("source_split")
    partition = payload.get("partition")
    label = payload.get("label")
    sample_id = payload.get("sample_id")

    if not isinstance(dataset_id, str):
        raise ValueError(f"Invalid or missing dataset_id in raw score row: {payload!r}")
    if not isinstance(source_split, str):
        raise ValueError(f"Invalid or missing source_split in raw score row: {payload!r}")
    if not isinstance(partition, str):
        raise ValueError(f"Invalid or missing partition in raw score row: {payload!r}")
    if not isinstance(label, str):
        raise ValueError(f"Invalid or missing label in raw score row: {payload!r}")
    if not isinstance(sample_id, str):
        raise ValueError(f"Invalid or missing sample_id in raw score row: {payload!r}")

    return (dataset_id, source_split, partition, label, sample_id)


def _model_run_spec_to_dict(model_run: ModelRunSpec) -> dict[str, Any]:
    """Serialize model run spec into JSON-friendly dictionary."""

    return {
        "run_id": model_run.run_id,
        "detector_id": model_run.detector_id,
        "adapter_module": model_run.adapter_module,
        "adapter_class": model_run.adapter_class,
        "config_path": str(model_run.config_path),
        "config_overrides": dict(model_run.config_overrides),
    }


def _with_tqdm(items: Iterable[Any], *, desc: str, unit: str) -> Iterable[Any]:
    """Wrap iterable with tqdm progress bar when tqdm is available."""

    if _tqdm is None:
        return items
    return _tqdm(items, desc=desc, unit=unit)


if __name__ == "__main__":
    raise SystemExit(main())
