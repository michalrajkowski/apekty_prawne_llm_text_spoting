"""Scratch helper: transform streaming JSONL detector scores into plotting-friendly structures."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Summarize raw detector scores for plotting.")
    parser.add_argument(
        "--raw-jsonl",
        type=Path,
        default=Path("scratch/detector_scoring/results/raw_scores.jsonl"),
        help="Input JSONL produced by run_detector_scores.py.",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=Path("scratch/detector_scoring/results/summary_scores.json"),
        help="Output JSON with matrix and long-form rows.",
    )
    parser.add_argument("--project-root", type=Path, default=Path.cwd(), help="Repository root directory.")
    return parser


def _read_jsonl_records(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        parsed = json.loads(line)
        if not isinstance(parsed, dict):
            raise ValueError(f"Expected JSON object rows in {path}.")
        records.append(parsed)
    return records


def _save_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def _extract_example_fields(meta_record: dict[str, Any]) -> tuple[list[str], list[str]]:
    examples = meta_record.get("examples")
    if not isinstance(examples, list):
        raise ValueError("Meta record must contain list field 'examples'.")

    example_ids: list[str] = []
    example_labels: list[str] = []
    for example in examples:
        if not isinstance(example, dict):
            raise ValueError("Every entry in meta 'examples' must be an object.")
        example_id = example.get("example_id")
        label = example.get("label")
        if not isinstance(example_id, str):
            raise ValueError("Each example must include string 'example_id'.")
        if not isinstance(label, str):
            raise ValueError("Each example must include string 'label'.")
        example_ids.append(example_id)
        example_labels.append(label)
    return example_ids, example_labels


def main() -> int:
    parser = _build_arg_parser()
    args = parser.parse_args()

    raw_path = (args.project_root / args.raw_jsonl).resolve()
    summary_path = (args.project_root / args.output_json).resolve()

    records = _read_jsonl_records(raw_path)
    if not records:
        raise ValueError(f"No records found in {raw_path}.")

    meta_records = [record for record in records if record.get("record_type") == "meta"]
    if not meta_records:
        raise ValueError("JSONL input must contain a 'meta' record.")
    meta_record = meta_records[0]

    example_ids, example_labels = _extract_example_fields(meta_record)

    score_runs = [record for record in records if record.get("record_type") == "score_run"]
    error_runs = [record for record in records if record.get("record_type") == "run_error"]

    model_ids: list[str] = []
    scores_matrix_by_model: list[list[float]] = []
    long_rows: list[dict[str, Any]] = []

    for score_run in score_runs:
        run_id = score_run.get("run_id")
        scores = score_run.get("scores")
        if not isinstance(run_id, str):
            raise ValueError("Each score run must include string 'run_id'.")
        if not isinstance(scores, list):
            raise ValueError("Each score run must include list 'scores'.")
        if len(scores) != len(example_ids):
            raise ValueError(
                f"Score run '{run_id}' has {len(scores)} scores, expected {len(example_ids)}."
            )

        normalized_scores = [float(value) for value in scores]
        model_ids.append(run_id)
        scores_matrix_by_model.append(normalized_scores)

        for index, score in enumerate(normalized_scores):
            long_rows.append(
                {
                    "example_id": example_ids[index],
                    "label": example_labels[index],
                    "model_id": run_id,
                    "score": score,
                }
            )

    summary: dict[str, Any] = {
        "example_ids": example_ids,
        "example_labels": example_labels,
        "model_ids": model_ids,
        "scores_matrix_by_model": scores_matrix_by_model,
        "long_rows": long_rows,
        "run_errors": error_runs,
        "raw_jsonl_path": str(raw_path),
    }
    _save_json(summary_path, summary)

    print(f"Saved summary scores to: {summary_path}")
    print(f"Examples: {len(example_ids)}")
    print(f"Successful model runs: {len(model_ids)}")
    print(f"Failed model runs: {len(error_runs)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
