"""Tests for post-processing global/local metrics from precomputed raw scores."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Sequence

import pandas as pd

from apm.experiments import global_local_runner as runner
from apm.experiments.global_local_postprocess import (
    GlobalLocalPostprocessRequest,
    run_postprocess,
)


def _write_split_rows(
    *,
    project_root: Path,
    dataset_id: str,
    source_split: str,
    partition: str,
    label: str,
    rows: Sequence[tuple[str, str]],
) -> None:
    target = (
        project_root
        / "data"
        / "interim"
        / "splits"
        / dataset_id
        / source_split
        / partition
        / label
        / "sampled_records.parquet"
    )
    target.parent.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(
        {
            "sample_id": [sample_id for sample_id, _text in rows],
            "text": [text for _sample_id, text in rows],
            "label": [label for _ in rows],
        }
    )
    frame.to_parquet(target, index=False)


def _write_detector_config(project_root: Path) -> None:
    config_dir = project_root / "configs" / "detectors"
    config_dir.mkdir(parents=True, exist_ok=True)
    payload = {"detector_id": "aigc_detector_env3"}
    (config_dir / "aigc_detector_env3.detector.json").write_text(
        json.dumps(payload, indent=2) + "\n",
        encoding="utf-8",
    )


def _load_jsonl(path: Path) -> list[dict[str, object]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def test_postprocess_recovers_outputs_and_drops_nonfinite_scores(tmp_path: Path) -> None:
    _write_detector_config(tmp_path)

    _write_split_rows(
        project_root=tmp_path,
        dataset_id="hc3",
        source_split="all_train",
        partition="train",
        label="human",
        rows=(
            ("hc3-train-human-1", "normal human sentence"),
            ("medicine:train:103:human:0", "Hello"),
        ),
    )
    _write_split_rows(
        project_root=tmp_path,
        dataset_id="hc3",
        source_split="all_train",
        partition="train",
        label="ai",
        rows=(
            ("hc3-train-ai-1", "ai generated answer"),
            ("hc3-train-ai-2", "ai synthetic content"),
        ),
    )
    _write_split_rows(
        project_root=tmp_path,
        dataset_id="hc3",
        source_split="all_train",
        partition="test",
        label="human",
        rows=(("hc3-test-human-1", "human heldout sample"),),
    )
    _write_split_rows(
        project_root=tmp_path,
        dataset_id="hc3",
        source_split="all_train",
        partition="test",
        label="ai",
        rows=(("hc3-test-ai-1", "ai heldout sample"),),
    )

    _write_split_rows(
        project_root=tmp_path,
        dataset_id="grid",
        source_split="filtered",
        partition="train",
        label="human",
        rows=(("grid-train-human-1", "grid human sample"),),
    )
    _write_split_rows(
        project_root=tmp_path,
        dataset_id="grid",
        source_split="filtered",
        partition="train",
        label="ai",
        rows=(("grid-train-ai-1", "grid ai sample"),),
    )
    _write_split_rows(
        project_root=tmp_path,
        dataset_id="grid",
        source_split="filtered",
        partition="test",
        label="human",
        rows=(("grid-test-human-1", "grid human holdout"),),
    )
    _write_split_rows(
        project_root=tmp_path,
        dataset_id="grid",
        source_split="filtered",
        partition="test",
        label="ai",
        rows=(("grid-test-ai-1", "grid ai holdout"),),
    )

    split_selections = runner._build_split_selections(
        hc3_splits=("all_train",),
        grid_splits=("filtered",),
    )
    records = runner._load_split_records(project_root=tmp_path, split_selections=split_selections)

    raw_dir = tmp_path / "runs" / "global_local_experiments" / "test_run"
    raw_dir.mkdir(parents=True, exist_ok=True)
    raw_scores_path = raw_dir / "raw_scores.jsonl"
    raw_rows: list[dict[str, object]] = []
    for record in records:
        score = 0.8 if record.label == "ai" else 0.2
        if record.sample_id == "medicine:train:103:human:0":
            score = float("nan")
        raw_rows.append(
            {
                "detector_run_id": "aigc_detector_env3",
                "dataset_id": record.dataset_id,
                "source_split": record.source_split,
                "partition": record.partition,
                "label": record.label,
                "sample_id": record.sample_id,
                "score": score,
            }
        )
    raw_scores_path.write_text(
        "\n".join(json.dumps(row, sort_keys=True) for row in raw_rows) + "\n",
        encoding="utf-8",
    )

    request = GlobalLocalPostprocessRequest(
        project_root=tmp_path,
        raw_scores_path=raw_scores_path,
        model_run_ids=("aigc_detector_env3",),
        hc3_splits=("all_train",),
        grid_splits=("filtered",),
        threshold_objective="balanced_accuracy",
        output_dir=raw_dir,
        run_id="test_run",
    )
    result = run_postprocess(request)

    assert result.global_results_path.exists()
    assert result.local_results_path.exists()
    assert result.summary_path.exists()
    assert result.config_snapshot_path.exists()

    global_rows = _load_jsonl(result.global_results_path)
    local_rows = _load_jsonl(result.local_results_path)
    assert len(global_rows) == 2
    assert len(local_rows) == 2

    by_scope_global = {str(row["scope_id"]): row for row in global_rows}
    assert int(by_scope_global["hc3"]["nonfinite_scores_dropped_train"]) == 1
    assert int(by_scope_global["hc3"]["nonfinite_scores_dropped_test"]) == 0
    assert int(by_scope_global["grid"]["nonfinite_scores_dropped_train"]) == 0
    assert int(by_scope_global["grid"]["nonfinite_scores_dropped_test"]) == 0

    by_scope_local = {str(row["scope_id"]): row for row in local_rows}
    assert int(by_scope_local["hc3:all_train"]["nonfinite_scores_dropped_train"]) == 1
    assert int(by_scope_local["grid:filtered"]["nonfinite_scores_dropped_train"]) == 0

    summary = json.loads(result.summary_path.read_text(encoding="utf-8"))
    assert int(summary["global_result_rows"]) == 2
    assert int(summary["local_result_rows"]) == 2
    assert int(summary["dropped_nonfinite_scores_train_total"]) == 2
