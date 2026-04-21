"""Integration-style tests for immutable experiment runner outputs."""

from __future__ import annotations

import json
import importlib
import types
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from apm.experiments.matrix import DatasetSpec
from apm.experiments.runner import ExperimentRequest, run_experiment


class _FakeDetector:
    """Deterministic fake detector for end-to-end runner tests."""

    @classmethod
    def initialize(cls, config: dict[str, Any]) -> "_FakeDetector":
        _ = config
        return cls()

    def predict_batch(self, texts: list[str]) -> list[float]:
        return [0.9 if " ai " in f" {text.lower()} " else 0.1 for text in texts]

    def delete(self) -> None:
        return None


def _write_label_parquet(
    *,
    project_root: Path,
    dataset_id: str,
    split: str,
    label: str,
    rows: list[dict[str, str]],
) -> None:
    target = (
        project_root
        / "data"
        / "interim"
        / "datasets"
        / dataset_id
        / split
        / label
        / "sampled_records.parquet"
    )
    target.parent.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(rows)
    frame.to_parquet(target, index=False)


def _write_detector_config(project_root: Path) -> None:
    config_dir = project_root / "configs" / "detectors"
    config_dir.mkdir(parents=True, exist_ok=True)
    config_payload = {
        "detector_id": "aigc_detector_env3",
        "model_id": "fake",
        "device": "cpu",
        "local_files_only": True,
        "trust_remote_code": False,
        "cache_dir": ".cache/huggingface",
    }
    (config_dir / "aigc_detector_env3.detector.json").write_text(
        json.dumps(config_payload, indent=2) + "\n",
        encoding="utf-8",
    )


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def test_run_experiment_writes_immutable_artifacts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    human_rows = [
        {"sample_id": f"h-{idx}", "text": f"human sample {idx}", "label": "human"}
        for idx in range(8)
    ]
    ai_rows = [
        {"sample_id": f"a-{idx}", "text": f"ai sample {idx}", "label": "ai"}
        for idx in range(8)
    ]
    _write_label_parquet(
        project_root=tmp_path,
        dataset_id="toy",
        split="train",
        label="human",
        rows=human_rows,
    )
    _write_label_parquet(
        project_root=tmp_path,
        dataset_id="toy",
        split="train",
        label="ai",
        rows=ai_rows,
    )
    _write_detector_config(project_root=tmp_path)

    original_import_module = importlib.import_module

    def _fake_import_module(name: str) -> Any:
        if name == "apm.detectors.adapters.aigc_detector_env3":
            return types.SimpleNamespace(AIGCDetectorEnv3=_FakeDetector)
        return original_import_module(name)

    monkeypatch.setattr("apm.experiments.runner.importlib.import_module", _fake_import_module)

    request = ExperimentRequest(
        project_root=tmp_path,
        dataset_specs=(DatasetSpec(dataset_id="toy", split="train"),),
        model_run_ids=("aigc_detector_env3",),
        train_examples_per_label=3,
        evaluation_examples_per_label=3,
        seed=42,
        threshold_objective="balanced_accuracy",
        output_root=tmp_path / "runs" / "experiments",
    )

    first = run_experiment(request)
    second = run_experiment(request)

    assert first.run_id != second.run_id
    assert first.run_dir.exists()
    assert second.run_dir.exists()

    for path in (
        first.split_assignments_path,
        first.predictions_path,
        first.thresholds_path,
        first.metrics_by_detector_path,
        first.metrics_overall_path,
        first.config_snapshot_path,
    ):
        assert path.exists()

    prediction_rows = [
        json.loads(line)
        for line in first.predictions_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert len(prediction_rows) == 12
    split_rows = [
        json.loads(line)
        for line in first.split_assignments_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert len(split_rows) == 12
    assert sum(1 for row in split_rows if row["split_name"] == "train") == 6
    assert sum(1 for row in split_rows if row["split_name"] == "evaluation") == 6

    metrics = _load_json(first.metrics_by_detector_path)
    eval_metrics = metrics["aigc_detector_env3"]["evaluation"]
    assert eval_metrics["accuracy"] >= 0.9
    assert metrics["aigc_detector_env3"]["probability_metrics"] is not None
    threshold_payload = _load_json(first.thresholds_path)["aigc_detector_env3"]
    assert threshold_payload["candidate_count"] >= 1
    assert isinstance(threshold_payload["candidate_metrics"], list)

    index_path = request.output_root / "index.jsonl"
    index_rows = [line for line in index_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert len(index_rows) == 2
