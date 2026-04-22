"""Tests for global/local experiment runner helpers."""

from __future__ import annotations

from pathlib import Path

from apm.experiments import global_local_runner as runner
from apm.experiments.matrix import ModelRunSpec


def test_evaluate_global_scopes_supports_hc3_only_when_grid_is_empty() -> None:
    model_run = ModelRunSpec(
        run_id="aigc_detector_env3",
        detector_id="aigc_detector_env3",
        adapter_module="apm.detectors.adapters.aigc_detector_env3",
        adapter_class="AIGCDetectorEnv3",
        config_path=Path("configs/detectors/aigc_detector_env3.detector.json"),
        config_overrides={},
    )
    records = [
        runner.ScoredTextRecord(
            dataset_id="hc3",
            source_split="aug_baseline",
            partition="train",
            label="human",
            sample_id="human:0001",
            text="h train",
        ),
        runner.ScoredTextRecord(
            dataset_id="hc3",
            source_split="aug_baseline",
            partition="train",
            label="ai",
            sample_id="ai:0001",
            text="a train",
        ),
        runner.ScoredTextRecord(
            dataset_id="hc3",
            source_split="aug_baseline",
            partition="test",
            label="human",
            sample_id="human:0002",
            text="h test",
        ),
        runner.ScoredTextRecord(
            dataset_id="hc3",
            source_split="aug_baseline",
            partition="test",
            label="ai",
            sample_id="ai:0002",
            text="a test",
        ),
    ]
    scores = {"aigc_detector_env3": [0.1, 0.9, 0.2, 0.8]}

    rows = runner._evaluate_global_scopes(
        records=records,
        model_runs=(model_run,),
        scores_by_detector=scores,
        threshold_objective="balanced_accuracy",
        hc3_splits=("aug_baseline",),
        grid_splits=(),
        progress_desc=None,
    )

    assert len(rows) == 1
    assert str(rows[0]["scope_id"]) == "hc3"


def test_validate_request_rejects_empty_hc3_and_grid_scopes() -> None:
    request = runner.GlobalLocalExperimentRequest(
        project_root=Path("."),
        model_run_ids=("aigc_detector_env3",),
        hc3_splits=(),
        grid_splits=(),
        threshold_objective="balanced_accuracy",
        batch_size=8,
        output_root=Path("runs/global_local_experiments"),
    )

    try:
        runner._validate_request(request)
    except ValueError as exc:
        assert "At least one of hc3_splits or grid_splits must be non-empty" in str(exc)
    else:
        raise AssertionError("Expected ValueError for empty scope selections.")
