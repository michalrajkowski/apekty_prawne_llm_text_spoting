"""Experiment package exports with lazy attribute loading."""

from __future__ import annotations

from typing import Any

__all__ = ["ExperimentRequest", "ExperimentResult", "run_experiment"]


def __getattr__(name: str) -> Any:
    """Lazily expose runner symbols without eager module import side effects."""

    if name in {"ExperimentRequest", "ExperimentResult", "run_experiment"}:
        from apm.experiments.runner import ExperimentRequest, ExperimentResult, run_experiment

        exports: dict[str, Any] = {
            "ExperimentRequest": ExperimentRequest,
            "ExperimentResult": ExperimentResult,
            "run_experiment": run_experiment,
        }
        return exports[name]
    raise AttributeError(f"module 'apm.experiments' has no attribute {name!r}")
