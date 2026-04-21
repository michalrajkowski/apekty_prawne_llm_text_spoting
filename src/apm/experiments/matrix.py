"""Experiment matrix helpers for dataset and detector run selection."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence


@dataclass(frozen=True, slots=True)
class DatasetSpec:
    """One dataset/split source specification for scoring."""

    dataset_id: str
    split: str


@dataclass(frozen=True, slots=True)
class ModelRunSpec:
    """One detector run configuration resolved from detector configs."""

    run_id: str
    detector_id: str
    adapter_module: str
    adapter_class: str
    config_path: Path
    config_overrides: Mapping[str, Any]


DEFAULT_DATASET_SPECS: tuple[DatasetSpec, ...] = (
    DatasetSpec(dataset_id="hc3", split="all_train"),
    DatasetSpec(dataset_id="grid", split="filtered"),
)

DEFAULT_MODEL_RUN_IDS: tuple[str, ...] = (
    "aigc_detector_env3",
    "seqxgpt:gpt2_medium",
    "seqxgpt:gpt_j_6b",
)

DETECTOR_ADAPTERS: dict[str, tuple[str, str]] = {
    "aigc_detector_env3": ("apm.detectors.adapters.aigc_detector_env3", "AIGCDetectorEnv3"),
    "aigc_detector_env3short": ("apm.detectors.adapters.aigc_detector_env3short", "AIGCDetectorEnv3Short"),
    "detectgpt_light": ("apm.detectors.adapters.detectgpt_light", "DetectGptLightDetector"),
    "fast_detectgpt": ("apm.detectors.adapters.fast_detectgpt", "FastDetectGptDetector"),
    "gltr_gpt2_small": ("apm.detectors.adapters.gltr_gpt2_small", "GLTRGpt2SmallDetector"),
    "ghostbuster": ("apm.detectors.adapters.ghostbuster", "GhostbusterDetector"),
    "radar_vicuna_7b": ("apm.detectors.adapters.radar_vicuna_7b", "RadarVicuna7BDetector"),
    "seqxgpt": ("apm.detectors.adapters.seqxgpt", "SeqXGPTDetector"),
    "synthid_text": ("apm.detectors.adapters.synthid_text", "SynthIDTextDetector"),
}


def parse_dataset_specs(raw_specs: Sequence[str]) -> tuple[DatasetSpec, ...]:
    """Parse dataset specs from `<dataset_id>:<split>` textual tokens."""

    if not raw_specs:
        return DEFAULT_DATASET_SPECS

    parsed: list[DatasetSpec] = []
    for token in raw_specs:
        normalized = token.strip()
        if not normalized:
            continue
        dataset_id, separator, split = normalized.partition(":")
        if separator != ":" or not dataset_id or not split:
            raise ValueError(
                f"Invalid dataset token {token!r}. Expected '<dataset_id>:<split>' format."
            )
        parsed.append(DatasetSpec(dataset_id=dataset_id, split=split))

    if not parsed:
        raise ValueError("At least one non-empty dataset spec is required.")
    return tuple(parsed)


def default_dataset_tokens() -> tuple[str, ...]:
    """Return default dataset specs in CLI-friendly string form."""

    return tuple(f"{dataset.dataset_id}:{dataset.split}" for dataset in DEFAULT_DATASET_SPECS)


def discover_model_runs(project_root: Path, selected_run_ids: Sequence[str]) -> tuple[ModelRunSpec, ...]:
    """Resolve selected detector run ids into concrete run specs."""

    if not selected_run_ids:
        raise ValueError("selected_run_ids cannot be empty.")

    config_dir = project_root / "configs" / "detectors"
    config_paths = sorted(config_dir.glob("*.detector.json"))
    discovered_runs: list[ModelRunSpec] = []

    for config_path in config_paths:
        config_data = json.loads(config_path.read_text(encoding="utf-8"))
        if not isinstance(config_data, dict):
            raise ValueError(f"Config must be a JSON object: {config_path}")

        detector_id = config_data.get("detector_id")
        if not isinstance(detector_id, str):
            raise ValueError(f"Missing string detector_id in {config_path}")
        if detector_id not in DETECTOR_ADAPTERS:
            continue

        adapter_module, adapter_class = DETECTOR_ADAPTERS[detector_id]
        if detector_id in {"seqxgpt", "synthid_text"}:
            variants = config_data.get("model_variants", [])
            if not isinstance(variants, list):
                raise ValueError(f"Config key 'model_variants' must be a list in {config_path}")
            for variant in variants:
                if not isinstance(variant, dict):
                    raise ValueError(f"Each model variant must be an object in {config_path}")
                variant_id = variant.get("variant_id")
                estimated_vram_gb = variant.get("estimated_vram_gb")
                if not isinstance(variant_id, str):
                    raise ValueError(f"Variant missing string 'variant_id' in {config_path}")
                if not isinstance(estimated_vram_gb, (int, float)):
                    raise ValueError(f"Variant missing numeric 'estimated_vram_gb' in {config_path}")

                run_id = f"{detector_id}:{variant_id}"
                discovered_runs.append(
                    ModelRunSpec(
                        run_id=run_id,
                        detector_id=detector_id,
                        adapter_module=adapter_module,
                        adapter_class=adapter_class,
                        config_path=config_path,
                        config_overrides={
                            "variant_id": variant_id,
                            "max_supported_vram_gb": float(estimated_vram_gb),
                        },
                    )
                )
        else:
            discovered_runs.append(
                ModelRunSpec(
                    run_id=detector_id,
                    detector_id=detector_id,
                    adapter_module=adapter_module,
                    adapter_class=adapter_class,
                    config_path=config_path,
                    config_overrides={},
                )
            )

    run_by_id: dict[str, ModelRunSpec] = {}
    for run in discovered_runs:
        if run.run_id in run_by_id:
            raise ValueError(f"Duplicate run_id discovered: {run.run_id}")
        run_by_id[run.run_id] = run

    selected_runs: list[ModelRunSpec] = []
    missing: list[str] = []
    for run_id in selected_run_ids:
        selected = run_by_id.get(run_id)
        if selected is None:
            missing.append(run_id)
            continue
        selected_runs.append(selected)

    if missing:
        available = ", ".join(sorted(run_by_id))
        raise ValueError(f"Unknown run_id(s): {', '.join(missing)}. Available run_ids: {available}")

    return tuple(selected_runs)
