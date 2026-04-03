"""Scratch runner: sample dataset texts, score detectors sequentially, and stream JSONL results.

Output privacy rule: never persist raw sample text in JSON artifacts.
"""

from __future__ import annotations

import argparse
import importlib
import json
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping


@dataclass(frozen=True)
class DatasetSpec:
    """One dataset+split source for sampling examples."""

    dataset_id: str
    split: str


@dataclass(frozen=True)
class Example:
    """One scored text example."""

    example_id: str
    label: str
    dataset_id: str
    split: str
    sample_id: str
    text: str


@dataclass(frozen=True)
class ModelRunSpec:
    """One detector run configuration."""

    run_id: str
    detector_id: str
    adapter_module: str
    adapter_class: str
    config_path: Path
    config_overrides: Mapping[str, Any]


DEFAULT_DATASETS: tuple[DatasetSpec, ...] = (
    DatasetSpec(dataset_id="hc3", split="all_train"),
    DatasetSpec(dataset_id="kaggle_llm_detect_ai_generated_text", split="train"),
    DatasetSpec(dataset_id="grid", split="filtered"),
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


def _bootstrap_src_path() -> Path:
    current = Path(__file__).resolve()
    for parent in current.parents:
        src_dir = parent / "src"
        if src_dir.is_dir():
            src_str = str(src_dir)
            if src_str not in sys.path:
                sys.path.insert(0, src_str)
            return src_dir
    raise RuntimeError("Cannot find repository 'src' directory for imports.")


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run detector scoring on sampled dataset examples.")
    parser.add_argument("--project-root", type=Path, default=Path.cwd(), help="Repository root directory.")
    parser.add_argument(
        "--examples-per-label",
        type=int,
        default=30,
        help="How many examples to load per label (human/ai) in total across configured datasets.",
    )
    parser.add_argument(
        "--output-jsonl",
        type=Path,
        default=Path("scratch/detector_scoring/results/raw_scores.jsonl"),
        help="Output JSONL path for streaming detector scores.",
    )
    return parser


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        parsed = json.loads(line)
        if not isinstance(parsed, dict):
            raise ValueError(f"Expected JSON object rows in {path}.")
        rows.append(parsed)
    return rows


def _load_examples(project_root: Path, examples_per_label: int) -> list[Example]:
    by_label: dict[str, list[Example]] = {"human": [], "ai": []}

    for label in ("human", "ai"):
        collected = 0
        for dataset in DEFAULT_DATASETS:
            source_path = (
                project_root
                / "data"
                / "raw"
                / "datasets"
                / dataset.dataset_id
                / dataset.split
                / label
                / "sampled_records.jsonl"
            )
            if not source_path.exists():
                raise FileNotFoundError(f"Missing sampled records file: {source_path}")
            rows = _read_jsonl(source_path)

            for row in rows:
                if collected >= examples_per_label:
                    break
                text = row.get("text")
                sample_id = row.get("sample_id")
                if not isinstance(text, str):
                    raise ValueError(f"Row missing string 'text' field in {source_path}.")
                if not isinstance(sample_id, str):
                    raise ValueError(f"Row missing string 'sample_id' field in {source_path}.")
                collected += 1
                by_label[label].append(
                    Example(
                        example_id=f"{label}_{collected}",
                        label=label,
                        dataset_id=dataset.dataset_id,
                        split=dataset.split,
                        sample_id=sample_id,
                        text=text,
                    )
                )

            if collected >= examples_per_label:
                break

        if collected < examples_per_label:
            raise ValueError(
                f"Need {examples_per_label} '{label}' examples across configured datasets, got {collected}."
            )

    # Keep strict order for downstream per-model plots: first all human, then all ai.
    return by_label["human"] + by_label["ai"]


def _build_model_runs(project_root: Path) -> list[ModelRunSpec]:
    config_dir = project_root / "configs" / "detectors"
    config_paths = sorted(config_dir.glob("*.detector.json"))
    runs: list[ModelRunSpec] = []

    for config_path in config_paths:
        config_data = json.loads(config_path.read_text(encoding="utf-8"))
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
                overrides: dict[str, Any] = {
                    "variant_id": variant_id,
                    "max_supported_vram_gb": float(estimated_vram_gb),
                }
                runs.append(
                    ModelRunSpec(
                        run_id=run_id,
                        detector_id=detector_id,
                        adapter_module=adapter_module,
                        adapter_class=adapter_class,
                        config_path=config_path,
                        config_overrides=overrides,
                    )
                )
        else:
            runs.append(
                ModelRunSpec(
                    run_id=detector_id,
                    detector_id=detector_id,
                    adapter_module=adapter_module,
                    adapter_class=adapter_class,
                    config_path=config_path,
                    config_overrides={},
                )
            )

    return runs


def _load_detector_class(module_name: str, class_name: str) -> type[Any]:
    module = importlib.import_module(module_name)
    detector_class = getattr(module, class_name)
    return detector_class


def _score_model_run(model_run: ModelRunSpec, examples: list[Example]) -> dict[str, Any]:
    detector_config = json.loads(model_run.config_path.read_text(encoding="utf-8"))
    merged_config = dict(detector_config)
    merged_config.update(model_run.config_overrides)

    detector_class = _load_detector_class(model_run.adapter_module, model_run.adapter_class)
    detector = detector_class.initialize(config=merged_config)
    texts = [example.text for example in examples]

    try:
        scores = detector.predict_batch(texts)
    finally:
        detector.delete()

    if len(scores) != len(examples):
        raise ValueError(
            f"Detector '{model_run.run_id}' returned {len(scores)} scores for {len(examples)} examples."
        )

    return {
        "run_id": model_run.run_id,
        "detector_id": model_run.detector_id,
        "config_path": str(model_run.config_path),
        "config_overrides": dict(model_run.config_overrides),
        "scores": [float(score) for score in scores],
    }


def _append_jsonl_record(path: Path, record: dict[str, Any], mode: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open(mode, encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def _examples_to_records(examples: list[Example]) -> list[dict[str, Any]]:
    return [
        {
            "example_id": example.example_id,
            "label": example.label,
            "dataset_id": example.dataset_id,
            "split": example.split,
            "sample_id": example.sample_id,
        }
        for example in examples
    ]


def main() -> int:
    _bootstrap_src_path()
    parser = _build_arg_parser()
    args = parser.parse_args()

    project_root = args.project_root.resolve()
    output_path = (project_root / args.output_jsonl).resolve()

    examples = _load_examples(project_root=project_root, examples_per_label=args.examples_per_label)
    model_runs = _build_model_runs(project_root=project_root)

    meta_record: dict[str, Any] = {
        "record_type": "meta",
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "dataset_specs": [
            {"dataset_id": spec.dataset_id, "split": spec.split}
            for spec in DEFAULT_DATASETS
        ],
        "examples_per_label": args.examples_per_label,
        "examples": _examples_to_records(examples),
    }
    _append_jsonl_record(output_path, meta_record, mode="w")

    completed_runs = 0
    failed_runs = 0
    for model_run in model_runs:
        print(f"[run] {model_run.run_id}")
        try:
            score_run = _score_model_run(model_run=model_run, examples=examples)
            _append_jsonl_record(
                output_path,
                {
                    "record_type": "score_run",
                    "created_at": datetime.now().isoformat(timespec="seconds"),
                    **score_run,
                },
                mode="a",
            )
            completed_runs += 1
        except Exception as error:  # noqa: BLE001 - scratch script should keep partial outputs.
            _append_jsonl_record(
                output_path,
                {
                    "record_type": "run_error",
                    "created_at": datetime.now().isoformat(timespec="seconds"),
                    "run_id": model_run.run_id,
                    "detector_id": model_run.detector_id,
                    "config_path": str(model_run.config_path),
                    "config_overrides": dict(model_run.config_overrides),
                    "error": str(error),
                },
                mode="a",
            )
            failed_runs += 1
            print(f"[error] {model_run.run_id}: {error}")

    print(f"Saved streaming scores to: {output_path}")
    print(f"Scored examples: {len(examples)}")
    print(f"Completed model runs: {completed_runs}")
    print(f"Failed model runs: {failed_runs}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
