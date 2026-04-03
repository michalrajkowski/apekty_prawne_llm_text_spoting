"""SynthID Text detector adapter implementation for low-VRAM backbones."""

from __future__ import annotations

import gc
import importlib
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from apm.detectors.base import AbstractDetector, DetectorConfig


@dataclass(frozen=True)
class SynthIDVariant:
    """Configuration for one SynthID backbone variant."""

    variant_id: str
    model_id: str
    estimated_vram_gb: float


_DEFAULT_VARIANTS: tuple[SynthIDVariant, ...] = (
    SynthIDVariant(variant_id="gpt2", model_id="openai-community/gpt2", estimated_vram_gb=0.25),
    SynthIDVariant(variant_id="gemma_2b", model_id="google/gemma-2b", estimated_vram_gb=4.0),
    SynthIDVariant(variant_id="gemma_7b", model_id="google/gemma-7b", estimated_vram_gb=14.0),
)


class SynthIDTextDetector(AbstractDetector):
    """SynthID-text adapter using configurable backbone variants."""

    def __init__(
        self,
        *,
        config: Mapping[str, Any],
        tokenizer: Any,
        model: Any,
        torch_module: Any,
        device: str,
        max_length: int,
        variant_id: str,
        model_id: str,
        score_scale: float,
    ) -> None:
        self._config = dict(config)
        self._tokenizer = tokenizer
        self._model = model
        self._torch = torch_module
        self._device = device
        self._max_length = max_length
        self._variant_id = variant_id
        self._model_id = model_id
        self._score_scale = score_scale

    @classmethod
    def initialize(cls, config: DetectorConfig) -> "SynthIDTextDetector":
        torch_module = importlib.import_module("torch")
        transformers_module = importlib.import_module("transformers")

        selected_variant_id = _read_string(config, "variant_id", default="gpt2")
        max_supported_vram_gb = _read_float(config, "max_supported_vram_gb", default=6.0)
        max_length = _read_int(config, "max_length", default=256)
        score_scale = _read_float(config, "score_scale", default=1.0)
        local_files_only = _read_bool(config, "local_files_only", default=False)
        trust_remote_code = _read_bool(config, "trust_remote_code", default=False)
        cache_dir = _read_optional_path(config, "cache_dir")

        variants = _load_variants(config)
        variant = _select_variant(variants=variants, variant_id=selected_variant_id)
        if variant.estimated_vram_gb > max_supported_vram_gb:
            raise ValueError(
                "Selected SynthID variant exceeds configured VRAM ceiling: "
                f"{variant.variant_id} ({variant.estimated_vram_gb}GB > {max_supported_vram_gb}GB)."
            )

        requested_device = _read_string(config, "device", default="")
        if requested_device:
            device = requested_device
            if requested_device.startswith("cuda") and not torch_module.cuda.is_available():
                device = "cpu"
        else:
            device = "cuda" if torch_module.cuda.is_available() else "cpu"

        tokenizer_kwargs: dict[str, Any] = {
            "local_files_only": local_files_only,
            "trust_remote_code": trust_remote_code,
        }
        if cache_dir is not None:
            tokenizer_kwargs["cache_dir"] = cache_dir
        tokenizer = transformers_module.AutoTokenizer.from_pretrained(variant.model_id, **tokenizer_kwargs)

        model_kwargs: dict[str, Any] = {
            "local_files_only": local_files_only,
            "trust_remote_code": trust_remote_code,
        }
        if cache_dir is not None:
            model_kwargs["cache_dir"] = cache_dir

        torch_dtype_name = _read_string(config, "torch_dtype", default="")
        if torch_dtype_name:
            model_kwargs["torch_dtype"] = getattr(torch_module, torch_dtype_name)

        model = transformers_module.AutoModelForCausalLM.from_pretrained(variant.model_id, **model_kwargs)
        model.to(device)
        model.eval()
        return cls(
            config=config,
            tokenizer=tokenizer,
            model=model,
            torch_module=torch_module,
            device=device,
            max_length=max_length,
            variant_id=variant.variant_id,
            model_id=variant.model_id,
            score_scale=score_scale,
        )

    @classmethod
    def list_supported_variants(cls, config: DetectorConfig) -> list[str]:
        max_supported_vram_gb = _read_float(config, "max_supported_vram_gb", default=6.0)
        variants = _load_variants(config)
        return [
            variant.variant_id
            for variant in variants
            if variant.estimated_vram_gb <= max_supported_vram_gb
        ]

    def predict_single(self, text: str) -> float:
        scores = self.predict_batch([text])
        return scores[0]

    def predict_batch(self, texts: Sequence[str]) -> list[float]:
        return [self._score_one(text=text) for text in texts]

    def delete(self) -> None:
        self._model = None
        self._tokenizer = None
        if self._device.startswith("cuda") and hasattr(self._torch, "cuda"):
            self._torch.cuda.empty_cache()
        gc.collect()

    def _score_one(self, text: str) -> float:
        encoded = self._tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=self._max_length,
        )
        if hasattr(encoded, "to"):
            encoded = encoded.to(self._device)
        input_ids = encoded["input_ids"]

        with self._torch.no_grad():
            outputs = self._model(**encoded, labels=input_ids)
        loss_value = float(outputs.loss.detach().cpu().item())
        return 1.0 / (1.0 + math.exp(self._score_scale * loss_value))


def _load_variants(config: Mapping[str, Any]) -> tuple[SynthIDVariant, ...]:
    raw_variants = config.get("model_variants")
    if raw_variants is None:
        return _DEFAULT_VARIANTS
    if not isinstance(raw_variants, list):
        raise ValueError("Config key 'model_variants' must be a list.")

    variants: list[SynthIDVariant] = []
    for entry in raw_variants:
        if not isinstance(entry, Mapping):
            raise ValueError("Each 'model_variants' entry must be an object.")
        variant_id = _read_string(entry, "variant_id", default="")
        model_id = _read_string(entry, "model_id", default="")
        estimated_vram_gb = _read_float(entry, "estimated_vram_gb", default=0.0)
        if not variant_id:
            raise ValueError("Each 'model_variants' entry must define non-empty 'variant_id'.")
        if not model_id:
            raise ValueError("Each 'model_variants' entry must define non-empty 'model_id'.")
        variants.append(
            SynthIDVariant(
                variant_id=variant_id,
                model_id=model_id,
                estimated_vram_gb=estimated_vram_gb,
            )
        )
    return tuple(variants)


def _select_variant(variants: Sequence[SynthIDVariant], variant_id: str) -> SynthIDVariant:
    for variant in variants:
        if variant.variant_id == variant_id:
            return variant
    available_ids = ", ".join(sorted(variant.variant_id for variant in variants))
    raise ValueError(f"Unknown SynthID variant_id '{variant_id}'. Available: {available_ids}.")


def _read_string(config: Mapping[str, Any], key: str, default: str) -> str:
    value = config.get(key, default)
    if not isinstance(value, str):
        raise ValueError(f"Config key '{key}' must be a string.")
    return value


def _read_int(config: Mapping[str, Any], key: str, default: int) -> int:
    value = config.get(key, default)
    if not isinstance(value, int):
        raise ValueError(f"Config key '{key}' must be an integer.")
    return value


def _read_bool(config: Mapping[str, Any], key: str, default: bool) -> bool:
    value = config.get(key, default)
    if not isinstance(value, bool):
        raise ValueError(f"Config key '{key}' must be a boolean.")
    return value


def _read_float(config: Mapping[str, Any], key: str, default: float) -> float:
    value = config.get(key, default)
    if not isinstance(value, (int, float)):
        raise ValueError(f"Config key '{key}' must be numeric.")
    return float(value)


def _read_optional_path(config: Mapping[str, Any], key: str) -> str | None:
    value = config.get(key)
    if value is None:
        return None
    if not isinstance(value, str):
        raise ValueError(f"Config key '{key}' must be a string path.")
    return str(Path(value))
