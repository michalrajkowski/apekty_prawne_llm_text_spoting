"""DetectGPT-style light detector adapter implementation."""

from __future__ import annotations

import gc
import hashlib
import importlib
import math
import random
from pathlib import Path
from typing import Any, Mapping, Sequence

from apm.detectors.base import AbstractDetector, DetectorConfig


class DetectGptLightDetector(AbstractDetector):
    """Lightweight DetectGPT-style detector for English text."""

    def __init__(
        self,
        *,
        config: Mapping[str, Any],
        torch_module: Any,
        device: str,
        causal_tokenizer: Any,
        causal_model: Any,
        perturbation_tokenizer: Any,
        perturbation_model: Any,
        max_length: int,
        perturb_max_length: int,
        perturbations_per_text: int,
        perturb_temperature: float,
        perturb_top_p: float,
        perturb_max_new_tokens: int,
        score_scale: float,
        seed: int,
    ) -> None:
        self._config = dict(config)
        self._torch = torch_module
        self._device = device
        self._causal_tokenizer = causal_tokenizer
        self._causal_model = causal_model
        self._perturbation_tokenizer = perturbation_tokenizer
        self._perturbation_model = perturbation_model
        self._max_length = max_length
        self._perturb_max_length = perturb_max_length
        self._perturbations_per_text = perturbations_per_text
        self._perturb_temperature = perturb_temperature
        self._perturb_top_p = perturb_top_p
        self._perturb_max_new_tokens = perturb_max_new_tokens
        self._score_scale = score_scale
        self._seed = seed

    @classmethod
    def initialize(cls, config: DetectorConfig) -> "DetectGptLightDetector":
        torch_module = importlib.import_module("torch")
        transformers_module = importlib.import_module("transformers")

        base_model_id = _read_string(config, "base_model_id", default="openai-community/gpt2-medium")
        perturbation_model_id = _read_string(config, "perturbation_model_id", default="google/flan-t5-small")
        max_length = _read_int(config, "max_length", default=256)
        perturb_max_length = _read_int(config, "perturb_max_length", default=128)
        perturbations_per_text = _read_int(config, "perturbations_per_text", default=3)
        perturb_temperature = _read_float(config, "perturb_temperature", default=1.0)
        perturb_top_p = _read_float(config, "perturb_top_p", default=0.95)
        perturb_max_new_tokens = _read_int(config, "perturb_max_new_tokens", default=64)
        score_scale = _read_float(config, "score_scale", default=1.0)
        seed = _read_int(config, "seed", default=42)
        local_files_only = _read_bool(config, "local_files_only", default=False)
        trust_remote_code = _read_bool(config, "trust_remote_code", default=False)
        cache_dir = _read_optional_path(config, "cache_dir")

        requested_device = _read_string(config, "device", default="")
        if requested_device:
            device = requested_device
            if requested_device.startswith("cuda") and not torch_module.cuda.is_available():
                device = "cpu"
        else:
            device = "cuda" if torch_module.cuda.is_available() else "cpu"

        common_tokenizer_kwargs: dict[str, Any] = {
            "local_files_only": local_files_only,
            "trust_remote_code": trust_remote_code,
        }
        if cache_dir is not None:
            common_tokenizer_kwargs["cache_dir"] = cache_dir

        common_model_kwargs: dict[str, Any] = {
            "local_files_only": local_files_only,
            "trust_remote_code": trust_remote_code,
        }
        if cache_dir is not None:
            common_model_kwargs["cache_dir"] = cache_dir

        torch_dtype_name = _read_string(config, "torch_dtype", default="")
        if torch_dtype_name:
            common_model_kwargs["torch_dtype"] = getattr(torch_module, torch_dtype_name)

        causal_tokenizer = transformers_module.AutoTokenizer.from_pretrained(base_model_id, **common_tokenizer_kwargs)
        causal_model = transformers_module.AutoModelForCausalLM.from_pretrained(base_model_id, **common_model_kwargs)
        causal_model.to(device)
        causal_model.eval()

        perturb_tokenizer_cls = getattr(transformers_module, "T5Tokenizer", transformers_module.AutoTokenizer)
        perturbation_tokenizer = perturb_tokenizer_cls.from_pretrained(
            perturbation_model_id,
            **common_tokenizer_kwargs,
        )
        perturbation_model = transformers_module.AutoModelForSeq2SeqLM.from_pretrained(
            perturbation_model_id,
            **common_model_kwargs,
        )
        perturbation_model.to(device)
        perturbation_model.eval()

        return cls(
            config=config,
            torch_module=torch_module,
            device=device,
            causal_tokenizer=causal_tokenizer,
            causal_model=causal_model,
            perturbation_tokenizer=perturbation_tokenizer,
            perturbation_model=perturbation_model,
            max_length=max_length,
            perturb_max_length=perturb_max_length,
            perturbations_per_text=perturbations_per_text,
            perturb_temperature=perturb_temperature,
            perturb_top_p=perturb_top_p,
            perturb_max_new_tokens=perturb_max_new_tokens,
            score_scale=score_scale,
            seed=seed,
        )

    def predict_single(self, text: str) -> float:
        scores = self.predict_batch([text])
        return scores[0]

    def predict_batch(self, texts: Sequence[str]) -> list[float]:
        return [self._score_one(text=text) for text in texts]

    def delete(self) -> None:
        self._causal_model = None
        self._causal_tokenizer = None
        self._perturbation_model = None
        self._perturbation_tokenizer = None
        if self._device.startswith("cuda") and hasattr(self._torch, "cuda"):
            self._torch.cuda.empty_cache()
        gc.collect()

    def _score_one(self, text: str) -> float:
        text_seed = _derive_text_seed(base_seed=self._seed, text=text)
        base_loss = self._causal_loss(text=text)
        perturbations = self._generate_perturbations(text=text, seed=text_seed)
        perturb_losses = [self._causal_loss(text=perturbed_text) for perturbed_text in perturbations]
        mean_perturb_loss = sum(perturb_losses) / len(perturb_losses)
        discrepancy = mean_perturb_loss - base_loss
        return 1.0 / (1.0 + math.exp(-self._score_scale * discrepancy))

    def _causal_loss(self, text: str) -> float:
        encoded = self._causal_tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=self._max_length,
        )
        if hasattr(encoded, "to"):
            encoded = encoded.to(self._device)
        input_ids = encoded["input_ids"]
        with self._torch.no_grad():
            outputs = self._causal_model(**encoded, labels=input_ids)
        return float(outputs.loss.detach().cpu().item())

    def _generate_perturbations(self, text: str, seed: int) -> list[str]:
        _seed_runtime(torch_module=self._torch, device=self._device, seed=seed)
        encoded = self._perturbation_tokenizer(
            [text] * self._perturbations_per_text,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self._perturb_max_length,
        )
        if hasattr(encoded, "to"):
            encoded = encoded.to(self._device)
        generate_kwargs: dict[str, Any] = {
            "do_sample": True,
            "top_p": self._perturb_top_p,
            "temperature": self._perturb_temperature,
            "max_new_tokens": self._perturb_max_new_tokens,
            "num_return_sequences": 1,
        }
        with self._torch.no_grad():
            generated = self._perturbation_model.generate(**encoded, **generate_kwargs)
        decoded = self._perturbation_tokenizer.batch_decode(generated, skip_special_tokens=True)
        return [entry for entry in decoded if entry.strip()]


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


def _derive_text_seed(base_seed: int, text: str) -> int:
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
    text_component = int(digest[:16], 16)
    return (base_seed + text_component) % 2_147_483_647


def _seed_runtime(torch_module: Any, device: str, seed: int) -> None:
    random.seed(seed)
    if hasattr(torch_module, "manual_seed"):
        torch_module.manual_seed(seed)
    if device.startswith("cuda") and hasattr(torch_module, "cuda") and hasattr(torch_module.cuda, "manual_seed_all"):
        torch_module.cuda.manual_seed_all(seed)
