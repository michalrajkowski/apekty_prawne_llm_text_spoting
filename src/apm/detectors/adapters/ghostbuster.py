"""Ghostbuster-style detector adapter returning AI probability scores."""

from __future__ import annotations

import gc
import importlib
import math
import string
from pathlib import Path
from typing import Any, Mapping, Sequence

from apm.detectors.base import AbstractDetector, DetectorConfig


class GhostbusterDetector(AbstractDetector):
    """Approximate Ghostbuster detector using weak-model feature aggregation.

    Upstream Ghostbuster uses symbolic features plus a trained classifier.
    This adapter reproduces the same contract with deterministic, model-based
    weak features and a logistic probability head.
    """

    def __init__(
        self,
        *,
        config: Mapping[str, Any],
        weak_tokenizer: Any,
        weak_model: Any,
        strong_tokenizer: Any,
        strong_model: Any,
        torch_module: Any,
        device: str,
        max_length: int,
        criterion_weight: float,
        repetition_weight: float,
        punctuation_weight: float,
        length_weight: float,
        bias: float,
    ) -> None:
        self._config = dict(config)
        self._weak_tokenizer = weak_tokenizer
        self._weak_model = weak_model
        self._strong_tokenizer = strong_tokenizer
        self._strong_model = strong_model
        self._torch = torch_module
        self._device = device
        self._max_length = max_length
        self._criterion_weight = criterion_weight
        self._repetition_weight = repetition_weight
        self._punctuation_weight = punctuation_weight
        self._length_weight = length_weight
        self._bias = bias

    @classmethod
    def initialize(cls, config: DetectorConfig) -> "GhostbusterDetector":
        torch_module = importlib.import_module("torch")
        transformers_module = importlib.import_module("transformers")

        weak_model_id = _read_string(config, "weak_model_id", default="distilgpt2")
        strong_model_id = _read_string(config, "strong_model_id", default="openai-community/gpt2")
        max_length = _read_int(config, "max_length", default=256)
        criterion_weight = _read_float(config, "criterion_weight", default=0.4)
        repetition_weight = _read_float(config, "repetition_weight", default=3.0)
        punctuation_weight = _read_float(config, "punctuation_weight", default=8.0)
        length_weight = _read_float(config, "length_weight", default=0.6)
        bias = _read_float(config, "bias", default=-1.0)
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

        tokenizer_kwargs: dict[str, Any] = {
            "local_files_only": local_files_only,
            "trust_remote_code": trust_remote_code,
        }
        if cache_dir is not None:
            tokenizer_kwargs["cache_dir"] = cache_dir

        model_kwargs: dict[str, Any] = {
            "local_files_only": local_files_only,
            "trust_remote_code": trust_remote_code,
        }
        if cache_dir is not None:
            model_kwargs["cache_dir"] = cache_dir

        torch_dtype_name = _read_string(config, "torch_dtype", default="")
        if torch_dtype_name:
            model_kwargs["torch_dtype"] = getattr(torch_module, torch_dtype_name)

        weak_tokenizer = transformers_module.AutoTokenizer.from_pretrained(weak_model_id, **tokenizer_kwargs)
        weak_model = transformers_module.AutoModelForCausalLM.from_pretrained(weak_model_id, **model_kwargs)
        weak_model.to(device)
        weak_model.eval()

        strong_tokenizer = transformers_module.AutoTokenizer.from_pretrained(strong_model_id, **tokenizer_kwargs)
        strong_model = transformers_module.AutoModelForCausalLM.from_pretrained(strong_model_id, **model_kwargs)
        strong_model.to(device)
        strong_model.eval()

        return cls(
            config=config,
            weak_tokenizer=weak_tokenizer,
            weak_model=weak_model,
            strong_tokenizer=strong_tokenizer,
            strong_model=strong_model,
            torch_module=torch_module,
            device=device,
            max_length=max_length,
            criterion_weight=criterion_weight,
            repetition_weight=repetition_weight,
            punctuation_weight=punctuation_weight,
            length_weight=length_weight,
            bias=bias,
        )

    def predict_single(self, text: str) -> float:
        scores = self.predict_batch([text])
        return scores[0]

    def predict_batch(self, texts: Sequence[str]) -> list[float]:
        return [self._score_one(text=text) for text in texts]

    def delete(self) -> None:
        self._weak_model = None
        self._weak_tokenizer = None
        self._strong_model = None
        self._strong_tokenizer = None
        if self._device.startswith("cuda") and hasattr(self._torch, "cuda"):
            self._torch.cuda.empty_cache()
        gc.collect()

    def _score_one(self, text: str) -> float:
        weak_loss = self._causal_loss(tokenizer=self._weak_tokenizer, model=self._weak_model, text=text)
        strong_loss = self._causal_loss(tokenizer=self._strong_tokenizer, model=self._strong_model, text=text)

        criterion = weak_loss - strong_loss
        repetition_signal, punctuation_ratio, token_count = _lexical_ai_signals(text)
        length_signal = float(token_count) / 100.0

        raw_value = (
            self._bias
            + self._criterion_weight * criterion
            + self._repetition_weight * repetition_signal
            - self._punctuation_weight * punctuation_ratio
            + self._length_weight * length_signal
        )
        return 1.0 / (1.0 + math.exp(-raw_value))

    def _causal_loss(self, *, tokenizer: Any, model: Any, text: str) -> float:
        encoded = tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=self._max_length,
        )
        if hasattr(encoded, "to"):
            encoded = encoded.to(self._device)
        input_ids = encoded["input_ids"]

        with self._torch.no_grad():
            outputs = model(**encoded, labels=input_ids)
        return float(outputs.loss.detach().cpu().item())


def _lexical_ai_signals(text: str) -> tuple[float, float, int]:
    lowered = text.lower()
    tokens = [token for token in lowered.split() if token]
    if not tokens:
        return 0.0, 0.0, 0

    unique_ratio = len(set(tokens)) / float(len(tokens))
    punctuation_count = sum(1 for character in text if character in string.punctuation)
    punctuation_ratio = punctuation_count / float(max(len(text), 1))

    repetition_signal = 1.0 - unique_ratio
    return repetition_signal, punctuation_ratio, len(tokens)


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
