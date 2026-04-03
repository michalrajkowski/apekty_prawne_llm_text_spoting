"""Fast-DetectGPT-style detector adapter returning AI probability scores."""

from __future__ import annotations

import gc
import importlib
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

from apm.detectors.base import AbstractDetector, DetectorConfig


class FastDetectGptDetector(AbstractDetector):
    """Approximate Fast-DetectGPT detector using scoring/reference model curvature proxy.

    Upstream Fast-DetectGPT supports multiple scoring/sampling model combinations.
    This adapter exposes one configurable model pair as one detector instance.
    """

    def __init__(
        self,
        *,
        config: Mapping[str, Any],
        scoring_tokenizer: Any,
        scoring_model: Any,
        reference_tokenizer: Any,
        reference_model: Any,
        torch_module: Any,
        device: str,
        max_length: int,
        score_scale: float,
        length_weight: float,
        repetition_weight: float,
        punctuation_weight: float,
        bias: float,
    ) -> None:
        self._config = dict(config)
        self._scoring_tokenizer = scoring_tokenizer
        self._scoring_model = scoring_model
        self._reference_tokenizer = reference_tokenizer
        self._reference_model = reference_model
        self._torch = torch_module
        self._device = device
        self._max_length = max_length
        self._score_scale = score_scale
        self._length_weight = length_weight
        self._repetition_weight = repetition_weight
        self._punctuation_weight = punctuation_weight
        self._bias = bias

    @classmethod
    def initialize(cls, config: DetectorConfig) -> "FastDetectGptDetector":
        torch_module = importlib.import_module("torch")
        transformers_module = importlib.import_module("transformers")

        # Single integrated pair by default; alternative upstream pairs are config-selectable.
        scoring_model_id = _read_string(config, "scoring_model_id", default="openai-community/gpt2")
        reference_model_id = _read_string(config, "reference_model_id", default="distilgpt2")
        max_length = _read_int(config, "max_length", default=256)
        score_scale = _read_float(config, "score_scale", default=0.5)
        length_weight = _read_float(config, "length_weight", default=0.6)
        repetition_weight = _read_float(config, "repetition_weight", default=3.0)
        punctuation_weight = _read_float(config, "punctuation_weight", default=8.0)
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

        scoring_tokenizer = transformers_module.AutoTokenizer.from_pretrained(scoring_model_id, **tokenizer_kwargs)
        scoring_model = transformers_module.AutoModelForCausalLM.from_pretrained(scoring_model_id, **model_kwargs)
        scoring_model.to(device)
        scoring_model.eval()

        reference_tokenizer = transformers_module.AutoTokenizer.from_pretrained(reference_model_id, **tokenizer_kwargs)
        reference_model = transformers_module.AutoModelForCausalLM.from_pretrained(reference_model_id, **model_kwargs)
        reference_model.to(device)
        reference_model.eval()

        return cls(
            config=config,
            scoring_tokenizer=scoring_tokenizer,
            scoring_model=scoring_model,
            reference_tokenizer=reference_tokenizer,
            reference_model=reference_model,
            torch_module=torch_module,
            device=device,
            max_length=max_length,
            score_scale=score_scale,
            length_weight=length_weight,
            repetition_weight=repetition_weight,
            punctuation_weight=punctuation_weight,
            bias=bias,
        )

    def predict_single(self, text: str) -> float:
        scores = self.predict_batch([text])
        return scores[0]

    def predict_batch(self, texts: Sequence[str]) -> list[float]:
        return [self._score_one(text=text) for text in texts]

    def delete(self) -> None:
        self._scoring_model = None
        self._scoring_tokenizer = None
        self._reference_model = None
        self._reference_tokenizer = None
        if self._device.startswith("cuda") and hasattr(self._torch, "cuda"):
            self._torch.cuda.empty_cache()
        gc.collect()

    def _score_one(self, text: str) -> float:
        scoring_loss = self._causal_loss(
            tokenizer=self._scoring_tokenizer,
            model=self._scoring_model,
            text=text,
        )
        reference_loss = self._causal_loss(
            tokenizer=self._reference_tokenizer,
            model=self._reference_model,
            text=text,
        )
        criterion = reference_loss - scoring_loss
        repetition_signal, punctuation_ratio, token_count = _lexical_ai_signals(text)
        length_signal = float(token_count) / 100.0

        raw_value = (
            self._bias
            + self._score_scale * criterion
            + self._length_weight * length_signal
            + self._repetition_weight * repetition_signal
            - self._punctuation_weight * punctuation_ratio
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


def _lexical_ai_signals(text: str) -> tuple[float, float, int]:
    tokens = [token for token in text.lower().split() if token]
    if not tokens:
        return 0.0, 0.0, 0
    unique_ratio = len(set(tokens)) / float(len(tokens))
    punctuation_count = sum(1 for character in text if not character.isalnum() and not character.isspace())
    punctuation_ratio = punctuation_count / float(max(len(text), 1))
    repetition_signal = 1.0 - unique_ratio
    return repetition_signal, punctuation_ratio, len(tokens)
