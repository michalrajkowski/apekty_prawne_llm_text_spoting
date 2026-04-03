"""GLTR-style detector adapter with GPT-2 small backend."""

from __future__ import annotations

import gc
import importlib
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

from apm.detectors.base import AbstractDetector, DetectorConfig


class GLTRGpt2SmallDetector(AbstractDetector):
    """GLTR-style detector for English text using GPT-2 small."""

    def __init__(
        self,
        *,
        config: Mapping[str, Any],
        tokenizer: Any,
        model: Any,
        torch_module: Any,
        device: str,
        max_length: int,
    ) -> None:
        self._config = dict(config)
        self._tokenizer = tokenizer
        self._model = model
        self._torch = torch_module
        self._device = device
        self._max_length = max_length

    @classmethod
    def initialize(cls, config: DetectorConfig) -> "GLTRGpt2SmallDetector":
        torch_module = importlib.import_module("torch")
        transformers_module = importlib.import_module("transformers")

        model_id = _read_string(config, "model_id", default="openai-community/gpt2")
        max_length = _read_int(config, "max_length", default=256)
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
        tokenizer = transformers_module.AutoTokenizer.from_pretrained(model_id, **tokenizer_kwargs)

        model_kwargs: dict[str, Any] = {
            "local_files_only": local_files_only,
            "trust_remote_code": trust_remote_code,
        }
        if cache_dir is not None:
            model_kwargs["cache_dir"] = cache_dir

        torch_dtype_name = _read_string(config, "torch_dtype", default="")
        if torch_dtype_name:
            model_kwargs["torch_dtype"] = getattr(torch_module, torch_dtype_name)

        model = transformers_module.AutoModelForCausalLM.from_pretrained(model_id, **model_kwargs)
        model.to(device)
        model.eval()
        return cls(
            config=config,
            tokenizer=tokenizer,
            model=model,
            torch_module=torch_module,
            device=device,
            max_length=max_length,
        )

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
        return 1.0 / (1.0 + math.exp(loss_value))


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


def _read_optional_path(config: Mapping[str, Any], key: str) -> str | None:
    value = config.get(key)
    if value is None:
        return None
    if not isinstance(value, str):
        raise ValueError(f"Config key '{key}' must be a string path.")
    return str(Path(value))
