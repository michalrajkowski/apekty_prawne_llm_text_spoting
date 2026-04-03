"""AIGC Detector V3-Short adapter implementation."""

from __future__ import annotations

import gc
import importlib
from pathlib import Path
from typing import Any, Mapping, Sequence

from apm.detectors.base import AbstractDetector, DetectorConfig


class AIGCDetectorEnv3Short(AbstractDetector):
    """Detector adapter for `yuchuantian/AIGC_detector_env3short`."""

    def __init__(
        self,
        *,
        config: Mapping[str, Any],
        tokenizer: Any,
        model: Any,
        torch_module: Any,
        device: str,
        ai_label_index: int,
        max_length: int,
    ) -> None:
        self._config = dict(config)
        self._tokenizer = tokenizer
        self._model = model
        self._torch = torch_module
        self._device = device
        self._ai_label_index = ai_label_index
        self._max_length = max_length

    @classmethod
    def initialize(cls, config: DetectorConfig) -> "AIGCDetectorEnv3Short":
        torch_module = importlib.import_module("torch")
        transformers_module = importlib.import_module("transformers")

        model_id = _read_string(config, "model_id", default="yuchuantian/AIGC_detector_env3short")
        ai_label_index = _read_int(config, "ai_label_index", default=1)
        max_length = _read_int(config, "max_length", default=512)
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

        model = transformers_module.AutoModelForSequenceClassification.from_pretrained(model_id, **model_kwargs)
        model.to(device)
        model.eval()
        return cls(
            config=config,
            tokenizer=tokenizer,
            model=model,
            torch_module=torch_module,
            device=device,
            ai_label_index=ai_label_index,
            max_length=max_length,
        )

    def predict_single(self, text: str) -> float:
        scores = self.predict_batch([text])
        return scores[0]

    def predict_batch(self, texts: Sequence[str]) -> list[float]:
        if not texts:
            return []
        encoded_batch = self._tokenizer(
            list(texts),
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self._max_length,
        )
        if hasattr(encoded_batch, "to"):
            encoded_batch = encoded_batch.to(self._device)

        with self._torch.no_grad():
            outputs = self._model(**encoded_batch)
            logits = outputs.logits
            if logits.shape[-1] == 1:
                score_tensor = self._torch.sigmoid(logits.squeeze(-1))
            else:
                probabilities = self._torch.softmax(logits, dim=-1)
                score_tensor = probabilities[:, self._ai_label_index]

        score_values = score_tensor.detach().cpu().tolist()
        if isinstance(score_values, (float, int)):
            return [float(score_values)]
        return [float(score) for score in score_values]

    def delete(self) -> None:
        self._model = None
        self._tokenizer = None
        if self._device.startswith("cuda") and hasattr(self._torch, "cuda"):
            self._torch.cuda.empty_cache()
        gc.collect()


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
