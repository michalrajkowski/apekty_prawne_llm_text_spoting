"""Specialized unit tests for the Ghostbuster detector adapter."""

from __future__ import annotations

import math
import types
from typing import Any

import pytest

from apm.detectors.adapters.ghostbuster import GhostbusterDetector

pytestmark = pytest.mark.detector_special


class FakeLossValue:
    """Loss tensor-like wrapper providing detach/cpu/item chain."""

    def __init__(self, value: float) -> None:
        self._value = value

    def detach(self) -> "FakeLossValue":
        return self

    def cpu(self) -> "FakeLossValue":
        return self

    def item(self) -> float:
        return self._value


class FakeBatch(dict[str, Any]):
    """Tokenizer output batch supporting `.to(...)`."""

    def to(self, _device: str) -> "FakeBatch":
        return self


class FakeTokenizer:
    """Tokenizer mock preserving source text for loss heuristics."""

    @classmethod
    def from_pretrained(cls, _model_id: str, **_kwargs: Any) -> "FakeTokenizer":
        return cls()

    def __call__(
        self,
        text: str,
        *,
        return_tensors: str,
        truncation: bool,
        max_length: int,
    ) -> FakeBatch:
        assert return_tensors == "pt"
        assert truncation is True
        assert max_length > 0
        return FakeBatch({"input_ids": [1, 2], "__text__": text})


class FakeModel:
    """Model mock with role-dependent losses for AI-vs-human texts."""

    def __init__(self, model_id: str) -> None:
        self._model_id = model_id
        self.device = "cpu"

    @classmethod
    def from_pretrained(cls, model_id: str, **_kwargs: Any) -> "FakeModel":
        return cls(model_id=model_id)

    def to(self, device: str) -> "FakeModel":
        self.device = device
        return self

    def eval(self) -> "FakeModel":
        return self

    def __call__(self, **batch: Any) -> Any:
        text = str(batch["__text__"]).lower()
        is_ai = "ai" in text

        if self._model_id == "distilgpt2":
            loss_value = 0.9 if is_ai else 1.2
        else:
            loss_value = 0.4 if is_ai else 1.1

        return types.SimpleNamespace(loss=FakeLossValue(loss_value))


class FakeNoGrad:
    """No-op context manager replacing torch.no_grad."""

    def __enter__(self) -> None:
        return None

    def __exit__(self, _exc_type: Any, _exc: Any, _tb: Any) -> bool:
        return False


class FakeCuda:
    """CUDA helper with call tracking."""

    def __init__(self, is_available: bool) -> None:
        self._is_available = is_available
        self.empty_cache_called = 0

    def is_available(self) -> bool:
        return self._is_available

    def empty_cache(self) -> None:
        self.empty_cache_called += 1


class FakeTorch:
    """Minimal subset of torch API used by adapter initialization/inference."""

    def __init__(self, cuda_available: bool) -> None:
        self.cuda = FakeCuda(is_available=cuda_available)

    def no_grad(self) -> FakeNoGrad:
        return FakeNoGrad()


def _install_fake_runtime(monkeypatch: pytest.MonkeyPatch, cuda_available: bool = True) -> FakeTorch:
    fake_torch = FakeTorch(cuda_available=cuda_available)
    fake_transformers = types.SimpleNamespace(
        AutoTokenizer=FakeTokenizer,
        AutoModelForCausalLM=FakeModel,
    )

    def fake_import_module(name: str) -> Any:
        if name == "torch":
            return fake_torch
        if name == "transformers":
            return fake_transformers
        raise ModuleNotFoundError(name)

    monkeypatch.setattr(
        "apm.detectors.adapters.ghostbuster.importlib.import_module",
        fake_import_module,
    )
    return fake_torch


def test_predict_batch_returns_scores_for_each_text(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_fake_runtime(monkeypatch)
    detector = GhostbusterDetector.initialize(config={"device": "cuda"})

    scores = detector.predict_batch(["this is ai text", "this is human writing"])

    assert len(scores) == 2
    assert all(isinstance(score, float) for score in scores)
    assert all(0.0 <= score <= 1.0 for score in scores)
    assert scores[0] > scores[1]


def test_predict_single_matches_batch_first_item(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_fake_runtime(monkeypatch)
    detector = GhostbusterDetector.initialize(config={"device": "cuda"})

    single_score = detector.predict_single("this is ai text")
    batch_score = detector.predict_batch(["this is ai text"])[0]

    assert math.isclose(single_score, batch_score, rel_tol=1e-12)


def test_delete_releases_cuda_cache_when_cuda_selected(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_torch = _install_fake_runtime(monkeypatch)
    detector = GhostbusterDetector.initialize(config={"device": "cuda"})

    detector.delete()

    assert fake_torch.cuda.empty_cache_called == 1
