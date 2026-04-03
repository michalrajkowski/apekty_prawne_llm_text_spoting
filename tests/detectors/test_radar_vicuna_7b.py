"""Specialized unit tests for the RADAR-Vicuna-7B detector adapter."""

from __future__ import annotations

import math
import types
from typing import Any

import pytest

from apm.detectors.adapters.radar_vicuna_7b import RadarVicuna7BDetector

pytestmark = pytest.mark.detector_special


class FakeTensor:
    """Minimal tensor-like object for deterministic math in tests."""

    def __init__(self, data: list[Any]) -> None:
        self.data = data

    @property
    def ndim(self) -> int:
        if self.data and isinstance(self.data[0], list):
            return 2
        return 1

    @property
    def shape(self) -> tuple[int, ...]:
        if self.ndim == 2:
            return (len(self.data), len(self.data[0]))
        return (len(self.data),)

    def squeeze(self, dim: int = -1) -> "FakeTensor":
        if dim == -1 and self.ndim == 2 and self.shape[1] == 1:
            return FakeTensor([float(row[0]) for row in self.data])
        return self

    def __getitem__(self, key: Any) -> "FakeTensor":
        if isinstance(key, tuple):
            row_selector, column_index = key
            if row_selector != slice(None):
                raise ValueError("Only full-row slices are supported in FakeTensor tests.")
            return FakeTensor([float(row[column_index]) for row in self.data])
        raise TypeError("Unsupported key type for FakeTensor.")

    def detach(self) -> "FakeTensor":
        return self

    def cpu(self) -> "FakeTensor":
        return self

    def tolist(self) -> list[Any]:
        return self.data


class FakeBatch(dict[str, Any]):
    """Tokenizer output batch supporting `.to(...)`."""

    def to(self, _device: str) -> "FakeBatch":
        return self


class FakeTokenizer:
    """Simple tokenizer mock returning deterministic batch payloads."""

    @classmethod
    def from_pretrained(cls, _model_id: str, **_kwargs: Any) -> "FakeTokenizer":
        return cls()

    def __call__(
        self,
        texts: list[str],
        *,
        return_tensors: str,
        padding: bool,
        truncation: bool,
        max_length: int,
    ) -> FakeBatch:
        assert return_tensors == "pt"
        assert padding is True
        assert truncation is True
        assert max_length > 0
        return FakeBatch({"__texts__": texts})


class FakeModel:
    """Model mock producing logits based on text content."""

    def __init__(self) -> None:
        self.device = "cpu"

    @classmethod
    def from_pretrained(cls, _model_id: str, **_kwargs: Any) -> "FakeModel":
        return cls()

    def to(self, device: str) -> "FakeModel":
        self.device = device
        return self

    def eval(self) -> "FakeModel":
        return self

    def __call__(self, **batch: Any) -> Any:
        logits: list[list[float]] = []
        for text in batch["__texts__"]:
            lowered = text.lower()
            if "ai" in lowered:
                logits.append([0.2, 2.2])
            else:
                logits.append([2.0, 0.2])
        return types.SimpleNamespace(logits=FakeTensor(logits))


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
    """Minimal subset of torch API used by the adapter."""

    def __init__(self, cuda_available: bool) -> None:
        self.cuda = FakeCuda(is_available=cuda_available)

    def no_grad(self) -> FakeNoGrad:
        return FakeNoGrad()

    def softmax(self, logits: FakeTensor, dim: int) -> FakeTensor:
        assert dim == -1
        rows: list[list[float]] = []
        for row in logits.data:
            row_max = max(row)
            exp_values = [math.exp(value - row_max) for value in row]
            total = sum(exp_values)
            rows.append([value / total for value in exp_values])
        return FakeTensor(rows)

    def sigmoid(self, logits: FakeTensor) -> FakeTensor:
        return FakeTensor([1.0 / (1.0 + math.exp(-float(value))) for value in logits.data])


def _install_fake_runtime(monkeypatch: pytest.MonkeyPatch, cuda_available: bool = True) -> FakeTorch:
    fake_torch = FakeTorch(cuda_available=cuda_available)
    fake_transformers = types.SimpleNamespace(
        AutoTokenizer=FakeTokenizer,
        AutoModelForSequenceClassification=FakeModel,
    )

    def fake_import_module(name: str) -> Any:
        if name == "torch":
            return fake_torch
        if name == "transformers":
            return fake_transformers
        raise ModuleNotFoundError(name)

    monkeypatch.setattr(
        "apm.detectors.adapters.radar_vicuna_7b.importlib.import_module",
        fake_import_module,
    )
    return fake_torch


def test_predict_batch_returns_scores_for_each_text(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_fake_runtime(monkeypatch)
    detector = RadarVicuna7BDetector.initialize(config={"device": "cuda", "ai_label_index": 1})

    scores = detector.predict_batch(["This is ai generated.", "This is human writing."])

    assert len(scores) == 2
    assert all(isinstance(score, float) for score in scores)
    assert scores[0] > scores[1]


def test_predict_single_matches_batch_first_item(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_fake_runtime(monkeypatch)
    detector = RadarVicuna7BDetector.initialize(config={"device": "cuda", "ai_label_index": 1})

    single_score = detector.predict_single("AI text sample")
    batch_score = detector.predict_batch(["AI text sample"])[0]

    assert single_score == batch_score


def test_delete_releases_cuda_cache_when_cuda_selected(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_torch = _install_fake_runtime(monkeypatch)
    detector = RadarVicuna7BDetector.initialize(config={"device": "cuda"})

    detector.delete()

    assert fake_torch.cuda.empty_cache_called == 1
