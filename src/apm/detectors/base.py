"""Core detector abstraction for unified AI-text detector integrations."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Mapping, Sequence, Self

DetectorConfig = Mapping[str, Any]


class AbstractDetector(ABC):
    """Shared contract that every detector adapter must implement."""

    @classmethod
    @abstractmethod
    def initialize(cls, config: DetectorConfig) -> Self:
        """Create a configured detector instance."""

    @abstractmethod
    def predict_single(self, text: str) -> float:
        """Predict a score for one text input."""

    @abstractmethod
    def predict_batch(self, texts: Sequence[str]) -> list[float]:
        """Predict scores for a batch of text inputs."""

    @abstractmethod
    def delete(self) -> None:
        """Release resources (including GPU memory) owned by the detector."""
