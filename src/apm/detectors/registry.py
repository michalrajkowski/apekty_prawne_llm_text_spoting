"""Registry/factory helpers for detector adapters using the shared interface."""

from __future__ import annotations

from typing import Mapping

from apm.detectors.base import AbstractDetector, DetectorConfig


class DetectorRegistry:
    """In-memory registry for detector adapter classes."""

    def __init__(self) -> None:
        self._detectors: dict[str, type[AbstractDetector]] = {}

    def register(self, detector_id: str, detector_cls: type[AbstractDetector]) -> None:
        """Register one detector adapter class under a stable detector id."""
        normalized_id = detector_id.strip()
        if not normalized_id:
            raise ValueError("detector_id must be a non-empty string.")
        if normalized_id in self._detectors:
            raise ValueError(f"Detector '{normalized_id}' is already registered.")
        self._detectors[normalized_id] = detector_cls

    def list_detectors(self) -> tuple[str, ...]:
        """List registered detector ids in deterministic order."""
        return tuple(sorted(self._detectors))

    def create(
        self,
        detector_id: str,
        config: DetectorConfig | None = None,
    ) -> AbstractDetector:
        """Create a detector instance through the adapter initialization method."""
        normalized_id = detector_id.strip()
        if normalized_id not in self._detectors:
            raise KeyError(f"Detector '{normalized_id}' is not registered.")
        detector_cls = self._detectors[normalized_id]
        resolved_config: Mapping[str, object]
        if config is None:
            resolved_config = {}
        else:
            resolved_config = config
        return detector_cls.initialize(resolved_config)
