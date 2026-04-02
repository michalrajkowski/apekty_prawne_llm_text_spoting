"""Dataset-specific adapter implementations."""

from apm.data.adapters.hc3_adapter import HC3Adapter, Hc3LoadCheckReport, Hc3SelectorValidationSummary, load_hc3_config
from apm.data.adapters.kaggle_llm_detect_ai_generated_text_adapter import (
    KaggleLlmDetectAiGeneratedTextAdapter,
    load_kaggle_config,
)

__all__ = [
    "HC3Adapter",
    "Hc3LoadCheckReport",
    "Hc3SelectorValidationSummary",
    "KaggleLlmDetectAiGeneratedTextAdapter",
    "load_hc3_config",
    "load_kaggle_config",
]
