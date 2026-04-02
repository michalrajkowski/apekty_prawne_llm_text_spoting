"""Data models for prompt history parsing and rendering."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Literal

Speaker = Literal["USER", "AGENT"]


@dataclass(frozen=True)
class ChatMessage:
    """Represents one parsed message block from PROMPTS_HISTORY.md."""

    timestamp: datetime
    speaker: Speaker
    text: str
    tags: tuple[str, ...]
    source_line: int
