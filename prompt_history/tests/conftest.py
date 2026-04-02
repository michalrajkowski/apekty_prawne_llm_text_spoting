"""Pytest configuration for prompt_history tests."""

from __future__ import annotations

from pathlib import Path
import sys

_TESTS_DIR = Path(__file__).resolve().parent
_SRC_DIR = _TESTS_DIR.parent / "src"

if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))
