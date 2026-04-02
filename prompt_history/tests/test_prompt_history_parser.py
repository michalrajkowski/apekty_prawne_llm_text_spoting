"""Tests for PROMPTS_HISTORY markdown parsing."""

from __future__ import annotations

import re

import pytest

from prompt_history.parser import PromptHistoryParseError, parse_prompt_history


_VALID_HISTORY = """# Prompt History

2026-04-02 16:00:00
USER:
```text
Line one

Line three
```
TAGS: [question, planning]
---
2026-04-02 16:01:00
AGENT:
```text
Answer here.
```
---
"""


def test_parse_prompt_history_valid_blocks() -> None:
    """Parser reads required message structure for USER and AGENT blocks."""

    messages = parse_prompt_history(_VALID_HISTORY)

    assert len(messages) == 2
    assert messages[0].speaker == "USER"
    assert messages[0].tags == ("question", "planning")
    assert messages[0].text == "Line one\n\nLine three"

    assert messages[1].speaker == "AGENT"
    assert messages[1].tags == ()
    assert messages[1].text == "Answer here."


def test_parse_prompt_history_reports_missing_tags_line() -> None:
    """Parser reports a line-specific error when USER tags are missing."""

    malformed = _VALID_HISTORY.replace("TAGS: [question, planning]\n", "", 1)

    with pytest.raises(PromptHistoryParseError) as exc_info:
        parse_prompt_history(malformed)

    assert re.search(r"Line \d+: Expected TAGS line", str(exc_info.value))


def test_parse_prompt_history_reports_missing_separator() -> None:
    """Parser fails with context if message separator is missing."""

    malformed = _VALID_HISTORY.replace("---\n", "", 1)

    with pytest.raises(PromptHistoryParseError) as exc_info:
        parse_prompt_history(malformed)

    assert "Expected separator line ---" in str(exc_info.value)


def test_parse_prompt_history_accepts_nested_fence_text() -> None:
    """Parser keeps inner triple-backtick examples inside message content."""

    history = """# Prompt History

2026-04-02 18:00:00
USER:
```text
Template:
```text
sample
```
TAGS: [demo]
---
still message text
```
TAGS: [execution]
---
"""
    messages = parse_prompt_history(history)

    assert len(messages) == 1
    assert "Template:\n```text\nsample\n```\nTAGS: [demo]\n---\nstill message text" == messages[0].text
