"""Tests for HTML rendering and deterministic tag colors."""

from __future__ import annotations

from datetime import datetime

from prompt_history.models import ChatMessage
from prompt_history.render import render_chat_html, tag_colors


def test_tag_colors_are_deterministic() -> None:
    """Tag color generation remains stable for repeated inputs."""

    first = tag_colors("execution")
    second = tag_colors("execution")
    different = tag_colors("planning")

    assert first == second
    assert first != different


def test_render_chat_html_contains_day_separator_and_preserves_newlines() -> None:
    """Rendered HTML includes day separators and explicit newline breaks."""

    messages = [
        ChatMessage(
            timestamp=datetime(2026, 4, 2, 10, 0, 0),
            speaker="USER",
            text="alpha\n\nbeta",
            tags=("question",),
            source_line=4,
        ),
        ChatMessage(
            timestamp=datetime(2026, 4, 3, 11, 0, 0),
            speaker="AGENT",
            text="reply",
            tags=(),
            source_line=12,
        ),
    ]

    html = render_chat_html(messages, user_avatar_url="https://github.com/michalrajkowski.png")

    assert "2026-04-02" in html
    assert "2026-04-03" in html
    assert "alpha<br /><br />beta" in html
    assert "tag-pill" in html
    assert "AI" in html
    assert "message-header" in html


def test_render_chat_html_shortens_very_long_messages() -> None:
    """Rendered HTML shortens very long messages to head + ellipsis + tail lines."""

    long_lines = [f"L{line_number:03d}" for line_number in range(1, 56)]
    long_text = "\n".join(long_lines)
    messages = [
        ChatMessage(
            timestamp=datetime(2026, 4, 2, 12, 0, 0),
            speaker="USER",
            text=long_text,
            tags=("execution",),
            source_line=4,
        )
    ]

    html = render_chat_html(messages, user_avatar_url="https://github.com/michalrajkowski.png")

    assert "L001" in html
    assert "L020" in html
    assert "L036" in html
    assert "L055" in html
    assert "<br />...<br />" in html
    assert "L021" not in html
    assert "L035" not in html
