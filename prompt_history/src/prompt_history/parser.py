"""Parser for PROMPTS_HISTORY.md chat-formatted archives."""

from __future__ import annotations

from datetime import datetime
import re

from prompt_history.models import ChatMessage, Speaker

_TIMESTAMP_RE = re.compile(r"^\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}$")
_TAGS_RE = re.compile(r"^TAGS:\s*\[(.*)\]\s*$")
_SPEAKER_LINES: dict[str, Speaker] = {"USER:": "USER", "AGENT:": "AGENT"}


class PromptHistoryParseError(ValueError):
    """Raised when PROMPTS_HISTORY.md does not match the required schema."""

    def __init__(self, line_number: int, details: str) -> None:
        super().__init__(f"Line {line_number}: {details}")
        self.line_number = line_number
        self.details = details


def parse_prompt_history(markdown_text: str) -> list[ChatMessage]:
    """Parse prompt history markdown into typed message records."""

    lines = markdown_text.splitlines()
    index = _skip_header(lines, 0)
    messages: list[ChatMessage] = []

    while index < len(lines):
        index = _skip_blank(lines, index)
        if index >= len(lines):
            break

        timestamp, source_line = _parse_timestamp(lines, index)
        index += 1

        speaker, index = _parse_speaker(lines, index)
        text, index = _parse_fenced_text(lines, index, speaker)

        tags: tuple[str, ...] = ()
        if speaker == "USER":
            tags, index = _parse_tags(lines, index)

        index = _parse_separator(lines, index)

        messages.append(
            ChatMessage(
                timestamp=timestamp,
                speaker=speaker,
                text=text,
                tags=tags,
                source_line=source_line,
            )
        )

    return messages


def _skip_header(lines: list[str], index: int) -> int:
    if index < len(lines) and lines[index].strip() == "# Prompt History":
        index += 1
    return _skip_blank(lines, index)


def _skip_blank(lines: list[str], index: int) -> int:
    while index < len(lines) and lines[index].strip() == "":
        index += 1
    return index


def _parse_timestamp(lines: list[str], index: int) -> tuple[datetime, int]:
    line = lines[index].strip()
    if not _TIMESTAMP_RE.match(line):
        raise PromptHistoryParseError(index + 1, "Expected timestamp in format YYYY-MM-DD HH:MM:SS.")
    return datetime.strptime(line, "%Y-%m-%d %H:%M:%S"), index + 1


def _parse_speaker(lines: list[str], index: int) -> tuple[Speaker, int]:
    if index >= len(lines):
        raise PromptHistoryParseError(index + 1, "Expected speaker line (USER: or AGENT:).")

    speaker_line = lines[index].strip()
    if speaker_line not in _SPEAKER_LINES:
        raise PromptHistoryParseError(index + 1, "Expected speaker line USER: or AGENT:.")

    return _SPEAKER_LINES[speaker_line], index + 1


def _parse_fenced_text(lines: list[str], index: int, speaker: Speaker) -> tuple[str, int]:
    if index >= len(lines) or lines[index].strip() != "```text":
        raise PromptHistoryParseError(index + 1, "Expected opening text fence ```text.")

    index += 1
    content_lines: list[str] = []
    candidate_error: PromptHistoryParseError | None = None

    while index < len(lines):
        if lines[index].strip() == "```":
            closure = _classify_fence_closure(lines, index + 1, speaker)
            if closure.is_valid:
                text = "\n".join(content_lines)
                return text, index + 1
            if closure.candidate_error is not None and candidate_error is None:
                candidate_error = closure.candidate_error
        content_lines.append(lines[index])
        index += 1

    if candidate_error is not None:
        raise candidate_error

    raise PromptHistoryParseError(index + 1, "Missing closing text fence ```.")


def _parse_tags(lines: list[str], index: int) -> tuple[tuple[str, ...], int]:
    if index >= len(lines):
        raise PromptHistoryParseError(index + 1, "Expected TAGS line for USER message.")

    match = _TAGS_RE.match(lines[index].strip())
    if match is None:
        raise PromptHistoryParseError(index + 1, "Expected TAGS line in format TAGS: [a, b].")

    raw_tags = match.group(1).strip()
    if raw_tags == "":
        return (), index + 1

    tags = tuple(tag.strip() for tag in raw_tags.split(",") if tag.strip() != "")
    return tags, index + 1


def _parse_separator(lines: list[str], index: int) -> int:
    if index >= len(lines) or lines[index].strip() != "---":
        raise PromptHistoryParseError(index + 1, "Expected separator line ---.")
    return index + 1


class _FenceClosureClassification:
    def __init__(self, is_valid: bool, candidate_error: PromptHistoryParseError | None) -> None:
        self.is_valid = is_valid
        self.candidate_error = candidate_error


def _classify_fence_closure(
    lines: list[str], after_fence_index: int, speaker: Speaker
) -> _FenceClosureClassification:
    if after_fence_index >= len(lines):
        return _FenceClosureClassification(False, None)

    next_line = lines[after_fence_index].strip()

    if speaker == "USER":
        tags_match = _TAGS_RE.match(next_line)
        if tags_match is not None:
            if after_fence_index + 1 >= len(lines):
                return _FenceClosureClassification(False, None)
            next_after_tags = lines[after_fence_index + 1].strip()
            if next_after_tags == "---":
                if _tail_starts_with_timestamp_or_eof(lines, after_fence_index + 2):
                    return _FenceClosureClassification(True, None)
                return _FenceClosureClassification(False, None)
            if _TIMESTAMP_RE.match(next_after_tags) is not None:
                return _FenceClosureClassification(
                    False,
                    PromptHistoryParseError(after_fence_index + 2, "Expected separator line ---."),
                )
            return _FenceClosureClassification(False, None)

        if next_line == "---":
            if _tail_starts_with_timestamp_or_eof(lines, after_fence_index + 1):
                return _FenceClosureClassification(
                    False,
                    PromptHistoryParseError(
                        after_fence_index + 1, "Expected TAGS line for USER message."
                    ),
                )
            return _FenceClosureClassification(False, None)

        if _TIMESTAMP_RE.match(next_line) is not None:
            return _FenceClosureClassification(
                False,
                PromptHistoryParseError(after_fence_index + 1, "Expected TAGS line for USER message."),
            )
        return _FenceClosureClassification(False, None)

    if next_line == "---":
        if _tail_starts_with_timestamp_or_eof(lines, after_fence_index + 1):
            return _FenceClosureClassification(True, None)
        return _FenceClosureClassification(False, None)

    if _TIMESTAMP_RE.match(next_line) is not None:
        return _FenceClosureClassification(
            False,
            PromptHistoryParseError(after_fence_index + 1, "Expected separator line ---."),
        )
    return _FenceClosureClassification(False, None)


def _tail_starts_with_timestamp_or_eof(lines: list[str], index: int) -> bool:
    next_index = _skip_blank(lines, index)
    if next_index >= len(lines):
        return True
    return _TIMESTAMP_RE.match(lines[next_index].strip()) is not None
