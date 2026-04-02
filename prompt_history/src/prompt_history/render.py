"""HTML renderer for prompt history chat views."""

from __future__ import annotations

from datetime import date
import hashlib
import html
from typing import Sequence

from prompt_history.models import ChatMessage

_MAX_HEAD_LINES = 20
_MAX_TAIL_LINES = 20


def render_chat_html(
    messages: Sequence[ChatMessage],
    user_avatar_url: str,
    title: str = "Prompt History",
) -> str:
    """Render parsed prompt history messages to a static HTML chat page."""

    rows: list[str] = []
    active_day: date | None = None

    for message in messages:
        message_day = message.timestamp.date()
        if message_day != active_day:
            rows.append(_render_day_separator(message_day.isoformat()))
            active_day = message_day

        rows.append(_render_message(message, user_avatar_url))

    content = "\n".join(rows)

    escaped_title = html.escape(title)
    return (
        "<!DOCTYPE html>\n"
        "<html lang=\"en\">\n"
        "<head>\n"
        "  <meta charset=\"utf-8\" />\n"
        "  <meta name=\"viewport\" content=\"width=device-width, initial-scale=1\" />\n"
        f"  <title>{escaped_title}</title>\n"
        f"  <style>{_CHAT_CSS}</style>\n"
        "</head>\n"
        "<body>\n"
        "  <main class=\"chat-shell\">\n"
        f"    <h1>{escaped_title}</h1>\n"
        f"    {content}\n"
        "  </main>\n"
        "</body>\n"
        "</html>\n"
    )


def tag_colors(tag: str) -> tuple[str, str, str]:
    """Return deterministic background, border, and text colors for one tag."""

    digest = hashlib.sha256(tag.encode("utf-8")).hexdigest()
    hue = int(digest[:8], 16) % 360
    bg = _hsl_to_hex(hue, 72, 92)
    border = _hsl_to_hex(hue, 58, 72)
    text_color = _hsl_to_hex(hue, 54, 24)
    return bg, border, text_color


def _render_day_separator(day_text: str) -> str:
    return f"<section class=\"day-divider\"><span>{html.escape(day_text)}</span></section>"


def _render_message(message: ChatMessage, user_avatar_url: str) -> str:
    speaker_class = "user" if message.speaker == "USER" else "agent"
    avatar = _render_avatar(message.speaker, user_avatar_url)
    tags_markup = _render_tags(message.tags) if message.speaker == "USER" and message.tags else ""

    timestamp_label = message.timestamp.strftime("%H:%M")
    message_text = _format_message_text(message.text)

    return (
        f"<article class=\"message-row {speaker_class}\">"
        f"<div class=\"message-header\">{avatar}<div class=\"message-meta\">"
        f"{html.escape(message.speaker)} {timestamp_label}"
        "</div></div>"
        f"<div class=\"message-content\">"
        f"<div class=\"bubble\"><div class=\"message-text\">{message_text}</div></div>"
        f"{tags_markup}"
        "</div>"
        "</article>"
    )


def _render_avatar(speaker: str, user_avatar_url: str) -> str:
    if speaker == "USER":
        escaped_url = html.escape(user_avatar_url)
        return f"<img class=\"avatar user-avatar\" src=\"{escaped_url}\" alt=\"User avatar\" />"
    return "<div class=\"avatar ai-avatar\" aria-label=\"AI avatar\">AI</div>"


def _render_tags(tags: Sequence[str]) -> str:
    rendered_tags: list[str] = []
    for tag in tags:
        bg, border, text_color = tag_colors(tag)
        rendered_tags.append(
            "<span class=\"tag-pill\" "
            f"style=\"background:{bg};border-color:{border};color:{text_color};\">"
            f"{html.escape(tag)}"
            "</span>"
        )
    tags_html = "".join(rendered_tags)
    return f"<div class=\"tags\">{tags_html}</div>"


def _format_message_text(message_text: str) -> str:
    display_text = _truncate_long_message(message_text)
    escaped_text = html.escape(display_text)
    return escaped_text.replace("\n", "<br />")


def _truncate_long_message(message_text: str) -> str:
    lines = message_text.split("\n")
    max_visible_lines = _MAX_HEAD_LINES + _MAX_TAIL_LINES
    if len(lines) <= max_visible_lines:
        return message_text

    shortened_lines = lines[:_MAX_HEAD_LINES] + ["..."] + lines[-_MAX_TAIL_LINES:]
    return "\n".join(shortened_lines)


def _hsl_to_hex(hue: int, saturation: int, lightness: int) -> str:
    """Convert HSL integer values to #RRGGBB."""

    sat = saturation / 100.0
    light = lightness / 100.0

    chroma = (1.0 - abs((2.0 * light) - 1.0)) * sat
    hue_prime = hue / 60.0
    second_component = chroma * (1.0 - abs((hue_prime % 2.0) - 1.0))

    red, green, blue = 0.0, 0.0, 0.0

    if 0.0 <= hue_prime < 1.0:
        red, green = chroma, second_component
    elif 1.0 <= hue_prime < 2.0:
        red, green = second_component, chroma
    elif 2.0 <= hue_prime < 3.0:
        green, blue = chroma, second_component
    elif 3.0 <= hue_prime < 4.0:
        green, blue = second_component, chroma
    elif 4.0 <= hue_prime < 5.0:
        red, blue = second_component, chroma
    elif 5.0 <= hue_prime < 6.0:
        red, blue = chroma, second_component

    match_lightness = light - (chroma / 2.0)
    red += match_lightness
    green += match_lightness
    blue += match_lightness

    red_int = max(0, min(255, int(round(red * 255.0))))
    green_int = max(0, min(255, int(round(green * 255.0))))
    blue_int = max(0, min(255, int(round(blue * 255.0))))

    return f"#{red_int:02x}{green_int:02x}{blue_int:02x}"


_CHAT_CSS = """
* { box-sizing: border-box; }
body {
  margin: 0;
  background: #f4f7fb;
  color: #16212f;
  font-family: "Segoe UI", Tahoma, Geneva, Verdana, sans-serif;
}
.chat-shell {
  max-width: 960px;
  margin: 0 auto;
  padding: 24px 16px 40px;
}
h1 {
  font-size: 1.6rem;
  margin: 0 0 20px;
}
.day-divider {
  text-align: center;
  margin: 18px 0;
}
.day-divider span {
  border: 1px solid #d8e0ea;
  border-radius: 999px;
  background: #ffffff;
  display: inline-block;
  font-size: 0.78rem;
  letter-spacing: 0.03em;
  padding: 6px 14px;
}
.message-row {
  margin: 16px 0;
  width: 100%;
}
.message-header {
  align-items: center;
  display: flex;
  gap: 10px;
}
.message-row.user .message-header {
  justify-content: flex-end;
}
.message-row.agent .message-header {
  justify-content: flex-start;
}
.message-row.user .message-header .avatar {
  order: 2;
}
.message-row.user .message-header .message-meta {
  order: 1;
}
.message-content {
  max-width: 80%;
  width: 80%;
}
.message-row.user .message-content {
  margin-left: auto;
}
.message-row.agent .message-content {
  margin-right: auto;
}
.bubble {
  border-radius: 16px;
  border: 1px solid #d8e0ea;
  padding: 14px 16px;
}
.message-row.user .bubble {
  background: #d9f1ff;
}
.message-row.agent .bubble {
  background: #eef2f7;
}
.message-meta {
  color: #526074;
  font-size: 0.88rem;
  font-weight: 600;
}
.message-text {
  line-height: 1.45;
  word-break: break-word;
}
.tags {
  display: flex;
  flex-wrap: wrap;
  gap: 10px;
  margin-top: 12px;
}
.tag-pill {
  border: 2px solid;
  border-radius: 999px;
  display: inline-block;
  font-size: 0.86rem;
  font-weight: 600;
  padding: 6px 14px;
}
.avatar {
  width: 44px;
  height: 44px;
  border-radius: 50%;
  border: 1px solid #d8e0ea;
  background: #ffffff;
  object-fit: cover;
}
.ai-avatar {
  align-items: center;
  background: #d5e9da;
  color: #114c2b;
  display: inline-flex;
  font-weight: 800;
  justify-content: center;
  letter-spacing: 1px;
}
@media (max-width: 740px) {
  .message-content {
    max-width: 92%;
    width: 92%;
  }
  .avatar {
    width: 34px;
    height: 34px;
  }
}
"""
