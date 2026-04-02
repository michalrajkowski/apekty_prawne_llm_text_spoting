"""CLI for building HTML chat views from PROMPTS_HISTORY.md."""

from __future__ import annotations

import argparse
import base64
import mimetypes
from pathlib import Path

from prompt_history.models import ChatMessage
from prompt_history.parser import parse_prompt_history
from prompt_history.render import render_chat_html

_DEFAULT_INPUT = Path("PROMPTS_HISTORY.md")
_DEFAULT_HTML = Path("runs/prompt_history.html")
_DEFAULT_AVATAR = "assets/michalrajkowski.png"


def build_prompt_history_outputs(
    input_path: Path,
    html_output_path: Path,
    user_avatar_url: str,
    title: str,
) -> None:
    """Build static HTML output from a prompt history markdown file."""

    markdown_text = input_path.read_text(encoding="utf-8")
    parsed_messages = parse_prompt_history(markdown_text)
    messages = [message for message in parsed_messages if not _is_excluded_message(message)]

    avatar_source = _resolve_avatar_source(user_avatar_url)
    html_document = render_chat_html(messages, user_avatar_url=avatar_source, title=title)

    html_output_path.parent.mkdir(parents=True, exist_ok=True)
    html_output_path.write_text(html_document, encoding="utf-8")


def _resolve_avatar_source(avatar_argument: str) -> str:
    avatar_path = Path(avatar_argument)
    if avatar_path.exists():
        media_type, _ = mimetypes.guess_type(avatar_path.name)
        if media_type is None:
            media_type = "application/octet-stream"
        encoded = base64.b64encode(avatar_path.read_bytes()).decode("ascii")
        return f"data:{media_type};base64,{encoded}"
    return avatar_argument


def _is_excluded_message(message: ChatMessage) -> bool:
    return message.text.lstrip().startswith("# AGENTS.md")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build static HTML chat view from PROMPTS_HISTORY.md."
    )
    parser.add_argument("--input", type=Path, default=_DEFAULT_INPUT, help="Input prompt history markdown file.")
    parser.add_argument(
        "--html-output",
        type=Path,
        default=_DEFAULT_HTML,
        help="Output path for generated HTML chat view.",
    )
    parser.add_argument(
        "--user-avatar-url",
        type=str,
        default=_DEFAULT_AVATAR,
        help="Avatar URL used for USER messages.",
    )
    parser.add_argument(
        "--title",
        type=str,
        default="Prompt History",
        help="Document title used in HTML header.",
    )
    return parser


def main() -> int:
    """Run the CLI entrypoint."""

    parser = _build_parser()
    args = parser.parse_args()

    build_prompt_history_outputs(
        input_path=args.input,
        html_output_path=args.html_output,
        user_avatar_url=args.user_avatar_url,
        title=args.title,
    )

    print(f"HTML written to: {args.html_output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
