"""Python bootstrap runner for prompt history HTML generation."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import subprocess
import sys

_PROJECT_ROOT = Path(__file__).resolve().parent
_VENV_DIR = _PROJECT_ROOT / ".task_history_venv"
_REQUIREMENTS_PATH = _PROJECT_ROOT / "requirements.txt"
_DEFAULT_INPUT = _PROJECT_ROOT / "PROMPTS_HISTORY.md"
_DEFAULT_HTML = _PROJECT_ROOT / "runs/prompt_history.html"
_DEFAULT_AVATAR = "assets/michalrajkowski.png"


def _build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build prompt history HTML output.")
    parser.add_argument("--input", type=Path, default=_DEFAULT_INPUT)
    parser.add_argument("--html-output", type=Path, default=_DEFAULT_HTML)
    parser.add_argument("--user-avatar-url", type=str, default=_DEFAULT_AVATAR)
    parser.add_argument("--title", type=str, default="Prompt History")
    return parser


def _venv_python_path(venv_dir: Path) -> Path:
    if os.name == "nt":
        return venv_dir / "Scripts" / "python.exe"
    return venv_dir / "bin" / "python"


def _ensure_venv(venv_dir: Path) -> Path:
    python_path = _venv_python_path(venv_dir)
    if not python_path.exists():
        subprocess.run([sys.executable, "-m", "venv", str(venv_dir)], check=True)
    return python_path


def _install_requirements(venv_python: Path, requirements_path: Path) -> None:
    subprocess.run([str(venv_python), "-m", "pip", "install", "--upgrade", "pip"], check=True)
    subprocess.run([str(venv_python), "-m", "pip", "install", "-r", str(requirements_path)], check=True)


def _run_builder(
    venv_python: Path,
    input_path: Path,
    html_output_path: Path,
    user_avatar_url: str,
    title: str,
) -> None:
    environment = os.environ.copy()
    source_path = _PROJECT_ROOT / "src"
    existing_python_path = environment.get("PYTHONPATH", "")
    if existing_python_path:
        environment["PYTHONPATH"] = f"{source_path}{os.pathsep}{existing_python_path}"
    else:
        environment["PYTHONPATH"] = str(source_path)

    command = [
        str(venv_python),
        "-m",
        "prompt_history.cli",
        "--input",
        str(input_path),
        "--html-output",
        str(html_output_path),
        "--user-avatar-url",
        user_avatar_url,
        "--title",
        title,
    ]
    subprocess.run(command, check=True, env=environment, cwd=str(_PROJECT_ROOT))


def main() -> int:
    """Create isolated venv, install requirements, and run the prompt-history builder."""

    parser = _build_argument_parser()
    args = parser.parse_args()

    venv_python = _ensure_venv(_VENV_DIR)
    _install_requirements(venv_python, _REQUIREMENTS_PATH)

    _run_builder(
        venv_python=venv_python,
        input_path=args.input,
        html_output_path=args.html_output,
        user_avatar_url=args.user_avatar_url,
        title=args.title,
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
