"""Shim entry point; use the installed CLIs after ``uv sync --extra export``."""

from __future__ import annotations


def main() -> None:
    print(
        "safetensors-to-litertlm: use the console scripts from a dev install.\n"
        "  uv sync --extra export\n"
        "  export-litertlm --backend gemma4 --help\n"
        "  gemma4-export-litertlm --help  # compatibility alias\n"
        "  litertlm-bundle --help\n"
        "See README.md for the full Edge Gallery / LiteRT-LM workflow."
    )


if __name__ == "__main__":
    main()
