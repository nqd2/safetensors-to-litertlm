"""Generic Safetensors -> LiteRTLM exporter entrypoint."""

from __future__ import annotations

import argparse
import sys

from safetensors_to_litertlm.converter.backends.registry import get_backend, list_backend_keys


def _extract_backend(raw_argv: list[str]) -> tuple[str, list[str]]:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--backend", default="gemma4", choices=list_backend_keys())
    ns, remaining = parser.parse_known_args(raw_argv)
    return ns.backend, remaining


def main(argv: list[str] | None = None) -> None:
    raw = list(sys.argv[1:] if argv is None else argv)
    backend_key, remaining = _extract_backend(raw)
    backend = get_backend(backend_key)
    args = backend.parse_args(remaining)
    setattr(args, "backend", backend_key)
    backend.run(args, raw)
