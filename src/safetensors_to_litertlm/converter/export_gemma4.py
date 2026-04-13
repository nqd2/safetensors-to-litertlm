"""Compatibility wrapper for legacy gemma4 CLI entrypoint."""

from __future__ import annotations

import sys

from safetensors_to_litertlm.converter.backends.gemma4 import (
    build_export_parser as _build_export_parser,
)
from safetensors_to_litertlm.converter.backends.gemma4 import (
    parse_export_args as _parse_export_args,
)
from safetensors_to_litertlm.converter.backends.gemma4 import (
    single_token_embedder_enabled as _single_token_embedder_enabled,
)
from safetensors_to_litertlm.converter.backends.gemma4 import (
    validate_behavior_parity_mode as _validate_behavior_parity_mode,
)
from safetensors_to_litertlm.converter.export import main as _generic_main


def main(argv: list[str] | None = None) -> None:
    extra = list(sys.argv[1:] if argv is None else argv)
    _generic_main(["--backend", "gemma4", *extra])


if __name__ == "__main__":
    main()
