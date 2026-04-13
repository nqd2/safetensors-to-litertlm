from __future__ import annotations

from safetensors_to_litertlm.converter.backends.base import ExportBackend
from safetensors_to_litertlm.converter.backends import gemma4

_BACKENDS: dict[str, ExportBackend] = {
    gemma4.BACKEND.key: gemma4.BACKEND,
}


def list_backend_keys() -> tuple[str, ...]:
    return tuple(sorted(_BACKENDS.keys()))


def get_backend(key: str) -> ExportBackend:
    try:
        return _BACKENDS[key]
    except KeyError as exc:
        available = ", ".join(list_backend_keys())
        raise KeyError(f"Unknown backend {key!r}. Available backends: {available}") from exc
