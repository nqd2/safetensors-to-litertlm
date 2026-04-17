from __future__ import annotations

import os

SKIP_VRAM_CHECK_ENV = "GEMMA4_EXPORT_SKIP_CUDA_VRAM_CHECK"
LOW_MEMORY_ENV = "GEMMA4_EXPORT_LOW_MEMORY"
SKIP_PER_LAYER_QUANT_ENV = "GEMMA4_EXPORT_SKIP_PER_LAYER_EMBEDDER_QUANT"
SKIP_PREFILL_DECODE_QUANT_ENV = "GEMMA4_EXPORT_SKIP_PREFILL_DECODE_QUANT"
DEFAULT_LOG_FILE = "gemma-export.logs"


def env_truthy(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in ("1", "true", "yes", "on")
