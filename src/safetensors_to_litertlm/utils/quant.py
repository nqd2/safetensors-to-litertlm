from __future__ import annotations

import contextlib
import os
from collections.abc import Iterator

from safetensors_to_litertlm.utils.env import (
    SKIP_PER_LAYER_QUANT_ENV,
    SKIP_PREFILL_DECODE_QUANT_ENV,
)

_PER_LAYER_EMBEDDER_TFLITE = "per_layer_embedder.tflite"
_PREFILL_DECODE_TFLITE = "model.tflite"


@contextlib.contextmanager
def maybe_selective_quant_skips(
    skip_prefill_decode: bool,
    skip_per_layer: bool,
) -> Iterator[None]:
    """Skip ai-edge-quantizer on selected flatbuffers (OOM). No true layer-by-layer API."""
    if not skip_prefill_decode and not skip_per_layer:
        yield
        return
    from litert_torch.generative.export_hf.core import export_lib as export_lib_mod

    original = export_lib_mod.maybe_quantize_model

    def _wrapped(model_path: str, quantization_recipe: str | None = None) -> str:
        if not quantization_recipe:
            return original(model_path, quantization_recipe)
        base = os.path.basename(model_path)
        if skip_prefill_decode and base == _PREFILL_DECODE_TFLITE:
            print(
                f"Skipping quantization for {_PREFILL_DECODE_TFLITE} "
                f"({SKIP_PREFILL_DECODE_QUANT_ENV}=1 or --skip-prefill-decode-quant / "
                "--ram-poor-export): prefill+decode stays FP32; .litertlm very large.",
                flush=True,
            )
            return model_path
        if skip_per_layer and base == _PER_LAYER_EMBEDDER_TFLITE:
            print(
                f"Skipping quantization for {_PER_LAYER_EMBEDDER_TFLITE} "
                f"({SKIP_PER_LAYER_QUANT_ENV}=1 or --skip-per-layer-embedder-quant / "
                "--ram-poor-export): subgraph stays FP32.",
                flush=True,
            )
            return model_path
        return original(model_path, quantization_recipe)

    export_lib_mod.maybe_quantize_model = _wrapped  # type: ignore[method-assign]
    try:
        yield
    finally:
        export_lib_mod.maybe_quantize_model = original  # type: ignore[method-assign]
