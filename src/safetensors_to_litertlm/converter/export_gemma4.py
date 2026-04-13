"""Export a local Gemma 4 HF folder to a bundled .litertlm using litert-torch hf_export.

Requires optional dependencies: ``uv sync --extra export``.

Upstream ``litert_torch.generative.export_hf.export.export`` performs conversion,
optional vision encoder export, tokenizer export, and LiteRT-LM packaging
(``LitertLmFileBuilder``) when ``bundle_litert_lm=True``.

Gemma 4 currently has no vision exportables in ``get_vision_exportables``; keep
``export_vision_encoder=False`` (default here) for text + per-layer embedder
bundles. Set ``--export-vision-encoder`` when upstream adds Gemma 4 vision.
"""

from __future__ import annotations

import argparse
import contextlib
import os
import sys
from collections.abc import Iterator
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import TextIO

from safetensors_to_litertlm.utils.env import (
    DEFAULT_LOG_FILE,
    LOW_MEMORY_ENV,
    SKIP_PER_LAYER_QUANT_ENV,
    SKIP_PREFILL_DECODE_QUANT_ENV,
    SKIP_VRAM_CHECK_ENV,
    env_truthy,
)
from safetensors_to_litertlm.utils.low_memory import (
    apply_low_memory_env,
    apply_low_memory_torch,
    low_memory_effective,
)
from safetensors_to_litertlm.utils.quant import maybe_selective_quant_skips
from safetensors_to_litertlm.utils.tee import TeeTextIO

# Full FP32 Gemma 4 multimodal placement on GPU needs far more than laptop VRAM.
_MIN_CUDA_TOTAL_VRAM_BYTES = 20 * (1024**3)


def _parse_prefill_lengths(s: str) -> list[int]:
    parts = [p.strip() for p in s.split(",") if p.strip()]
    return [int(p) for p in parts]


class _QuantPresetAction(argparse.Action):
    """Apply shorthand recipes; argv order matters (later flags override earlier)."""

    _PRESETS: dict[str, tuple[str, str]] = {
        "int8": ("dynamic_wi8_afp32", "weight_only_wi8_afp32"),
        "int4": ("dynamic_wi4_afp32", "weight_only_wi4_afp32"),
    }

    def __call__(
        self,
        parser: argparse.ArgumentParser,
        namespace: argparse.Namespace,
        values: str | None,
        option_string: str | None = None,
    ) -> None:
        if values is None:
            return
        text_r, vision_r = self._PRESETS[values]
        setattr(namespace, self.dest, values)
        namespace.quantization_recipe = text_r
        namespace.vision_encoder_quantization_recipe = vision_r


@dataclass(frozen=True)
class _ExportProfile:
    key: str
    quantization_recipe: str
    vision_encoder_quantization_recipe: str


_EXPORT_PROFILES: dict[str, _ExportProfile] = {
    "litert-community-int8": _ExportProfile(
        key="litert-community-int8",
        quantization_recipe="dynamic_wi8_afp32",
        vision_encoder_quantization_recipe="weight_only_wi8_afp32",
    )
}


def _build_export_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Export local Gemma 4 HF checkpoint to model.litertlm"
    )
    p.add_argument(
        "--profile",
        choices=tuple(_EXPORT_PROFILES.keys()),
        default=None,
        help=(
            "Apply a reproducible export profile (sets default recipes). "
            "Explicit flags later in argv override these defaults."
        ),
    )
    p.add_argument(
        "--log-file",
        default=DEFAULT_LOG_FILE,
        help=(
            f"Append stdout/stderr to this file as well as the terminal (default: {DEFAULT_LOG_FILE}). "
            "Opens before imports so early logs are captured."
        ),
    )
    p.add_argument(
        "--no-log-tee",
        action="store_true",
        help="Do not tee output to --log-file (terminal only).",
    )
    p.add_argument(
        "--model-path",
        required=True,
        help="Directory with config.json, model.safetensors, tokenizer.json, etc.",
    )
    p.add_argument(
        "--output-dir",
        required=True,
        help="Directory for exported artifacts (includes model.litertlm when bundling)",
    )
    p.add_argument(
        "--task",
        default="image_text_to_text",
        choices=("image_text_to_text", "text_generation"),
        help="Use image_text_to_text for Gemma4ForConditionalGeneration (default)",
    )
    p.add_argument(
        "--prefill-lengths",
        default="256,512",
        help="Comma-separated prefill chunk sizes (e.g. 256,512,1024)",
    )
    p.add_argument(
        "--cache-length",
        type=int,
        default=8192,
        help="KV cache / max context slot size for export",
    )
    p.add_argument(
        "--quant-preset",
        choices=("int8", "int4"),
        default=None,
        action=_QuantPresetAction,
        help=(
            "Shorthand for ai-edge-quantizer recipes: int4 -> dynamic_wi4_afp32 + "
            "weight_only_wi4_afp32; int8 -> dynamic_wi8_afp32 + weight_only_wi8_afp32. "
            "Argv order matters: a later --quantization-recipe or "
            "--vision-encoder-quantization-recipe overrides the preset for that field."
        ),
    )
    p.add_argument(
        "--quantization-recipe",
        default="dynamic_wi8_afp32",
        help=(
            "Text stack recipe (ai_edge_quantizer.recipe), e.g. dynamic_wi8_afp32, "
            "dynamic_wi4_afp32, weight_only_wi4_afp32"
        ),
    )
    p.add_argument(
        "--vision-encoder-quantization-recipe",
        default="weight_only_wi8_afp32",
        help=(
            "Vision encoder when --export-vision-encoder is set "
            "(e.g. weight_only_wi8_afp32, weight_only_wi4_afp32)"
        ),
    )
    p.add_argument(
        "--export-vision-encoder",
        action="store_true",
        help="Export vision TFLite subgraphs (requires upstream Gemma 4 vision support)",
    )
    p.add_argument(
        "--no-bundle-litert-lm",
        action="store_true",
        help="Only emit intermediate TFLite/tokenizer; skip .litertlm packaging",
    )
    p.add_argument(
        "--trust-remote-code",
        action="store_true",
        help="Pass trust_remote_code=True when loading the HF model",
    )
    p.add_argument(
        "--keep-temporary-files",
        action="store_true",
        help="Keep litert-torch work_dir contents under output-dir",
    )
    p.add_argument(
        "--use-jinja-template",
        default=True,
        action=argparse.BooleanOptionalAction,
        help="Use tokenizer chat_template as jinja in LlmMetadata (default: true)",
    )
    p.add_argument(
        "--device",
        default="cpu",
        metavar="DEVICE",
        help=(
            'PyTorch device for the loaded model during export, e.g. "cpu" (default), '
            '"cuda", or "cuda:0". Experimental for CUDA: litert-torch lowering to TFLite '
            "is still mostly CPU-bound; GPU may speed torch.export / attention on some "
            "steps but can OOM (FP32 Gemma 4 needs substantial VRAM) or fail if the "
            "converter expects CPU tensors. GPUs with under 20 GiB total VRAM are "
            f"refused unless {SKIP_VRAM_CHECK_ENV}=1."
        ),
    )
    p.add_argument(
        "--low-memory",
        action="store_true",
        help=(
            "Reduce RAM during export: cap BLAS/OpenMP threads, limit PyTorch thread pools, "
            "and enable litert-torch experimental_lightweight_conversion (may affect speed "
            f"or compatibility). Same as env {LOW_MEMORY_ENV}=1."
        ),
    )
    p.add_argument(
        "--skip-per-layer-embedder-quant",
        action="store_true",
        help=(
            "Do not run ai-edge-quantizer on per_layer_embedder.tflite (often where RAM "
            "spikes). The subgraph stays FP32; output .litertlm is much larger. Same as env "
            f"{SKIP_PER_LAYER_QUANT_ENV}=1."
        ),
    )
    p.add_argument(
        "--skip-prefill-decode-quant",
        action="store_true",
        help=(
            "Do not run ai-edge-quantizer on model.tflite (prefill+decode stack). Use when "
            "quantization dies in numpy/min-max (RAM). Graph stays FP32; bundle huge. Same as env "
            f"{SKIP_PREFILL_DECODE_QUANT_ENV}=1. "
            "There is no supported layer-by-layer quant in ai-edge-quantizer."
        ),
    )
    p.add_argument(
        "--ram-poor-export",
        action="store_true",
        help=(
            "Same as --skip-prefill-decode-quant --skip-per-layer-embedder-quant. "
            "Still quantizes embedder.tflite. For lowest RAM through the quantizer."
        ),
    )
    return p


def _parse_export_args(raw_argv: list[str]) -> argparse.Namespace:
    """Parse argv with optional profile defaults applied, preserving 'later flags win'."""
    parser = _build_export_parser()
    argv = list(raw_argv)
    profile: _ExportProfile | None = None
    if "--profile" in argv:
        idx = argv.index("--profile")
        if idx + 1 < len(argv):
            profile = _EXPORT_PROFILES.get(argv[idx + 1])

    if profile is not None:
        parser.set_defaults(
            quantization_recipe=profile.quantization_recipe,
            vision_encoder_quantization_recipe=profile.vision_encoder_quantization_recipe,
            profile=profile.key,
        )

    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    raw = list(sys.argv[1:] if argv is None else argv)
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument(
        "--log-file",
        default=DEFAULT_LOG_FILE,
        help=argparse.SUPPRESS,
    )
    pre.add_argument("--no-log-tee", action="store_true", help=argparse.SUPPRESS)
    pre_ns, _ = pre.parse_known_args(raw)

    low_memory = low_memory_effective(raw)
    if low_memory:
        apply_low_memory_env()

    log_fp: TextIO | None = None
    saved_out, saved_err = sys.stdout, sys.stderr
    try:
        if not pre_ns.no_log_tee:
            log_path = os.path.abspath(pre_ns.log_file)
            parent = os.path.dirname(log_path)
            if parent:
                os.makedirs(parent, exist_ok=True)
            log_fp = open(log_path, "a", encoding="utf-8", buffering=1)
            log_fp.write(
                f"\n===== gemma4-export-litertlm {datetime.now(timezone.utc).isoformat()} =====\n"
            )
            log_fp.write(f"argv: {raw!r}\n")
            log_fp.flush()
            sys.stdout = TeeTextIO(saved_out, log_fp)
            sys.stderr = TeeTextIO(saved_err, log_fp)
            print(
                f"Logging to {log_path} (and terminal); use --no-log-tee to disable.",
                flush=True,
            )

        try:
            from litert_torch.generative.export_hf import export as export_mod

            if low_memory:
                apply_low_memory_torch()
                print(
                    "Low-memory mode: parallel BLAS/OpenMP capped, "
                    "experimental_lightweight_conversion enabled "
                    f"(also set {LOW_MEMORY_ENV}=1 to enable via env).",
                    flush=True,
                )
        except ImportError as e:
            print(
                "litert-torch is not installed. Run: uv sync --extra export",
                file=sys.stderr,
            )
            raise SystemExit(1) from e

        args = _parse_export_args(raw)

        low_memory_run = bool(args.low_memory) or env_truthy(LOW_MEMORY_ENV)
        skip_per_layer_quant = (
            bool(args.ram_poor_export)
            or bool(args.skip_per_layer_embedder_quant)
            or env_truthy(SKIP_PER_LAYER_QUANT_ENV)
        )
        skip_prefill_decode_quant = (
            bool(args.ram_poor_export)
            or bool(args.skip_prefill_decode_quant)
            or env_truthy(SKIP_PREFILL_DECODE_QUANT_ENV)
        )
        profile_key = getattr(args, "profile", None)
        if profile_key:
            print(
                "[export-profile] "
                f"profile={profile_key} "
                f"quantization_recipe={args.quantization_recipe} "
                f"vision_encoder_quantization_recipe={args.vision_encoder_quantization_recipe} "
                f"skip_prefill_decode_quant={skip_prefill_decode_quant} "
                f"skip_per_layer_embedder_quant={skip_per_layer_quant}",
                flush=True,
            )
        prefill_lengths = _parse_prefill_lengths(args.prefill_lengths)

        @contextlib.contextmanager
        def _maybe_cuda_model(device_str: str) -> Iterator[None]:
            import torch
            from litert_torch.generative.export_hf.core import export_lib as export_lib_mod

            dev = torch.device(device_str)
            if dev.type == "cpu":
                yield
                return
            if dev.type != "cuda":
                print(
                    f'Only "cpu" and "cuda" devices are supported; got {device_str!r}.',
                    file=sys.stderr,
                )
                raise SystemExit(2)
            if not torch.cuda.is_available():
                print(
                    "CUDA is not available (install a CUDA torch wheel).",
                    file=sys.stderr,
                )
                raise SystemExit(1)

            cuda_idx = dev.index if dev.index is not None else 0
            total_mem = torch.cuda.get_device_properties(cuda_idx).total_memory
            if (
                total_mem < _MIN_CUDA_TOTAL_VRAM_BYTES
                and os.environ.get(SKIP_VRAM_CHECK_ENV, "").strip() != "1"
            ):
                gib = total_mem / (1024.0**3)
                need = _MIN_CUDA_TOTAL_VRAM_BYTES / (1024.0**3)
                print(
                    f"CUDA device {cuda_idx} has ~{gib:.1f} GiB total VRAM; "
                    f"FP32 Gemma 4 export on GPU expects at least ~{need:.0f} GiB "
                    "(full model .to(cuda) will OOM on typical 6–12 GiB laptops).\n"
                    "Run without --device (CPU, default), or set "
                    f"{SKIP_VRAM_CHECK_ENV}=1 to force GPU anyway (not recommended).",
                    file=sys.stderr,
                )
                raise SystemExit(1)

            original = export_lib_mod.load_model

            def load_model_on_device(*a, **kw):
                artifacts = original(*a, **kw)
                artifacts.model = artifacts.model.to(dev)
                return artifacts

            export_lib_mod.load_model = load_model_on_device  # type: ignore[method-assign]
            try:
                print(
                    "Using experimental CUDA placement on "
                    f"{torch.cuda.get_device_name(cuda_idx)} "
                    f"(VRAM: {total_mem // (1024**3)} GiB)."
                )
                yield
            finally:
                export_lib_mod.load_model = original  # type: ignore[method-assign]

        # Gemma 4 requires external embedder; vision=False does not auto-enable it in config.
        with _maybe_cuda_model(args.device):
            with maybe_selective_quant_skips(
                skip_prefill_decode_quant,
                skip_per_layer_quant,
            ):
                export_mod.export(
                    model=args.model_path,
                    output_dir=args.output_dir,
                    task=args.task,
                    keep_temporary_files=args.keep_temporary_files,
                    trust_remote_code=args.trust_remote_code,
                    prefill_lengths=prefill_lengths,
                    cache_length=args.cache_length,
                    quantization_recipe=args.quantization_recipe,
                    externalize_embedder=True,
                    single_token_embedder=True,
                    split_cache=False,
                    use_jinja_template=args.use_jinja_template,
                    bundle_litert_lm=not args.no_bundle_litert_lm,
                    export_vision_encoder=args.export_vision_encoder,
                    vision_encoder_quantization_recipe=args.vision_encoder_quantization_recipe,
                    experimental_lightweight_conversion=low_memory_run,
                )
    finally:
        if log_fp is not None:
            sys.stdout = saved_out
            sys.stderr = saved_err
            log_fp.flush()
            log_fp.close()


if __name__ == "__main__":
    main()
