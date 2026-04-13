"""Gemma 4 backend for Safetensors -> LiteRTLM export."""

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

_MIN_CUDA_TOTAL_VRAM_BYTES = 20 * (1024**3)


def _parse_prefill_lengths(s: str) -> list[int]:
    parts = [p.strip() for p in s.split(",") if p.strip()]
    return [int(p) for p in parts]


class _QuantPresetAction(argparse.Action):
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
class ExportProfile:
    key: str
    quantization_recipe: str
    vision_encoder_quantization_recipe: str


EXPORT_PROFILES: dict[str, ExportProfile] = {
    "litert-community-int8": ExportProfile(
        key="litert-community-int8",
        quantization_recipe="dynamic_wi8_afp32",
        vision_encoder_quantization_recipe="weight_only_wi8_afp32",
    )
}


def build_export_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Export local Gemma 4 HF checkpoint to model.litertlm"
    )
    p.add_argument(
        "--profile",
        choices=tuple(EXPORT_PROFILES.keys()),
        default=None,
        help=(
            "Apply a reproducible export profile (sets default recipes). "
            "Explicit flags later in argv override these defaults."
        ),
    )
    p.add_argument("--behavior-parity-mode", action="store_true")
    p.add_argument("--log-file", default=DEFAULT_LOG_FILE)
    p.add_argument("--no-log-tee", action="store_true")
    p.add_argument("--model-path", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--task", default="image_text_to_text", choices=("image_text_to_text", "text_generation"))
    p.add_argument("--prefill-lengths", default="256,512")
    p.add_argument("--cache-length", type=int, default=8192)
    p.add_argument(
        "--quant-preset",
        choices=("int8", "int4"),
        default=None,
        action=_QuantPresetAction,
    )
    p.add_argument("--quantization-recipe", default="dynamic_wi8_afp32")
    p.add_argument("--vision-encoder-quantization-recipe", default="weight_only_wi8_afp32")
    p.add_argument("--export-vision-encoder", action="store_true")
    p.add_argument("--no-bundle-litert-lm", action="store_true")
    p.add_argument("--trust-remote-code", action="store_true")
    p.add_argument("--keep-temporary-files", action="store_true")
    p.add_argument(
        "--use-jinja-template",
        default=True,
        action=argparse.BooleanOptionalAction,
    )
    p.add_argument("--device", default="cpu", metavar="DEVICE")
    p.add_argument("--low-memory", action="store_true")
    p.add_argument("--skip-per-layer-embedder-quant", action="store_true")
    p.add_argument("--skip-per-layer-embedder-export", action="store_true")
    p.add_argument("--skip-prefill-decode-quant", action="store_true")
    p.add_argument("--ram-poor-export", action="store_true")
    return p


def parse_export_args(raw_argv: list[str]) -> argparse.Namespace:
    parser = build_export_parser()
    argv = list(raw_argv)
    profile: ExportProfile | None = None
    if "--profile" in argv:
        idx = argv.index("--profile")
        if idx + 1 < len(argv):
            profile = EXPORT_PROFILES.get(argv[idx + 1])
    if profile is not None:
        parser.set_defaults(
            quantization_recipe=profile.quantization_recipe,
            vision_encoder_quantization_recipe=profile.vision_encoder_quantization_recipe,
            profile=profile.key,
        )
    return parser.parse_args(argv)


def single_token_embedder_enabled(args: argparse.Namespace) -> bool:
    return not bool(args.skip_per_layer_embedder_export)


def validate_behavior_parity_mode(args: argparse.Namespace) -> None:
    if not args.behavior_parity_mode:
        return
    violations: list[str] = []
    if args.skip_per_layer_embedder_export:
        violations.append("--skip-per-layer-embedder-export")
    if args.skip_per_layer_embedder_quant:
        violations.append("--skip-per-layer-embedder-quant")
    if args.skip_prefill_decode_quant:
        violations.append("--skip-prefill-decode-quant")
    if args.ram_poor_export:
        violations.append("--ram-poor-export")
    if not args.use_jinja_template:
        violations.append("--no-use-jinja-template")
    if violations:
        raise SystemExit(
            "Behavior parity mode rejects flags that can alter refusal/style behavior: "
            f"{', '.join(violations)}. Remove these flags or disable --behavior-parity-mode."
        )


class Gemma4Backend:
    key = "gemma4"
    description = "Gemma 4 Hugging Face safetensors checkpoint"

    def parse_args(self, raw_argv: list[str]) -> argparse.Namespace:
        return parse_export_args(raw_argv)

    def run(self, args: argparse.Namespace, raw_argv: list[str]) -> None:
        low_memory = low_memory_effective(raw_argv)
        if low_memory:
            apply_low_memory_env()

        log_fp: TextIO | None = None
        saved_out, saved_err = sys.stdout, sys.stderr
        try:
            if not args.no_log_tee:
                log_path = os.path.abspath(args.log_file)
                parent = os.path.dirname(log_path)
                if parent:
                    os.makedirs(parent, exist_ok=True)
                log_fp = open(log_path, "a", encoding="utf-8", buffering=1)
                log_fp.write(
                    f"\n===== export-litertlm backend=gemma4 {datetime.now(timezone.utc).isoformat()} =====\n"
                )
                log_fp.write(f"argv: {raw_argv!r}\n")
                log_fp.flush()
                sys.stdout = TeeTextIO(saved_out, log_fp)
                sys.stderr = TeeTextIO(saved_err, log_fp)
                print(
                    f"Logging to {log_path} (and terminal); use --no-log-tee to disable.",
                    flush=True,
                )

            self._run_with_dependencies(args, low_memory)
        finally:
            if log_fp is not None:
                sys.stdout = saved_out
                sys.stderr = saved_err
                log_fp.flush()
                log_fp.close()

    def _run_with_dependencies(self, args: argparse.Namespace, low_memory: bool) -> None:
        try:
            from litert_torch.generative.export_hf import export as export_mod
            if low_memory:
                apply_low_memory_torch()
        except ImportError as e:
            print("litert-torch is not installed. Run: uv sync --extra export", file=sys.stderr)
            raise SystemExit(1) from e

        validate_behavior_parity_mode(args)
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
        prefill_lengths = _parse_prefill_lengths(args.prefill_lengths)

        with _maybe_cuda_model(args.device):
            with _maybe_skip_per_layer_exportables(args.skip_per_layer_embedder_export):
                with maybe_selective_quant_skips(skip_prefill_decode_quant, skip_per_layer_quant):
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
                        single_token_embedder=single_token_embedder_enabled(args),
                        split_cache=False,
                        use_jinja_template=args.use_jinja_template,
                        bundle_litert_lm=not args.no_bundle_litert_lm,
                        export_vision_encoder=args.export_vision_encoder,
                        vision_encoder_quantization_recipe=args.vision_encoder_quantization_recipe,
                        experimental_lightweight_conversion=low_memory_run,
                    )


@contextlib.contextmanager
def _maybe_cuda_model(device_str: str) -> Iterator[None]:
    import torch
    from litert_torch.generative.export_hf.core import export_lib as export_lib_mod

    dev = torch.device(device_str)
    if dev.type == "cpu":
        yield
        return
    if dev.type != "cuda":
        print(f'Only "cpu" and "cuda" devices are supported; got {device_str!r}.', file=sys.stderr)
        raise SystemExit(2)
    if not torch.cuda.is_available():
        print("CUDA is not available (install a CUDA torch wheel).", file=sys.stderr)
        raise SystemExit(1)

    cuda_idx = dev.index if dev.index is not None else 0
    total_mem = torch.cuda.get_device_properties(cuda_idx).total_memory
    if total_mem < _MIN_CUDA_TOTAL_VRAM_BYTES and os.environ.get(SKIP_VRAM_CHECK_ENV, "").strip() != "1":
        gib = total_mem / (1024.0**3)
        need = _MIN_CUDA_TOTAL_VRAM_BYTES / (1024.0**3)
        print(
            f"CUDA device {cuda_idx} has ~{gib:.1f} GiB total VRAM; FP32 Gemma 4 export on GPU expects at least ~{need:.0f} GiB.",
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
        yield
    finally:
        export_lib_mod.load_model = original  # type: ignore[method-assign]


@contextlib.contextmanager
def _maybe_skip_per_layer_exportables(skip_per_layer_export: bool) -> Iterator[None]:
    if not skip_per_layer_export:
        yield
        return
    from litert_torch.generative.export_hf.model_ext import exportables as model_exportables

    original = model_exportables.get_additional_exportables

    def _without_per_layer(model_config):
        exportables = dict(original(model_config))
        exportables.pop("per_layer_embedder", None)
        return exportables

    model_exportables.get_additional_exportables = _without_per_layer  # type: ignore[method-assign]
    try:
        yield
    finally:
        model_exportables.get_additional_exportables = original  # type: ignore[method-assign]


BACKEND = Gemma4Backend()
