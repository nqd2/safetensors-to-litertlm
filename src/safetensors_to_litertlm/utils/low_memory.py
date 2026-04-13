from __future__ import annotations

import argparse
import os

from safetensors_to_litertlm.utils.env import LOW_MEMORY_ENV, env_truthy


def low_memory_from_argv(argv: list[str]) -> bool:
    p = argparse.ArgumentParser(add_help=False)
    p.add_argument("--low-memory", action="store_true")
    ns, _ = p.parse_known_args(argv)
    return bool(ns.low_memory)


def low_memory_effective(argv: list[str]) -> bool:
    return low_memory_from_argv(argv) or env_truthy(LOW_MEMORY_ENV)


def apply_low_memory_env() -> None:
    """Reduce parallel allocator pressure before importing torch/litert-torch."""
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
    os.environ.setdefault("MALLOC_ARENA_MAX", "2")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


def apply_low_memory_torch() -> None:
    import torch

    torch.set_num_threads(1)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass
