# safetensors-to-litertlm

Convert a local **Hugging Face Gemma 4** checkpoint (e.g. `huihui-ai/Huihui-gemma-4-E2B-it-abliterated`) into a **`model.litertlm`** bundle for [LiteRT-LM](https://ai.google.dev/edge/litert-lm) and the [Google AI Edge Gallery](https://github.com/google-ai-edge/gallery).

This repo provides a CLI export pipeline built on `litert-torch`’s Gemma 4 `hf_export` path, plus a small bundling helper.

## Prerequisites

- Python 3.12+
- [uv](https://docs.astral.sh/uv/) (recommended)
- Enough RAM + disk for the checkpoint (E2B is multi-GB) and a long, mostly CPU-bound export (often tens of minutes to over an hour).

## Quickstart (recommended)

From the repo root, with a local HF-style folder containing `config.json`, `model.safetensors`, `tokenizer.json`, etc.:

```bash
uv sync
uv sync --extra export

gemma4-export-litertlm \
  --model-path ./Huihui-gemma-4-E2B-it-abliterated \
  --output-dir ./out \
  --prefill-lengths 256,512 \
  --cache-length 8192
```

On success, `./out/model.litertlm` is produced (bundling is on by default). Temporary intermediate artifacts are removed unless you pass `--keep-temporary-files`.

## Install

Base runtime:

```bash
uv sync
```

Export pipeline (litert-torch from Git + torch/transformers/torchvision/…):

```bash
uv sync --extra export
```

Optional (Linux): JAX with CUDA 13 pip wheels:

```bash
uv sync --extra export --extra jax-cuda13
```

## Export to `.litertlm`

Canonical export (good default for most machines, including 6–8 GB NVIDIA GPUs): keep the model on **CPU** during export.

```bash
gemma4-export-litertlm \
  --model-path ./Huihui-gemma-4-E2B-it-abliterated \
  --output-dir ./out \
  --prefill-lengths 256,512 \
  --cache-length 8192
```

### Logging

By default, stdout and stderr are appended to `./gemma-export.logs` as well as the terminal (line-buffered).

- Change log path: `--log-file /path/to/export.log`
- Disable tee-to-file: `--no-log-tee` (terminal only)

Native code that writes directly to OS file descriptors may still bypass Python’s redirection; for a full capture use something like `2>&1 | tee -a gemma-export.logs` as well.

## Quantization (smaller `.litertlm`)

### INT4 weight quantization

Recipes come from **`ai_edge_quantizer.recipe`** (used by litert-torch during TFLite quantization):

| Recipe | Role |
|--------|------|
| `dynamic_wi4_afp32` | INT4 weights, FP32 activations; matches the style of the default `dynamic_wi8_afp32` but smaller on disk. |
| `weight_only_wi4_afp32` | INT4 weights with dequantization before ops; often **better quality** than strict dynamic INT4, at a possible latency cost. |

**Preset (recommended):** `--quant-preset int4` sets the text stack to `dynamic_wi4_afp32` and the vision encoder recipe to `weight_only_wi4_afp32` (when `--export-vision-encoder` is enabled). `--quant-preset int8` restores the usual `dynamic_wi8_afp32` + `weight_only_wi8_afp32` pair.

```bash
gemma4-export-litertlm \
  --model-path ./Huihui-gemma-4-E2B-it-abliterated \
  --output-dir ./out-int4 \
  --quant-preset int4
```

Equivalent explicit flags:

```bash
gemma4-export-litertlm \
  --model-path ./Huihui-gemma-4-E2B-it-abliterated \
  --output-dir ./out-int4 \
  --quantization-recipe dynamic_wi4_afp32 \
  --vision-encoder-quantization-recipe weight_only_wi4_afp32
```

**Export-time GPU vs on-device GPU:** INT4 does **not** remove the need to load the **FP32** HF checkpoint in PyTorch during export, so `--device cuda` still needs large VRAM (see [NVIDIA GPU](#nvidia-gpu-optional-experimental)). INT4 primarily shrinks the **bundled** model. Whether your phone/desktop GPU runs the quantized graph efficiently depends on the LiteRT-LM / TFLite delegate for that build—not on the recipe name alone.

**Quality checks:** After export, run `litert-lm run … --backend cpu` (and your target delegate if any). If you see pad-only or garbage text (a known class of upstream issues for some Gemma exports), try `--quantization-recipe weight_only_wi4_afp32` for the text stack, or fall back to `dynamic_wi8_afp32`.

## Long runs (nohup / tmux)

A full Gemma 4 export often takes tens of minutes to well over an hour. Use `nohup`, `tmux`, or `screen` so closing the terminal/IDE does not kill the job. Example:

```bash
nohup uv run gemma4-export-litertlm --no-log-tee \
  --model-path ./Huihui-gemma-4-E2B-it-abliterated \
  --output-dir ./out-int4 \
  --quant-preset int4 \
  --low-memory \
  --ram-poor-export >> out-int4/export.log 2>&1 &
```

## Troubleshooting (OOM / low RAM)

If the process exits with no Python traceback while the log stops at *Run LiteRT Converter Passes* (or similar), the OS may have killed it for memory (OOM). Try (roughly) in this order:

- **Lower peak memory**: `--low-memory` (caps parallel math libraries, enables litert-torch `experimental_lightweight_conversion`) or `GEMMA4_EXPORT_LOW_MEMORY=1`
- **Reduce signatures**: `--prefill-lengths 256` only (dropping `512` lowers memory, but yields fewer prefill signatures)
- **System-level**: add swap, close other apps, use a machine with more RAM

INT4 does not remove peak RAM during conversion; it mainly affects bundle size.

If the log stops during **`Quantize model`** right after `per_layer_embedder.tflite` is written, the quantizer is often RAM-heavy:

- Use `--skip-per-layer-embedder-quant` (or `GEMMA4_EXPORT_SKIP_PER_LAYER_EMBEDDER_QUANT=1`) to skip quantizing only that subgraph.
- That subgraph stays FP32, so `model.litertlm` becomes much larger—prefer re-running quant on a machine with more RAM when you need a smaller bundle.

If quantization fails inside `model.tflite` (prefill+decode):

- Use `--skip-prefill-decode-quant` (or `GEMMA4_EXPORT_SKIP_PREFILL_DECODE_QUANT=1`) to skip quant on that graph only.
- `--ram-poor-export` turns on both skips above (embedder is still quantized).

`ai-edge-quantizer` does not expose a true “one layer at a time” mode; skipping whole subgraphs is the practical low-RAM workaround.

## Compare bundle sizes

```bash
ls -lh ./out/model.litertlm ./out-int4/model.litertlm
```

## Validate with LiteRT-LM CLI

Smoke inference (CPU backend):

```bash
litert-lm run ./out-int4/model.litertlm --prompt "Hello" --backend cpu
```

Sanity-check the LiteRT-LM CLI with a reference Gemma 4 `.litertlm` (download on first run):

```bash
litert-lm run --from-huggingface-repo litert-community/gemma-4-E2B-it-litert-lm \
  gemma-4-E2B-it.litertlm --prompt "Hi" --backend cpu
```

Import for a stable local name:

```bash
litert-lm import ./out/model.litertlm huihui-gemma4-e2b
litert-lm run huihui-gemma4-e2b --prompt "Hi"
```

**Note:** Image/audio/video in the app requires the corresponding TFLite sections inside the bundle. This repo’s default export keeps `--export-vision-encoder` off because Gemma 4 vision export in litert-torch is still gated on upstream `get_vision_exportables`; text + embedder + per-layer embedder paths match the current HF export pipeline.

## Useful flags

| Flag | Purpose |
|------|---------|
| `--log-file` | Tee stdout/stderr to this file (default `gemma-export.logs`). |
| `--no-log-tee` | Terminal only; no log file. |
| `--no-bundle-litert-lm` | Emit TFLite + tokenizer + metadata only; pack later with `litertlm-bundle`. |
| `--export-vision-encoder` | Vision subgraphs (only when upstream adds Gemma 4 vision in `get_vision_exportables`). |
| `--trust-remote-code` | If the checkpoint requires custom code. |
| `--quant-preset` | Shorthand: `int4` or `int8` (sets text + vision encoder recipes; later `--quantization-recipe` / `--vision-encoder-quantization-recipe` override). |
| `--quantization-recipe` | Text stack recipe (default `dynamic_wi8_afp32`; e.g. `dynamic_wi4_afp32`, `weight_only_wi4_afp32`). |
| `--vision-encoder-quantization-recipe` | Vision encoder when `--export-vision-encoder` is set (default `weight_only_wi8_afp32`). |
| `--low-memory` | Lower RAM during export (thread caps + `experimental_lightweight_conversion`; same as `GEMMA4_EXPORT_LOW_MEMORY=1`). |
| `--skip-per-layer-embedder-quant` | Skip quantizer on `per_layer_embedder.tflite` only (avoids common OOM); subgraph stays FP32; huge `.litertlm`. Same as `GEMMA4_EXPORT_SKIP_PER_LAYER_EMBEDDER_QUANT=1`. |
| `--skip-prefill-decode-quant` | Skip quantizer on `model.tflite` (prefill+decode). Same as `GEMMA4_EXPORT_SKIP_PREFILL_DECODE_QUANT=1`. |
| `--ram-poor-export` | Both skips above; still quantizes `embedder.tflite`. |

## Optional: bundle after a non-bundled export

```bash
litertlm-bundle --artifact-dir ./path/to/work_dir --output ./model.litertlm
```

Adjust `--prefill-decode`, `--embedder`, `--tokenizer`, and `--llm-metadata` if your filenames differ.

## Google AI Edge Gallery (device)

1. Copy `model.litertlm` to the phone (USB, cloud, or `adb push`).
2. Open the [Edge Gallery](https://github.com/google-ai-edge/gallery) app and import or open the file per the current in-app flow (see the Gallery README; it changes by release).

Example:

```bash
adb push out/model.litertlm /sdcard/Download/model.litertlm
```

On device, pick the file from Downloads or the path your Gallery version expects.

## Hardware notes

### Google Tensor NPU vs GPU / CPU

LiteRT-LM’s documented **LLM NPU** path today targets **Qualcomm** and **MediaTek** (SoC-specific bundles and runtimes). **Google Tensor** NPU for generic LiteRT is separate and largely **experimental** (see [NPU acceleration with LiteRT](https://ai.google.dev/edge/litert/next/npu) and [Run LLMs using LiteRT-LM](https://ai.google.dev/edge/litert/next/litert_lm_npu)). Expect **GPU or CPU** on Pixel-class hardware for a generic `.litertlm` until Tensor-specific compiled artifacts and SDK workflows match your SoC.

### JAX on NVIDIA (CUDA 13)

[JAX’s installation guide](https://jax.readthedocs.io/en/latest/installation.html) recommends **CUDA 13 GPU** installs via pip wheels on **Linux** (x86_64 or aarch64):

```bash
pip install --upgrade pip
pip install --upgrade "jax[cuda13]"
```

With **uv** (same environment as this project):

```bash
uv sync --extra export --extra jax-cuda13
# or, into the active venv only:
uv pip install -U "jax[cuda13]"
```

**Requirements (from JAX docs):** NVIDIA driver **≥ 580** on Linux for CUDA 13; GPU **SM ≥ 7.5**. If JAX picks up the wrong CUDA libraries, **unset `LD_LIBRARY_PATH`** so bundled wheels win. For a **system CUDA** install instead of pip-shipped CUDA, use `pip install --upgrade "jax[cuda13-local]"` (see the same doc section).

**Note:** `litert-torch` already depends on CPU `jax`; adding `jax[cuda13]` swaps in GPU plugin wheels where supported. This mainly helps **JAX-backed** code paths; the Gemma export pipeline is still largely **PyTorch + CPU lowering** unless a component explicitly uses JAX on GPU.

### NVIDIA GPU (optional, experimental)

The LiteRT Torch **HF export** path is built around **CPU** tracing and MLIR/TFLite lowering; Google’s docs often describe this stack as CPU-oriented. You can still try placing the **PyTorch model on CUDA** so `torch.export` and forward passes run on the GPU, which sometimes reduces wall time on a strong GPU (depends on PyTorch/litert-torch versions):

```bash
# Requires a CUDA-enabled PyTorch wheel (the default uv lock pulls torch+cu13 on Linux).
export CUDA_VISIBLE_DEVICES=0   # optional: pick one GPU
gemma4-export-litertlm \
  --model-path ./Huihui-gemma-4-E2B-it-abliterated \
  --output-dir ./out \
  --device cuda
```

**6–8 GB consumer GPUs (e.g. RTX 4050 Laptop) will OOM** when moving the full **FP32** Gemma 4 checkpoint to CUDA; use the default **CPU** export above, or a **24 GB+** GPU class machine for `--device cuda`. The CLI refuses CUDA when total VRAM is under **20 GiB** unless you set `GEMMA4_EXPORT_SKIP_CUDA_VRAM_CHECK=1` (still likely to OOM).

**Caveats:** large **FP32** checkpoints need **a lot of VRAM**; you may hit OOM. If export fails, use `--device cpu` (default). Heavy steps (MLIR lowering, quantization, bundling) may remain CPU-bound regardless.

## Toolchain notes

- **litert-torch** is pinned to a **Git revision** that includes Gemma 4 `hf_export` (PyPI `0.8.0` did not ship `model_ext/gemma4`).
- Locked versions are in [`uv.lock`](uv.lock); primary runtime deps include `litert-lm==0.10.1`, `transformers`, `torch`, `torchvision` (needed for `AutoImageProcessor` on `image_text_to_text`), and nightly `ai-edge-litert` / quantizer wheels pulled in by litert-torch.

## References

- [LiteRT-LM overview](https://ai.google.dev/edge/litert-lm)
- [LiteRT-LM CLI](https://ai.google.dev/edge/litert-lm/cli)
- [Convert PyTorch GenAI models](https://ai.google.dev/edge/litert/conversion/pytorch/genai)
- [litert-community/gemma-4-E2B-it-litert-lm](https://huggingface.co/litert-community/gemma-4-E2B-it-litert-lm) (reference `.litertlm`)

## Entry points

After `uv sync --extra export`:

- `gemma4-export-litertlm`: export Gemma 4 checkpoint → `model.litertlm`
- `litertlm-bundle`: bundle artifacts into a `.litertlm` (when exporting without bundling)
- `inspector`: inspect a `.litertlm` / export artifacts

`python -m safetensors_to_litertlm` prints a short pointer to the commands above.
