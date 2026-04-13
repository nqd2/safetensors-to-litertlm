# safetensors-to-litertlm

Convert local safetensors-style Hugging Face checkpoints into LiteRT-LM `.litertlm` bundles for on-device inference and Edge Gallery workflows.

This repository now uses a model-agnostic export entrypoint with pluggable backends. `gemma4` is the first backend implemented.

## Supported Backends

| Backend | Status | Notes |
|---|---|---|
| `gemma4` | available | Full export pipeline wired to `litert-torch` Gemma 4 `hf_export`. |
| others | planned | Add new backend modules under `src/safetensors_to_litertlm/converter/backends/`. |

## Prerequisites

- Python 3.12+
- [uv](https://docs.astral.sh/uv/) (recommended)
- Enough RAM and disk for your checkpoint and conversion artifacts

## Install

```bash
uv sync
uv sync --extra export
```

Optional Linux extra for JAX CUDA 13 wheels:

```bash
uv sync --extra export --extra jax-cuda13
```

## Quickstart (Generic CLI)

```bash
export-litertlm --backend gemma4 \
  --model-path ./my-model \
  --output-dir ./out \
  --prefill-lengths 256,512 \
  --cache-length 8192
```

On success, `./out/model.litertlm` is created (bundling enabled by default).

Backward-compatible alias:

```bash
gemma4-export-litertlm --model-path ./my-model --output-dir ./out
```

## Gemma4 Backend Notes

- Default task is `image_text_to_text` for Gemma 4 exports.
- `--export-vision-encoder` remains optional and depends on upstream `litert-torch` support.
- For constrained hosts, use `--low-memory` and selective skip-quant flags.
- `--behavior-parity-mode` rejects known drift-prone shortcut flags.

### Common Gemma4 recipes

INT4 preset:

```bash
export-litertlm --backend gemma4 \
  --model-path ./my-model \
  --output-dir ./out-int4 \
  --quant-preset int4
```

INT8 profile:

```bash
export-litertlm --backend gemma4 \
  --profile litert-community-int8 \
  --model-path ./my-model \
  --output-dir ./out-int8
```

## Logs

By default, output is tee'd to `./gemma-export.logs`.

- Change path: `--log-file /path/to/export.log`
- Disable file tee: `--no-log-tee`

## Bundle Later (if exported with no bundle)

```bash
litertlm-bundle --artifact-dir ./path/to/work_dir --output ./model.litertlm
```

## Validate Bundle

```bash
litert-lm run ./out/model.litertlm --prompt "Hello" --backend cpu
```

## Hardware Notes

- Export is still mostly CPU-oriented even when using `--device cuda`.
- Full FP32 model placement can require very large VRAM.
- Quantization recipes reduce output size, not necessarily export-time peak memory.

## Entry Points

- `export-litertlm`: generic model-family export command
- `gemma4-export-litertlm`: compatibility alias for Gemma 4 backend
- `litertlm-bundle`: package artifacts into `.litertlm`
- `inspector`: inspect `.litertlm` / export artifacts
- `behavior-benchmark`, `parity-diff`: behavior and parity analysis helpers

`python -m safetensors_to_litertlm` prints a quick command reference.
