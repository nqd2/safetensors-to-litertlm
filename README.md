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
- TranslateGemma (`google/translategemma-4b-it`) currently hits an upstream Gemma3 vision export limitation (`vision_tower` missing).
- By default, `--auto-fallback-text-only` is enabled: if that known vision error occurs during multimodal export, the tool retries once with `--task text_generation` and vision export disabled.
- Use `--no-auto-fallback-text-only` if you want strict failure instead of fallback.
- `--multimodal-intent` controls planner behavior:
  - `legacy` (default): preserve existing task/vision flags and fallback behavior.
  - `best-effort`: try multimodal when capability probe indicates support, else downgrade to text-only.
  - `strict`: require multimodal capability; fail fast if probe says unsupported.
  - `text-only`: always export text-generation artifacts.
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

Strict multimodal attempt (no auto fallback):

```bash
export-litertlm --backend gemma4 \
  --model-path google/translategemma-4b-it \
  --output-dir ./out-mm-strict \
  --task image_text_to_text \
  --export-vision-encoder \
  --no-auto-fallback-text-only
```

Best-effort multimodal planner (recommended for mixed model families):

```bash
export-litertlm --backend gemma4 \
  --model-path google/translategemma-4b-it \
  --output-dir ./out-mm-best-effort \
  --multimodal-intent best-effort \
  --task image_text_to_text \
  --export-vision-encoder
```

Force strict multimodal planner (fail-fast if probe says unsupported):

```bash
export-litertlm --backend gemma4 \
  --model-path google/translategemma-4b-it \
  --output-dir ./out-mm-strict-intent \
  --multimodal-intent strict \
  --task image_text_to_text \
  --export-vision-encoder
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
