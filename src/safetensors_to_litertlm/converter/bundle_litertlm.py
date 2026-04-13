"""Re-pack a directory of exported LiteRT artifacts into ``model.litertlm``.

Use this only if you ran export with ``--no-bundle-litertlm`` and need to bundle
later. Normal flow: use ``gemma4-export-litertlm`` with bundling enabled
(default); it calls the same ``LitertLmFileBuilder`` path inside litert-torch.

Paths are resolved relative to *artifact_dir* (typically the export work_dir
left by ``--keep-temporary-files``).
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


def main(argv: list[str] | None = None) -> None:
    try:
        from ai_edge_litert.internal import litertlm_builder
    except ImportError as e:
        print(
            "ai-edge-litert (from litert-torch export extra) is required.",
            file=sys.stderr,
        )
        raise SystemExit(1) from e

    p = argparse.ArgumentParser(description="Pack TFLite + tokenizer + metadata → .litertlm")
    p.add_argument("--artifact-dir", required=True, help="Directory containing exported files")
    p.add_argument(
        "--output",
        default="model.litertlm",
        help="Output .litertlm path (default: ./model.litertlm)",
    )
    p.add_argument(
        "--prefill-decode",
        default="model_quantized.tflite",
        help="Relative name of prefill+decode flatbuffer (often model_quantized.tflite after quant)",
    )
    p.add_argument(
        "--embedder",
        default="embedder.tflite",
        help="Relative name of embedder flatbuffer",
    )
    p.add_argument(
        "--per-layer-embedder",
        default="per_layer_embedder.tflite",
        help="Relative name of per-layer embedder (if present)",
    )
    p.add_argument(
        "--tokenizer",
        default="tokenizer.json",
        help="Relative tokenizer path (.json for HF, .model for SentencePiece)",
    )
    p.add_argument(
        "--llm-metadata",
        default="llm_metadata.pb",
        help="Relative path to protobuf LlmMetadata",
    )
    args = p.parse_args(argv)

    root = Path(args.artifact_dir).resolve()

    def rp(name: str) -> str:
        return str(root / name)

    builder = litertlm_builder.LitertLmFileBuilder()
    builder.add_system_metadata(
        litertlm_builder.Metadata(
            key="Authors",
            value="safetensors-to-litertlm",
            dtype=litertlm_builder.DType.STRING,
        )
    )
    meta_path = rp(args.llm_metadata)
    if not os.path.isfile(meta_path):
        print(f"Missing {meta_path}", file=sys.stderr)
        raise SystemExit(1)
    builder.add_llm_metadata(meta_path)

    tok_path = rp(args.tokenizer)
    if not os.path.isfile(tok_path):
        print(f"Missing {tok_path}", file=sys.stderr)
        raise SystemExit(1)
    if tok_path.endswith(".json"):
        builder.add_hf_tokenizer(tok_path)
    else:
        builder.add_sentencepiece_tokenizer(tok_path)

    pd = rp(args.prefill_decode)
    if not os.path.isfile(pd):
        for fallback in ("model_quantized.tflite", "model.tflite"):
            cand = str(root / fallback)
            if os.path.isfile(cand):
                pd = cand
                break
        else:
            print(f"Missing prefill/decode TFLite: {pd}", file=sys.stderr)
            raise SystemExit(1)
    builder.add_tflite_model(pd, litertlm_builder.TfLiteModelType.PREFILL_DECODE)

    emb = rp(args.embedder)
    if os.path.isfile(emb):
        builder.add_tflite_model(emb, litertlm_builder.TfLiteModelType.EMBEDDER)
    else:
        print(f"Embedder TFLite not found, skipping: {emb}")

    ple = rp(args.per_layer_embedder)
    if os.path.isfile(ple):
        builder.add_tflite_model(ple, litertlm_builder.TfLiteModelType.PER_LAYER_EMBEDDER)
    else:
        print(f"Per-layer embedder TFLite not found, skipping: {ple}")

    out = Path(args.output).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "wb") as f:
        builder.build(f)
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
