"""Inspect ``.litertlm`` bundles, ``.tflite`` files, or HF checkpoint folders."""

from __future__ import annotations

import argparse
import contextlib
import json
import mmap
import os
import struct
from pathlib import Path

_LITERTLM_MAGIC = b"LITERTLM"
_HEADER_END_LOCATION_BYTE_OFFSET = 24
_HEADER_BEGIN_BYTE_OFFSET = 32
_TFLITE_SECTION_DATA_TYPE = 3


def _inspect_tflite_content(model_content: bytes, title: str) -> None:
    import tensorflow as tf

    interpreter = tf.lite.Interpreter(model_content=model_content)
    interpreter.allocate_tensors()

    print("\n" + "=" * 40)
    print(title)
    print("=" * 40)
    print("MODEL INPUTS")
    print("-" * 40)
    for i, detail in enumerate(interpreter.get_input_details()):
        print(f"[{i}] Name:  {detail.get('name', 'N/A')}")
        print(f"    Shape: {detail.get('shape', 'N/A')}")
        print(f"    Type:  {detail.get('dtype', 'N/A')}")

    print("\nMODEL OUTPUTS")
    print("-" * 40)
    for i, detail in enumerate(interpreter.get_output_details()):
        print(f"[{i}] Name:  {detail.get('name', 'N/A')}")
        print(f"    Shape: {detail.get('shape', 'N/A')}")
        print(f"    Type:  {detail.get('dtype', 'N/A')}")

    print("\nINTERNAL TENSORS & ARCHITECTURE (First 20)")
    print("-" * 40)
    tensor_details = interpreter.get_tensor_details()
    shown = min(20, len(tensor_details))
    for i, tensor in enumerate(tensor_details[:shown]):
        print(f"Tensor {i}: {tensor['name']}")
        print(f"  Shape: {tensor['shape']} | Type: {tensor['dtype']}")
    if len(tensor_details) > 20:
        print(f"\n... and {len(tensor_details) - 20} more tensors hidden.")
    print(f"Total Tensors: {len(tensor_details)}")


def _inspect_litertlm_bundle(model_path: str) -> None:
    try:
        from ai_edge_litert.internal import litertlm_core
        from ai_edge_litert.internal import litertlm_header_schema_py_generated as schema
    except ImportError:
        print(
            "Failed to parse .litertlm container: ai_edge_litert is missing.\n"
            "Install dependencies with: uv sync --extra export"
        )
        return

    with open(model_path, "rb") as fh:
        mm = mmap.mmap(fh.fileno(), 0, access=mmap.ACCESS_READ)
        try:
            header_end = struct.unpack_from(
                "<Q", mm, _HEADER_END_LOCATION_BYTE_OFFSET
            )[0]
            header = mm[_HEADER_BEGIN_BYTE_OFFSET:header_end]
            root = schema.LiteRTLMMetaData.GetRootAs(header, 0)
            section_meta = root.SectionMetadata()
            section_count = section_meta.ObjectsLength() if section_meta is not None else 0

            version = struct.unpack_from("<III", mm, 8)
            print(
                "LiteRT-LM bundle detected "
                f"(version {version[0]}.{version[1]}.{version[2]}, sections: {section_count})"
            )

            for idx in range(section_count):
                section = section_meta.Objects(idx)
                begin = int(section.BeginOffset())
                end = int(section.EndOffset())
                data_type = int(section.DataType())
                section_size = end - begin
                try:
                    section_name = litertlm_core.any_section_data_type_to_string(data_type)
                except ValueError:
                    section_name = f"Unknown({data_type})"
                print(
                    f"\nSection[{idx}] type={section_name} offset=[{begin},{end}) size={section_size} bytes"
                )

                if data_type != _TFLITE_SECTION_DATA_TYPE:
                    continue

                if begin < 0 or end > mm.size() or begin >= end:
                    print("Skipping invalid section range.")
                    continue

                model_content = bytes(mm[begin:end])
                try:
                    _inspect_tflite_content(
                        model_content=model_content,
                        title=f"TFLITE SECTION {idx}",
                    )
                except Exception as exc:
                    print(f"Failed to parse embedded TFLite section {idx}: {exc}")
        finally:
            mm.close()


def _inspect_hf_checkpoint_dir(model_dir: str) -> None:
    root = Path(model_dir)
    config_path = root / "config.json"
    safetensors_paths = sorted(root.glob("*.safetensors"))

    if not config_path.is_file() and not safetensors_paths:
        print(
            "Directory does not look like a supported checkpoint.\n"
            "Expected config.json and/or *.safetensors."
        )
        return

    print("\n" + "=" * 40)
    print("HUGGING FACE CHECKPOINT")
    print("=" * 40)
    print(f"Path: {root}")

    if config_path.is_file():
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                cfg = json.load(f)
            model_type = cfg.get("model_type", "N/A")
            arch = cfg.get("architectures", [])
            hidden_size = cfg.get("hidden_size", "N/A")
            num_layers = cfg.get("num_hidden_layers", "N/A")
            vocab_size = cfg.get("vocab_size", "N/A")
            print(f"Model type: {model_type}")
            print(f"Architectures: {arch if arch else 'N/A'}")
            print(f"Hidden size: {hidden_size}")
            print(f"Num layers: {num_layers}")
            print(f"Vocab size: {vocab_size}")
        except Exception as exc:
            print(f"Failed to parse config.json: {exc}")

    if not safetensors_paths:
        print("No .safetensors weight file found.")
        return

    total_bytes = sum(p.stat().st_size for p in safetensors_paths)
    print(f"\nSafetensors shards: {len(safetensors_paths)}")
    print(f"Total weights size: {total_bytes / (1024 * 1024 * 1024):.2f} GiB")

    try:
        from safetensors import safe_open
    except ImportError:
        print(
            "Install `safetensors` to inspect tensor-level metadata "
            "(e.g. with `uv sync --extra export`)."
        )
        return

    shown = 0
    total_tensors = 0
    print("\nFirst tensors:")
    for shard in safetensors_paths:
        with safe_open(str(shard), framework="pt", device="cpu") as f:
            keys = list(f.keys())
            total_tensors += len(keys)
            for key in keys:
                if shown >= 20:
                    break
                tensor = f.get_tensor(key)
                print(f"- {key}: shape={tuple(tensor.shape)} dtype={tensor.dtype}")
                shown += 1
        if shown >= 20:
            break

    if total_tensors > shown:
        print(f"... and {total_tensors - shown} more tensors hidden.")
    print(f"Total tensors: {total_tensors}")


def inspect_litert_model(model_path: str) -> None:
    if not os.path.exists(model_path):
        print(f"Error: File '{model_path}' not found.")
        return

    if os.path.isdir(model_path):
        _inspect_hf_checkpoint_dir(model_path)
        return

    print(
        f"Loading model: {model_path} (Size: {os.path.getsize(model_path) / (1024 * 1024):.2f} MB)"
    )

    with open(model_path, "rb") as f:
        magic = f.read(8)

    if magic == _LITERTLM_MAGIC:
        _inspect_litertlm_bundle(model_path)
        return

    try:
        with open(model_path, "rb") as f:
            _inspect_tflite_content(model_content=f.read(), title="TFLITE MODEL")
    except Exception as e:
        print(f"Failed to parse model metadata: {e}")


def _resolve_report_path(model_path: Path) -> Path:
    try:
        relative_path = model_path.resolve().relative_to(Path.cwd())
    except ValueError:
        relative_path = Path(*model_path.parts[1:]) if model_path.is_absolute() else model_path

    is_directory = model_path.is_dir() or not model_path.suffix
    report_root = relative_path if is_directory else relative_path.parent
    report_name = f"{model_path.name if is_directory else model_path.stem}-inspection.md"
    return Path.cwd() / "inspections" / report_root / report_name


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(
        description="Inspect a LiteRT bundle, TFLite flatbuffer, or HF checkpoint folder and save to Markdown."
    )
    p.add_argument(
        "model_path",
        help="Path to .litertlm, .tflite, or HF checkpoint directory",
    )
    args = p.parse_args(argv)
    model_path = Path(args.model_path)
    if not model_path.exists():
        print(f"Error: Path '{model_path}' not found.")
        return

    out_file = _resolve_report_path(model_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)

    print(f"[*] Inspecting model: {model_path}")
    print(f"[*] Generating report: {out_file}")

    with out_file.open("w", encoding="utf-8") as f:
        f.write(f"# Inspection Report: `{model_path.stem if model_path.is_file() else model_path.name}`\n\n")
        f.write(f"**Source:** `{model_path}`\n\n")
        f.write("```text\n")
        with contextlib.redirect_stdout(f):
            inspect_litert_model(str(model_path))
        f.write("```\n")

    print("[*] Done!")


if __name__ == "__main__":
    main()
