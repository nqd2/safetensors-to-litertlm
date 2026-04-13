"""Diff helper for tokenizer template and inspection topology."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


def _read_chat_template(tokenizer_json_path: str) -> str:
    data = json.loads(Path(tokenizer_json_path).read_text(encoding="utf-8"))
    template = data.get("chat_template", "")
    return template if isinstance(template, str) else ""


def _inspect_facts(inspection_path: str) -> dict[str, object]:
    text = Path(inspection_path).read_text(encoding="utf-8")
    section_count_match = re.search(r"sections:\s*(\d+)\)", text)
    sections = int(section_count_match.group(1)) if section_count_match else -1
    has_per_layer_input = "per_layer_embeddings" in text or "per_layer_embedder" in text
    has_int8 = "numpy.int8" in text
    return {
        "sections": sections,
        "has_per_layer_input": has_per_layer_input,
        "has_int8": has_int8,
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Compare tokenizer template and LiteRT inspection topology."
    )
    parser.add_argument("--source-tokenizer", required=True, help="Path to source tokenizer.json")
    parser.add_argument("--source-inspection", required=True, help="Inspection report path")
    parser.add_argument("--target-inspection", required=True, help="Inspection report path")
    args = parser.parse_args(argv)

    src_template = _read_chat_template(args.source_tokenizer)
    src = _inspect_facts(args.source_inspection)
    tgt = _inspect_facts(args.target_inspection)

    result = {
        "chat_template_present": bool(src_template.strip()),
        "source": src,
        "target": tgt,
        "diff": {
            "sections_delta": int(tgt["sections"]) - int(src["sections"]),
            "per_layer_dependency_changed": bool(src["has_per_layer_input"])
            != bool(tgt["has_per_layer_input"]),
            "int8_presence_changed": bool(src["has_int8"]) != bool(tgt["has_int8"]),
        },
    }
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
