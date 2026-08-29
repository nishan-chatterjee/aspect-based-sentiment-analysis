#!/usr/bin/env python3
"""Bundle the small tokenizer/config assets needed for offline reconstruction."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

from transformers import AutoConfig

from model_registry import CHECKPOINTS, LANGUAGES, MODES, MODEL_SPECS, model_spec


SCRIPT_DIR = Path(__file__).resolve().parent
TOKENIZER_CANDIDATES = (
    "sentencepiece.bpe.model",
    "spm.model",
    "spiece.model",
    "vocab.txt",
    "tokenizer.json",
)
TOKENIZER_METADATA = (
    "tokenizer_config.json",
    "special_tokens_map.json",
    "added_tokens.json",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-model-root", type=Path, required=True)
    parser.add_argument(
        "--model-root", type=Path, default=SCRIPT_DIR.parent / "models"
    )
    parser.add_argument("--model", action="append", choices=sorted(MODEL_SPECS))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    base_root = args.base_model_root.resolve()
    model_root = args.model_root.resolve()
    selected = set(args.model or MODEL_SPECS)

    for model_name in MODEL_SPECS:
        if model_name not in selected:
            continue
        if MODEL_SPECS[model_name]["backend"] == "bge":
            print(f"SKIP {model_name}: no releasable trained MLP head", flush=True)
            continue
        for language in LANGUAGES:
            if not any(
                CHECKPOINTS[(model_name, language, mode)]["available"] for mode in MODES
            ):
                continue
            spec = model_spec(model_name, language)
            source = base_root / spec["local_base_dir"]
            if not (source / "config.json").is_file():
                raise FileNotFoundError(
                    f"Missing local base-model config for {model_name}/{language}: {source}"
                )
            destination = model_root / model_name / language / "base_model"
            destination.mkdir(parents=True, exist_ok=True)
            config = AutoConfig.from_pretrained(source, local_files_only=True)
            config.save_pretrained(destination)
            # Prefer the compact native tokenizer model. Saving a fast tokenizer
            # with recent Transformers can expand it into a ~17 MB, million-line
            # JSON file; the native assets are both smaller and fully sufficient.
            tokenizer_asset = (
                "tokenizer.json"
                if model_name == "slavic-specific" and language == "slovenian"
                else next(
                    (name for name in TOKENIZER_CANDIDATES if (source / name).is_file()),
                    None,
                )
            )
            if tokenizer_asset is None:
                raise FileNotFoundError(f"No tokenizer asset found in {source}")
            keep = {tokenizer_asset, "config.json", "source.json"}
            for name in TOKENIZER_METADATA:
                if (source / name).is_file():
                    shutil.copy2(source / name, destination / name)
                    keep.add(name)
            shutil.copy2(source / tokenizer_asset, destination / tokenizer_asset)
            for candidate in (*TOKENIZER_CANDIDATES, *TOKENIZER_METADATA):
                stale = destination / candidate
                if stale.is_file() and candidate not in keep:
                    stale.unlink()
            metadata = {
                "model": model_name,
                "language": language,
                "upstream_base_model": spec["base_model"],
                "purpose": "Tokenizer and architecture configuration for offline checkpoint reconstruction",
                "contains_upstream_model_weights": False,
            }
            (destination / "source.json").write_text(
                json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
            )
            print(f"BUNDLED {model_name}/{language}: {destination}", flush=True)


if __name__ == "__main__":
    main()
