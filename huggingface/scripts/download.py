#!/usr/bin/env python3
"""Download private AspectBench repositories into the canonical local layout."""

from __future__ import annotations

import argparse
from pathlib import Path

from huggingface_hub import snapshot_download

from model_registry import CHECKPOINTS, LANGUAGES, MODES, MODEL_SPECS


SCRIPT_DIR = Path(__file__).resolve().parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, default=SCRIPT_DIR.parent / "models")
    parser.add_argument("--model", action="append", choices=sorted(MODEL_SPECS))
    parser.add_argument("--revision", default="main")
    parser.add_argument("--token", help="Normally omitted; the saved HF token is used.")
    parser.add_argument("--force-download", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    downloadable = [
        model_name
        for model_name in MODEL_SPECS
        if any(
            CHECKPOINTS[(model_name, language, mode)]["available"]
            for language in LANGUAGES
            for mode in MODES
        )
    ]
    selected = args.model or downloadable
    args.output_root.mkdir(parents=True, exist_ok=True)
    for model_name in selected:
        if model_name not in downloadable:
            raise SystemExit(
                f"{model_name} has no saved checkpoint weights and therefore has no "
                "downloadable model repository. See models/manifest.json."
            )
        spec = MODEL_SPECS[model_name]
        destination = args.output_root / model_name
        print(f"Downloading {spec['hf_repo']} -> {destination}", flush=True)
        snapshot_download(
            repo_id=spec["hf_repo"],
            repo_type="model",
            revision=args.revision,
            token=args.token,
            local_dir=destination,
            force_download=args.force_download,
        )


if __name__ == "__main__":
    main()
