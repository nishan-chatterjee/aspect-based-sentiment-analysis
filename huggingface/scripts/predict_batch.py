#!/usr/bin/env python3
"""Run batched AspectBench predictions from JSON or JSONL."""

from __future__ import annotations

import argparse
from pathlib import Path

from inference import InferenceEngine, load_records, write_json
from model_registry import LANGUAGES, MODES, MODEL_SPECS


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_MODEL_ROOT = SCRIPT_DIR.parent / "models"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Predict a JSON/JSONL list whose articles contain <aspect>...</aspect>."
    )
    parser.add_argument("--model-name", required=True, choices=sorted(MODEL_SPECS))
    parser.add_argument("--language", required=True, choices=LANGUAGES)
    parser.add_argument("--mode", required=True, choices=MODES)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--model-root", type=Path, default=DEFAULT_MODEL_ROOT)
    parser.add_argument("--base-model-root", type=Path)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--mc-passes", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    engine = InferenceEngine(
        model_name=args.model_name,
        language=args.language,
        mode=args.mode,
        model_root=args.model_root,
        base_model_root=args.base_model_root,
        device=args.device,
    )
    predictions = engine.predict_batch(
        load_records(args.input),
        batch_size=args.batch_size,
        mc_passes=args.mc_passes,
        seed=args.seed,
    )
    write_json(predictions, args.output)
    if args.output is not None:
        print(f"Wrote {len(predictions)} predictions to {args.output.resolve()}")


if __name__ == "__main__":
    main()
