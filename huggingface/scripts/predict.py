#!/usr/bin/env python3
"""Run one AspectBench prediction and print a JSON object."""

from __future__ import annotations

import argparse
from pathlib import Path

from inference import InferenceEngine, write_json
from model_registry import LANGUAGES, MODES, MODEL_SPECS


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_MODEL_ROOT = SCRIPT_DIR.parent / "models"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Predict sentiment for one article containing <aspect>...</aspect>."
    )
    parser.add_argument("--model-name", required=True, choices=sorted(MODEL_SPECS))
    parser.add_argument("--language", required=True, choices=LANGUAGES)
    parser.add_argument("--mode", required=True, choices=MODES)
    parser.add_argument("--article", required=True)
    parser.add_argument(
        "--aspect",
        help="Optional explicit target; otherwise the first tagged aspect is used.",
    )
    parser.add_argument("--sentiment", type=int, choices=(-1, 0, 1))
    parser.add_argument("--model-root", type=Path, default=DEFAULT_MODEL_ROOT)
    parser.add_argument(
        "--base-model-root",
        type=Path,
        help=(
            "Optional legacy base-model cache. Bundled model assets are preferred "
            "automatically when present."
        ),
    )
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, or cuda:N")
    parser.add_argument(
        "--mc-passes",
        type=int,
        default=0,
        help="0 disables MC dropout; otherwise use at least 2 passes.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    record = {"article": args.article}
    if args.aspect is not None:
        record["aspect"] = args.aspect
    if args.sentiment is not None:
        record["sentiment"] = args.sentiment
    engine = InferenceEngine(
        model_name=args.model_name,
        language=args.language,
        mode=args.mode,
        model_root=args.model_root,
        base_model_root=args.base_model_root,
        device=args.device,
    )
    write_json(
        engine.predict(record, mc_passes=args.mc_passes, seed=args.seed),
        args.output,
    )


if __name__ == "__main__":
    main()
