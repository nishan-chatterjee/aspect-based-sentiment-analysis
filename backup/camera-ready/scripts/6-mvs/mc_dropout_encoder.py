#!/usr/bin/env python3
"""Run MC dropout for a minimum-viable-set checkpoint directory."""

from __future__ import annotations

import argparse
import gc
import importlib.util
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import torch
from transformers import logging as hf_logging


ROOT_DIR = Path(os.environ.get("ABSA_RELEASE_ROOT", Path(__file__).resolve().parents[2])).resolve()
THIS_DIR = Path(__file__).resolve().parent


def load_uncertainty_module() -> Any:
    module_path = ROOT_DIR / "scripts" / "3-uncertainty" / "predict_uncertainty_experts.py"
    spec = importlib.util.spec_from_file_location("predict_uncertainty_experts", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not import {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules["predict_uncertainty_experts"] = module
    spec.loader.exec_module(module)
    return module


U = load_uncertainty_module()


DEFAULTS = {
    "longformer": {"default_batch_size": 8, "default_max_len": 4096},
    "mdeberta": {"default_batch_size": 64, "default_max_len": 512},
    "slavic_specific": {"default_batch_size": 64, "default_max_len": 512},
}

SUBSET_DATA_SPECS = {
    "sr": {
        "train_val": "additional-tasks/data/sr_train_val_{run}.json",
        "test": "additional-tasks/data/sr_test.json",
    },
    "sh": {
        "train_val": "additional-tasks/data/sh_train_val_{run}.json",
        "test": "additional-tasks/data/sh_test.json",
    },
    "hr": {
        "train_val": "additional-tasks/data/hr_train_val_{run}.json",
        "test": "additional-tasks/data/hr_test.json",
    },
    "bs": {
        "train_val": "additional-tasks/data/bs_train_val_{run}.json",
        "test": "additional-tasks/data/bs_test.json",
    },
    "sr_latin": {
        "train_val": "additional-tasks/data/sr_latin_train_val_{run}.json",
        "test": "additional-tasks/data/sr_latin_test.json",
    },
    "sr_cyrillic": {
        "train_val": "additional-tasks/data/sr_cyrillic_train_val_{run}.json",
        "test": "additional-tasks/data/sr_cyrillic_test.json",
    },
}
SUPPORTED_LANGUAGES = ("slovenian", "serbian", *SUBSET_DATA_SPECS)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--approach",
        default="slavic_specific",
        choices=["longformer", "mdeberta", "slavic_specific"],
    )
    parser.add_argument("--language", required=True, choices=SUPPORTED_LANGUAGES)
    parser.add_argument("--percentage_tag", required=True)
    parser.add_argument("--checkpoint_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--data_dir", default=str(ROOT_DIR / "data"))
    parser.add_argument("--model_root", default=str(ROOT_DIR / "models"))
    parser.add_argument("--splits", nargs="+", default=["test"], choices=U.SPLIT_NAMES)
    parser.add_argument("--num_mc_samples", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_len", type=int, default=None)
    parser.add_argument("--limit_items", type=int, default=None)
    parser.add_argument("--checkpoint_limit", type=int, default=None)
    parser.add_argument("--progress_every", type=int, default=25)
    parser.add_argument("--skip_completed", action="store_true")
    parser.add_argument("--no_amp", action="store_true")
    parser.add_argument("--local_files_only", action="store_true", default=True)
    return parser.parse_args()


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(U.to_jsonable(data), f, indent=2, ensure_ascii=False)


def collect_checkpoints(checkpoint_dir: Path, limit: int | None) -> list[Path]:
    checkpoints = sorted(checkpoint_dir.glob("best_model_*.pt"))
    if limit is not None:
        checkpoints = checkpoints[:limit]
    if not checkpoints:
        raise FileNotFoundError(f"No best_model_*.pt checkpoints found in {checkpoint_dir}")
    return checkpoints


def expert_name(approach: str) -> str:
    return f"{approach}_masked"


def configure_subset_support(args: argparse.Namespace) -> None:
    """Teach the comparison uncertainty helpers about the locally stored subset data."""
    if args.language not in SUBSET_DATA_SPECS:
        return
    original_input_path = U.input_path_for_split
    original_model_path = U.encoder_model_path

    def subset_input_path(_data_dir: Path, language: str, split_name: str) -> Path:
        if language not in SUBSET_DATA_SPECS:
            return original_input_path(_data_dir, language, split_name)
        spec = SUBSET_DATA_SPECS[language]
        if split_name == "test":
            return ROOT_DIR / spec["test"]
        run_index = split_name.rsplit("_", 1)[-1]
        return ROOT_DIR / spec["train_val"].format(run=run_index)

    def subset_model_path(comparison_module: Any, model_root: Path, approach: str, language: str) -> Path:
        if approach == "slavic_specific" and language in SUBSET_DATA_SPECS:
            return model_root / "classla_bcms-bertic"
        if approach == "slavic_specific" and language != "slovenian":
            return model_root / "classla_bcms-bertic"
        return original_model_path(comparison_module, model_root, approach, language)

    U.input_path_for_split = subset_input_path
    U.encoder_model_path = subset_model_path


def main() -> None:
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    hf_logging.set_verbosity_error()
    args = parse_args()
    configure_subset_support(args)
    args.expert = expert_name(args.approach)
    U.set_seed(args.seed)

    checkpoint_dir = Path(args.checkpoint_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoints = collect_checkpoints(checkpoint_dir, args.checkpoint_limit)
    defaults = DEFAULTS[args.approach]
    spec = {
        "backend": "encoder",
        "approach": args.approach,
        "json_key": f"{args.approach}/masked",
        "default_batch_size": defaults["default_batch_size"],
        "default_max_len": defaults["default_max_len"],
    }
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("--- Minimum Viable Set MC Dropout ---", flush=True)
    print(f"Expert: {args.expert}", flush=True)
    print(f"Language: {args.language}", flush=True)
    print(f"Percentage tag: {args.percentage_tag}", flush=True)
    print(f"Checkpoint dir: {checkpoint_dir}", flush=True)
    print(f"Output dir: {output_dir}", flush=True)
    print(f"Checkpoints: {[str(p) for p in checkpoints]}", flush=True)
    print(f"MC samples per checkpoint: {args.num_mc_samples}", flush=True)
    if torch.cuda.is_available():
        print(f"CUDA device: {torch.cuda.get_device_name(0)}", flush=True)
    started_at = time.time()

    states, active_records = U.prepare_split_states(args, spec, checkpoints, output_dir)
    if active_records:
        all_results = U.run_encoder_uncertainty_multi(args, spec, active_records, checkpoints, device)
    else:
        all_results = {}
    split_metrics = U.finalize_split_states(args, spec, checkpoints, states, all_results)

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    success = {
        "expert": args.expert,
        "approach": args.approach,
        "language": args.language,
        "percentage_tag": args.percentage_tag,
        "backend": spec["backend"],
        "json_key": spec["json_key"],
        "probabilities_key": f"{spec['json_key']}/probabilities",
        "uncertainty_key": f"{spec['json_key']}/uncertainty",
        "checkpoint_dir": str(checkpoint_dir),
        "checkpoints": [str(path) for path in checkpoints],
        "num_checkpoints": len(checkpoints),
        "num_mc_samples_per_checkpoint": args.num_mc_samples,
        "total_stochastic_samples": len(checkpoints) * args.num_mc_samples,
        "splits": split_metrics,
        "elapsed_seconds": time.time() - started_at,
    }
    write_json(output_dir / "_SUCCESS.json", success)
    print(f"Wrote success marker: {output_dir / '_SUCCESS.json'}", flush=True)


if __name__ == "__main__":
    main()
