#!/usr/bin/env python3
"""Train masked encoder experts on a percentage of train/validation data.

This is an isolated minimum-viable-annotation experiment. It reuses the
comparison encoder baseline implementation for data preparation, model loading,
metrics, and prediction JSON shape, but adds deterministic stratified
subsampling and validation-patience early stopping.
"""

from __future__ import annotations

import argparse
import copy
import datetime as dt
import gc
import importlib.util
import json
import math
import os
import random
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
from tqdm.auto import tqdm
from transformers import logging as hf_logging


ROOT_DIR = Path(os.environ.get("ABSA_RELEASE_ROOT", Path(__file__).resolve().parents[2])).resolve()
THIS_DIR = Path(__file__).resolve().parent
LABELS = (-1, 0, 1)
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


def load_comparison_module() -> Any:
    module_path = ROOT_DIR / "scripts" / "2-experts" / "8.1 additional_encoder_baselines.py"
    spec = importlib.util.spec_from_file_location("additional_encoder_baselines", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not import {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules["additional_encoder_baselines"] = module
    spec.loader.exec_module(module)
    return module


R = load_comparison_module()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Progressive-data masked encoder training for minimum viable set estimates."
    )
    parser.add_argument(
        "--approach",
        default="slavic_specific",
        choices=["longformer", "mdeberta", "slavic_specific"],
    )
    parser.add_argument(
        "--split",
        required=True,
        choices=["slovenian", "serbian", "sr", "sh", "hr", "bs", "sr_latin", "sr_cyrillic"],
    )
    parser.add_argument("--percent", type=float, default=100.0)
    parser.add_argument("--output_tag", default=None)
    parser.add_argument(
        "--controlled_train_val_template",
        default=None,
        help="Explicit train/val JSON template with {run_index}; bypasses percentage subsampling.",
    )
    parser.add_argument("--test_path", default=None)
    parser.add_argument("--data_dir", default=str(ROOT_DIR / "data"))
    parser.add_argument("--model_root", default=str(ROOT_DIR / "models"))
    parser.add_argument("--model_path", default=None)
    parser.add_argument("--output_root", default=str(THIS_DIR / "reviews"))
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--min_epochs", type=int, default=4)
    parser.add_argument("--early_stop_patience", type=int, default=3)
    parser.add_argument(
        "--early_stop_metric",
        default="f1_macro",
        choices=["f1_macro", "qwk", "accuracy", "loss"],
        help="Validation metric used for checkpointing and early stopping.",
    )
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--grad_accum_steps", type=int, default=2)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--max_len", type=int, default=None)
    parser.add_argument("--dropout_rate", type=float, default=None)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--subset_seed", type=int, default=None)
    parser.add_argument("--run_indices", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--mask_aspect", action="store_true", default=True)
    parser.add_argument("--no_mask_aspect", action="store_false", dest="mask_aspect")
    parser.add_argument("--no_amp", action="store_true")
    parser.add_argument("--gradient_checkpointing", action="store_true")
    parser.add_argument("--skip_completed", action="store_true")
    parser.add_argument("--test_only", action="store_true")
    parser.add_argument("--subset_min_per_class", type=int, default=1)
    parser.add_argument("--limit_train", type=int, default=None)
    parser.add_argument("--limit_val", type=int, default=None)
    parser.add_argument(
        "--max_initial_nonfinite_batches",
        type=int,
        default=10,
    )
    parser.add_argument("--local_files_only", action="store_true", default=True)
    return parser.parse_args()


def percent_tag(percent: float) -> str:
    if percent <= 0 or percent > 100:
        raise ValueError("--percent must be in (0, 100].")
    if float(percent).is_integer():
        return "pct_%03d" % int(percent)
    text = ("%.4f" % percent).rstrip("0").rstrip(".").replace(".", "p")
    return f"pct_{text}"


def output_tag(args: argparse.Namespace) -> str:
    return args.output_tag or percent_tag(args.percent)


def to_jsonable(value: Any) -> Any:
    return R.to_jsonable(value)


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(to_jsonable(data), f, indent=4, ensure_ascii=False)


def metric_improved(metric: float, best: float | None, metric_name: str) -> bool:
    if not np.isfinite(metric):
        return False
    if best is None:
        return True
    if metric_name == "loss":
        return metric < best
    return metric > best


def completion_marker(output_dir: Path) -> Path:
    return output_dir / "_SUCCESS.json"


def is_complete(output_dir: Path, run_indices: list[int], percent: float) -> bool:
    marker = completion_marker(output_dir)
    if not marker.exists():
        return False
    try:
        data = json.loads(marker.read_text(encoding="utf-8"))
    except Exception:
        return False
    if abs(float(data.get("percent", -1)) - float(percent)) > 1e-9:
        return False
    for run_index in run_indices:
        if not (output_dir / f"best_model_{run_index}.pt").exists():
            return False
        if not (output_dir / f"training_metrics_{run_index}.json").exists():
            return False
        if not (output_dir / f"test_predictions_{run_index}.json").exists():
            return False
    return (output_dir / "test_metrics_summary.json").exists()


def default_model_path(args: argparse.Namespace) -> Path:
    if args.model_path:
        return Path(args.model_path)
    if args.approach == "slavic_specific" and args.split != "slovenian":
        return Path(args.model_root) / "classla_bcms-bertic"
    return R.default_model_path(args)


def load_controlled_train_val(template: str, run_index: int) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    path = Path(template.format(run_index=run_index, run=run_index))
    data = json.loads(path.read_text(encoding="utf-8"))
    if "train_indices" in data and "val_indices" in data:
        source_path = Path(data["source_path"])
        source_data = json.loads(source_path.read_text(encoding="utf-8"))
        source_items = list(source_data.get("train", [])) + list(source_data.get("val", []))
        train = [source_items[int(idx)] for idx in data.get("train_indices", [])]
        val = [source_items[int(idx)] for idx in data.get("val_indices", [])]
    else:
        train = data.get("train", [])
        val = data.get("val", [])
    if not train or not val:
        raise ValueError(f"Controlled subset missing train/val data: {path}")
    return train, val, {"path": str(path), "metadata": data.get("metadata", {})}


def load_test_data(args: argparse.Namespace) -> list[dict[str, Any]]:
    if args.test_path:
        path = Path(args.test_path)
        data = json.loads(path.read_text(encoding="utf-8"))
        test = data.get("test", [])
        if not test:
            raise ValueError(f"Controlled test path missing test data: {path}")
        print("Loading controlled test data from: %s" % path, flush=True)
        return test
    if args.split not in SUBSET_DATA_SPECS:
        return R.load_split_data(args.data_dir, args.split, split_index=None)
    path = ROOT_DIR / SUBSET_DATA_SPECS[args.split]["test"]
    data = json.loads(path.read_text(encoding="utf-8"))
    test = data.get("test", [])
    if not test:
        raise ValueError(f"Subset test path missing test data: {path}")
    print("Loading subset test data from: %s" % path, flush=True)
    return test


def load_train_val_data(args: argparse.Namespace, run_index: int) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Load either the original corpus splits or the language/script subset splits."""
    if args.split not in SUBSET_DATA_SPECS:
        return R.load_split_data(args.data_dir, args.split, run_index)
    path = ROOT_DIR / SUBSET_DATA_SPECS[args.split]["train_val"].format(run=run_index)
    data = json.loads(path.read_text(encoding="utf-8"))
    train = data.get("train", [])
    val = data.get("val", [])
    if not train or not val:
        raise ValueError(f"Subset train/val path missing train or val data: {path}")
    print("Loading subset train/val data from: %s" % path, flush=True)
    return train, val


def stratified_subset(
    records: list[dict[str, Any]],
    percent: float,
    seed: int,
    min_per_class: int,
    limit: int | None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if percent >= 100.0 and limit is None:
        counts = Counter(item.get("sentiment") for item in records)
        return list(records), {
            "requested_percent": percent,
            "source_count": len(records),
            "selected_count": len(records),
            "label_counts": dict(counts),
            "selection": "full",
        }

    rng = random.Random(seed)
    grouped: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for item in records:
        grouped[int(item.get("sentiment", 0))].append(item)
    for label_records in grouped.values():
        rng.shuffle(label_records)

    target_total = max(1, int(round(len(records) * percent / 100.0)))
    if limit is not None:
        target_total = min(target_total, limit)

    selected: list[dict[str, Any]] = []
    selected_ids: set[int] = set()

    for label in LABELS:
        label_records = grouped.get(label, [])
        take = min(min_per_class, len(label_records))
        for item in label_records[:take]:
            selected.append(item)
            selected_ids.add(id(item))

    proportional_counts: dict[int, int] = {}
    for label, label_records in grouped.items():
        proportional_counts[label] = int(math.floor(len(label_records) * percent / 100.0))

    for label in LABELS:
        label_records = grouped.get(label, [])
        current = sum(1 for item in selected if item.get("sentiment") == label)
        need = max(0, proportional_counts.get(label, 0) - current)
        for item in label_records:
            if need <= 0:
                break
            if id(item) in selected_ids:
                continue
            selected.append(item)
            selected_ids.add(id(item))
            need -= 1

    remaining = [item for label in LABELS for item in grouped.get(label, []) if id(item) not in selected_ids]
    rng.shuffle(remaining)
    for item in remaining:
        if len(selected) >= target_total:
            break
        selected.append(item)
        selected_ids.add(id(item))

    if limit is not None and len(selected) > limit:
        selected = selected[:limit]

    rng.shuffle(selected)
    counts = Counter(item.get("sentiment") for item in selected)
    return selected, {
        "requested_percent": percent,
        "source_count": len(records),
        "target_count": target_total,
        "selected_count": len(selected),
        "label_counts": dict(counts),
        "selection": "stratified",
        "min_per_class": min_per_class,
        "seed": seed,
    }


def move_batch_to_device(batch: dict[str, torch.Tensor], device: torch.device) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
    return R.move_batch_to_device(batch, device)


def train_one_run_dynamic(
    model: torch.nn.Module,
    train_loader: torch.utils.data.DataLoader,
    val_loader: torch.utils.data.DataLoader,
    args: argparse.Namespace,
    device: torch.device,
    run_index: int,
    checkpoint_path: Path,
    run_args: dict[str, Any],
) -> dict[str, Any]:
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min" if args.early_stop_metric == "loss" else "max",
        factor=0.2,
        patience=2,
    )
    use_amp = torch.cuda.is_available() and not args.no_amp
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    best_metric: float | None = None
    best_epoch: int | None = None
    bad_epochs = 0
    run_metrics: dict[str, Any] = {"train": [], "eval": []}

    optimizer.zero_grad(set_to_none=True)
    print("\n--- Starting progressive training for run %s ---" % run_index, flush=True)
    print("Best checkpoint path: %s" % checkpoint_path, flush=True)

    for epoch in range(args.epochs):
        epoch_start = time.time()
        model.train()
        total_loss = 0.0
        total_items = 0
        nan_batches = 0
        grad_norm_sum = 0.0
        grad_updates = 0
        progress = tqdm(
            train_loader,
            desc="Run %s epoch %s/%s" % (run_index, epoch + 1, args.epochs),
            leave=False,
            disable=R.env_flag("TQDM_DISABLE"),
        )

        for step, batch in enumerate(progress):
            inputs, labels = move_batch_to_device(batch, device)
            with torch.cuda.amp.autocast(enabled=use_amp):
                outputs = model(**inputs, labels=labels)
                raw_loss = outputs.loss
                loss = raw_loss / args.grad_accum_steps

            if not torch.isfinite(raw_loss):
                nan_batches += 1
                if nan_batches <= 5:
                    print(
                        "Warning: non-finite loss at run %s epoch %s step %s"
                        % (run_index, epoch + 1, step),
                        flush=True,
                    )
                if total_items == 0 and nan_batches >= args.max_initial_nonfinite_batches:
                    raise FloatingPointError(
                        "Initial training batches produced non-finite losses."
                    )
                optimizer.zero_grad(set_to_none=True)
                continue

            scaler.scale(loss).backward()
            batch_size = labels.size(0)
            total_loss += float(raw_loss.item()) * batch_size
            total_items += batch_size

            should_step = (step + 1) % args.grad_accum_steps == 0 or (step + 1) == len(train_loader)
            if should_step:
                scaler.unscale_(optimizer)
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                grad_norm_sum += float(grad_norm)
                grad_updates += 1
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)

            progress.set_postfix(loss="%.4f" % float(raw_loss.item()))

        avg_train_loss = total_loss / total_items if total_items else 0.0
        avg_grad_norm = grad_norm_sum / grad_updates if grad_updates else 0.0
        val_loss, val_metrics, val_report, _ = R.evaluate(
            model, val_loader, device, use_amp, desc="Validation"
        )
        current_metric = val_loss if args.early_stop_metric == "loss" else val_metrics[args.early_stop_metric]
        scheduler.step(current_metric if np.isfinite(current_metric) else (1e9 if args.early_stop_metric == "loss" else -1e9))

        train_row = {
            "run_index": run_index,
            "epoch": epoch + 1,
            "loss": avg_train_loss,
            "grad_norm": avg_grad_norm,
            "learning_rate": optimizer.param_groups[0]["lr"],
            "timestamp": dt.datetime.now().isoformat(),
            "type": "train",
        }
        eval_row = {
            "run_index": run_index,
            "epoch": epoch + 1,
            "loss": val_loss,
            **val_metrics,
            "early_stop_metric": args.early_stop_metric,
            "early_stop_metric_value": current_metric,
            "timestamp": dt.datetime.now().isoformat(),
            "type": "eval",
        }
        run_metrics["train"].append(to_jsonable(train_row))
        run_metrics["eval"].append(to_jsonable(eval_row))

        print(
            "Run %s epoch %s/%s (%.1fs): train_loss=%.4f val_loss=%.4f val_%s=%.4f best=%s bad_epochs=%s"
            % (
                run_index,
                epoch + 1,
                args.epochs,
                time.time() - epoch_start,
                avg_train_loss,
                val_loss,
                args.early_stop_metric,
                current_metric,
                "%.4f" % best_metric if best_metric is not None else "none",
                bad_epochs,
            ),
            flush=True,
        )
        print(val_report, flush=True)

        if metric_improved(current_metric, best_metric, args.early_stop_metric):
            best_metric = float(current_metric)
            best_epoch = epoch + 1
            bad_epochs = 0
            checkpoint = {
                "epoch": epoch + 1,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "best_metric": best_metric,
                "early_stop_metric": args.early_stop_metric,
                "val_f1_macro": val_metrics.get("f1_macro"),
                "val_metrics": to_jsonable(val_metrics),
                "run_index": run_index,
                "args": to_jsonable(run_args),
            }
            torch.save(checkpoint, checkpoint_path)
            print(
                "Saved new best checkpoint for run %s at epoch %s with val_%s %.4f"
                % (run_index, epoch + 1, args.early_stop_metric, best_metric),
                flush=True,
            )
        else:
            bad_epochs += 1

        if epoch + 1 >= args.min_epochs and bad_epochs >= args.early_stop_patience:
            reason = (
                "validation %s did not improve for %s epochs after peak epoch %s"
                % (args.early_stop_metric, args.early_stop_patience, best_epoch)
            )
            print("Early stopping run %s: %s" % (run_index, reason), flush=True)
            run_metrics["early_stop"] = {
                "stopped": True,
                "reason": reason,
                "best_epoch": best_epoch,
                "best_metric": best_metric,
                "completed_epochs": epoch + 1,
            }
            break

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    run_metrics.setdefault(
        "early_stop",
        {
            "stopped": False,
            "reason": "max_epochs_reached",
            "best_epoch": best_epoch,
            "best_metric": best_metric,
            "completed_epochs": len(run_metrics["eval"]),
        },
    )
    return run_metrics


def main() -> None:
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    hf_logging.set_verbosity_error()
    args = parse_args()
    args.max_len = args.max_len or R.default_max_len(args.approach)
    if args.subset_seed is None:
        args.subset_seed = args.seed

    R.set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_path = default_model_path(args)
    variant_name = "masked" if args.mask_aspect else "unmasked"
    tag = output_tag(args)
    output_dir = Path(args.output_root) / args.approach / variant_name / args.split / tag
    output_dir.mkdir(parents=True, exist_ok=True)

    run_args = vars(args).copy()
    run_args.update(
        {
            "resolved_model_path": str(model_path),
            "variant": variant_name,
            "percentage_tag": tag,
            "output_dir": str(output_dir),
        }
    )

    print("--- Minimum Viable Set Encoder Training ---", flush=True)
    print("Approach: %s" % args.approach, flush=True)
    print("Language: %s" % args.split, flush=True)
    print("Variant: %s" % variant_name, flush=True)
    print("Percent/design: %s (%s)" % (args.percent, tag), flush=True)
    print("Output dir: %s" % output_dir, flush=True)
    print("Device: %s" % device, flush=True)
    if torch.cuda.is_available():
        print("CUDA device: %s" % torch.cuda.get_device_name(0), flush=True)

    if args.skip_completed and is_complete(output_dir, args.run_indices, args.percent):
        print("Skipping complete percentage run: %s" % output_dir, flush=True)
        return

    best_model_paths: dict[int, str | None] = {}
    subset_summaries: dict[str, Any] = {}
    started_at = time.time()

    if not args.test_only:
        for run_index in args.run_indices:
            checkpoint_path = output_dir / f"best_model_{run_index}.pt"
            metrics_path = output_dir / f"training_metrics_{run_index}.json"
            if args.skip_completed and checkpoint_path.exists() and metrics_path.exists():
                best_model_paths[run_index] = str(checkpoint_path)
                print("Skipping completed run %s." % run_index, flush=True)
                continue

            print("\n===== Training split run %s =====" % run_index, flush=True)
            R.set_seed(args.seed + run_index)
            if args.controlled_train_val_template:
                train_subset, val_subset, controlled_meta = load_controlled_train_val(
                    args.controlled_train_val_template, run_index
                )
                if args.limit_train is not None:
                    train_subset = train_subset[: args.limit_train]
                if args.limit_val is not None:
                    val_subset = val_subset[: args.limit_val]
                train_summary = {
                    "selection": "controlled",
                    "selected_count": len(train_subset),
                    "label_counts": dict(Counter(item.get("sentiment") for item in train_subset)),
                    **controlled_meta,
                }
                val_summary = {
                    "selection": "controlled",
                    "selected_count": len(val_subset),
                    "label_counts": dict(Counter(item.get("sentiment") for item in val_subset)),
                    **controlled_meta,
                }
            else:
                train_raw, val_raw = load_train_val_data(args, run_index)
                train_subset, train_summary = stratified_subset(
                    train_raw,
                    args.percent,
                    args.subset_seed + run_index * 101,
                    args.subset_min_per_class,
                    args.limit_train,
                )
                val_subset, val_summary = stratified_subset(
                    val_raw,
                    args.percent,
                    args.subset_seed + run_index * 101 + 17,
                    args.subset_min_per_class,
                    args.limit_val,
                )
            subset_summaries[str(run_index)] = {
                "train": train_summary,
                "val": val_summary,
            }
            print("Train subset: %s" % train_summary, flush=True)
            print("Val subset: %s" % val_summary, flush=True)

            tokenizer, model = R.load_tokenizer_and_model(args, model_path)
            model.to(device)
            train_dataset = R.ABSADataset(train_subset, args.mask_aspect)
            val_dataset = R.ABSADataset(val_subset, args.mask_aspect)
            train_loader = R.make_loader(train_dataset, tokenizer, args, shuffle=True)
            val_loader = R.make_loader(val_dataset, tokenizer, args, shuffle=False)

            run_metrics = train_one_run_dynamic(
                model,
                train_loader,
                val_loader,
                args,
                device,
                run_index,
                checkpoint_path,
                run_args,
            )
            best_model_paths[run_index] = str(checkpoint_path) if checkpoint_path.exists() else None
            write_json(
                metrics_path,
                {
                    "arguments": run_args,
                    "run_index": run_index,
                    "percentage": args.percent,
                    "percentage_tag": tag,
                    "subset": subset_summaries[str(run_index)],
                    "train_metrics": run_metrics["train"],
                    "eval_metrics": run_metrics["eval"],
                    "early_stop": run_metrics["early_stop"],
                    "best_model_path": best_model_paths[run_index],
                },
            )
            print("Training metrics saved to: %s" % metrics_path, flush=True)

            del model, tokenizer, train_loader, val_loader, train_dataset, val_dataset
            del train_subset, val_subset, run_metrics
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    else:
        for run_index in args.run_indices:
            checkpoint_path = output_dir / f"best_model_{run_index}.pt"
            best_model_paths[run_index] = str(checkpoint_path) if checkpoint_path.exists() else None

    print("\n===== Testing best checkpoints on complete test split =====", flush=True)
    test_raw = load_test_data(args)
    all_test_metrics: dict[str, Any] = {}
    successful_results: list[dict[str, Any]] = []
    use_amp = torch.cuda.is_available() and not args.no_amp

    for run_index in args.run_indices:
        checkpoint_path = best_model_paths.get(run_index)
        print("\n--- Evaluating run %s checkpoint ---" % run_index, flush=True)
        if not checkpoint_path or not Path(checkpoint_path).exists():
            all_test_metrics[f"model_{run_index}"] = {
                "error": "Model path not found or model was not saved/found."
            }
            continue

        tokenizer, model = R.load_tokenizer_and_model(args, model_path)
        checkpoint = R.torch_load(checkpoint_path, map_location="cpu")
        state_dict = checkpoint.get("model_state_dict", checkpoint)
        model.load_state_dict(state_dict)
        model.to(device)
        test_dataset = R.ABSADataset(test_raw, args.mask_aspect)
        test_loader = R.make_loader(
            test_dataset,
            tokenizer,
            args,
            shuffle=False,
            batch_size=max(1, args.batch_size * 2),
        )
        test_loss, test_metrics, test_report, test_preds = R.evaluate(
            model, test_loader, device, use_amp, desc="Testing"
        )
        print(
            "Test run %s: loss=%.4f macro_f1=%.4f accuracy=%.4f qwk=%.4f"
            % (
                run_index,
                test_loss,
                test_metrics["f1_macro"],
                test_metrics["accuracy"],
                test_metrics["qwk"],
            ),
            flush=True,
        )
        print(test_report, flush=True)

        run_result = {
            "model_run_index": run_index,
            "model_path": checkpoint_path,
            "test_loss": test_loss,
            "percentage": args.percent,
            "percentage_tag": tag,
            **test_metrics,
        }
        all_test_metrics[f"model_{run_index}"] = to_jsonable(run_result)
        successful_results.append(test_metrics)

        predictions_path = output_dir / f"test_predictions_{run_index}.json"
        test_copy = copy.deepcopy(test_raw)
        original_scale_preds = [R.ID_TO_ORIGINAL_LABEL[int(pred)] for pred in test_preds]
        if len(test_copy) == len(original_scale_preds):
            for item, pred in zip(test_copy, original_scale_preds):
                item["prediction"] = int(pred)
            write_json(predictions_path, test_copy)
        else:
            all_test_metrics[f"model_{run_index}"]["prediction_error"] = "Prediction count mismatch."

        del model, tokenizer, test_dataset, test_loader, checkpoint
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    all_test_metrics["average_performance"] = R.summarize_average_performance(successful_results)
    all_test_metrics["percentage"] = args.percent
    all_test_metrics["percentage_tag"] = tag
    all_test_metrics["subset_summaries"] = subset_summaries
    write_json(output_dir / "test_metrics_summary.json", all_test_metrics)

    success = {
        "approach": args.approach,
        "language": args.split,
        "variant": variant_name,
        "percent": args.percent,
        "percentage_tag": tag,
        "run_indices": args.run_indices,
        "output_dir": str(output_dir),
        "elapsed_seconds": time.time() - started_at,
        "checkpoints": [best_model_paths.get(i) for i in args.run_indices],
        "test_metrics_summary": str(output_dir / "test_metrics_summary.json"),
    }
    write_json(completion_marker(output_dir), success)
    print("Wrote success marker: %s" % completion_marker(output_dir), flush=True)


if __name__ == "__main__":
    main()
