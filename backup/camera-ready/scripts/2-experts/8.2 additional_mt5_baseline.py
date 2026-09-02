#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""additional-comparison mT5 baseline for document-level ABSA.

The model is trained as a text-to-text classifier. At evaluation time, it scores
the three allowed label strings by conditional token loss and chooses the best
label, avoiding invalid free-form generations.
"""

import argparse
import copy
import datetime as dt
import gc
import json
import os
import random
import re
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    cohen_kappa_score,
    precision_recall_fscore_support,
)
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, logging as hf_logging


ROOT_DIR = Path(os.environ.get("ABSA_RELEASE_ROOT", Path(__file__).resolve().parents[2])).resolve()
ASPECT_TOKEN = "[ASPECT]"
LABEL_TO_ID = {-1: 0, 0: 1, 1: 2}
ID_TO_ORIGINAL_LABEL = {0: -1, 1: 0, 2: 1}
ID_TO_LABEL_TEXT = {0: "negative", 1: "neutral", 2: "positive"}
FIXED_LABEL_IDS = [0, 1, 2]
TARGET_NAMES = ["Negative (0)", "Neutral (1)", "Positive (2)"]
DEFAULT_MT5_MODEL_DIR = "google_mt5-base"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train additional-comparison mT5 text-to-text ABSA baseline."
    )
    parser.add_argument(
        "--split",
        required=True,
        choices=["slovenian", "serbian"],
        help="Language split to use.",
    )
    parser.add_argument(
        "--mask_aspect",
        action="store_true",
        help="Replace tagged aspect mentions and the paired aspect name with [ASPECT].",
    )
    parser.add_argument("--data_dir", default=str(ROOT_DIR / "data"))
    parser.add_argument("--model_root", default=str(ROOT_DIR / "models"))
    parser.add_argument("--model_path", default=None)
    parser.add_argument("--output_root", default=str(ROOT_DIR / "reviews"))
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--grad_accum_steps", type=int, default=8)
    parser.add_argument("--lr", type=float, default=3e-5)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--max_source_len", type=int, default=512)
    parser.add_argument("--max_target_len", type=int, default=8)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--run_indices", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--test_only", action="store_true")
    parser.add_argument(
        "--skip_completed",
        action="store_true",
        help="Skip split-runs whose checkpoint and full training metrics already exist.",
    )
    parser.add_argument("--no_amp", action="store_true")
    parser.add_argument("--gradient_checkpointing", action="store_true")
    parser.add_argument(
        "--max_initial_nonfinite_batches",
        type=int,
        default=10,
        help="Abort a run if this many initial training batches produce non-finite losses.",
    )
    parser.add_argument(
        "--local_files_only",
        action="store_true",
        default=True,
        help="Load Hugging Face artifacts from local files only.",
    )
    return parser.parse_args()


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def to_jsonable(value):
    if isinstance(value, dict):
        return {str(k): to_jsonable(v) for k, v in value.items()}
    if isinstance(value, list):
        return [to_jsonable(v) for v in value]
    if isinstance(value, tuple):
        return [to_jsonable(v) for v in value]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, Path):
        return str(value)
    return value


def torch_load(path, map_location):
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


def env_flag(name, default=False):
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def nonfinite_parameter_names(model, max_names=8):
    bad_names = []
    for name, param in model.named_parameters():
        if not torch.isfinite(param).all():
            bad_names.append(name)
            if len(bad_names) >= max_names:
                break
    return bad_names


def assert_finite_model_parameters(model, context):
    bad_names = nonfinite_parameter_names(model)
    if bad_names:
        raise FloatingPointError(
            "%s contains non-finite parameters: %s"
            % (context, ", ".join(bad_names))
        )


def is_run_complete(output_dir, run_index, expected_epochs):
    checkpoint_path = output_dir / ("best_model_%s.pt" % run_index)
    metrics_path = output_dir / ("training_metrics_%s.json" % run_index)
    if not checkpoint_path.exists() or not metrics_path.exists():
        return False, "missing checkpoint or training metrics"
    try:
        with metrics_path.open("r", encoding="utf-8") as f:
            metrics = json.load(f)
    except Exception as exc:
        return False, "could not read metrics: %s" % exc
    train_count = len(metrics.get("train_metrics", []))
    eval_count = len(metrics.get("eval_metrics", []))
    if train_count < expected_epochs or eval_count < expected_epochs:
        return (
            False,
            "incomplete epochs: train=%s eval=%s expected=%s"
            % (train_count, eval_count, expected_epochs),
        )
    best_path = metrics.get("best_model_path")
    if best_path and not Path(best_path).exists() and not checkpoint_path.exists():
        return False, "best_model_path in metrics does not exist"
    return True, "complete"


def default_model_path(args):
    if args.model_path:
        return Path(args.model_path)
    return Path(args.model_root) / DEFAULT_MT5_MODEL_DIR


def load_split_data(data_dir, language, split_index=None):
    data_dir = Path(data_dir)
    if split_index is None:
        file_path = data_dir / ("%s_test_complete.json" % language)
        key = "test"
    else:
        file_path = data_dir / ("%s_train_val_complete_%s.json" % (language, split_index))
        key = None

    print("Loading data from: %s" % file_path)
    with file_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    if key == "test":
        test_data = data.get("test", [])
        if not test_data:
            raise ValueError("Missing or empty test split in %s" % file_path)
        return test_data

    train_data = data.get("train", [])
    val_data = data.get("val", [])
    if not train_data or not val_data:
        raise ValueError("Missing train/val split in %s" % file_path)
    return train_data, val_data


def prepare_article_and_aspect(item, mask_aspect):
    article = item.get("article", "") or ""
    aspect = item.get("aspect", "") or ""
    if mask_aspect:
        article = re.sub(r"<aspect>.*?</aspect>", ASPECT_TOKEN, article, flags=re.DOTALL)
        aspect = ASPECT_TOKEN
    else:
        article = article.replace("<aspect>", "").replace("</aspect>", "")
    return article.strip(), aspect.strip()


def build_source_text(article, aspect):
    return "classify sentiment\naspect: %s\narticle: %s" % (aspect, article)


class MT5ABSADataset(Dataset):
    def __init__(self, data, mask_aspect):
        self.data = data
        self.mask_aspect = mask_aspect

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        article, aspect = prepare_article_and_aspect(item, self.mask_aspect)
        original_label = item.get("sentiment", 0)
        mapped_label = LABEL_TO_ID.get(original_label, 1)
        return {
            "source": build_source_text(article, aspect),
            "label_id": mapped_label,
            "target": ID_TO_LABEL_TEXT[mapped_label],
        }


class MT5Collator:
    def __init__(self, tokenizer, max_source_len, max_target_len):
        self.tokenizer = tokenizer
        self.max_source_len = max_source_len
        self.max_target_len = max_target_len

    def encode_targets(self, targets):
        try:
            target_encoding = self.tokenizer(
                text_target=targets,
                truncation=True,
                max_length=self.max_target_len,
                padding=True,
                return_tensors="pt",
            )
        except TypeError:
            with self.tokenizer.as_target_tokenizer():
                target_encoding = self.tokenizer(
                    targets,
                    truncation=True,
                    max_length=self.max_target_len,
                    padding=True,
                    return_tensors="pt",
                )
        labels = target_encoding["input_ids"]
        labels[labels == self.tokenizer.pad_token_id] = -100
        return labels

    def __call__(self, features):
        sources = [f["source"] for f in features]
        targets = [f["target"] for f in features]
        label_ids = torch.tensor([f["label_id"] for f in features], dtype=torch.long)
        source_encoding = self.tokenizer(
            sources,
            truncation=True,
            max_length=self.max_source_len,
            padding=True,
            return_tensors="pt",
        )
        source_encoding["labels"] = self.encode_targets(targets)
        source_encoding["class_labels"] = label_ids
        return source_encoding


def load_tokenizer_and_model(args, model_path):
    tokenizer = AutoTokenizer.from_pretrained(
        model_path, local_files_only=args.local_files_only, use_fast=True
    )
    model = AutoModelForSeq2SeqLM.from_pretrained(
        model_path, local_files_only=args.local_files_only, torch_dtype=torch.float32
    )

    if args.mask_aspect:
        added = tokenizer.add_special_tokens({"additional_special_tokens": [ASPECT_TOKEN]})
        if added:
            print("Added %s special token(s) to tokenizer." % added)
    current_vocab_size = model.get_input_embeddings().weight.shape[0]
    tokenizer_vocab_size = len(tokenizer)
    if tokenizer_vocab_size > current_vocab_size:
        print(
            "Resizing token embeddings from %s to %s."
            % (current_vocab_size, tokenizer_vocab_size)
        )
        model.resize_token_embeddings(tokenizer_vocab_size)
    elif tokenizer_vocab_size < current_vocab_size:
        print(
            "Tokenizer size (%s) is smaller than model embeddings (%s); keeping model embeddings."
            % (tokenizer_vocab_size, current_vocab_size)
        )

    if args.gradient_checkpointing and hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()
        if hasattr(model.config, "use_cache"):
            model.config.use_cache = False
        print("Enabled gradient checkpointing.")

    model.float()
    assert_finite_model_parameters(model, "Loaded mT5 model")
    print("Model parameter dtype after load: %s" % next(model.parameters()).dtype)

    return tokenizer, model


def make_loader(dataset, tokenizer, args, shuffle, batch_size=None):
    collator = MT5Collator(
        tokenizer=tokenizer,
        max_source_len=args.max_source_len,
        max_target_len=args.max_target_len,
    )
    return DataLoader(
        dataset,
        batch_size=batch_size or args.batch_size,
        shuffle=shuffle,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
        collate_fn=collator,
    )


def move_batch_to_device(batch, device):
    class_labels = batch.pop("class_labels").to(device)
    labels = batch.pop("labels").to(device)
    inputs = {k: v.to(device) for k, v in batch.items()}
    return inputs, labels, class_labels


def make_candidate_label_tensor(tokenizer, label_text, batch_size, max_target_len, device):
    encoded = tokenizer(
        [label_text],
        add_special_tokens=True,
        truncation=True,
        max_length=max_target_len,
        return_tensors="pt",
    )
    ids = encoded["input_ids"][0].tolist()
    ids = ids[:max_target_len]
    labels = torch.full(
        (batch_size, len(ids)),
        fill_value=-100,
        dtype=torch.long,
        device=device,
    )
    labels[:, : len(ids)] = torch.tensor(ids, dtype=torch.long, device=device)
    return labels


def score_candidate_labels(model, tokenizer, inputs, batch_size, args, use_amp):
    candidate_losses = []
    for class_id in FIXED_LABEL_IDS:
        label_text = ID_TO_LABEL_TEXT[class_id]
        labels = make_candidate_label_tensor(
            tokenizer, label_text, batch_size, args.max_target_len, inputs["input_ids"].device
        )
        decoder_input_ids = model._shift_right(labels)
        with torch.cuda.amp.autocast(enabled=use_amp):
            outputs = model(
                **inputs,
                decoder_input_ids=decoder_input_ids,
                use_cache=False,
            )
            logits = outputs.logits
            token_losses = F.cross_entropy(
                logits.transpose(1, 2),
                labels,
                ignore_index=-100,
                reduction="none",
            )
            denom = (labels != -100).sum(dim=1).clamp_min(1)
            per_example_loss = token_losses.sum(dim=1) / denom
        candidate_losses.append(per_example_loss.detach())
    stacked = torch.stack(candidate_losses, dim=1)
    return torch.argmin(stacked, dim=1)


def evaluate(model, tokenizer, dataloader, device, args, use_amp, desc="Evaluating"):
    model.eval()
    total_loss = 0.0
    total_items = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in tqdm(
            dataloader, desc=desc, leave=False, disable=env_flag("TQDM_DISABLE")
        ):
            inputs, labels, class_labels = move_batch_to_device(batch, device)
            with torch.cuda.amp.autocast(enabled=use_amp):
                outputs = model(**inputs, labels=labels, use_cache=False)
                loss = outputs.loss
            batch_size = class_labels.size(0)
            preds = score_candidate_labels(
                model, tokenizer, inputs, batch_size, args, use_amp
            )
            total_loss += float(loss.item()) * batch_size
            total_items += batch_size
            all_preds.extend(preds.detach().cpu().numpy().tolist())
            all_labels.extend(class_labels.detach().cpu().numpy().tolist())

    avg_loss = total_loss / total_items if total_items else 0.0
    accuracy = accuracy_score(all_labels, all_preds)
    precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
        all_labels,
        all_preds,
        labels=FIXED_LABEL_IDS,
        average="macro",
        zero_division=0,
    )
    precision_micro, recall_micro, f1_micro, _ = precision_recall_fscore_support(
        all_labels,
        all_preds,
        labels=FIXED_LABEL_IDS,
        average="micro",
        zero_division=0,
    )
    precision_weighted, recall_weighted, f1_weighted, _ = precision_recall_fscore_support(
        all_labels,
        all_preds,
        labels=FIXED_LABEL_IDS,
        average="weighted",
        zero_division=0,
    )
    qwk = cohen_kappa_score(
        all_labels, all_preds, labels=FIXED_LABEL_IDS, weights="quadratic"
    )
    report_dict = classification_report(
        all_labels,
        all_preds,
        labels=FIXED_LABEL_IDS,
        target_names=TARGET_NAMES,
        zero_division=0,
        output_dict=True,
    )
    report_str = classification_report(
        all_labels,
        all_preds,
        labels=FIXED_LABEL_IDS,
        target_names=TARGET_NAMES,
        zero_division=0,
    )
    metrics = {
        "loss": avg_loss,
        "accuracy": accuracy,
        "precision_macro": precision_macro,
        "recall_macro": recall_macro,
        "f1_macro": f1_macro,
        "precision_micro": precision_micro,
        "recall_micro": recall_micro,
        "f1_micro": f1_micro,
        "precision_weighted": precision_weighted,
        "recall_weighted": recall_weighted,
        "f1_weighted": f1_weighted,
        "qwk": qwk,
        "per_class_report": report_dict,
    }
    return avg_loss, metrics, report_str, all_preds


def train_one_run(
    model,
    tokenizer,
    train_loader,
    val_loader,
    args,
    device,
    run_index,
    checkpoint_path,
    run_args,
):
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.2, patience=2
    )
    use_amp = torch.cuda.is_available() and not args.no_amp
    print("Mixed precision enabled: %s" % use_amp)
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    best_val_f1 = -1.0
    run_metrics = {"train": [], "eval": []}
    optimizer.zero_grad(set_to_none=True)

    print("\n--- Starting mT5 training for run %s ---" % run_index)
    print("Best checkpoint path: %s" % checkpoint_path)

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
            disable=env_flag("TQDM_DISABLE"),
        )

        for step, batch in enumerate(progress):
            inputs, labels, class_labels = move_batch_to_device(batch, device)
            with torch.cuda.amp.autocast(enabled=use_amp):
                outputs = model(**inputs, labels=labels, use_cache=False)
                raw_loss = outputs.loss
                loss = raw_loss / args.grad_accum_steps

            if not torch.isfinite(raw_loss):
                nan_batches += 1
                if nan_batches <= 5:
                    print(
                        "Warning: non-finite loss at run %s epoch %s step %s"
                        % (run_index, epoch + 1, step)
                    )
                if total_items == 0 and nan_batches >= args.max_initial_nonfinite_batches:
                    raise FloatingPointError(
                        "First %s training batches for run %s epoch %s produced non-finite losses. "
                        "Aborting early; run diagnostics before resubmitting."
                        % (nan_batches, run_index, epoch + 1)
                    )
                optimizer.zero_grad(set_to_none=True)
                continue

            scaler.scale(loss).backward()
            batch_size = class_labels.size(0)
            total_loss += float(raw_loss.item()) * batch_size
            total_items += batch_size

            should_step = (
                (step + 1) % args.grad_accum_steps == 0
                or (step + 1) == len(train_loader)
            )
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
        val_loss, val_metrics, val_report, _ = evaluate(
            model, tokenizer, val_loader, device, args, use_amp, desc="Validation"
        )
        val_f1 = val_metrics["f1_macro"]
        scheduler_loss = val_loss if np.isfinite(val_loss) else 1e9
        scheduler.step(scheduler_loss)

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
            "timestamp": dt.datetime.now().isoformat(),
            "type": "eval",
        }
        run_metrics["train"].append(to_jsonable(train_row))
        run_metrics["eval"].append(to_jsonable(eval_row))

        print(
            "Run %s epoch %s/%s (%.1fs): train_loss=%.4f val_loss=%.4f val_macro_f1=%.4f val_qwk=%.4f nonfinite_batches=%s"
            % (
                run_index,
                epoch + 1,
                args.epochs,
                time.time() - epoch_start,
                avg_train_loss,
                val_loss,
                val_f1,
                val_metrics["qwk"],
                nan_batches,
            )
        )
        print(val_report)

        if not np.isfinite(val_loss) or not np.isfinite(val_f1):
            print(
                "Skipping checkpoint save for run %s epoch %s due to non-finite validation metrics."
                % (run_index, epoch + 1)
            )
        elif val_f1 > best_val_f1:
            best_val_f1 = val_f1
            checkpoint = {
                "epoch": epoch + 1,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "val_f1_macro": best_val_f1,
                "val_metrics": to_jsonable(val_metrics),
                "run_index": run_index,
                "args": to_jsonable(run_args),
            }
            torch.save(checkpoint, checkpoint_path)
            print(
                "Saved new best checkpoint for run %s at epoch %s with val macro F1 %.4f"
                % (run_index, epoch + 1, best_val_f1)
            )

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return run_metrics


def summarize_average_performance(test_results):
    if not test_results:
        return {"error": "No models available or evaluation failed for all."}

    def valid_values(key):
        return [r[key] for r in test_results if r.get(key) is not None]

    f1_scores = valid_values("f1_macro")
    acc_scores = valid_values("accuracy")
    qwk_scores = valid_values("qwk")
    return {
        "f1_macro_mean": float(np.mean(f1_scores)) if f1_scores else 0.0,
        "f1_macro_std": float(np.std(f1_scores)) if len(f1_scores) > 1 else 0.0,
        "num_models_f1": len(f1_scores),
        "accuracy_mean": float(np.mean(acc_scores)) if acc_scores else 0.0,
        "accuracy_std": float(np.std(acc_scores)) if len(acc_scores) > 1 else 0.0,
        "num_models_accuracy": len(acc_scores),
        "qwk_mean": float(np.mean(qwk_scores)) if qwk_scores else 0.0,
        "qwk_std": float(np.std(qwk_scores)) if len(qwk_scores) > 1 else 0.0,
        "num_models_qwk": len(qwk_scores),
    }


def main():
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    hf_logging.set_verbosity_error()
    args = parse_args()
    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_path = default_model_path(args)
    variant_name = "masked" if args.mask_aspect else "unmasked"
    output_dir = Path(args.output_root) / "mt5" / variant_name / args.split
    output_dir.mkdir(parents=True, exist_ok=True)

    run_args = vars(args).copy()
    run_args.update(
        {
            "approach": "mt5",
            "resolved_model_path": str(model_path),
            "variant": variant_name,
            "output_dir": str(output_dir),
            "prediction_rule": "minimum average conditional token loss over negative/neutral/positive",
        }
    )

    print("--- Comparison mT5 Baseline ---")
    print("Language: %s" % args.split)
    print("Variant: %s" % variant_name)
    print("Model path: %s" % model_path)
    print("Output dir: %s" % output_dir)
    print("Device: %s" % device)
    if torch.cuda.is_available():
        print("CUDA device: %s" % torch.cuda.get_device_name(0))
    print("Max source length: %s" % args.max_source_len)
    print("Batch size: %s; grad accumulation: %s" % (args.batch_size, args.grad_accum_steps))

    best_model_paths = {}

    if not args.test_only:
        for run_index in args.run_indices:
            checkpoint_path = output_dir / ("best_model_%s.pt" % run_index)
            metrics_path = output_dir / ("training_metrics_%s.json" % run_index)
            if args.skip_completed:
                complete, reason = is_run_complete(output_dir, run_index, args.epochs)
                if complete:
                    best_model_paths[run_index] = str(checkpoint_path)
                    print(
                        "\n===== Skipping completed split run %s (%s) ====="
                        % (run_index, reason)
                    )
                    continue
                print(
                    "\n===== Split run %s is not complete; rerunning (%s) ====="
                    % (run_index, reason)
                )
            print("\n===== Training split run %s =====" % run_index)
            set_seed(args.seed + run_index)
            train_raw, val_raw = load_split_data(args.data_dir, args.split, run_index)
            tokenizer, model = load_tokenizer_and_model(args, model_path)
            model.to(device)

            train_dataset = MT5ABSADataset(train_raw, args.mask_aspect)
            val_dataset = MT5ABSADataset(val_raw, args.mask_aspect)
            train_loader = make_loader(train_dataset, tokenizer, args, shuffle=True)
            val_loader = make_loader(val_dataset, tokenizer, args, shuffle=False)

            run_metrics = train_one_run(
                model,
                tokenizer,
                train_loader,
                val_loader,
                args,
                device,
                run_index,
                checkpoint_path,
                run_args,
            )
            best_model_paths[run_index] = str(checkpoint_path) if checkpoint_path.exists() else None
            with metrics_path.open("w", encoding="utf-8") as f:
                json.dump(
                    to_jsonable(
                        {
                            "arguments": run_args,
                            "run_index": run_index,
                            "train_metrics": run_metrics["train"],
                            "eval_metrics": run_metrics["eval"],
                            "best_model_path": best_model_paths[run_index],
                        }
                    ),
                    f,
                    indent=4,
                    ensure_ascii=False,
                )
            print("Training metrics saved to: %s" % metrics_path)

            del model, tokenizer, train_loader, val_loader, train_dataset, val_dataset
            del train_raw, val_raw, run_metrics
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    else:
        print("\n*** Test-only mode: locating existing checkpoints ***")
        for run_index in args.run_indices:
            checkpoint_path = output_dir / ("best_model_%s.pt" % run_index)
            best_model_paths[run_index] = str(checkpoint_path) if checkpoint_path.exists() else None
            print("Run %s checkpoint: %s" % (run_index, best_model_paths[run_index]))

    print("\n===== Testing best checkpoints =====")
    test_raw = load_split_data(args.data_dir, args.split, split_index=None)
    all_test_metrics = {}
    successful_results = []
    use_amp = torch.cuda.is_available() and not args.no_amp

    for run_index in args.run_indices:
        checkpoint_path = best_model_paths.get(run_index)
        print("\n--- Evaluating run %s checkpoint ---" % run_index)
        if not checkpoint_path:
            all_test_metrics["model_%s" % run_index] = {
                "error": "Model path not found or model was not saved/found."
            }
            print("Skipping run %s; checkpoint not found." % run_index)
            continue

        tokenizer, model = load_tokenizer_and_model(args, model_path)
        checkpoint = torch_load(checkpoint_path, map_location="cpu")
        state_dict = checkpoint.get("model_state_dict", checkpoint)
        model.load_state_dict(state_dict)
        model.to(device)

        test_dataset = MT5ABSADataset(test_raw, args.mask_aspect)
        test_loader = make_loader(
            test_dataset,
            tokenizer,
            args,
            shuffle=False,
            batch_size=max(1, args.batch_size * 2),
        )
        test_loss, test_metrics, test_report, test_preds = evaluate(
            model, tokenizer, test_loader, device, args, use_amp, desc="Testing"
        )
        print(
            "Test run %s: loss=%.4f macro_f1=%.4f accuracy=%.4f qwk=%.4f"
            % (
                run_index,
                test_loss,
                test_metrics["f1_macro"],
                test_metrics["accuracy"],
                test_metrics["qwk"],
            )
        )
        print(test_report)

        run_result = {
            "model_run_index": run_index,
            "model_path": checkpoint_path,
            "test_loss": test_loss,
            **test_metrics,
        }
        all_test_metrics["model_%s" % run_index] = to_jsonable(run_result)
        successful_results.append(test_metrics)

        predictions_path = output_dir / ("test_predictions_%s.json" % run_index)
        test_copy = copy.deepcopy(test_raw)
        original_scale_preds = [ID_TO_ORIGINAL_LABEL[int(pred)] for pred in test_preds]
        if len(test_copy) == len(original_scale_preds):
            for item, pred in zip(test_copy, original_scale_preds):
                item["prediction"] = int(pred)
            with predictions_path.open("w", encoding="utf-8") as f:
                json.dump(test_copy, f, indent=4, ensure_ascii=False)
            print("Test predictions saved to: %s" % predictions_path)
        else:
            all_test_metrics["model_%s" % run_index][
                "prediction_error"
            ] = "Prediction count mismatch."

        del model, tokenizer, test_dataset, test_loader, checkpoint
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    all_test_metrics["average_performance"] = summarize_average_performance(
        successful_results
    )
    metrics_summary_path = output_dir / "test_metrics_summary.json"
    with metrics_summary_path.open("w", encoding="utf-8") as f:
        json.dump(to_jsonable(all_test_metrics), f, indent=4, ensure_ascii=False)
    print("\nCombined test metrics summary saved to: %s" % metrics_summary_path)
    print("Experiment complete.")


if __name__ == "__main__":
    main()
