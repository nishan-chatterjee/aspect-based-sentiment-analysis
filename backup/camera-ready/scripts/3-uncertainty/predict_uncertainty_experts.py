#!/usr/bin/env python3
"""Generate checkpoint-ensemble + MC-dropout uncertainty predictions.

This script implements Option A from reviews/scratchpad/uncertainty_modelling_plan.md:
use every available best checkpoint for an expert and run MC dropout for each
checkpoint. The final probability distribution is the mean over
checkpoint x stochastic samples.
"""

from __future__ import annotations

import argparse
import copy
import gc
import importlib.util
import json
import math
import os
import random
import re
import statistics
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm
from transformers import AutoModel, AutoTokenizer, logging as hf_logging


ROOT_DIR = Path(os.environ.get("ABSA_RELEASE_ROOT", Path(__file__).resolve().parents[2])).resolve()
ASPECT_TOKEN = "[ASPECT]"
HAN_ASPECT_PLACEHOLDER = "[ASPECT_TARGET]"
LABEL_TO_ID = {-1: 0, 0: 1, 1: 2}
ID_TO_LABEL = {0: -1, 1: 0, 2: 1}
PROB_LABELS = ("Negative", "Neutral", "Positive")
SPLIT_NAMES = ("test", "train_val_0", "train_val_1", "train_val_2")

EXPERT_SPECS: dict[str, dict[str, Any]] = {
    "han_xlmr_masked": {
        "backend": "han_gcm",
        "json_key": "global-context-modelling/simplified-dart-xlmr",
        "checkpoint_dir": "results/global-context-modelling/simplified-dart-xlmr/{language}",
        "default_batch_size": 32,
    },
    "longformer_masked": {
        "backend": "encoder",
        "approach": "longformer",
        "json_key": "longformer/masked",
        "checkpoint_dir": "reviews/longformer/masked/{language}",
        "default_batch_size": 8,
        "default_max_len": 4096,
    },
    "mdeberta_masked": {
        "backend": "encoder",
        "approach": "mdeberta",
        "json_key": "mdeberta/masked",
        "checkpoint_dir": "reviews/mdeberta/masked/{language}",
        "default_batch_size": 64,
        "default_max_len": 512,
    },
    "slavic_specific_masked": {
        "backend": "encoder",
        "approach": "slavic_specific",
        "json_key": "slavic_specific/masked",
        "checkpoint_dir": "reviews/slavic_specific/masked/{language}",
        "default_batch_size": 64,
        "default_max_len": 512,
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--expert", required=True, choices=sorted(EXPERT_SPECS))
    parser.add_argument("--language", required=True, choices=["slovenian", "serbian"])
    parser.add_argument("--data_dir", default=str(ROOT_DIR / "data"))
    parser.add_argument("--output_root", default=str(ROOT_DIR / "reviews" / "uncertainty"))
    parser.add_argument("--model_root", default=str(ROOT_DIR / "models"))
    parser.add_argument("--splits", nargs="+", default=list(SPLIT_NAMES), choices=SPLIT_NAMES)
    parser.add_argument("--num_mc_samples", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_len", type=int, default=None)
    parser.add_argument("--limit_items", type=int, default=None)
    parser.add_argument("--checkpoint_limit", type=int, default=None)
    parser.add_argument(
        "--progress_every",
        type=int,
        default=25,
        help="Print one compact progress line every N batches when tqdm is disabled.",
    )
    parser.add_argument("--skip_completed", action="store_true")
    parser.add_argument("--no_amp", action="store_true")
    parser.add_argument("--local_files_only", action="store_true", default=True)
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def torch_load(path: Path, map_location: str | torch.device) -> Any:
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


def to_jsonable(value: Any) -> Any:
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


def env_flag(name: str, default: bool = False) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(to_jsonable(data), f, indent=2, ensure_ascii=False)


def input_path_for_split(data_dir: Path, language: str, split_name: str) -> Path:
    if split_name == "test":
        return data_dir / f"{language}_test_complete.json"
    suffix = split_name.rsplit("_", 1)[-1]
    return data_dir / f"{language}_train_val_complete_{suffix}.json"


def output_path_for_split(output_dir: Path, language: str, split_name: str) -> Path:
    if split_name == "test":
        return output_dir / f"{language}_test_complete.json"
    suffix = split_name.rsplit("_", 1)[-1]
    return output_dir / f"{language}_train_val_complete_{suffix}.json"


def collect_records(node: Any, records: list[dict[str, Any]]) -> None:
    if isinstance(node, dict):
        if "uuid" in node and "article" in node and "sentiment" in node:
            records.append(node)
        else:
            for value in node.values():
                collect_records(value, records)
    elif isinstance(node, list):
        for item in node:
            collect_records(item, records)


def find_checkpoints(spec: dict[str, Any], language: str, limit: int | None) -> list[Path]:
    checkpoint_dir = ROOT_DIR / spec["checkpoint_dir"].format(language=language)
    checkpoints = sorted(checkpoint_dir.glob("best_model_*.pt"))
    if limit is not None:
        checkpoints = checkpoints[:limit]
    if not checkpoints:
        raise FileNotFoundError(f"No best_model_*.pt checkpoints found in {checkpoint_dir}")
    return checkpoints


def iter_with_progress(loader: DataLoader, desc: str, progress_every: int):
    total = len(loader)
    use_tqdm = not env_flag("TQDM_DISABLE")
    if use_tqdm:
        yield from tqdm(loader, desc=desc, mininterval=30, leave=False)
        return

    started_at = time.time()
    for batch_idx, batch in enumerate(loader, start=1):
        yield batch
        should_log = (
            progress_every > 0
            and (batch_idx == 1 or batch_idx == total or batch_idx % progress_every == 0)
        )
        if should_log:
            elapsed = time.time() - started_at
            rate = batch_idx / elapsed if elapsed > 0 else 0.0
            eta = (total - batch_idx) / rate if rate > 0 else 0.0
            print(
                f"[progress] {desc}: batch {batch_idx}/{total} "
                f"elapsed={elapsed / 60:.1f}m eta={eta / 60:.1f}m",
                flush=True,
            )


def probs_to_uncertainty(predictions: list[int], prob_arrays: list[np.ndarray]) -> tuple[int, dict[str, float], dict[str, Any]]:
    if not prob_arrays:
        raise ValueError("No probability samples were collected.")
    total_samples = len(prob_arrays)
    mean_probs = np.mean(np.stack(prob_arrays, axis=0), axis=0).astype(np.float64)
    mean_probs = mean_probs / max(float(np.sum(mean_probs)), 1e-12)
    pred_from_mean = int(np.argmax(mean_probs))
    final_label = ID_TO_LABEL[pred_from_mean]

    sample_entropies = []
    for probs in prob_arrays:
        probs64 = probs.astype(np.float64)
        sample_entropies.append(float(-np.sum(probs64 * np.log2(probs64 + 1e-12))))
    predictive_entropy = float(-np.sum(mean_probs * np.log2(mean_probs + 1e-12)))
    expected_entropy = float(np.mean(sample_entropies))
    mutual_information = float(max(0.0, predictive_entropy - expected_entropy))

    counts = Counter(predictions)
    mode_count = max(counts.values()) if counts else 0
    return (
        int(final_label),
        {label: float(mean_probs[idx]) for idx, label in enumerate(PROB_LABELS)},
        {
            "confidence_score": float(np.max(mean_probs)),
            "vote_confidence": float(mode_count / total_samples),
            "variation_ratio": float(1.0 - (mode_count / total_samples)),
            "prediction_distribution": {
                "-1": int(counts.get(-1, 0)),
                "0": int(counts.get(0, 0)),
                "1": int(counts.get(1, 0)),
            },
            "predictive_entropy": predictive_entropy,
            "expected_entropy": expected_entropy,
            "mutual_information": mutual_information,
            "total_mc_samples": int(total_samples),
        },
    )


def annotate_records(
    records: list[dict[str, Any]],
    per_uuid_results: dict[str, dict[str, list[Any]]],
    json_key: str,
) -> None:
    prob_key = f"{json_key}/probabilities"
    uncertainty_key = f"{json_key}/uncertainty"
    for item in records:
        uuid = item.get("uuid")
        result = per_uuid_results.get(uuid)
        if not result:
            item[json_key] = "ERROR_NOT_PROCESSED"
            continue
        prediction, mean_probs, uncertainty = probs_to_uncertainty(
            result["predictions"],
            result["probabilities"],
        )
        item[json_key] = prediction
        item[prob_key] = mean_probs
        item[uncertainty_key] = uncertainty


def summarize_records(records: list[dict[str, Any]], json_key: str) -> dict[str, Any]:
    y_true = []
    y_pred = []
    for item in records:
        gold = item.get("sentiment")
        pred = item.get(json_key)
        if gold in LABEL_TO_ID and pred in LABEL_TO_ID:
            y_true.append(LABEL_TO_ID[gold])
            y_pred.append(LABEL_TO_ID[pred])
    if not y_true:
        return {"num_items": 0}
    from sklearn.metrics import accuracy_score, cohen_kappa_score, precision_recall_fscore_support

    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=[0, 1, 2], average="macro", zero_division=0
    )
    return {
        "num_items": len(y_true),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision_macro": float(precision),
        "recall_macro": float(recall),
        "f1_macro": float(f1),
        "qwk": float(cohen_kappa_score(y_true, y_pred, labels=[0, 1, 2], weights="quadratic")),
    }


def load_encoder_module():
    module_path = ROOT_DIR / "scripts" / "2-experts" / "8.1 additional_encoder_baselines.py"
    spec = importlib.util.spec_from_file_location("additional_encoder_baselines", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not import {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules["additional_encoder_baselines"] = module
    spec.loader.exec_module(module)
    return module


class EncoderInferenceDataset(Dataset):
    def __init__(self, records: list[dict[str, Any]], comparison_module: Any, mask_aspect: bool):
        self.records = records
        self.comparison_module = comparison_module
        self.mask_aspect = mask_aspect

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        item = self.records[idx]
        article, aspect = self.comparison_module.prepare_article_and_aspect(item, self.mask_aspect)
        return {
            "idx": idx,
            "article": article,
            "aspect": aspect,
            "label": LABEL_TO_ID.get(item.get("sentiment"), 1),
        }


class EncoderInferenceCollator:
    def __init__(self, tokenizer: Any, max_len: int, use_global_attention: bool):
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.use_global_attention = use_global_attention

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, Any]:
        encoding = self.tokenizer(
            [f["article"] for f in features],
            text_pair=[f["aspect"] for f in features],
            truncation=True,
            max_length=self.max_len,
            padding=True,
            return_tensors="pt",
        )
        if self.use_global_attention:
            global_attention_mask = torch.zeros_like(encoding["attention_mask"])
            global_attention_mask[:, 0] = 1
            encoding["global_attention_mask"] = global_attention_mask
        encoding["idx"] = torch.tensor([f["idx"] for f in features], dtype=torch.long)
        return encoding


def encoder_model_path(comparison_module: Any, model_root: Path, approach: str, language: str) -> Path:
    if approach == "slavic_specific":
        key = f"slavic_specific:{language}"
    else:
        key = approach
    return model_root / comparison_module.DEFAULT_MODEL_DIRS[key]


def build_encoder_model(
    comparison_module: Any,
    approach: str,
    language: str,
    model_path: Path,
    max_len: int,
    local_files_only: bool,
) -> tuple[Any, nn.Module]:
    ns = argparse.Namespace(
        approach=approach,
        split=language,
        model_path=str(model_path),
        model_root=str(model_path.parent),
        max_len=max_len,
        dropout_rate=None,
        gradient_checkpointing=False,
        mask_aspect=True,
        local_files_only=local_files_only,
    )
    return comparison_module.load_tokenizer_and_model(ns, model_path)


def run_encoder_uncertainty_multi(
    args: argparse.Namespace,
    spec: dict[str, Any],
    split_records: dict[str, list[dict[str, Any]]],
    checkpoints: list[Path],
    device: torch.device,
) -> dict[str, dict[str, dict[str, list[Any]]]]:
    comparison_module = load_encoder_module()
    approach = spec["approach"]
    max_len = args.max_len or spec["default_max_len"]
    model_path = encoder_model_path(comparison_module, Path(args.model_root), approach, args.language)
    batch_size = args.batch_size or spec["default_batch_size"]
    use_amp = torch.cuda.is_available() and not args.no_amp
    results = {
        split_name: {item["uuid"]: {"predictions": [], "probabilities": []} for item in records}
        for split_name, records in split_records.items()
    }
    loaders: dict[str, DataLoader] = {}
    reference_tokenizer_len: int | None = None

    for checkpoint_index, checkpoint_path in enumerate(checkpoints, start=1):
        print(f"Loading encoder checkpoint {checkpoint_index}/{len(checkpoints)}: {checkpoint_path}", flush=True)
        tokenizer, model = build_encoder_model(
            comparison_module,
            approach,
            args.language,
            model_path,
            max_len,
            args.local_files_only,
        )
        if reference_tokenizer_len is None:
            reference_tokenizer_len = len(tokenizer)
            loaders = {
                split_name: DataLoader(
                    EncoderInferenceDataset(records, comparison_module, mask_aspect=True),
                    batch_size=batch_size,
                    shuffle=False,
                    num_workers=args.num_workers,
                    pin_memory=torch.cuda.is_available(),
                    collate_fn=EncoderInferenceCollator(
                        tokenizer,
                        max_len=max_len,
                        use_global_attention=approach == "longformer",
                    ),
                )
                for split_name, records in split_records.items()
            }
        elif len(tokenizer) != reference_tokenizer_len:
            raise ValueError("Tokenizer size changed between encoder model loads.")

        checkpoint = torch_load(checkpoint_path, map_location="cpu")
        state_dict = checkpoint.get("model_state_dict", checkpoint)
        model.load_state_dict(state_dict)
        model.to(device)
        model.train()

        with torch.no_grad():
            for split_name, loader in loaders.items():
                records = split_records[split_name]
                desc = f"{checkpoint_path.name} {split_name}"
                for batch in iter_with_progress(loader, desc, args.progress_every):
                    indices = batch.pop("idx").numpy().tolist()
                    model_inputs = {k: v.to(device) for k, v in batch.items()}
                    for _mc in range(args.num_mc_samples):
                        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp):
                            outputs = model(**model_inputs)
                            logits = outputs.logits
                        probs = F.softmax(logits.float(), dim=-1).detach().cpu().numpy()
                        pred_ids = np.argmax(probs, axis=1)
                        for row_idx, pred_id, prob in zip(indices, pred_ids, probs):
                            uuid = records[row_idx]["uuid"]
                            results[split_name][uuid]["predictions"].append(ID_TO_LABEL[int(pred_id)])
                            results[split_name][uuid]["probabilities"].append(prob.astype(np.float64))

        del model, checkpoint, tokenizer
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return results


def run_encoder_uncertainty(
    args: argparse.Namespace,
    spec: dict[str, Any],
    records: list[dict[str, Any]],
    checkpoints: list[Path],
    device: torch.device,
) -> dict[str, dict[str, list[Any]]]:
    comparison_module = load_encoder_module()
    approach = spec["approach"]
    max_len = args.max_len or spec["default_max_len"]
    model_path = encoder_model_path(comparison_module, Path(args.model_root), approach, args.language)
    tokenizer, _ = build_encoder_model(
        comparison_module,
        approach,
        args.language,
        model_path,
        max_len,
        args.local_files_only,
    )
    batch_size = args.batch_size or spec["default_batch_size"]
    dataset = EncoderInferenceDataset(records, comparison_module, mask_aspect=True)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
        collate_fn=EncoderInferenceCollator(
            tokenizer,
            max_len=max_len,
            use_global_attention=approach == "longformer",
        ),
    )
    results = {item["uuid"]: {"predictions": [], "probabilities": []} for item in records}
    use_amp = torch.cuda.is_available() and not args.no_amp

    del _
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    for checkpoint_path in checkpoints:
        print(f"Loading encoder checkpoint: {checkpoint_path}")
        tokenizer_for_model, model = build_encoder_model(
            comparison_module,
            approach,
            args.language,
            model_path,
            max_len,
            args.local_files_only,
        )
        if len(tokenizer_for_model) != len(tokenizer):
            raise ValueError("Tokenizer size changed between encoder model loads.")
        checkpoint = torch_load(checkpoint_path, map_location="cpu")
        state_dict = checkpoint.get("model_state_dict", checkpoint)
        model.load_state_dict(state_dict)
        model.to(device)
        model.train()

        with torch.no_grad():
            for batch in iter_with_progress(loader, f"{checkpoint_path.name}", args.progress_every):
                indices = batch.pop("idx").numpy().tolist()
                model_inputs = {k: v.to(device) for k, v in batch.items()}
                for _mc in range(args.num_mc_samples):
                    with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp):
                        outputs = model(**model_inputs)
                        logits = outputs.logits
                    probs = F.softmax(logits.float(), dim=-1).detach().cpu().numpy()
                    pred_ids = np.argmax(probs, axis=1)
                    for row_idx, pred_id, prob in zip(indices, pred_ids, probs):
                        uuid = records[row_idx]["uuid"]
                        results[uuid]["predictions"].append(ID_TO_LABEL[int(pred_id)])
                        results[uuid]["probabilities"].append(prob.astype(np.float64))

        del model, checkpoint, tokenizer_for_model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return results


ASPECT_TAG_START = "<aspect>"
ASPECT_TAG_END = "</aspect>"


class SimplifiedDARTModel(nn.Module):
    def __init__(
        self,
        model_name: str,
        tokenizer_len: int,
        interaction_layers: int,
        interaction_heads: int,
        aggregation_heads: int,
        max_sentences: int,
        final_mlp_hidden_dim: int,
        dropout_rate: float,
        num_classes: int = 3,
        freeze_base: bool = False,
    ):
        super().__init__()
        self.base_model = AutoModel.from_pretrained(model_name)
        self.base_model.resize_token_embeddings(tokenizer_len)
        self.config = self.base_model.config
        self.hidden_dim = self.config.hidden_size
        self.dropout = nn.Dropout(dropout_rate)
        if freeze_base:
            for param in self.base_model.parameters():
                param.requires_grad = False
        self.sentence_pos_embedding = nn.Embedding(max_sentences + 1, self.hidden_dim, padding_idx=0)
        interact_encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.hidden_dim,
            nhead=interaction_heads,
            dim_feedforward=self.hidden_dim * 4,
            dropout=dropout_rate,
            activation="relu",
            batch_first=True,
        )
        self.sentence_interact_transformer = nn.TransformerEncoder(
            interact_encoder_layer,
            num_layers=interaction_layers,
        )
        self.global_aggregation_attention = nn.MultiheadAttention(
            embed_dim=self.hidden_dim,
            num_heads=aggregation_heads,
            dropout=dropout_rate,
            batch_first=True,
        )
        self.classifier = nn.Sequential(
            nn.Linear(self.hidden_dim, final_mlp_hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(final_mlp_hidden_dim, num_classes),
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        sentence_mask: torch.Tensor,
        sentence_position_ids: torch.Tensor,
        aspect_target_token_id: torch.Tensor,
    ) -> torch.Tensor:
        batch_size, num_max_sentences, num_max_tokens = input_ids.shape
        input_ids_flat = input_ids.view(-1, num_max_tokens)
        attention_mask_flat = attention_mask.view(-1, num_max_tokens)
        base_outputs = self.base_model(input_ids=input_ids_flat, attention_mask=attention_mask_flat)
        cls_embeddings_flat = base_outputs.last_hidden_state[:, 0, :]
        cls_embeddings = cls_embeddings_flat.view(batch_size, num_max_sentences, self.hidden_dim)
        pos_embs = self.sentence_pos_embedding(sentence_position_ids)
        cls_embeddings_with_pos = self.dropout(cls_embeddings + pos_embs)
        padding_mask = sentence_mask == 0
        contextualized = self.sentence_interact_transformer(
            cls_embeddings_with_pos,
            src_key_padding_mask=padding_mask,
        )
        contextualized *= sentence_mask.unsqueeze(-1).float()
        aspect_emb = self.base_model.get_input_embeddings()(aspect_target_token_id.to(input_ids.device))
        if aspect_emb.ndim > 2:
            aspect_emb = aspect_emb.squeeze(0)
        if aspect_emb.ndim == 1:
            aspect_emb = aspect_emb.unsqueeze(0)
        global_query = aspect_emb.unsqueeze(0).repeat(batch_size, 1, 1)
        global_attn_output, _ = self.global_aggregation_attention(
            query=global_query,
            key=contextualized,
            value=contextualized,
            key_padding_mask=padding_mask,
        )
        representation = self.dropout(global_attn_output.squeeze(1))
        return self.classifier(representation)


def load_spacy_model(language: str) -> Any:
    if language == "slovenian":
        model_name = "sl_core_news_sm"
    else:
        model_name = "hr_core_news_sm"
    try:
        import spacy

        nlp = spacy.load(model_name)
        if "sentencizer" not in nlp.pipe_names and "senter" not in nlp.pipe_names:
            nlp.add_pipe("sentencizer", first=True)
        return nlp
    except Exception as exc:
        print(
            f"Warning: could not load spaCy model {model_name!r} ({exc}). "
            "Using regex sentence splitting fallback for HAN inference.",
            flush=True,
        )
        return None


def replace_han_aspect_with_placeholder(text: str) -> str:
    start_idx = text.find(ASPECT_TAG_START)
    end_idx = text.find(ASPECT_TAG_END)
    while -1 < start_idx < end_idx:
        text = text[:start_idx] + HAN_ASPECT_PLACEHOLDER + text[end_idx + len(ASPECT_TAG_END) :]
        start_idx = text.find(ASPECT_TAG_START)
        end_idx = text.find(ASPECT_TAG_END)
    return text.replace(ASPECT_TAG_START, "").replace(ASPECT_TAG_END, "")


def split_document(raw_text: str, nlp_model: Any) -> list[str]:
    if not raw_text:
        return []
    if nlp_model is None:
        normalized = re.sub(r"\n+", "\n", raw_text.strip())
        chunks = re.split(r"(?<=[.!?])\s+|\n+", normalized)
        return [chunk.strip() for chunk in chunks if chunk.strip()]
    doc = nlp_model(raw_text.strip())
    return [sent.text.strip() for sent in doc.sents if sent.text.strip()]


def preprocess_han_item(
    item: dict[str, Any],
    tokenizer: Any,
    spacy_nlp: Any,
    max_seq_length: int,
    max_sentences: int,
    use_aspect_marker: bool,
) -> dict[str, torch.Tensor]:
    article = replace_han_aspect_with_placeholder(item.get("article", "") or "")
    sentences = split_document(article, spacy_nlp)[:max_sentences]
    all_input_ids = []
    all_attention_masks = []
    sentence_pos_ids = []
    for sent_idx, sentence in enumerate(sentences):
        if use_aspect_marker:
            text = f"{HAN_ASPECT_PLACEHOLDER} {tokenizer.sep_token} {sentence}"
        else:
            text = sentence
        encoding = tokenizer(
            text,
            add_special_tokens=True,
            max_length=max_seq_length,
            padding="max_length",
            truncation=True,
            return_attention_mask=True,
            return_tensors="pt",
        )
        all_input_ids.append(encoding["input_ids"].squeeze(0))
        all_attention_masks.append(encoding["attention_mask"].squeeze(0))
        sentence_pos_ids.append(sent_idx + 1)
    pad_needed = max_sentences - len(all_input_ids)
    if pad_needed > 0:
        pad_ids = torch.full((max_seq_length,), tokenizer.pad_token_id or 0, dtype=torch.long)
        pad_mask = torch.zeros((max_seq_length,), dtype=torch.long)
        all_input_ids.extend([pad_ids] * pad_needed)
        all_attention_masks.extend([pad_mask] * pad_needed)
        sentence_pos_ids.extend([0] * pad_needed)
    sentence_mask = torch.zeros(max_sentences, dtype=torch.long)
    if sentences:
        sentence_mask[: len(sentences)] = 1
    return {
        "input_ids": torch.stack(all_input_ids),
        "attention_mask": torch.stack(all_attention_masks),
        "sentence_mask": sentence_mask,
        "sentence_position_ids": torch.tensor(sentence_pos_ids, dtype=torch.long),
    }


class HanInferenceDataset(Dataset):
    def __init__(self, records: list[dict[str, Any]], tokenizer: Any, spacy_nlp: Any, run_args: argparse.Namespace):
        self.records = records
        self.tokenizer = tokenizer
        self.spacy_nlp = spacy_nlp
        self.run_args = run_args

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        item = self.records[idx]
        processed = preprocess_han_item(
            item,
            self.tokenizer,
            self.spacy_nlp,
            max_seq_length=getattr(self.run_args, "max_seq_length", 96),
            max_sentences=getattr(self.run_args, "max_sentences", 32),
            use_aspect_marker=getattr(self.run_args, "use_aspect_marker", False),
        )
        processed["idx"] = idx
        return processed


def han_collate(batch: list[dict[str, Any]]) -> dict[str, Any]:
    indices = torch.tensor([item.pop("idx") for item in batch], dtype=torch.long)
    collated = torch.utils.data.default_collate(batch)
    collated["idx"] = indices
    return collated


def build_han_model(run_args: argparse.Namespace, tokenizer_len: int) -> SimplifiedDARTModel:
    return SimplifiedDARTModel(
        model_name=getattr(run_args, "model_name", "xlm-roberta-base"),
        tokenizer_len=tokenizer_len,
        interaction_layers=getattr(run_args, "interaction_layers", 2),
        interaction_heads=getattr(run_args, "interaction_heads", 8),
        aggregation_heads=getattr(run_args, "aggregation_heads", 4),
        max_sentences=getattr(run_args, "max_sentences", 32),
        final_mlp_hidden_dim=getattr(run_args, "final_mlp_hidden_dim", 256),
        dropout_rate=getattr(run_args, "dropout_rate", 0.2),
    )


def run_han_uncertainty(
    args: argparse.Namespace,
    spec: dict[str, Any],
    records: list[dict[str, Any]],
    checkpoints: list[Path],
    device: torch.device,
) -> dict[str, dict[str, list[Any]]]:
    first_checkpoint = torch_load(checkpoints[0], map_location="cpu")
    run_args = argparse.Namespace(**first_checkpoint.get("args", {}))
    model_name = getattr(run_args, "model_name", "xlm-roberta-base")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.add_special_tokens({"additional_special_tokens": [HAN_ASPECT_PLACEHOLDER]})
    aspect_id = tokenizer.convert_tokens_to_ids(HAN_ASPECT_PLACEHOLDER)
    if aspect_id == tokenizer.unk_token_id:
        raise RuntimeError(f"{HAN_ASPECT_PLACEHOLDER} resolved to UNK.")
    aspect_tensor = torch.tensor(aspect_id, device=device)
    spacy_nlp = load_spacy_model(args.language)
    batch_size = args.batch_size or spec["default_batch_size"]
    dataset = HanInferenceDataset(records, tokenizer, spacy_nlp, run_args)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
        collate_fn=han_collate,
    )
    results = {item["uuid"]: {"predictions": [], "probabilities": []} for item in records}
    use_amp = torch.cuda.is_available() and not args.no_amp

    del first_checkpoint
    gc.collect()

    for checkpoint_path in checkpoints:
        print(f"Loading HAN checkpoint: {checkpoint_path}")
        checkpoint = torch_load(checkpoint_path, map_location="cpu")
        model = build_han_model(run_args, len(tokenizer))
        model.load_state_dict(checkpoint["model_state_dict"])
        model.to(device)
        model.train()
        with torch.no_grad():
            for batch in iter_with_progress(loader, f"{checkpoint_path.name}", args.progress_every):
                indices = batch.pop("idx").numpy().tolist()
                model_inputs = {k: v.to(device) for k, v in batch.items()}
                for _mc in range(args.num_mc_samples):
                    with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp):
                        logits = model(**model_inputs, aspect_target_token_id=aspect_tensor)
                    probs = F.softmax(logits.float(), dim=-1).detach().cpu().numpy()
                    pred_ids = np.argmax(probs, axis=1)
                    for row_idx, pred_id, prob in zip(indices, pred_ids, probs):
                        uuid = records[row_idx]["uuid"]
                        results[uuid]["predictions"].append(ID_TO_LABEL[int(pred_id)])
                        results[uuid]["probabilities"].append(prob.astype(np.float64))
        del model, checkpoint
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return results


def run_han_uncertainty_multi(
    args: argparse.Namespace,
    spec: dict[str, Any],
    split_records: dict[str, list[dict[str, Any]]],
    checkpoints: list[Path],
    device: torch.device,
) -> dict[str, dict[str, dict[str, list[Any]]]]:
    first_checkpoint = torch_load(checkpoints[0], map_location="cpu")
    run_args = argparse.Namespace(**first_checkpoint.get("args", {}))
    model_name = getattr(run_args, "model_name", "xlm-roberta-base")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.add_special_tokens({"additional_special_tokens": [HAN_ASPECT_PLACEHOLDER]})
    aspect_id = tokenizer.convert_tokens_to_ids(HAN_ASPECT_PLACEHOLDER)
    if aspect_id == tokenizer.unk_token_id:
        raise RuntimeError(f"{HAN_ASPECT_PLACEHOLDER} resolved to UNK.")
    aspect_tensor = torch.tensor(aspect_id, device=device)
    spacy_nlp = load_spacy_model(args.language)
    batch_size = args.batch_size or spec["default_batch_size"]
    use_amp = torch.cuda.is_available() and not args.no_amp
    loaders = {
        split_name: DataLoader(
            HanInferenceDataset(records, tokenizer, spacy_nlp, run_args),
            batch_size=batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=torch.cuda.is_available(),
            collate_fn=han_collate,
        )
        for split_name, records in split_records.items()
    }
    results = {
        split_name: {item["uuid"]: {"predictions": [], "probabilities": []} for item in records}
        for split_name, records in split_records.items()
    }
    del first_checkpoint
    gc.collect()

    for checkpoint_index, checkpoint_path in enumerate(checkpoints, start=1):
        print(f"Loading HAN checkpoint {checkpoint_index}/{len(checkpoints)}: {checkpoint_path}", flush=True)
        checkpoint = torch_load(checkpoint_path, map_location="cpu")
        model = build_han_model(run_args, len(tokenizer))
        model.load_state_dict(checkpoint["model_state_dict"])
        model.to(device)
        model.train()
        with torch.no_grad():
            for split_name, loader in loaders.items():
                records = split_records[split_name]
                desc = f"{checkpoint_path.name} {split_name}"
                for batch in iter_with_progress(loader, desc, args.progress_every):
                    indices = batch.pop("idx").numpy().tolist()
                    model_inputs = {k: v.to(device) for k, v in batch.items()}
                    for _mc in range(args.num_mc_samples):
                        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp):
                            logits = model(**model_inputs, aspect_target_token_id=aspect_tensor)
                        probs = F.softmax(logits.float(), dim=-1).detach().cpu().numpy()
                        pred_ids = np.argmax(probs, axis=1)
                        for row_idx, pred_id, prob in zip(indices, pred_ids, probs):
                            uuid = records[row_idx]["uuid"]
                            results[split_name][uuid]["predictions"].append(ID_TO_LABEL[int(pred_id)])
                            results[split_name][uuid]["probabilities"].append(prob.astype(np.float64))
        del model, checkpoint
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return results


def is_split_complete(path: Path, json_key: str, expected_samples: int) -> bool:
    if not path.exists():
        return False
    try:
        data = load_json(path)
    except Exception:
        return False
    records: list[dict[str, Any]] = []
    collect_records(data, records)
    if not records:
        return False
    uncertainty_key = f"{json_key}/uncertainty"
    for item in records[: min(20, len(records))]:
        uncertainty = item.get(uncertainty_key)
        if not isinstance(uncertainty, dict):
            return False
        if int(uncertainty.get("total_mc_samples", -1)) != expected_samples:
            return False
    return True


def prepare_split_states(
    args: argparse.Namespace,
    spec: dict[str, Any],
    checkpoints: list[Path],
    output_dir: Path,
) -> tuple[list[dict[str, Any]], dict[str, list[dict[str, Any]]]]:
    json_key = spec["json_key"]
    expected_samples = len(checkpoints) * args.num_mc_samples
    states = []
    active_records = {}
    for split_name in args.splits:
        input_path = input_path_for_split(Path(args.data_dir), args.language, split_name)
        output_path = output_path_for_split(output_dir, args.language, split_name)
        metrics_path = output_path.with_name(output_path.stem + "_metrics.json")
        if args.skip_completed and is_split_complete(output_path, json_key, expected_samples):
            print(f"Skipping complete split: {split_name} -> {output_path}", flush=True)
            states.append(
                {
                    "split": split_name,
                    "input_path": input_path,
                    "output_path": output_path,
                    "metrics_path": metrics_path,
                    "skipped": True,
                    "records": [],
                    "data_copy": None,
                }
            )
            continue

        original_data = load_json(input_path)
        data_copy = copy.deepcopy(original_data)
        records: list[dict[str, Any]] = []
        collect_records(data_copy, records)
        if args.limit_items is not None:
            records = records[: args.limit_items]
        if not records:
            raise ValueError(f"No processable records found in {input_path}")
        print(f"Queued split {split_name}: {len(records)} records from {input_path}", flush=True)
        states.append(
            {
                "split": split_name,
                "input_path": input_path,
                "output_path": output_path,
                "metrics_path": metrics_path,
                "skipped": False,
                "records": records,
                "data_copy": data_copy,
            }
        )
        active_records[split_name] = records
    return states, active_records


def finalize_split_states(
    args: argparse.Namespace,
    spec: dict[str, Any],
    checkpoints: list[Path],
    states: list[dict[str, Any]],
    all_results: dict[str, dict[str, dict[str, list[Any]]]],
) -> list[dict[str, Any]]:
    json_key = spec["json_key"]
    expected_samples = len(checkpoints) * args.num_mc_samples
    split_metrics = []
    for state in states:
        split_name = state["split"]
        if state["skipped"]:
            split_metrics.append(
                {
                    "split": split_name,
                    "output_path": str(state["output_path"]),
                    "skipped": True,
                }
            )
            continue

        records = state["records"]
        annotate_records(records, all_results[split_name], json_key)
        metrics = summarize_records(records, json_key)
        metrics.update(
            {
                "split": split_name,
                "expert": args.expert,
                "language": args.language,
                "json_key": json_key,
                "num_checkpoints": len(checkpoints),
                "num_mc_samples_per_checkpoint": args.num_mc_samples,
                "total_stochastic_samples": expected_samples,
                "input_path": str(state["input_path"]),
                "output_path": str(state["output_path"]),
            }
        )
        if args.limit_items is None:
            write_json(state["output_path"], state["data_copy"])
        else:
            write_json(state["output_path"], {"records": records})
        write_json(state["metrics_path"], metrics)
        split_metrics.append(metrics)
    return split_metrics


def process_split(
    args: argparse.Namespace,
    spec: dict[str, Any],
    split_name: str,
    checkpoints: list[Path],
    device: torch.device,
    output_dir: Path,
) -> dict[str, Any]:
    json_key = spec["json_key"]
    expected_samples = len(checkpoints) * args.num_mc_samples
    input_path = input_path_for_split(Path(args.data_dir), args.language, split_name)
    output_path = output_path_for_split(output_dir, args.language, split_name)
    metrics_path = output_path.with_name(output_path.stem + "_metrics.json")

    if args.skip_completed and is_split_complete(output_path, json_key, expected_samples):
        print(f"Skipping complete split: {split_name} -> {output_path}")
        return {"split": split_name, "output_path": str(output_path), "skipped": True}

    original_data = load_json(input_path)
    data_copy = copy.deepcopy(original_data)
    records: list[dict[str, Any]] = []
    collect_records(data_copy, records)
    if args.limit_items is not None:
        records = records[: args.limit_items]
    if not records:
        raise ValueError(f"No processable records found in {input_path}")

    print(f"Processing {len(records)} records from {input_path}")
    if spec["backend"] == "encoder":
        per_uuid_results = run_encoder_uncertainty(args, spec, records, checkpoints, device)
    elif spec["backend"] == "han_gcm":
        per_uuid_results = run_han_uncertainty(args, spec, records, checkpoints, device)
    else:
        raise ValueError(f"Unknown backend: {spec['backend']}")

    annotate_records(records, per_uuid_results, json_key)
    metrics = summarize_records(records, json_key)
    metrics.update(
        {
            "split": split_name,
            "expert": args.expert,
            "language": args.language,
            "json_key": json_key,
            "num_checkpoints": len(checkpoints),
            "num_mc_samples_per_checkpoint": args.num_mc_samples,
            "total_stochastic_samples": expected_samples,
            "input_path": str(input_path),
            "output_path": str(output_path),
        }
    )

    if args.limit_items is None:
        write_json(output_path, data_copy)
    else:
        limited_output = {"records": records}
        write_json(output_path, limited_output)
    write_json(metrics_path, metrics)
    return metrics


def main() -> None:
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    hf_logging.set_verbosity_error()
    args = parse_args()
    set_seed(args.seed)
    spec = EXPERT_SPECS[args.expert]
    checkpoints = find_checkpoints(spec, args.language, args.checkpoint_limit)
    output_dir = Path(args.output_root) / args.expert / args.language
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("--- Expert uncertainty prediction ---")
    print(f"Expert: {args.expert}")
    print(f"Language: {args.language}")
    print(f"Backend: {spec['backend']}")
    print(f"JSON key: {spec['json_key']}")
    print(f"Output dir: {output_dir}")
    print(f"Device: {device}")
    if torch.cuda.is_available():
        print(f"CUDA device: {torch.cuda.get_device_name(0)}")
    print(f"Checkpoints: {[str(p) for p in checkpoints]}")
    print(f"MC samples per checkpoint: {args.num_mc_samples}")
    print(f"Batch size: {args.batch_size or spec['default_batch_size']}")
    started_at = time.time()

    states, active_records = prepare_split_states(args, spec, checkpoints, output_dir)
    if active_records:
        if spec["backend"] == "encoder":
            all_results = run_encoder_uncertainty_multi(args, spec, active_records, checkpoints, device)
        elif spec["backend"] == "han_gcm":
            all_results = run_han_uncertainty_multi(args, spec, active_records, checkpoints, device)
        else:
            raise ValueError(f"Unknown backend: {spec['backend']}")
    else:
        all_results = {}
    split_metrics = finalize_split_states(args, spec, checkpoints, states, all_results)
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    success = {
        "expert": args.expert,
        "language": args.language,
        "backend": spec["backend"],
        "json_key": spec["json_key"],
        "probabilities_key": f"{spec['json_key']}/probabilities",
        "uncertainty_key": f"{spec['json_key']}/uncertainty",
        "checkpoints": [str(path) for path in checkpoints],
        "num_checkpoints": len(checkpoints),
        "num_mc_samples_per_checkpoint": args.num_mc_samples,
        "total_stochastic_samples": len(checkpoints) * args.num_mc_samples,
        "splits": split_metrics,
        "elapsed_seconds": time.time() - started_at,
    }
    write_json(output_dir / "_SUCCESS.json", success)
    print(f"Wrote success marker: {output_dir / '_SUCCESS.json'}")


if __name__ == "__main__":
    main()
