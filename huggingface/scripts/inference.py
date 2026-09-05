"""Unified single-checkpoint inference for all AspectBench model families."""

from __future__ import annotations

import json
import math
import random
import re
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import (
    AutoConfig,
    AutoModel,
    AutoModelForSeq2SeqLM,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    LongformerConfig,
    LongformerForSequenceClassification,
    PreTrainedTokenizerFast,
)

from model_registry import CHECKPOINTS, MODEL_SPECS, model_spec, weight_path


ASPECT_TOKEN = "[ASPECT]"
HAN_ASPECT_TOKEN = "[ASPECT_TARGET]"
ASPECT_PATTERN = re.compile(r"<aspect>(.*?)</aspect>", flags=re.DOTALL)
ID_TO_SENTIMENT = {0: -1, 1: 0, 2: 1}
SENTIMENT_NAMES = {-1: "negative", 0: "neutral", 1: "positive"}
CLASS_KEYS = ("-1 (negative)", "0 (neutral)", "1 (positive)")
LABEL_TEXTS = ("negative", "neutral", "positive")


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def prepare_record(record: dict[str, Any], mode: str) -> dict[str, Any]:
    if not isinstance(record, dict):
        raise TypeError("Each input record must be a JSON object/dict.")
    article = record.get("article")
    if not isinstance(article, str) or not article.strip():
        raise ValueError("Each record needs a non-empty string field named 'article'.")
    tagged = [value.strip() for value in ASPECT_PATTERN.findall(article)]
    if not tagged or any(not value for value in tagged):
        raise ValueError("The article must contain non-empty <aspect>...</aspect> tags.")
    remainder = ASPECT_PATTERN.sub("", article)
    if "<aspect>" in remainder or "</aspect>" in remainder:
        raise ValueError("Aspect tags are unbalanced or nested.")
    explicit_aspect = record.get("aspect")
    if explicit_aspect is not None and (
        not isinstance(explicit_aspect, str) or not explicit_aspect.strip()
    ):
        raise ValueError("Optional 'aspect' must be a non-empty string.")
    aspect = explicit_aspect.strip() if explicit_aspect else tagged[0]
    gold = record.get("sentiment")
    if gold is not None and gold not in (-1, 0, 1):
        raise ValueError("Optional 'sentiment' must be -1, 0, or 1.")
    if mode == "masked":
        model_article = ASPECT_PATTERN.sub(ASPECT_TOKEN, article).strip()
        model_aspect = ASPECT_TOKEN
    else:
        model_article = article.replace("<aspect>", "").replace("</aspect>", "").strip()
        model_aspect = aspect
    return {
        "article": article,
        "tagged_aspects": tagged,
        "aspect": aspect,
        "model_article": model_article,
        "model_aspect": model_aspect,
        "gold_sentiment": gold,
    }


def load_records(path: str | Path) -> list[dict[str, Any]]:
    input_path = Path(path)
    if input_path.suffix.lower() == ".jsonl":
        with input_path.open("r", encoding="utf-8") as handle:
            records = [json.loads(line) for line in handle if line.strip()]
    else:
        with input_path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        if isinstance(payload, list):
            records = payload
        elif isinstance(payload, dict) and "article" in payload:
            records = [payload]
        elif isinstance(payload, dict):
            records = next(
                (
                    payload[key]
                    for key in ("records", "test", "train", "val")
                    if isinstance(payload.get(key), list)
                ),
                [],
            )
        else:
            records = []
    if not records:
        raise ValueError("Input has no records.")
    return records


def write_json(payload: Any, path: str | Path | None = None) -> None:
    rendered = json.dumps(payload, indent=2, ensure_ascii=False)
    if path is None:
        print(rendered)
    else:
        Path(path).write_text(rendered + "\n", encoding="utf-8")


def _enable_dropout_only(model: nn.Module) -> None:
    model.eval()
    dropout_types = (
        nn.Dropout,
        nn.Dropout1d,
        nn.Dropout2d,
        nn.Dropout3d,
        nn.AlphaDropout,
        nn.FeatureAlphaDropout,
    )
    for module in model.modules():
        if isinstance(module, dropout_types):
            module.train()


def _entropy_bits(probabilities: np.ndarray) -> float:
    safe = np.clip(probabilities.astype(np.float64), 1e-12, 1.0)
    return float(-np.sum(safe * np.log2(safe)))


def _uncertainty(samples: np.ndarray, predictions: list[int]) -> dict[str, Any]:
    mean_probs = samples.mean(axis=0)
    predictive_entropy = _entropy_bits(mean_probs)
    sorted_probs = np.sort(mean_probs)
    result: dict[str, Any] = {
        "predictive_entropy_bits": predictive_entropy,
        "normalized_predictive_entropy": predictive_entropy / math.log2(3),
        "confidence": float(mean_probs.max()),
        "top_two_probability_margin": float(sorted_probs[-1] - sorted_probs[-2]),
    }
    if len(samples) > 1:
        expected_entropy = float(np.mean([_entropy_bits(row) for row in samples]))
        counts = Counter(predictions)
        agreement = max(counts.values()) / len(predictions)
        result["mc_dropout"] = {
            "passes": len(samples),
            "expected_entropy_bits": expected_entropy,
            "mutual_information_bits": max(0.0, predictive_entropy - expected_entropy),
            "prediction_agreement": float(agreement),
            "variation_ratio": float(1.0 - agreement),
            "vote_counts": {
                str(label): int(counts.get(label, 0)) for label in (-1, 0, 1)
            },
        }
    return result


def _resolve_base(
    spec: dict[str, Any],
    base_model_root: str | Path | None,
    bundled_assets: Path | None = None,
) -> str:
    if bundled_assets is not None and (bundled_assets / "config.json").is_file():
        return str(bundled_assets)
    if base_model_root:
        candidate = Path(base_model_root) / spec["local_base_dir"]
        if candidate.is_dir():
            return str(candidate)
    return spec["base_model"]


def _load_tokenizer(base_path: str, model_name: str, language: str, max_length: int):
    path = Path(base_path)
    local_only = path.is_dir()
    if model_name == "slavic-specific" and language == "slovenian" and local_only:
        return PreTrainedTokenizerFast(
            tokenizer_file=str(path / "tokenizer.json"),
            bos_token="<s>",
            eos_token="</s>",
            sep_token="</s>",
            cls_token="<s>",
            unk_token="<unk>",
            pad_token="<pad>",
            mask_token="<mask>",
            model_max_length=max_length,
        )
    return AutoTokenizer.from_pretrained(
        base_path,
        local_files_only=local_only,
        use_fast=True,
        fix_mistral_regex=True,
    )


def _load_state(path: Path) -> dict[str, torch.Tensor]:
    try:
        state = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
    except TypeError:
        state = torch.load(path, map_location="cpu")
    if not isinstance(state, dict) or not state:
        raise ValueError(f"Weight file is not a model state dictionary: {path}")
    return state


def _resize_embeddings(model: nn.Module, size: int) -> None:
    try:
        model.resize_token_embeddings(size, mean_resizing=False)
    except TypeError:
        model.resize_token_embeddings(size)


def _resize_embeddings_to_checkpoint(
    model: nn.Module, state: dict[str, torch.Tensor]
) -> None:
    """Match base-model vocabulary padding used by the original training run."""
    if not hasattr(model, "get_input_embeddings"):
        return
    embedding = model.get_input_embeddings()
    candidates = [
        value
        for key, value in state.items()
        if value.ndim == 2
        and (
            key.endswith("word_embeddings.weight")
            or key == "shared.weight"
            or key.endswith(".shared.weight")
        )
        and value.shape[1] == embedding.weight.shape[1]
    ]
    if not candidates:
        return
    checkpoint_vocab = max(value.shape[0] for value in candidates)
    if checkpoint_vocab == embedding.weight.shape[0]:
        return
    _resize_embeddings(model, checkpoint_vocab)


class SimplifiedDARTModel(nn.Module):
    def __init__(
        self,
        base_config: Any,
        tokenizer_len: int,
        interaction_layers: int = 2,
        interaction_heads: int = 8,
        aggregation_heads: int = 4,
        max_sentences: int = 128,
        final_mlp_hidden_dim: int = 256,
        dropout_rate: float = 0.2,
    ) -> None:
        super().__init__()
        self.base_model = AutoModel.from_config(base_config)
        _resize_embeddings(self.base_model, tokenizer_len)
        self.hidden_dim = base_config.hidden_size
        self.dropout = nn.Dropout(dropout_rate)
        self.sentence_pos_embedding = nn.Embedding(
            max_sentences + 1, self.hidden_dim, padding_idx=0
        )
        layer = nn.TransformerEncoderLayer(
            d_model=self.hidden_dim,
            nhead=interaction_heads,
            dim_feedforward=self.hidden_dim * 4,
            dropout=dropout_rate,
            activation="relu",
            batch_first=True,
        )
        self.sentence_interact_transformer = nn.TransformerEncoder(
            layer, num_layers=interaction_layers, enable_nested_tensor=False
        )
        self.global_aggregation_attention = nn.MultiheadAttention(
            self.hidden_dim, aggregation_heads, dropout=dropout_rate, batch_first=True
        )
        self.classifier = nn.Sequential(
            nn.Linear(self.hidden_dim, final_mlp_hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(final_mlp_hidden_dim, 3),
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        sentence_mask: torch.Tensor,
        sentence_position_ids: torch.Tensor,
        aspect_target_token_id: torch.Tensor,
    ) -> torch.Tensor:
        batch_size, sentence_count, token_count = input_ids.shape
        outputs = self.base_model(
            input_ids=input_ids.view(-1, token_count),
            attention_mask=attention_mask.view(-1, token_count),
        )
        cls = outputs.last_hidden_state[:, 0, :].view(
            batch_size, sentence_count, self.hidden_dim
        )
        cls = self.dropout(cls + self.sentence_pos_embedding(sentence_position_ids))
        padding_mask = sentence_mask == 0
        contextualized = self.sentence_interact_transformer(
            cls, src_key_padding_mask=padding_mask
        )
        contextualized *= sentence_mask.unsqueeze(-1).float()
        aspect_embedding = self.base_model.get_input_embeddings()(
            aspect_target_token_id.to(input_ids.device)
        )
        if aspect_embedding.ndim == 1:
            aspect_embedding = aspect_embedding.unsqueeze(0)
        query = aspect_embedding.unsqueeze(0).repeat(batch_size, 1, 1)
        aggregated, _ = self.global_aggregation_attention(
            query=query,
            key=contextualized,
            value=contextualized,
            key_padding_mask=padding_mask,
        )
        return self.classifier(self.dropout(aggregated.squeeze(1)))


def _load_spacy(language: str):
    model_name = "sl_core_news_sm" if language == "slovenian" else "hr_core_news_sm"
    try:
        import spacy

        nlp = spacy.load(
            model_name,
            disable=["tagger", "parser", "attribute_ruler", "lemmatizer", "ner"],
        )
        if "sentencizer" not in nlp.pipe_names and "senter" not in nlp.pipe_names:
            nlp.add_pipe("sentencizer")
        return nlp
    except Exception as exc:
        print(f"Warning: {model_name} unavailable ({exc}); using regex sentence splitting.")
        return None


def _sentences(text: str, nlp: Any) -> list[str]:
    if nlp is not None:
        return [sentence.text.strip() for sentence in nlp(text.strip()).sents if sentence.text.strip()]
    return [
        part.strip()
        for part in re.split(r"(?<=[.!?])\s+|\n+", text.strip())
        if part.strip()
    ]


class InferenceEngine:
    def __init__(
        self,
        model_name: str,
        language: str,
        mode: str,
        model_root: str | Path,
        base_model_root: str | Path | None = None,
        device: str = "auto",
    ) -> None:
        if model_name not in MODEL_SPECS:
            raise ValueError(f"Unknown model: {model_name}")
        if language not in ("hbs", "slovenian") or mode not in ("masked", "unmasked"):
            raise ValueError("language must be hbs/slovenian and mode masked/unmasked")
        selection = CHECKPOINTS[(model_name, language, mode)]
        self.weight_path = weight_path(model_root, model_name, language, mode)
        if not selection["available"] and not self.weight_path.is_file():
            raise FileNotFoundError(selection["unavailable_reason"])
        if not self.weight_path.is_file():
            raise FileNotFoundError(self.weight_path)
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        if device.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but unavailable.")
        self.device = torch.device(device)
        self.model_name = model_name
        self.language = language
        self.mode = mode
        self.spec = model_spec(model_name, language)
        self.base_path = _resolve_base(
            self.spec,
            base_model_root,
            bundled_assets=self.weight_path.parent / "base_model",
        )
        self.backend = self.spec["backend"]
        self.tokenizer = _load_tokenizer(
            self.base_path, model_name, language, self.spec["max_length"]
        )
        self.spacy_nlp = None
        self.aspect_token_id = None
        self._load_model()

    def _load_model(self) -> None:
        local_only = Path(self.base_path).is_dir()
        config = AutoConfig.from_pretrained(self.base_path, local_files_only=local_only)
        state = _load_state(self.weight_path)
        if self.backend == "han":
            self.tokenizer.add_special_tokens(
                {"additional_special_tokens": [HAN_ASPECT_TOKEN]}
            )
            model = SimplifiedDARTModel(
                config,
                len(self.tokenizer),
                max_sentences=self.spec["max_sentences"],
            )
            self.aspect_token_id = self.tokenizer.convert_tokens_to_ids(HAN_ASPECT_TOKEN)
            self.spacy_nlp = _load_spacy(self.language)
        elif self.backend == "mt5":
            model = AutoModelForSeq2SeqLM.from_config(config)
            if self.mode == "masked":
                self.tokenizer.add_special_tokens(
                    {"additional_special_tokens": [ASPECT_TOKEN]}
                )
                _resize_embeddings(model, len(self.tokenizer))
        elif self.backend == "bge":
            self._load_bge(state)
            return
        elif self.model_name == "longformer":
            config_dict = config.to_dict()
            config_dict.update(
                num_labels=3,
                problem_type="single_label_classification",
                id2label={0: "negative", 1: "neutral", 2: "positive"},
                label2id={"negative": 0, "neutral": 1, "positive": 2},
            )
            long_config = LongformerConfig(**config_dict)
            if isinstance(long_config.attention_window, int):
                long_config.attention_window = [
                    long_config.attention_window
                ] * long_config.num_hidden_layers
            model = LongformerForSequenceClassification(long_config)
            if self.mode == "masked":
                self.tokenizer.add_special_tokens(
                    {"additional_special_tokens": [ASPECT_TOKEN]}
                )
                _resize_embeddings(model, len(self.tokenizer))
        else:
            config.num_labels = 3
            config.problem_type = "single_label_classification"
            config.id2label = {0: "negative", 1: "neutral", 2: "positive"}
            config.label2id = {"negative": 0, "neutral": 1, "positive": 2}
            model = AutoModelForSequenceClassification.from_config(config)
            if self.mode == "masked":
                self.tokenizer.add_special_tokens(
                    {"additional_special_tokens": [ASPECT_TOKEN]}
                )
                _resize_embeddings(model, len(self.tokenizer))
        _resize_embeddings_to_checkpoint(model, state)
        model.load_state_dict(state, strict=True)
        del state
        self.model = model.to(self.device)
        self.model.eval()

    def _load_bge(self, state: dict[str, torch.Tensor]) -> None:
        from sentence_transformers import SentenceTransformer

        self.embedding_model = SentenceTransformer(self.base_path, device=str(self.device))
        linear_weights = [
            (key, value)
            for key, value in state.items()
            if key.endswith(".weight") and value.ndim == 2
        ]
        linear_weights.sort(key=lambda item: item[0])
        if len(linear_weights) != 3:
            raise ValueError("Expected three BGE MLP linear layers.")
        dims = [linear_weights[0][1].shape[1]] + [value.shape[0] for _, value in linear_weights]
        model = nn.Sequential(
            nn.Linear(dims[0], dims[1]),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(dims[1], dims[2]),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(dims[2], dims[3]),
        )
        cleaned = {
            key.removeprefix("classifier."): value for key, value in state.items()
        }
        model.load_state_dict(cleaned, strict=True)
        self.model = model.to(self.device)
        self.model.eval()

    def _encoder_inputs(self, prepared: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        tokenizer_kwargs: dict[str, Any] = {}
        if self.spec.get("global_attention"):
            attention_window = self.spec.get("attention_window", 512)
            tokenizer_kwargs["pad_to_multiple_of"] = attention_window
        encoded = self.tokenizer(
            [item["model_article"] for item in prepared],
            text_pair=[item["model_aspect"] for item in prepared],
            truncation=True,
            max_length=self.spec["max_length"],
            padding=True,
            return_tensors="pt",
            **tokenizer_kwargs,
        )
        if self.spec.get("global_attention"):
            global_attention = torch.zeros_like(encoded["attention_mask"])
            global_attention[:, 0] = 1
            encoded["global_attention_mask"] = global_attention
        return {key: value.to(self.device) for key, value in encoded.items()}

    def _han_inputs(self, prepared: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        max_sentences = self.spec["max_sentences"]
        max_length = self.spec["max_length"]
        rows = []
        for item in prepared:
            article = ASPECT_PATTERN.sub(HAN_ASPECT_TOKEN, item["article"])
            sentences = _sentences(article, self.spacy_nlp)[:max_sentences]
            ids, masks, positions = [], [], []
            for index, sentence in enumerate(sentences):
                text = f"{HAN_ASPECT_TOKEN} {self.tokenizer.sep_token} {sentence}"
                encoded = self.tokenizer(
                    text,
                    add_special_tokens=True,
                    max_length=max_length,
                    padding="max_length",
                    truncation=True,
                    return_tensors="pt",
                )
                ids.append(encoded["input_ids"].squeeze(0))
                masks.append(encoded["attention_mask"].squeeze(0))
                positions.append(index + 1)
            pad_count = max_sentences - len(ids)
            pad_ids = torch.full(
                (max_length,), self.tokenizer.pad_token_id or 0, dtype=torch.long
            )
            pad_mask = torch.zeros(max_length, dtype=torch.long)
            ids.extend([pad_ids] * pad_count)
            masks.extend([pad_mask] * pad_count)
            positions.extend([0] * pad_count)
            sentence_mask = torch.zeros(max_sentences, dtype=torch.long)
            sentence_mask[: len(sentences)] = 1
            rows.append(
                {
                    "input_ids": torch.stack(ids),
                    "attention_mask": torch.stack(masks),
                    "sentence_mask": sentence_mask,
                    "sentence_position_ids": torch.tensor(positions, dtype=torch.long),
                }
            )
        return {
            key: torch.stack([row[key] for row in rows]).to(self.device) for key in rows[0]
        }

    def _mt5_sources(self, prepared: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        sources = [
            f"classify sentiment\naspect: {item['model_aspect']}\narticle: {item['model_article']}"
            for item in prepared
        ]
        encoded = self.tokenizer(
            sources,
            truncation=True,
            max_length=self.spec["max_length"],
            padding=True,
            return_tensors="pt",
        )
        return {key: value.to(self.device) for key, value in encoded.items()}

    def _mt5_probabilities(self, inputs: dict[str, torch.Tensor]) -> torch.Tensor:
        losses = []
        batch_size = inputs["input_ids"].shape[0]
        for label in LABEL_TEXTS:
            ids = self.tokenizer(
                label,
                add_special_tokens=True,
                truncation=True,
                max_length=self.spec["max_target_length"],
                return_tensors="pt",
            )["input_ids"][0].to(self.device)
            labels = ids.unsqueeze(0).repeat(batch_size, 1)
            decoder_input_ids = self.model._shift_right(labels)
            logits = self.model(
                **inputs, decoder_input_ids=decoder_input_ids, use_cache=False
            ).logits
            token_losses = F.cross_entropy(
                logits.transpose(1, 2), labels, reduction="none"
            )
            losses.append(token_losses.mean(dim=1))
        return torch.softmax(-torch.stack(losses, dim=1), dim=1)

    def _bge_inputs(self, prepared: list[dict[str, Any]]) -> torch.Tensor:
        texts = []
        for item in prepared:
            if self.mode == "masked":
                text = ASPECT_PATTERN.sub("[ASPECT_MENTION]", item["article"])
                text = f"{text.strip()} [ASPECT_NAME]"
            else:
                text = item["article"].replace("<aspect>", "").replace("</aspect>", "")
            texts.append(text.strip())
        embeddings = self.embedding_model.encode(texts, normalize_embeddings=True)
        return torch.tensor(embeddings, dtype=torch.float32, device=self.device)

    def _sample_probabilities(
        self, prepared: list[dict[str, Any]], passes: int, seed: int
    ) -> np.ndarray:
        if passes == 1 or passes < 0:
            raise ValueError("mc_passes must be 0 or at least 2.")
        count = passes or 1
        if passes:
            _seed_everything(seed)
            _enable_dropout_only(self.model)
        else:
            self.model.eval()
        if self.backend == "encoder":
            inputs = self._encoder_inputs(prepared)
        elif self.backend == "han":
            inputs = self._han_inputs(prepared)
        elif self.backend == "mt5":
            inputs = self._mt5_sources(prepared)
        else:
            inputs = self._bge_inputs(prepared)
        rows = []
        with torch.inference_mode():
            for _ in range(count):
                if self.backend == "encoder":
                    probs = torch.softmax(self.model(**inputs).logits, dim=-1)
                elif self.backend == "han":
                    logits = self.model(
                        **inputs,
                        aspect_target_token_id=torch.tensor(
                            [self.aspect_token_id], device=self.device
                        ),
                    )
                    probs = torch.softmax(logits, dim=-1)
                elif self.backend == "mt5":
                    probs = self._mt5_probabilities(inputs)
                else:
                    probs = torch.softmax(self.model(inputs), dim=-1)
                rows.append(probs.float().cpu().numpy())
        self.model.eval()
        return np.stack(rows, axis=0)

    def predict_batch(
        self,
        records: Iterable[dict[str, Any]],
        batch_size: int = 8,
        mc_passes: int = 0,
        seed: int = 42,
    ) -> list[dict[str, Any]]:
        records_list = list(records)
        if not records_list or batch_size < 1:
            raise ValueError("Input must be non-empty and batch_size >= 1.")
        outputs = []
        for start in range(0, len(records_list), batch_size):
            prepared = [
                prepare_record(record, self.mode)
                for record in records_list[start : start + batch_size]
            ]
            samples = self._sample_probabilities(prepared, mc_passes, seed + start)
            for index, item in enumerate(prepared):
                item_samples = samples[:, index, :]
                mean_probs = item_samples.mean(axis=0)
                sample_labels = [
                    ID_TO_SENTIMENT[int(value)] for value in item_samples.argmax(axis=1)
                ]
                predicted = ID_TO_SENTIMENT[int(mean_probs.argmax())]
                outputs.append(
                    {
                        "model": self.model_name,
                        "language": self.language,
                        "mode": self.mode,
                        "input_article": item["article"],
                        "tagged_aspects": item["tagged_aspects"],
                        "aspect_used": item["model_aspect"],
                        "gold_sentiment": item["gold_sentiment"],
                        "predicted_sentiment": predicted,
                        "predicted_sentiment_name": SENTIMENT_NAMES[predicted],
                        "class_probabilities": {
                            key: float(mean_probs[class_id])
                            for class_id, key in enumerate(CLASS_KEYS)
                        },
                        "uncertainty_across_classes": _uncertainty(
                            item_samples, sample_labels
                        ),
                        "inference": {
                            "device": str(self.device),
                            "mc_dropout": bool(mc_passes),
                            "weight_file": str(self.weight_path),
                        },
                    }
                )
        return outputs

    def predict(
        self, record: dict[str, Any], mc_passes: int = 0, seed: int = 42
    ) -> dict[str, Any]:
        return self.predict_batch(
            [record], batch_size=1, mc_passes=mc_passes, seed=seed
        )[0]
