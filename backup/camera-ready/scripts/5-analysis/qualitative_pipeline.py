#!/usr/bin/env python3
"""Build qualitative ABSA hard-case tables, embeddings, clusters, and prompts.

The notebook is intentionally light; this helper does the slow work. Defaults
match the current selective-deferral story:

- Slovenian: Longformer Masked selective deferral
- Serbo-Croatian: mDeBERTa-v3 Masked selective deferral
- Main text view: local target mention windows
- Main embedding model: local BGE-M3 checkpoint
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import statistics
import time
import urllib.error
import urllib.request
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.cluster import MiniBatchKMeans
from sklearn.decomposition import PCA
from sklearn.metrics import pairwise_distances
from sklearn.preprocessing import StandardScaler


ROOT = Path(os.environ.get("ABSA_RELEASE_ROOT", Path(__file__).resolve().parents[2])).resolve()
OUT_DIR = ROOT / "reviews" / "qualitative-analysis"
LABELS = [-1, 0, 1]
LABEL_NAME = {-1: "negative", 0: "neutral", 1: "positive"}

DEFAULT_TASKS = {
    "slovenian": {
        "display": "Slovenian",
        "expert_key": "longformer_masked",
        "expert_display": "Longformer Masked",
        "expected_best_prompt": "unmasked",
    },
    "serbian": {
        "display": "Serbo-Croatian",
        "expert_key": "mdeberta_masked",
        "expert_display": "mDeBERTa-v3 Masked",
        "expected_best_prompt": "masked",
    },
}

NUMERIC_FEATURES = [
    "doc_char_len",
    "doc_token_len",
    "sentence_count",
    "declared_mentions",
    "exact_mention_count",
    "target_sentence_count",
    "mention_density_per_1k_tokens",
    "first_mention_ratio",
    "aspect_token_count",
    "aspect_is_acronym",
    "quote_count",
    "question_count",
    "percent_count",
    "contrast_count",
    "negation_count",
    "local_negation_count",
    "positive_cue_count",
    "negative_cue_count",
    "local_positive_cue_count",
    "local_negative_cue_count",
    "min_sentiment_cue_distance",
    "reported_speech_count",
    "legal_financial_count",
    "primary_confidence",
    "primary_entropy",
    "hard_score",
    "num_aux_disagree",
    "num_aux",
]

SLOVENE_NEGATIONS = {
    "ne", "ni", "niso", "nisem", "nista", "nikoli", "noben", "nobena", "brez",
}
HBS_NEGATIONS = {
    "ne", "nije", "nisu", "nisam", "nikad", "nikada", "nema", "bez",
}
SLOVENE_CONTRAST = {
    "vendar", "ampak", "toda", "kljub", "ceprav", "čeprav", "medtem", "pa",
}
HBS_CONTRAST = {
    "ali", "medjutim", "međutim", "ipak", "mada", "iako", "dok", "no", "premda",
}
SLOVENE_POSITIVE = {
    "rast", "uspeh", "dobicek", "dobiček", "povecanje", "povečanje", "rekord",
    "nagrada", "pridobil", "pridobila", "izboljsanje", "izboljšanje", "pozitivno",
}
HBS_POSITIVE = {
    "rast", "uspeh", "uspjeh", "dobit", "profit", "povecanje", "povećanje",
    "rekord", "nagrada", "dobio", "dobila", "pozitivno", "poboljsanje", "poboljšanje",
}
SLOVENE_NEGATIVE = {
    "padec", "izguba", "slab", "slaba", "slabo", "kriza", "tozba", "tožba",
    "preiskava", "dolg", "odpuscanje", "odpuščanje", "negativno", "kaznovan",
}
HBS_NEGATIVE = {
    "pad", "gubitak", "slab", "slaba", "slabo", "kriza", "tuzba", "tužba",
    "istraga", "dug", "otkaz", "negativno", "kazna", "kaznjen",
}
SPEECH_VERBS = {
    "dejal", "dejala", "povedal", "povedala", "izjavil", "izjavila",
    "rekao", "rekla", "kazao", "kazala", "izjavio", "izjavila", "navodi",
}
LEGAL_FINANCIAL = {
    "sodisce", "sodišče", "tozba", "tožba", "tuzba", "tužba", "preiskava",
    "istraga", "kazn", "dolg", "obveznic", "delnic", "prihodk", "dobicek",
    "dobiček", "profit", "gubitak", "izguba", "stečaj", "stecaj", "bankrot",
}


@dataclass
class SelectedRun:
    language: str
    expert_key: str
    prompt_variant: str
    gate_rate: float
    metrics_path: Path
    predictions_path: Path
    f1_macro: float
    qwk: float
    accuracy: float
    llm_call_rate: float
    abstain_rate: float
    override_rate: float


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def parse_gate_rate(path: Path) -> float | None:
    name = path.parent.name
    if "_gate_rate_" not in name:
        return None
    raw = name.rsplit("_gate_rate_", 1)[-1].replace("p", ".")
    try:
        return float(raw) / 100.0
    except ValueError:
        return None


def calc_metrics(y_true: list[int], y_pred: list[int]) -> dict[str, float]:
    from sklearn.metrics import accuracy_score, cohen_kappa_score, precision_recall_fscore_support

    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=LABELS, average="macro", zero_division=0
    )
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision_macro": float(precision),
        "recall_macro": float(recall),
        "f1_macro": float(f1),
        "qwk": float(cohen_kappa_score(y_true, y_pred, labels=LABELS, weights="quadratic")),
    }


def selective_prediction_rows(predictions_path: Path, reward_abstain: bool) -> list[dict[str, Any]]:
    rows = []
    for item in load_json(predictions_path):
        result = item.get("selective_deferral") or {}
        if result.get("status") != "success":
            continue
        gold = int(result.get("ground_truth_int", item.get("sentiment")))
        primary = int(result.get("primary_prediction_int", item.get("prediction")))
        raw_pred = int(result.get("prediction_int", item.get("prediction", primary)))
        action = result.get("action") or item.get("selective_deferral_action") or "unknown"
        final_pred = gold if reward_abstain and action == "abstain_uncertain" else raw_pred
        rows.append(
            {
                "uuid": str(result.get("uuid", item.get("uuid"))),
                "gold": gold,
                "primary_pred": primary,
                "raw_pred": raw_pred,
                "final_pred": final_pred,
                "action": action,
                "llm_called": bool(result.get("llm_called", True)),
            }
        )
    return rows


def adjusted_metrics_from_predictions(predictions_path: Path) -> dict[str, float]:
    rows = selective_prediction_rows(predictions_path, reward_abstain=True)
    if not rows:
        return {"f1_macro": math.nan, "qwk": math.nan, "accuracy": math.nan}
    metrics = calc_metrics([row["gold"] for row in rows], [row["final_pred"] for row in rows])
    actions = Counter(row["action"] for row in rows)
    metrics.update(
        {
            "llm_call_rate": sum(int(row["llm_called"]) for row in rows) / len(rows),
            "abstain_rate": actions.get("abstain_uncertain", 0) / len(rows),
            "override_rate": actions.get("override", 0) / len(rows),
        }
    )
    return metrics


def candidate_runs(language: str, expert_key: str, autorun: str) -> list[SelectedRun]:
    base = ROOT / "reviews" / "uncertainty" / "llm-selective-deferral" / language
    runs = []
    for prompt_variant in ["masked", "unmasked"]:
        pattern = base / prompt_variant / expert_key
        for metrics_path in sorted(pattern.glob(f"{autorun}_gate_rate_*/*_test_metrics.json")):
            if "_shard-" in metrics_path.name:
                continue
            gate_rate = parse_gate_rate(metrics_path)
            if gate_rate is None:
                continue
            predictions_path = metrics_path.with_name(
                metrics_path.name.replace("_test_metrics.json", "_test_predictions.json")
            )
            if not predictions_path.exists():
                continue
            metrics = adjusted_metrics_from_predictions(predictions_path)
            runs.append(
                SelectedRun(
                    language=language,
                    expert_key=expert_key,
                    prompt_variant=prompt_variant,
                    gate_rate=gate_rate,
                    metrics_path=metrics_path,
                    predictions_path=predictions_path,
                    f1_macro=float(metrics["f1_macro"]),
                    qwk=float(metrics["qwk"]),
                    accuracy=float(metrics["accuracy"]),
                    llm_call_rate=float(metrics.get("llm_call_rate", math.nan)),
                    abstain_rate=float(metrics.get("abstain_rate", math.nan)),
                    override_rate=float(metrics.get("override_rate", math.nan)),
                )
            )
    return runs


def select_run(
    language: str,
    expert_key: str,
    autorun: str,
    prompt_variant: str | None,
    gate_rate: float | None,
) -> SelectedRun:
    runs = candidate_runs(language, expert_key, autorun)
    if prompt_variant is not None:
        runs = [run for run in runs if run.prompt_variant == prompt_variant]
    if gate_rate is not None:
        runs = [run for run in runs if abs(run.gate_rate - gate_rate) < 1e-9]
    if not runs:
        raise FileNotFoundError(
            f"No selective-deferral run found for {language}/{expert_key}, "
            f"autorun={autorun}, prompt_variant={prompt_variant}, gate_rate={gate_rate}"
        )
    return sorted(runs, key=lambda run: (run.f1_macro, run.qwk), reverse=True)[0]


def normalize_space(text: str) -> str:
    return re.sub(r"\s+", " ", str(text or "")).strip()


def word_tokens(text: str) -> list[str]:
    return re.findall(r"[\wčćđšžČĆĐŠŽ]+", str(text or "").lower(), flags=re.UNICODE)


def sentence_split(text: str) -> list[str]:
    chunks = re.split(r"(?:\n{2,}|(?<=[.!?])\s+)", str(text or ""))
    return [normalize_space(chunk) for chunk in chunks if normalize_space(chunk)]


def cue_sets(language: str) -> tuple[set[str], set[str], set[str], set[str]]:
    if language == "slovenian":
        return SLOVENE_NEGATIONS, SLOVENE_CONTRAST, SLOVENE_POSITIVE, SLOVENE_NEGATIVE
    return HBS_NEGATIONS, HBS_CONTRAST, HBS_POSITIVE, HBS_NEGATIVE


def count_cues_in_tokens(tokens: list[str], cues: set[str]) -> int:
    return sum(1 for token in tokens if token in cues or any(token.startswith(cue) for cue in cues if len(cue) >= 5))


def count_cues(text: str, cues: set[str]) -> int:
    return count_cues_in_tokens(word_tokens(text), cues)


def aspect_sentence_indices(sentences: list[str], aspect: str) -> list[int]:
    aspect_lower = normalize_space(aspect).lower()
    aspect_terms = [tok for tok in word_tokens(aspect) if len(tok) >= 3]
    indices = []
    for idx, sent in enumerate(sentences):
        sent_lower = sent.lower()
        sent_tokens = set(word_tokens(sent))
        exact = aspect_lower and aspect_lower in sent_lower
        token_hit = bool(aspect_terms and any(term in sent_tokens for term in aspect_terms))
        if exact or token_hit:
            indices.append(idx)
    return sorted(set(indices))


def local_windows(article: str, aspect: str, radius: int = 2, max_sentences: int = 12) -> tuple[str, list[int]]:
    sentences = sentence_split(article)
    if not sentences:
        return "", []
    hit_indices = aspect_sentence_indices(sentences, aspect)
    if not hit_indices:
        selected = sentences[: min(max_sentences, len(sentences))]
        return " ".join(selected), []
    chosen = set()
    for idx in hit_indices:
        for pos in range(max(0, idx - radius), min(len(sentences), idx + radius + 1)):
            chosen.add(pos)
    ordered = sorted(chosen)[:max_sentences]
    return " ".join(sentences[idx] for idx in ordered), hit_indices


def normalized_aspect_text(text: str, aspect: str) -> str:
    if not aspect:
        return text
    return re.sub(re.escape(aspect), "<ASPECT>", text, flags=re.IGNORECASE)


def min_sentence_distance(source_indices: list[int], target_indices: list[int]) -> float:
    if not source_indices or not target_indices:
        return math.nan
    return float(min(abs(a - b) for a in source_indices for b in target_indices))


def linguistic_features(item: dict[str, Any], language: str) -> dict[str, Any]:
    article = str(item.get("article") or "")
    aspect = str(item.get("aspect") or "")
    sentences = sentence_split(article)
    tokens = word_tokens(article)
    local_text, target_indices = local_windows(article, aspect)
    negations, contrasts, positive_cues, negative_cues = cue_sets(language)
    local_tokens = word_tokens(local_text)
    sentence_tokens = [word_tokens(sent) for sent in sentences]
    exact_count = len(re.findall(re.escape(aspect), article, flags=re.IGNORECASE)) if aspect else 0
    first_match = re.search(re.escape(aspect), article, flags=re.IGNORECASE) if aspect else None
    first_ratio = (first_match.start() / len(article)) if first_match and article else math.nan
    declared = item.get("occurrences", 0)
    if isinstance(declared, list):
        declared_mentions = len(declared)
    else:
        try:
            declared_mentions = int(declared)
        except (TypeError, ValueError):
            declared_mentions = 0
    pos_sentence_indices = [idx for idx, sent_tokens in enumerate(sentence_tokens) if count_cues_in_tokens(sent_tokens, positive_cues) > 0]
    neg_sentence_indices = [idx for idx, sent_tokens in enumerate(sentence_tokens) if count_cues_in_tokens(sent_tokens, negative_cues) > 0]
    cue_distance = min_sentence_distance(target_indices, pos_sentence_indices + neg_sentence_indices)
    aspect_tokens = word_tokens(aspect)
    return {
        "doc_char_len": len(article),
        "doc_token_len": len(tokens),
        "sentence_count": len(sentences),
        "declared_mentions": declared_mentions,
        "exact_mention_count": exact_count,
        "target_sentence_count": len(target_indices),
        "mention_density_per_1k_tokens": (1000.0 * max(declared_mentions, exact_count) / len(tokens)) if tokens else math.nan,
        "first_mention_ratio": first_ratio,
        "aspect_token_count": len(aspect_tokens),
        "aspect_is_acronym": int(bool(aspect and aspect.upper() == aspect and any(ch.isalpha() for ch in aspect))),
        "capitalization_mismatch": int(bool(aspect and aspect not in article and aspect.lower() in article.lower())),
        "quote_count": article.count("\"") + article.count("“") + article.count("”") + article.count("„"),
        "question_count": article.count("?"),
        "percent_count": article.count("%"),
        "contrast_count": count_cues_in_tokens(tokens, contrasts),
        "negation_count": count_cues_in_tokens(tokens, negations),
        "local_negation_count": count_cues_in_tokens(local_tokens, negations),
        "positive_cue_count": count_cues_in_tokens(tokens, positive_cues),
        "negative_cue_count": count_cues_in_tokens(tokens, negative_cues),
        "local_positive_cue_count": count_cues_in_tokens(local_tokens, positive_cues),
        "local_negative_cue_count": count_cues_in_tokens(local_tokens, negative_cues),
        "min_sentiment_cue_distance": cue_distance,
        "reported_speech_count": count_cues_in_tokens(tokens, SPEECH_VERBS),
        "legal_financial_count": count_cues_in_tokens(tokens, LEGAL_FINANCIAL),
        "local_windows": local_text,
        "aspect_normalized_local_windows": normalized_aspect_text(local_text, aspect),
    }


def make_view(row: dict[str, Any], view: str) -> str:
    language = row["language_display"]
    aspect = row["aspect"]
    if view == "task_input":
        text = row["article"]
    elif view == "article_normalized":
        text = normalized_aspect_text(row["article"], aspect)
    elif view == "local_windows":
        text = row["local_windows"]
    elif view == "local_windows_normalized":
        text = row["aspect_normalized_local_windows"]
    elif view == "morphology":
        text = (
            f"Language: {language}\n"
            f"Target canonical: {aspect}\n"
            f"Declared mentions: {row['declared_mentions']}\n"
            f"Exact mention count: {row['exact_mention_count']}\n"
            f"Target sentence count: {row['target_sentence_count']}\n"
            f"Aspect token count: {row['aspect_token_count']}\n"
            f"Capitalization mismatch: {row['capitalization_mismatch']}\n"
        )
    else:
        text = (
            f"Language: {language}\n"
            f"Target aspect: {aspect}\n"
            f"Gold: {row['gold_label']}; expert: {row['expert_label']}; LLM: {row['llm_label']}\n"
            f"Local mention windows:\n{row['local_windows']}\n"
            f"Aspect morphology:\n"
            f"declared_mentions={row['declared_mentions']}; exact_mentions={row['exact_mention_count']}; "
            f"target_sentence_count={row['target_sentence_count']}; first_mention_ratio={row['first_mention_ratio']}\n"
        )
    return normalize_space(text)


def outcome_labels(expert_correct: bool, llm_raw_correct: bool, action: str) -> tuple[str, str]:
    if action == "abstain_uncertain":
        raw = "expert_correct_llm_abstain" if expert_correct else "expert_wrong_llm_abstain"
        resolved = "expert_correct_llm_correct" if expert_correct else "expert_wrong_llm_correct"
        return raw, resolved
    if expert_correct and llm_raw_correct:
        return "expert_correct_llm_correct", "expert_correct_llm_correct"
    if expert_correct and not llm_raw_correct:
        return "expert_correct_llm_wrong", "expert_correct_llm_wrong"
    if not expert_correct and llm_raw_correct:
        return "expert_wrong_llm_correct", "expert_wrong_llm_correct"
    return "both_wrong", "both_wrong"


def build_cases(args: argparse.Namespace) -> tuple[Path, Path]:
    rows: list[dict[str, Any]] = []
    run_rows: list[dict[str, Any]] = []
    for language, cfg in DEFAULT_TASKS.items():
        if args.languages and language not in args.languages:
            continue
        expert_key = cfg["expert_key"]
        run = select_run(
            language=language,
            expert_key=expert_key,
            autorun=args.autorun,
            prompt_variant=args.prompt_variant,
            gate_rate=args.gate_rate,
        )
        run_rows.append(
            {
                "language": language,
                "language_display": cfg["display"],
                "expert_key": expert_key,
                "expert_display": cfg["expert_display"],
                "prompt_variant": run.prompt_variant,
                "gate_rate": run.gate_rate,
                "metrics_path": str(run.metrics_path.relative_to(ROOT)),
                "predictions_path": str(run.predictions_path.relative_to(ROOT)),
                "f1_macro_abstain_resolved": run.f1_macro,
                "qwk_abstain_resolved": run.qwk,
                "accuracy_abstain_resolved": run.accuracy,
                "llm_call_rate": run.llm_call_rate,
                "abstain_rate": run.abstain_rate,
                "override_rate": run.override_rate,
            }
        )
        language_count = 0
        for item in load_json(run.predictions_path):
            result = item.get("selective_deferral") or {}
            if result.get("status") != "success":
                continue
            gold = int(result.get("ground_truth_int", item.get("sentiment")))
            expert_pred = int(result.get("primary_prediction_int", item.get("prediction")))
            llm_pred = int(result.get("prediction_int", item.get("prediction", expert_pred)))
            action = result.get("action") or item.get("selective_deferral_action") or "unknown"
            llm_resolved_pred = gold if action == "abstain_uncertain" else llm_pred
            expert_correct = expert_pred == gold
            llm_raw_correct = llm_pred == gold
            llm_resolved_correct = llm_resolved_pred == gold
            raw_outcome, resolved_outcome = outcome_labels(expert_correct, llm_raw_correct, action)
            features = linguistic_features(item, language)
            row = {
                "case_id": f"{language}:{item.get('uuid')}",
                "uuid": str(item.get("uuid")),
                "uuid_Kliping": str(item.get("uuid_Kliping", "")),
                "language": language,
                "language_display": cfg["display"],
                "expert_key": expert_key,
                "expert_display": cfg["expert_display"],
                "prompt_variant": run.prompt_variant,
                "gate_rate": run.gate_rate,
                "aspect": str(item.get("aspect") or ""),
                "article": str(item.get("article") or ""),
                "gold": gold,
                "gold_label": LABEL_NAME.get(gold, str(gold)),
                "expert_pred": expert_pred,
                "expert_label": LABEL_NAME.get(expert_pred, str(expert_pred)),
                "llm_pred": llm_pred,
                "llm_label": LABEL_NAME.get(llm_pred, str(llm_pred)),
                "llm_resolved_pred": llm_resolved_pred,
                "llm_resolved_label": LABEL_NAME.get(llm_resolved_pred, str(llm_resolved_pred)),
                "action": action,
                "llm_called": bool(result.get("llm_called", True)),
                "gate_decision": result.get("gate_decision", ""),
                "expert_correct": expert_correct,
                "llm_raw_correct": llm_raw_correct,
                "llm_resolved_correct": llm_resolved_correct,
                "error_type_raw": raw_outcome,
                "error_type_resolved": resolved_outcome,
                "primary_confidence": float(result.get("primary_confidence", math.nan)),
                "primary_entropy": float(result.get("primary_entropy", math.nan)),
                "hard_score": float(result.get("hard_score", math.nan)),
                "num_aux_disagree": int(result.get("num_aux_disagree", 0) or 0),
                "num_aux": int(result.get("num_aux", 0) or 0),
                **features,
            }
            row["text_local_windows"] = make_view(row, "local_windows")
            row["text_local_windows_normalized"] = make_view(row, "local_windows_normalized")
            row["text_morphology"] = make_view(row, "morphology")
            row["text_combined"] = make_view(row, "combined")
            rows.append(row)
            language_count += 1
            if args.max_cases_per_language and language_count >= args.max_cases_per_language:
                break
    args.output_dir.mkdir(parents=True, exist_ok=True)
    cases_path = args.output_dir / "cases.jsonl"
    runs_path = args.output_dir / "selected_runs.json"
    write_jsonl(cases_path, rows)
    write_json(runs_path, run_rows)
    pd.DataFrame(rows).drop(columns=["article"], errors="ignore").to_csv(
        args.output_dir / "cases_compact.csv", index=False
    )
    print(f"Wrote {len(rows)} cases to {cases_path}")
    print(f"Wrote selected run metadata to {runs_path}")
    return cases_path, runs_path


def load_cases(output_dir: Path) -> pd.DataFrame:
    path = output_dir / "cases.jsonl"
    if not path.exists():
        raise FileNotFoundError(f"Missing {path}; run --step cases first")
    return pd.DataFrame(read_jsonl(path))


def model_slug(model_path: str | Path) -> str:
    path = Path(model_path)
    return path.name.replace("/", "_")


def embedding_paths(output_dir: Path, model_path: str | Path, view: str) -> tuple[Path, Path]:
    slug = model_slug(model_path)
    return (
        output_dir / f"embeddings_{slug}_{view}.npy",
        output_dir / f"embedding_index_{slug}_{view}.csv",
    )


def encode_with_sentence_transformers(texts: list[str], args: argparse.Namespace) -> np.ndarray | None:
    if args.backend not in {"auto", "sentence-transformers"}:
        return None
    try:
        from sentence_transformers import SentenceTransformer
    except ModuleNotFoundError:
        if args.backend == "sentence-transformers":
            raise
        return None
    model = SentenceTransformer(str(args.embedding_model), device=args.device)
    encoded_texts = texts
    if "multilingual-e5" in str(args.embedding_model).lower():
        encoded_texts = [f"passage: {text}" for text in texts]
    return model.encode(
        encoded_texts,
        batch_size=args.batch_size,
        normalize_embeddings=True,
        show_progress_bar=True,
    )


def mean_pool(last_hidden_state: Any, attention_mask: Any) -> Any:
    import torch

    mask = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
    summed = torch.sum(last_hidden_state * mask, dim=1)
    denom = torch.clamp(mask.sum(dim=1), min=1e-9)
    return summed / denom


def encode_with_transformers(texts: list[str], args: argparse.Namespace) -> np.ndarray:
    import torch
    import torch.nn.functional as F
    from transformers import AutoModel, AutoTokenizer

    trust_remote_code = bool(args.trust_remote_code or "jina" in str(args.embedding_model).lower())
    tokenizer = AutoTokenizer.from_pretrained(str(args.embedding_model), trust_remote_code=trust_remote_code)
    model = AutoModel.from_pretrained(str(args.embedding_model), trust_remote_code=trust_remote_code)
    model.to(args.device)
    model.eval()
    encoded_chunks = []
    prefix = "passage: " if "multilingual-e5" in str(args.embedding_model).lower() else ""
    with torch.inference_mode():
        for start in range(0, len(texts), args.batch_size):
            batch_texts = [prefix + text for text in texts[start : start + args.batch_size]]
            batch = tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=args.max_length,
                return_tensors="pt",
            )
            batch = {key: value.to(args.device) for key, value in batch.items()}
            outputs = model(**batch)
            pooled = mean_pool(outputs.last_hidden_state, batch["attention_mask"])
            pooled = F.normalize(pooled, p=2, dim=1)
            encoded_chunks.append(pooled.detach().cpu().numpy())
            print(f"embedded {min(start + args.batch_size, len(texts))}/{len(texts)}", flush=True)
    return np.vstack(encoded_chunks)


def build_embeddings(args: argparse.Namespace) -> tuple[Path, Path]:
    df = load_cases(args.output_dir)
    text_col = f"text_{args.view}"
    if text_col in df.columns:
        texts = df[text_col].fillna("").astype(str).tolist()
    elif args.view in {"task_input", "article_normalized"}:
        texts = [make_view(row, args.view) for row in df.to_dict("records")]
    else:
        raise KeyError(f"Missing view column {text_col}. Available text columns: {[c for c in df.columns if c.startswith('text_')]}")
    embeddings = encode_with_sentence_transformers(texts, args)
    if embeddings is None:
        embeddings = encode_with_transformers(texts, args)
    embeddings = np.asarray(embeddings, dtype=np.float32)
    emb_path, index_path = embedding_paths(args.output_dir, args.embedding_model, args.view)
    np.save(emb_path, embeddings)
    df[["case_id", "language", "uuid", "expert_key", "prompt_variant", "gate_rate", "llm_called", "error_type_resolved"]].to_csv(
        index_path, index=False
    )
    print(f"Wrote embeddings {embeddings.shape} to {emb_path}")
    print(f"Wrote embedding index to {index_path}")
    return emb_path, index_path


def scope_mask(df: pd.DataFrame, scope: str) -> pd.Series:
    if scope == "all":
        return pd.Series(True, index=df.index)
    if scope in {"hard", "llm_called"}:
        return df["llm_called"].astype(bool)
    if scope == "errors":
        return (~df["expert_correct"].astype(bool)) | (~df["llm_raw_correct"].astype(bool))
    if scope == "expert_wrong":
        return ~df["expert_correct"].astype(bool)
    if scope == "sentiment_rich":
        return (df["gold"] != 0) | (df["expert_pred"] != 0) | (df["llm_pred"] != 0)
    raise ValueError(f"Unknown scope: {scope}")


def reduce_for_clustering(X: np.ndarray, args: argparse.Namespace) -> tuple[np.ndarray, np.ndarray, str]:
    try:
        import umap

        reducer = umap.UMAP(
            n_neighbors=args.umap_neighbors,
            min_dist=0.0,
            n_components=args.cluster_components,
            metric="cosine",
            random_state=args.seed,
        )
        cluster_space = reducer.fit_transform(X)
        viz = umap.UMAP(
            n_neighbors=args.umap_neighbors,
            min_dist=0.1,
            n_components=2,
            metric="cosine",
            random_state=args.seed,
        ).fit_transform(X)
        return cluster_space, viz, "umap"
    except ModuleNotFoundError:
        n_components = min(args.cluster_components, X.shape[1], max(2, X.shape[0] - 1))
        scaled = StandardScaler(with_mean=False).fit_transform(X)
        pca = PCA(n_components=n_components, random_state=args.seed)
        cluster_space = pca.fit_transform(scaled)
        viz = cluster_space[:, :2] if cluster_space.shape[1] >= 2 else np.c_[cluster_space[:, 0], np.zeros(len(cluster_space))]
        return cluster_space, viz, "pca_fallback"


def cluster_labels(cluster_space: np.ndarray, args: argparse.Namespace) -> tuple[np.ndarray, str]:
    try:
        import hdbscan

        clusterer = hdbscan.HDBSCAN(
            min_cluster_size=args.min_cluster_size,
            min_samples=args.min_samples,
            metric="euclidean",
        )
        return clusterer.fit_predict(cluster_space), "hdbscan"
    except ModuleNotFoundError:
        n_clusters = min(args.n_clusters, max(2, len(cluster_space) // max(args.min_cluster_size, 1)))
        clusterer = MiniBatchKMeans(n_clusters=n_clusters, random_state=args.seed, n_init="auto")
        return clusterer.fit_predict(cluster_space), "minibatch_kmeans_fallback"


def top_values(series: pd.Series, n: int = 5) -> str:
    counts = series.fillna("").astype(str).value_counts().head(n)
    return "; ".join(f"{idx}={count}" for idx, count in counts.items())


def cluster_summaries(clustered: pd.DataFrame) -> pd.DataFrame:
    global_means = clustered[NUMERIC_FEATURES].apply(pd.to_numeric, errors="coerce").mean()
    global_stds = clustered[NUMERIC_FEATURES].apply(pd.to_numeric, errors="coerce").std().replace(0, np.nan)
    rows = []
    for cluster_id, group in clustered.groupby("cluster"):
        numeric = group[NUMERIC_FEATURES].apply(pd.to_numeric, errors="coerce")
        z = ((numeric.mean() - global_means) / global_stds).replace([np.inf, -np.inf], np.nan).dropna()
        top_enriched = "; ".join(f"{idx}={value:+.2f}z" for idx, value in z.abs().sort_values(ascending=False).head(6).items())
        rows.append(
            {
                "cluster": int(cluster_id),
                "n": int(len(group)),
                "language_mix": top_values(group["language_display"], 4),
                "resolved_error_mix": top_values(group["error_type_resolved"], 5),
                "raw_error_mix": top_values(group["error_type_raw"], 5),
                "action_mix": top_values(group["action"], 5),
                "gold_mix": top_values(group["gold_label"], 3),
                "top_aspects": top_values(group["aspect"], 8),
                "mean_confidence": float(pd.to_numeric(group["primary_confidence"], errors="coerce").mean()),
                "mean_doc_tokens": float(pd.to_numeric(group["doc_token_len"], errors="coerce").mean()),
                "mean_declared_mentions": float(pd.to_numeric(group["declared_mentions"], errors="coerce").mean()),
                "enriched_numeric_features": top_enriched,
            }
        )
    return pd.DataFrame(rows).sort_values(["cluster"]).reset_index(drop=True)


def representative_rows(clustered: pd.DataFrame, cluster_space: np.ndarray, n: int) -> list[dict[str, Any]]:
    reps = []
    for cluster_id in sorted(clustered["cluster"].unique()):
        if int(cluster_id) == -1:
            continue
        positions = np.where(clustered["cluster"].to_numpy() == cluster_id)[0]
        if len(positions) == 0:
            continue
        centroid = cluster_space[positions].mean(axis=0, keepdims=True)
        distances = pairwise_distances(cluster_space[positions], centroid, metric="euclidean").ravel()
        for rank, local_pos in enumerate(np.argsort(distances)[:n], start=1):
            row = clustered.iloc[positions[local_pos]]
            reps.append(
                {
                    "cluster": int(cluster_id),
                    "rank": rank,
                    "case_id": row["case_id"],
                    "language": row["language_display"],
                    "uuid": row["uuid"],
                    "aspect": row["aspect"],
                    "gold": row["gold_label"],
                    "expert": row["expert_label"],
                    "llm": row["llm_label"],
                    "action": row["action"],
                    "error_type_resolved": row["error_type_resolved"],
                    "primary_confidence": row["primary_confidence"],
                    "local_windows": row["local_windows"][:1200],
                }
            )
    return reps


def run_clustering(args: argparse.Namespace) -> tuple[Path, Path, Path]:
    df = load_cases(args.output_dir)
    emb_path, index_path = embedding_paths(args.output_dir, args.embedding_model, args.view)
    if not emb_path.exists():
        raise FileNotFoundError(f"Missing {emb_path}; run --step embed first")
    embeddings = np.load(emb_path)
    index = pd.read_csv(index_path)
    if len(df) != len(embeddings) or len(index) != len(embeddings):
        raise ValueError("Cases, embedding index, and embedding matrix have different lengths")
    mask = scope_mask(df, args.scope)
    scoped = df.loc[mask].reset_index(drop=True)
    X = embeddings[mask.to_numpy()]
    if args.sample_size and len(scoped) > args.sample_size:
        rng = np.random.default_rng(args.seed)
        keep = np.sort(rng.choice(len(scoped), size=args.sample_size, replace=False))
        scoped = scoped.iloc[keep].reset_index(drop=True)
        X = X[keep]
    cluster_space, viz, reducer_name = reduce_for_clustering(X, args)
    labels, clusterer_name = cluster_labels(cluster_space, args)
    clustered = scoped.copy()
    clustered["cluster"] = labels.astype(int)
    clustered["umap_x"] = viz[:, 0]
    clustered["umap_y"] = viz[:, 1]
    slug = model_slug(args.embedding_model)
    prefix = f"{slug}_{args.view}_{args.scope}"
    clusters_path = args.output_dir / f"clusters_{prefix}.csv"
    summary_path = args.output_dir / f"cluster_summary_{prefix}.csv"
    reps_path = args.output_dir / f"cluster_representatives_{prefix}.jsonl"
    keep_cols = [
        "case_id", "uuid", "language", "language_display", "expert_key", "expert_display",
        "prompt_variant", "gate_rate", "aspect", "gold", "gold_label", "expert_pred",
        "expert_label", "llm_pred", "llm_label", "llm_resolved_pred", "llm_resolved_label",
        "action", "llm_called", "expert_correct", "llm_raw_correct", "llm_resolved_correct",
        "error_type_raw", "error_type_resolved", "cluster", "umap_x", "umap_y",
        *NUMERIC_FEATURES,
        "local_windows",
    ]
    clustered[keep_cols].to_csv(clusters_path, index=False)
    summary = cluster_summaries(clustered)
    summary.insert(1, "reducer", reducer_name)
    summary.insert(2, "clusterer", clusterer_name)
    summary.to_csv(summary_path, index=False)
    write_jsonl(reps_path, representative_rows(clustered, cluster_space, args.representatives_per_cluster))
    print(f"Wrote clustered rows to {clusters_path}")
    print(f"Wrote cluster summary to {summary_path}")
    print(f"Wrote representatives to {reps_path}")
    return clusters_path, summary_path, reps_path


def make_cluster_prompts(args: argparse.Namespace) -> Path:
    slug = model_slug(args.embedding_model)
    prefix = f"{slug}_{args.view}_{args.scope}"
    reps_path = args.output_dir / f"cluster_representatives_{prefix}.jsonl"
    summary_path = args.output_dir / f"cluster_summary_{prefix}.csv"
    if not reps_path.exists() or not summary_path.exists():
        raise FileNotFoundError("Missing clustering outputs; run --step cluster first")
    reps = pd.DataFrame(read_jsonl(reps_path))
    summary = pd.read_csv(summary_path)
    prompt_rows = []
    for cluster_id, group in reps.groupby("cluster"):
        cluster_summary = summary[summary["cluster"] == cluster_id].iloc[0].to_dict()
        examples = []
        for _, row in group.head(args.prompt_examples_per_cluster).iterrows():
            examples.append(
                "\n".join(
                    [
                        f"Example {int(row['rank'])}: case_id={row['case_id']}, language={row['language']}, aspect={row['aspect']}",
                        f"Gold={row['gold']}; expert={row['expert']}; LLM={row['llm']}; action={row['action']}; outcome={row['error_type_resolved']}",
                        f"Local context: {row['local_windows']}",
                    ]
                )
            )
        prompt = (
            "You are labeling clusters for document-level, target-specific ABSA error analysis. "
            "Focus on recurring linguistic, discourse, or annotation phenomena. Do not infer from labels alone.\n\n"
            f"Cluster summary: {json.dumps(cluster_summary, ensure_ascii=False)}\n\n"
            + "\n\n".join(examples)
            + "\n\nReturn: (1) a short cluster label, (2) 3-5 candidate explanations, "
            "(3) which examples support each explanation, and (4) what should be manually checked."
        )
        prompt_rows.append({"cluster": int(cluster_id), "prompt": prompt})
    out = args.output_dir / f"cluster_label_prompts_{prefix}.jsonl"
    write_jsonl(out, prompt_rows)
    print(f"Wrote {len(prompt_rows)} cluster-label prompts to {out}")
    return out


def openai_chat_completion(
    api_base: str,
    model: str,
    prompt: str,
    max_tokens: int,
    temperature: float,
    timeout: int,
) -> dict[str, Any]:
    url = api_base.rstrip("/") + "/chat/completions"
    payload = {
        "model": model,
        "messages": [
            {
                "role": "system",
                "content": (
                    "You label qualitative error-analysis clusters for document-level, "
                    "target-specific ABSA. Be concise, evidence-grounded, and cautious."
                ),
            },
            {"role": "user", "content": prompt},
        ],
        "max_tokens": max_tokens,
        "temperature": temperature,
    }
    data = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=data,
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {os.environ.get('OPENAI_API_KEY', 'EMPTY')}",
        },
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def label_clusters(args: argparse.Namespace) -> Path:
    slug = model_slug(args.embedding_model)
    prefix = f"{slug}_{args.view}_{args.scope}"
    prompts_path = args.output_dir / f"cluster_label_prompts_{prefix}.jsonl"
    if not prompts_path.exists():
        raise FileNotFoundError(f"Missing {prompts_path}; run --step prompts first")
    prompt_rows = read_jsonl(prompts_path)
    if args.max_cluster_labels:
        prompt_rows = prompt_rows[: args.max_cluster_labels]
    output_rows = []
    for idx, row in enumerate(prompt_rows, start=1):
        print(f"labeling cluster {row['cluster']} ({idx}/{len(prompt_rows)})", flush=True)
        try:
            response = openai_chat_completion(
                api_base=args.llm_api_base,
                model=args.llm_model,
                prompt=row["prompt"],
                max_tokens=args.llm_max_tokens,
                temperature=args.llm_temperature,
                timeout=args.llm_timeout,
            )
            content = response["choices"][0]["message"]["content"]
            status = "success"
            error = ""
        except (urllib.error.URLError, TimeoutError, KeyError, json.JSONDecodeError) as exc:
            content = ""
            status = "error"
            error = repr(exc)
        output_rows.append(
            {
                "cluster": int(row["cluster"]),
                "status": status,
                "label_text": content,
                "error": error,
                "model": args.llm_model,
                "api_base": args.llm_api_base,
            }
        )
        if args.llm_sleep:
            time.sleep(args.llm_sleep)
    out = args.output_dir / f"cluster_labels_{prefix}.jsonl"
    write_jsonl(out, output_rows)
    pd.DataFrame(output_rows).to_csv(args.output_dir / f"cluster_labels_{prefix}.csv", index=False)
    print(f"Wrote {len(output_rows)} cluster labels to {out}")
    return out


def run_bertopic(args: argparse.Namespace) -> tuple[Path, Path]:
    try:
        from bertopic import BERTopic
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError("BERTopic is not installed in this environment") from exc
    df = load_cases(args.output_dir)
    emb_path, index_path = embedding_paths(args.output_dir, args.embedding_model, args.view)
    if not emb_path.exists():
        raise FileNotFoundError(f"Missing {emb_path}; run --step embed first")
    embeddings = np.load(emb_path)
    if len(df) != len(embeddings):
        raise ValueError("Cases and embedding matrix have different lengths")
    mask = scope_mask(df, args.scope)
    scoped = df.loc[mask].reset_index(drop=True)
    X = embeddings[mask.to_numpy()]
    if args.sample_size and len(scoped) > args.sample_size:
        rng = np.random.default_rng(args.seed)
        keep = np.sort(rng.choice(len(scoped), size=args.sample_size, replace=False))
        scoped = scoped.iloc[keep].reset_index(drop=True)
        X = X[keep]
    texts = scoped[f"text_{args.view}"].fillna("").astype(str).tolist()
    topic_model = BERTopic(
        language="multilingual",
        calculate_probabilities=False,
        verbose=True,
        min_topic_size=args.min_cluster_size,
    )
    topics, _ = topic_model.fit_transform(texts, embeddings=X)
    scoped = scoped.copy()
    scoped["topic"] = topics
    slug = model_slug(args.embedding_model)
    prefix = f"{slug}_{args.view}_{args.scope}"
    topic_info_path = args.output_dir / f"bertopic_topics_{prefix}.csv"
    topic_docs_path = args.output_dir / f"bertopic_docs_{prefix}.csv"
    topic_model.get_topic_info().to_csv(topic_info_path, index=False)
    keep_cols = [
        "case_id", "uuid", "language_display", "aspect", "gold_label", "expert_label",
        "llm_label", "action", "error_type_resolved", "topic", "primary_confidence",
        "hard_score", "local_windows",
    ]
    scoped[keep_cols].to_csv(topic_docs_path, index=False)
    print(f"Wrote BERTopic topic info to {topic_info_path}")
    print(f"Wrote BERTopic document-topic rows to {topic_docs_path}")
    return topic_info_path, topic_docs_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--step",
        choices=["cases", "embed", "cluster", "prompts", "label", "topics", "all"],
        default="all",
    )
    parser.add_argument("--output-dir", type=Path, default=OUT_DIR)
    parser.add_argument("--languages", nargs="*", choices=sorted(DEFAULT_TASKS), default=None)
    parser.add_argument("--autorun", default="medium")
    parser.add_argument("--prompt-variant", choices=["masked", "unmasked"], default=None)
    parser.add_argument("--gate-rate", type=float, default=None, help="Use decimal gate rate, e.g. 0.10. Default selects best available run.")
    parser.add_argument("--max-cases-per-language", type=int, default=None, help="Smoke-test limiter for case extraction.")
    parser.add_argument("--embedding-model", type=Path, default=ROOT / "models" / "embeddings" / "bge-m3")
    parser.add_argument(
        "--view",
        choices=["task_input", "article_normalized", "local_windows", "local_windows_normalized", "morphology", "combined"],
        default="local_windows",
    )
    parser.add_argument("--backend", choices=["auto", "sentence-transformers", "transformers"], default="auto")
    parser.add_argument("--device", default="cuda" if os.environ.get("CUDA_VISIBLE_DEVICES") else "cpu")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-length", type=int, default=2048)
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--scope", choices=["all", "hard", "llm_called", "errors", "expert_wrong", "sentiment_rich"], default="llm_called")
    parser.add_argument("--sample-size", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--umap-neighbors", type=int, default=30)
    parser.add_argument("--cluster-components", type=int, default=10)
    parser.add_argument("--min-cluster-size", type=int, default=25)
    parser.add_argument("--min-samples", type=int, default=10)
    parser.add_argument("--n-clusters", type=int, default=20)
    parser.add_argument("--representatives-per-cluster", type=int, default=8)
    parser.add_argument("--prompt-examples-per-cluster", type=int, default=8)
    parser.add_argument("--llm-api-base", default=os.environ.get("LLM_API_BASE", "http://127.0.0.1:8000/v1"))
    parser.add_argument("--llm-model", default=os.environ.get("LLM_MODEL", "gemma27b"))
    parser.add_argument("--llm-max-tokens", type=int, default=768)
    parser.add_argument("--llm-temperature", type=float, default=0.2)
    parser.add_argument("--llm-timeout", type=int, default=180)
    parser.add_argument("--llm-sleep", type=float, default=0.0)
    parser.add_argument("--max-cluster-labels", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir = args.output_dir.resolve()
    args.embedding_model = args.embedding_model.resolve()
    if args.step in {"cases", "all"}:
        build_cases(args)
    if args.step in {"embed", "all"}:
        build_embeddings(args)
    if args.step in {"cluster", "all"}:
        run_clustering(args)
    if args.step in {"prompts", "all"}:
        make_cluster_prompts(args)
    if args.step in {"label", "all"}:
        label_clusters(args)
    if args.step in {"topics"}:
        run_bertopic(args)


if __name__ == "__main__":
    main()
