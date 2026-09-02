#!/usr/bin/env python3
"""Build final ABSA result tables and a lightweight analysis notebook.

Run from the repository root:

    python3 reviews/build_final_results_notebook.py

The script intentionally keeps an explicit source map. There are many
experiments in results/ and reviews/ that are useful for development but should
not automatically flow into paper tables.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
from collections import Counter
from pathlib import Path
from typing import Any

ROOT = Path(os.environ.get("ABSA_RELEASE_ROOT", Path(__file__).resolve().parents[2])).resolve()
OUT_DIR = ROOT / "reviews" / "scratchpad"
NOTEBOOK_PATH = ROOT / "reviews" / "8.0 final-result-analysis.ipynb"
MARKDOWN_PATH = ROOT / "reviews" / "8.0 final-result-analysis.md"

LANGUAGES = ("slovenian", "serbian")
LABELS = [-1, 0, 1]
CLASS_KEYS = {
    -1: ("Negative (0)", "negative", "-1"),
    0: ("Neutral (1)", "neutral", "0"),
    1: ("Positive (2)", "positive", "1"),
}

UNSEEN_ASPECTS = {
    "slovenian": {
        "A1 Slovenija",
        "Prva osebna zavarovalnica",
        "Mlinotest",
        "Audi",
        "Renault",
        "Energetika Ljubljana",
        "Cupra",
        "Nissan",
        "Addiko banka",
        "Grawe",
        "Delavska hranilnica",
    },
    "serbian": {
        "mts",
        "Knez Petrol",
        "Generali",
        "Mobi Banka",
        "Philip Moris",
        "JTI",
        "Uniqa",
        "Delta",
        "API bank",
        "AXA",
    },
}

METRIC_COLUMNS = [
    ("accuracy", "Accuracy", True, 2),
    ("precision_macro", "Precision", True, 2),
    ("recall_macro", "Recall", True, 2),
    ("f1_macro", "F1", True, 2),
    ("qwk", "QWK", False, 3),
    ("negative_f1", "Negative F1", True, 2),
    ("neutral_f1", "Neutral F1", True, 2),
    ("positive_f1", "Positive F1", True, 2),
]

EXPERT_EXCLUDED_GROUPS = {"Prompting LLMs", "Automatic Baselines"}
EXPERT_DISPLAY_NAMES = {
    "han_xlmr_masked": "HAN + XLMR masked",
    "longformer_masked": "Longformer masked",
    "mdeberta_masked": "mDeBERTa-v3 masked",
    "slavic_specific_masked": "SloBERTa/BERTic masked",
}
EXPERT_STABILITY_COLUMNS = [
    "Rank",
    "Group",
    "Strategy",
    "F1",
    "F1 SD",
    "QWK",
    "QWK SD",
    "F1 - SD",
    "QWK - SD",
    "Conservative Composite",
    "Source",
]
JOINT_SCORING_COLUMNS = [
    "Index",
    "Group",
    "Strategy",
    "F1",
    "QWK",
    "Aggregated Score",
    "Source",
]
LLM_DEFER_COLUMNS = [
    "Calibration",
    "Language",
    "Expert",
    "Prompt Variant",
    "Autorun",
    "Accuracy",
    "Precision",
    "Recall",
    "F1",
    "QWK",
    "Negative F1",
    "Neutral F1",
    "Positive F1",
    "F1 (Seen)",
    "F1 (Unseen)",
    "QWK (Seen)",
    "QWK (Unseen)",
    "N",
    "Source",
]
BEST_LLM_DEFER_COLUMNS = [
    "Language",
    "Expert",
    "Best Calibration",
    "Prompt Variant",
    "Autorun",
    "Accuracy",
    "F1",
    "QWK",
    "Source Dir",
    "Source",
]
BEST_LLM_DEFER_SEEN_UNSEEN_COLUMNS = [
    "Language",
    "Expert",
    "Best Calibration",
    "Prompt Variant",
    "Autorun",
    "F1 (Seen)",
    "F1 (Unseen)",
    "QWK (Seen)",
    "QWK (Unseen)",
    "Source Dir",
    "Source",
]

SEEN_UNSEEN_COLUMNS = [
    ("f1_seen", "F1 (Seen)", True, 2),
    ("f1_unseen", "F1 (Unseen)", True, 2),
    ("qwk_seen", "QWK (Seen)", False, 3),
    ("qwk_unseen", "QWK (Unseen)", False, 3),
]


TABLE_SPECS: list[dict[str, Any]] = [
    {
        "key": "bge_m3_mlp_whole",
        "group": "Document Embeddings + MLP",
        "strategy": "Baseline",
        "kind": "summary",
        "dir_candidates": ["results/bge-m3_mlp/whole/{language}"],
        "notes": "Paper Baseline row; bge-m3_hybrid is intentionally excluded.",
    },
    {
        "key": "bge_m3_mlp_masked",
        "group": "Document Embeddings + MLP",
        "strategy": "Masked",
        "kind": "summary",
        "dir_candidates": ["results/bge-m3_mlp/masked/{language}"],
    },
    {
        "key": "bge_m3_mlp_filtered",
        "group": "Document Embeddings + MLP",
        "strategy": "Extractive",
        "kind": "summary",
        "dir_candidates": ["results/bge-m3_mlp/filtered/{language}"],
    },
    {
        "key": "xlmr_no_summary",
        "group": "Fine-tuning XLMR",
        "strategy": "Truncated",
        "kind": "summary",
        "dir_candidates": ["results/xlmr-saved/no_summary/{language}"],
    },
    {
        "key": "xlmr_extractive_summary",
        "group": "Fine-tuning XLMR",
        "strategy": "Extractive",
        "kind": "summary",
        "dir_candidates": ["results/xlmr-saved/extractive_summary/{language}"],
    },
    {
        "key": "xlmr_gemma_summary",
        "group": "Fine-tuning XLMR",
        "strategy": "LLM Summary",
        "kind": "summary",
        "dir_candidates": ["results/xlmr-saved/gemma-3-27b-summary/{language}"],
        "notes": "Mapped by matching the old paper scores; gams-9b and textrank are excluded.",
    },
    {
        "key": "xlmr_gemma_summary_masked",
        "group": "Fine-tuning XLMR",
        "strategy": "LLM Summary + Masked",
        "kind": "summary",
        "dir_candidates": ["results/xlmr-saved/gemma-3-27b-summary-masked/{language}"],
    },
    {
        "key": "han_with_aspect_markers",
        "group": "HAN + XLMR",
        "strategy": "Baseline",
        "kind": "summary",
        "dir_candidates": ["results/global-context-modelling/with-aspect-markers/{language}"],
        "notes": "This is the old paper Baseline row.",
    },
    {
        "key": "han_simplified_dart",
        "group": "HAN + XLMR",
        "strategy": "Masked",
        "kind": "summary",
        "dir_candidates": ["results/global-context-modelling/simplified-dart-xlmr/{language}"],
        "notes": "This is the old paper Masked row.",
    },
    {
        "key": "longformer_unmasked",
        "group": "Longformer",
        "strategy": "Baseline",
        "kind": "summary",
        "dir_candidates": [
            "reviews/longformer/unmasked/{language}",
            "results/longformer/unmasked/{language}",
        ],
    },
    {
        "key": "longformer_masked",
        "group": "Longformer",
        "strategy": "Masked",
        "kind": "summary",
        "dir_candidates": [
            "reviews/longformer/masked/{language}",
            "results/longformer/masked/{language}",
        ],
    },
    {
        "key": "mdeberta_unmasked",
        "group": "mDeBERTa-v3",
        "strategy": "Baseline",
        "kind": "summary",
        "dir_candidates": [
            "reviews/mdeberta/unmasked/{language}",
            "results/mdeberta/unmasked/{language}",
        ],
    },
    {
        "key": "mdeberta_masked",
        "group": "mDeBERTa-v3",
        "strategy": "Masked",
        "kind": "summary",
        "dir_candidates": [
            "reviews/mdeberta/masked/{language}",
            "results/mdeberta/masked/{language}",
        ],
    },
    {
        "key": "mt5_unmasked",
        "group": "mT5",
        "strategy": "Baseline",
        "kind": "summary",
        "dir_candidates": [
            "reviews/mt5/unmasked/{language}",
            "results/mt5/unmasked/{language}",
        ],
    },
    {
        "key": "mt5_masked",
        "group": "mT5",
        "strategy": "Masked",
        "kind": "summary",
        "dir_candidates": [
            "reviews/mt5/masked/{language}",
            "results/mt5/masked/{language}",
        ],
    },
    {
        "key": "slavic_specific_unmasked",
        "group": "Language-specific Encoder",
        "strategy": "SloBERTa/BERTic",
        "kind": "summary",
        "dir_candidates": [
            "reviews/slavic_specific/unmasked/{language}",
            "results/slavic_specific/unmasked/{language}",
            "results/slavic_specifc/unmasked/{language}",
        ],
    },
    {
        "key": "slavic_specific_masked",
        "group": "Language-specific Encoder",
        "strategy": "SloBERTa/BERTic + Masked",
        "kind": "summary",
        "dir_candidates": [
            "reviews/slavic_specific/masked/{language}",
            "results/slavic_specific/masked/{language}",
            "results/slavic_specifc/masked/{language}",
        ],
    },
    {
        "key": "llm_direct_prompting",
        "group": "Prompting LLMs",
        "strategy": "Direct Prompting",
        "kind": "llm",
        "dir_candidates": [
            "results/large-language-models/{language}/direct-prompting",
            "results/llms/{language}/direct-prompting",
        ],
    },
    {
        "key": "llm_dspy_direct_prompting",
        "group": "Prompting LLMs",
        "strategy": "DSPy Direct Prompting",
        "kind": "llm",
        "dir_candidates": [
            "results/large-language-models/{language}/direct-dspy-prompting",
            "results/llms/{language}/direct-dspy-prompting",
        ],
    },
    {
        "key": "llm_dspy_cot",
        "group": "Prompting LLMs",
        "strategy": "DSPy COT",
        "kind": "llm",
        "dir_candidates": [
            "results/large-language-models/{language}/dspy-cot",
            "results/llms/{language}/dspy-cot",
        ],
    },
    {
        "key": "llm_softmax_fusion",
        "group": "Prompting LLMs",
        "strategy": "Softmax Fusion",
        "kind": "llm",
        "dir_candidates": [
            "results/large-language-models/{language}/dspy-plm-augmented-cot-with-plm-reliability-signature-and-softmax",
            "results/llms/{language}/dspy-plm-augmented-cot-with-plm-reliability-signature-and-softmax",
        ],
    },
    {
        "key": "llm_softmax_fusion_masked",
        "group": "Prompting LLMs",
        "strategy": "Softmax Fusion + Masked",
        "kind": "llm",
        "dir_candidates": [
            "results/large-language-models/{language}/dspy-plm-augmented-cot-with-plm-reliability-signature-and-softmax-masked",
            "results/llms/{language}/dspy-plm-augmented-cot-with-plm-reliability-signature-and-softmax-masked",
        ],
    },
    {
        "key": "llm_uncertainty_fusion",
        "group": "Prompting LLMs",
        "strategy": "Uncertainty Fusion",
        "kind": "llm",
        "dir_candidates": [
            "results/large-language-models/{language}/dspy-plm-augmented-cot-with-uncertainty",
            "results/llms/{language}/dspy-plm-augmented-cot-with-uncertainty",
        ],
    },
    {
        "key": "llm_uncertainty_fusion_masked",
        "group": "Prompting LLMs",
        "strategy": "Uncertainty Fusion + Masked",
        "kind": "llm",
        "dir_candidates": [
            "results/large-language-models/{language}/dspy-plm-augmented-cot-with-uncertainty-masked",
            "results/llms/{language}/dspy-plm-augmented-cot-with-uncertainty-masked",
        ],
    },
    {
        "key": "document_sentiment_classifier",
        "group": "Automatic Baselines",
        "strategy": "Document Sentiment Classifier",
        "kind": "prediction",
        "prediction_candidates": ["results/luka-and-boshkos-baseline/{language}/test_predictions.json"],
        "qwk_mode": "unweighted",
        "notes": "The current paper table reports unweighted kappa for this legacy baseline.",
    },
    {
        "key": "majority_class",
        "group": "Automatic Baselines",
        "strategy": "Majority Class",
        "kind": "majority",
    },
]


def rel(path: Path | None) -> str | None:
    if path is None:
        return None
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def resolve_dir(spec: dict[str, Any], language: str, must_have: str | None = None) -> Path | None:
    for template in spec.get("dir_candidates", []):
        directory = ROOT / template.format(language=language)
        if not directory.exists():
            continue
        if must_have and not (directory / must_have).exists():
            continue
        return directory
    return None


def resolve_file_from_templates(templates: list[str], language: str) -> Path | None:
    for template in templates:
        path = ROOT / template.format(language=language)
        if path.exists():
            return path
    return None


def first_matching(directory: Path | None, suffix: str) -> Path | None:
    if directory is None or not directory.exists():
        return None
    matches = sorted(p for p in directory.iterdir() if p.name.endswith(suffix))
    return matches[0] if matches else None


def clean_label(value: Any) -> int | None:
    if value is None:
        return None
    if isinstance(value, str):
        value = value.strip().lower()
        mapping = {
            "negative": -1,
            "neg": -1,
            "-1": -1,
            "neutral": 0,
            "neu": 0,
            "0": 0,
            "positive": 1,
            "pos": 1,
            "1": 1,
        }
        return mapping.get(value)
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def mean_std(values: list[float | None]) -> dict[str, Any]:
    clean = [
        float(v)
        for v in values
        if v is not None and not (isinstance(v, float) and math.isnan(v))
    ]
    if not clean:
        return {"mean": None, "std": None, "n": 0, "values": []}
    mean = sum(clean) / len(clean)
    var = sum((v - mean) ** 2 for v in clean) / len(clean)
    return {
        "mean": mean,
        "std": math.sqrt(var),
        "n": len(clean),
        "values": clean,
    }


def format_stat(stat: dict[str, Any], pct: bool, digits: int, latex: bool = False) -> str:
    mean = stat.get("mean")
    if mean is None:
        return r"\textit{NA}" if latex else "NA"
    std = stat.get("std")
    n = stat.get("n", 0)
    scale = 100 if pct else 1
    mean_s = f"{mean * scale:.{digits}f}"
    if n and n > 1 and std is not None:
        std_s = f"{std * scale:.{digits}f}"
        sep = r" \pm " if latex else " +/- "
        return f"{mean_s} ({sep}{std_s})"
    return mean_s


def get_class_f1(report: dict[str, Any], label: int) -> float | None:
    for key in CLASS_KEYS[label]:
        class_data = report.get(key)
        if isinstance(class_data, dict) and class_data.get("f1-score") is not None:
            return float(class_data["f1-score"])
    return None


def metric_values_from_summary(summary: dict[str, Any]) -> dict[str, dict[str, Any]]:
    values: dict[str, list[float | None]] = {name: [] for name, *_ in METRIC_COLUMNS}
    for run_key, run_data in sorted(summary.items()):
        if not run_key.startswith("model_") or not isinstance(run_data, dict):
            continue
        values["accuracy"].append(run_data.get("accuracy"))
        values["precision_macro"].append(run_data.get("precision_macro"))
        values["recall_macro"].append(run_data.get("recall_macro"))
        values["f1_macro"].append(run_data.get("f1_macro"))
        values["qwk"].append(run_data.get("qwk"))
        report = run_data.get("per_class_report", {})
        values["negative_f1"].append(get_class_f1(report, -1))
        values["neutral_f1"].append(get_class_f1(report, 0))
        values["positive_f1"].append(get_class_f1(report, 1))
    return {key: mean_std(vals) for key, vals in values.items()}


def metric_values_from_single(metrics: dict[str, Any]) -> dict[str, dict[str, Any]]:
    report = metrics.get("per_class_report", {})
    values = {
        "accuracy": [metrics.get("accuracy")],
        "precision_macro": [metrics.get("precision_macro")],
        "recall_macro": [metrics.get("recall_macro")],
        "f1_macro": [metrics.get("f1_macro")],
        "qwk": [metrics.get("qwk")],
        "negative_f1": [get_class_f1(report, -1)],
        "neutral_f1": [get_class_f1(report, 0)],
        "positive_f1": [get_class_f1(report, 1)],
    }
    return {key: mean_std(vals) for key, vals in values.items()}


def display_metric(metrics: dict[str, Any], key: str, pct: bool = True, digits: int = 2) -> str:
    value = metrics.get(key)
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "NA"
    scale = 100 if pct else 1
    return f"{float(value) * scale:.{digits}f}"


def safe_qwk(y_true: list[int], y_pred: list[int]) -> float:
    if len(set(y_true)) < 2 or len(set(y_pred)) < 2:
        return math.nan
    label_to_idx = {label: idx for idx, label in enumerate(LABELS)}
    n_labels = len(LABELS)
    observed = [[0.0 for _ in LABELS] for _ in LABELS]
    true_hist = [0.0 for _ in LABELS]
    pred_hist = [0.0 for _ in LABELS]
    n = 0
    for gold, pred in zip(y_true, y_pred):
        if gold not in label_to_idx or pred not in label_to_idx:
            continue
        i = label_to_idx[gold]
        j = label_to_idx[pred]
        observed[i][j] += 1.0
        true_hist[i] += 1.0
        pred_hist[j] += 1.0
        n += 1
    if n == 0:
        return math.nan
    weighted_observed = 0.0
    weighted_expected = 0.0
    denom = float((n_labels - 1) ** 2)
    for i in range(n_labels):
        for j in range(n_labels):
            weight = ((i - j) ** 2) / denom
            expected = true_hist[i] * pred_hist[j] / n
            weighted_observed += weight * observed[i][j]
            weighted_expected += weight * expected
    if weighted_expected == 0:
        return math.nan
    return 1.0 - (weighted_observed / weighted_expected)


def safe_unweighted_kappa(y_true: list[int], y_pred: list[int]) -> float:
    if len(set(y_true)) < 2 or len(set(y_pred)) < 2:
        return math.nan
    n = 0
    observed_agree = 0
    true_hist = Counter()
    pred_hist = Counter()
    for gold, pred in zip(y_true, y_pred):
        if gold not in LABELS or pred not in LABELS:
            continue
        n += 1
        true_hist[gold] += 1
        pred_hist[pred] += 1
        if gold == pred:
            observed_agree += 1
    if n == 0:
        return math.nan
    po = observed_agree / n
    pe = sum(true_hist[label] * pred_hist[label] for label in LABELS) / (n * n)
    if pe == 1.0:
        return math.nan
    return (po - pe) / (1.0 - pe)


def per_label_scores(y_true: list[int], y_pred: list[int]) -> dict[int, dict[str, float]]:
    scores = {}
    for label in LABELS:
        tp = sum(1 for gold, pred in zip(y_true, y_pred) if gold == label and pred == label)
        fp = sum(1 for gold, pred in zip(y_true, y_pred) if gold != label and pred == label)
        fn = sum(1 for gold, pred in zip(y_true, y_pred) if gold == label and pred != label)
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
        scores[label] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
        }
    return scores


def compute_metrics(
    y_true: list[int],
    y_pred: list[int],
    qwk_mode: str = "quadratic",
) -> dict[str, float]:
    label_scores = per_label_scores(y_true, y_pred)
    n = len(y_true)
    accuracy = sum(1 for gold, pred in zip(y_true, y_pred) if gold == pred) / n if n else math.nan
    kappa = safe_unweighted_kappa(y_true, y_pred) if qwk_mode == "unweighted" else safe_qwk(y_true, y_pred)
    return {
        "accuracy": accuracy,
        "precision_macro": sum(label_scores[label]["precision"] for label in LABELS) / len(LABELS),
        "recall_macro": sum(label_scores[label]["recall"] for label in LABELS) / len(LABELS),
        "f1_macro": sum(label_scores[label]["f1"] for label in LABELS) / len(LABELS),
        "qwk": kappa,
        "negative_f1": label_scores[-1]["f1"],
        "neutral_f1": label_scores[0]["f1"],
        "positive_f1": label_scores[1]["f1"],
    }


def metric_values_from_predictions(
    records: list[dict[str, Any]],
    qwk_mode: str = "quadratic",
) -> dict[str, dict[str, Any]]:
    y_true: list[int] = []
    y_pred: list[int] = []
    for item in records:
        gold = clean_label(item.get("sentiment"))
        pred = clean_label(item.get("prediction"))
        if gold is None or pred is None:
            continue
        y_true.append(gold)
        y_pred.append(pred)
    metrics = compute_metrics(y_true, y_pred, qwk_mode=qwk_mode)
    return {key: mean_std([metrics[key]]) for key, *_ in METRIC_COLUMNS}


def load_test_records(language: str) -> list[dict[str, Any]]:
    data = load_json(ROOT / "data" / f"{language}_test_complete.json")
    if isinstance(data, dict):
        return data["test"]
    return data


def majority_records(language: str) -> list[dict[str, Any]]:
    records = load_test_records(language)
    labels = [clean_label(item.get("sentiment")) for item in records]
    labels = [label for label in labels if label is not None]
    majority_label = Counter(labels).most_common(1)[0][0]
    output = []
    for item in records:
        new_item = dict(item)
        new_item["prediction"] = majority_label
        output.append(new_item)
    return output


def summary_prediction_paths(directory: Path | None) -> list[Path]:
    if directory is None:
        return []
    return sorted(directory.glob("test_predictions_*.json"))


def prediction_paths_for_spec(spec: dict[str, Any], language: str) -> list[Path]:
    kind = spec["kind"]
    if kind == "summary":
        directory = resolve_dir(spec, language, "test_metrics_summary.json")
        return summary_prediction_paths(directory)
    if kind == "llm":
        directory = resolve_dir(spec, language)
        path = first_matching(directory, "_predictions.json")
        return [path] if path else []
    if kind == "prediction":
        path = resolve_file_from_templates(spec.get("prediction_candidates", []), language)
        return [path] if path else []
    return []


def normalize_aspect(value: Any) -> str:
    return str(value or "").strip().casefold()


def split_seen_unseen(records: list[dict[str, Any]], language: str) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    unseen = {normalize_aspect(a) for a in UNSEEN_ASPECTS[language]}
    seen_records = []
    unseen_records = []
    for item in records:
        if normalize_aspect(item.get("aspect")) in unseen:
            unseen_records.append(item)
        else:
            seen_records.append(item)
    return seen_records, unseen_records


def metrics_for_record_subset(
    records: list[dict[str, Any]],
    qwk_mode: str = "quadratic",
) -> dict[str, float]:
    y_true = []
    y_pred = []
    for item in records:
        gold = clean_label(item.get("sentiment"))
        pred = clean_label(item.get("prediction"))
        if gold is None or pred is None:
            continue
        y_true.append(gold)
        y_pred.append(pred)
    if not y_true:
        return {"f1_macro": math.nan, "qwk": math.nan}
    metrics = compute_metrics(y_true, y_pred, qwk_mode=qwk_mode)
    return {"f1_macro": metrics["f1_macro"], "qwk": metrics["qwk"]}


def majority_metrics_for_subset(records: list[dict[str, Any]]) -> dict[str, float]:
    labels = [clean_label(item.get("sentiment")) for item in records]
    labels = [label for label in labels if label is not None]
    if not labels:
        return {"f1_macro": math.nan, "qwk": math.nan}
    majority_label = Counter(labels).most_common(1)[0][0]
    return compute_metrics(labels, [majority_label] * len(labels))


def build_overall_row(spec: dict[str, Any], language: str) -> dict[str, Any]:
    kind = spec["kind"]
    source = None
    missing = False
    if kind == "summary":
        directory = resolve_dir(spec, language, "test_metrics_summary.json")
        source = directory / "test_metrics_summary.json" if directory else None
        metrics = metric_values_from_summary(load_json(source)) if source else {}
        missing = source is None
    elif kind == "llm":
        directory = resolve_dir(spec, language)
        source = first_matching(directory, "_metrics.json")
        metrics = metric_values_from_single(load_json(source)) if source else {}
        missing = source is None
    elif kind == "prediction":
        source = resolve_file_from_templates(spec.get("prediction_candidates", []), language)
        metrics = (
            metric_values_from_predictions(load_json(source), qwk_mode=spec.get("qwk_mode", "quadratic"))
            if source
            else {}
        )
        missing = source is None
    elif kind == "majority":
        metrics = metric_values_from_predictions(majority_records(language))
    else:
        raise ValueError(f"Unknown kind: {kind}")

    row = {
        "key": spec["key"],
        "group": spec["group"],
        "strategy": spec["strategy"],
        "language": language,
        "kind": kind,
        "source": rel(source),
        "missing": missing,
        "metrics": metrics,
    }
    row["display"] = {
        display_name: format_stat(metrics.get(name, mean_std([])), pct, digits)
        for name, display_name, pct, digits in METRIC_COLUMNS
    }
    return row


def build_seen_unseen_row(spec: dict[str, Any], language: str) -> dict[str, Any]:
    kind = spec["kind"]
    values = {
        "f1_seen": [],
        "f1_unseen": [],
        "qwk_seen": [],
        "qwk_unseen": [],
    }
    sources: list[str] = []

    if kind == "majority":
        paths: list[Path] = []
        seen_records, unseen_records = split_seen_unseen(load_test_records(language), language)
        seen_metrics = majority_metrics_for_subset(seen_records)
        unseen_metrics = majority_metrics_for_subset(unseen_records)
        values["f1_seen"].append(seen_metrics["f1_macro"])
        values["f1_unseen"].append(unseen_metrics["f1_macro"])
        values["qwk_seen"].append(seen_metrics["qwk"])
        values["qwk_unseen"].append(unseen_metrics["qwk"])
        record_sets = []
    else:
        paths = prediction_paths_for_spec(spec, language)
        record_sets = [load_json(path) for path in paths]

    for idx, records in enumerate(record_sets):
        if not isinstance(records, list):
            continue
        seen_records, unseen_records = split_seen_unseen(records, language)
        seen_metrics = metrics_for_record_subset(
            seen_records, qwk_mode=spec.get("qwk_mode", "quadratic")
        )
        unseen_metrics = metrics_for_record_subset(
            unseen_records, qwk_mode=spec.get("qwk_mode", "quadratic")
        )
        values["f1_seen"].append(seen_metrics["f1_macro"])
        values["f1_unseen"].append(unseen_metrics["f1_macro"])
        values["qwk_seen"].append(seen_metrics["qwk"])
        values["qwk_unseen"].append(unseen_metrics["qwk"])
        if idx < len(paths):
            sources.append(rel(paths[idx]) or "")

    metrics = {key: mean_std(vals) for key, vals in values.items()}
    row = {
        "key": spec["key"],
        "group": spec["group"],
        "strategy": spec["strategy"],
        "language": language,
        "kind": kind,
        "sources": sources,
        "missing": False if kind == "majority" else not record_sets,
        "metrics": metrics,
    }
    row["display"] = {
        display_name: format_stat(metrics.get(name, mean_std([])), pct, digits)
        for name, display_name, pct, digits in SEEN_UNSEEN_COLUMNS
    }
    return row


def rows_to_display(rows: list[dict[str, Any]], columns: list[tuple[str, str, bool, int]]) -> list[dict[str, Any]]:
    display_rows = []
    for row in rows:
        out = {
            "Group": row["group"],
            "Strategy": row["strategy"],
            "Source": row.get("source") or "; ".join(row.get("sources", [])),
        }
        for _, display_name, _, _ in columns:
            out[display_name] = row["display"].get(display_name, "NA")
        display_rows.append(out)
    return display_rows


def minmax(value: float, values: list[float]) -> float:
    low = min(values)
    high = max(values)
    if math.isclose(low, high):
        return 1.0
    return (value - low) / (high - low)


def build_expert_stability_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    experts = [
        row
        for row in rows
        if row["group"] not in EXPERT_EXCLUDED_GROUPS
        and numeric_mean(row, "f1_macro") is not None
        and numeric_mean(row, "qwk") is not None
    ]
    scored = []
    for row in experts:
        f1_mean = numeric_mean(row, "f1_macro") or 0.0
        qwk_mean = numeric_mean(row, "qwk") or 0.0
        f1_sd = metric_std(row, "f1_macro") or 0.0
        qwk_sd = metric_std(row, "qwk") or 0.0
        scored.append(
            {
                "key": row["key"],
                "group": row["group"],
                "strategy": row["strategy"],
                "source": row.get("source"),
                "f1_mean": f1_mean,
                "f1_sd": f1_sd,
                "qwk_mean": qwk_mean,
                "qwk_sd": qwk_sd,
                "conservative_f1": f1_mean - f1_sd,
                "conservative_qwk": qwk_mean - qwk_sd,
            }
        )

    f1_values = [row["conservative_f1"] for row in scored]
    qwk_values = [row["conservative_qwk"] for row in scored]
    for row in scored:
        row["conservative_composite"] = 0.5 * minmax(row["conservative_f1"], f1_values) + 0.5 * minmax(
            row["conservative_qwk"], qwk_values
        )

    scored.sort(
        key=lambda row: (
            -row["conservative_composite"],
            -row["f1_mean"],
            -row["qwk_mean"],
            row["f1_sd"],
            row["strategy"],
        )
    )

    display_rows = []
    for rank, row in enumerate(scored, start=1):
        display_rows.append(
            {
                "Rank": rank,
                "Group": row["group"],
                "Strategy": row["strategy"],
                "F1": f"{row['f1_mean'] * 100:.2f}",
                "F1 SD": f"{row['f1_sd'] * 100:.2f}",
                "QWK": f"{row['qwk_mean']:.3f}",
                "QWK SD": f"{row['qwk_sd']:.3f}",
                "F1 - SD": f"{(row['conservative_f1']) * 100:.2f}",
                "QWK - SD": f"{row['conservative_qwk']:.3f}",
                "Conservative Composite": f"{row['conservative_composite']:.3f}",
                "Source": row["source"] or "",
            }
        )
    return display_rows


def build_joint_scoring_rows(stability_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "Index": row["Rank"],
            "Group": row["Group"],
            "Strategy": row["Strategy"],
            "F1": row["F1"],
            "QWK": row["QWK"],
            "Aggregated Score": row["Conservative Composite"],
            "Source": row["Source"],
        }
        for row in stability_rows
    ]


def parse_new_defer_metric_path(path: Path, calibration_name: str) -> dict[str, str]:
    parts = path.relative_to(ROOT).parts
    # reviews/uncertainty/<root>/<expert>/<language>/<variant>/<autorun>/<file>
    return {
        "calibration": calibration_name,
        "expert": parts[-5],
        "language": parts[-4],
        "prompt_variant": parts[-3],
        "autorun": parts[-2],
    }


def parse_old_defer_metric_path(path: Path) -> dict[str, str]:
    parts = path.relative_to(ROOT).parts
    language = parts[2]
    directory_name = parts[3]
    metrics = load_json(path)
    return {
        "calibration": "old-calibration",
        "expert": "han_xlmr_masked",
        "language": language,
        "prompt_variant": "masked" if directory_name.endswith("-masked") else "unmasked",
        "autorun": str(metrics.get("dspy_autorun_setting") or ("heavy" if language == "slovenian" else "medium")),
    }


def llm_defer_row_from_metrics(path: Path, parsed: dict[str, str]) -> dict[str, Any]:
    metrics = load_json(path)
    report = metrics.get("per_class_report", {})
    pred_path = prediction_path_for_metric_source(rel(path) or "")
    if pred_path is not None:
        records = load_json(pred_path)
        seen_records, unseen_records = split_seen_unseen(records, parsed["language"])
        seen_metrics = metrics_for_record_subset(seen_records)
        unseen_metrics = metrics_for_record_subset(unseen_records)
        f1_seen = "NA" if math.isnan(seen_metrics["f1_macro"]) else f"{seen_metrics['f1_macro'] * 100:.2f}"
        f1_unseen = "NA" if math.isnan(unseen_metrics["f1_macro"]) else f"{unseen_metrics['f1_macro'] * 100:.2f}"
        qwk_seen = "NA" if math.isnan(seen_metrics["qwk"]) else f"{seen_metrics['qwk']:.3f}"
        qwk_unseen = "NA" if math.isnan(unseen_metrics["qwk"]) else f"{unseen_metrics['qwk']:.3f}"
    else:
        f1_seen = f1_unseen = qwk_seen = qwk_unseen = "NA"
    row = {
        "Calibration": parsed["calibration"],
        "Language": parsed["language"].title(),
        "Expert": EXPERT_DISPLAY_NAMES.get(parsed["expert"], parsed["expert"]),
        "Prompt Variant": parsed["prompt_variant"],
        "Autorun": parsed["autorun"],
        "Accuracy": display_metric(metrics, "accuracy", True, 2),
        "Precision": display_metric(metrics, "precision_macro", True, 2),
        "Recall": display_metric(metrics, "recall_macro", True, 2),
        "F1": display_metric(metrics, "f1_macro", True, 2),
        "QWK": display_metric(metrics, "qwk", False, 3),
        "Negative F1": display_metric({"negative_f1": get_class_f1(report, -1)}, "negative_f1", True, 2),
        "Neutral F1": display_metric({"neutral_f1": get_class_f1(report, 0)}, "neutral_f1", True, 2),
        "Positive F1": display_metric({"positive_f1": get_class_f1(report, 1)}, "positive_f1", True, 2),
        "F1 (Seen)": f1_seen,
        "F1 (Unseen)": f1_unseen,
        "QWK (Seen)": qwk_seen,
        "QWK (Unseen)": qwk_unseen,
        "N": metrics.get("num_samples_evaluated", ""),
        "Source": rel(path) or "",
    }
    row["_sort_language"] = parsed["language"]
    row["_sort_expert"] = parsed["expert"]
    row["_sort_variant"] = parsed["prompt_variant"]
    row["_sort_autorun"] = parsed["autorun"]
    return row


def build_llm_defer_rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    new_roots = [
        ("new-calibration", ROOT / "reviews" / "uncertainty" / "llm-dspy-calibration-cot"),
        ("new-rich-calibration", ROOT / "reviews" / "uncertainty" / "rich-llm-dspy-calibration-cot"),
    ]
    for calibration_name, root in new_roots:
        if not root.exists():
            continue
        for path in sorted(root.glob("*/*/*/*/*_test_metrics.json")):
            if "_shard-" in path.name:
                continue
            parsed = parse_new_defer_metric_path(path, calibration_name)
            if parsed["language"] not in LANGUAGES:
                continue
            rows.append(llm_defer_row_from_metrics(path, parsed))

    old_patterns = [
        "results/large-language-models/serbian/dspy-plm-augmented-cot-with-uncertainty-masked/*_metrics.json",
        "results/large-language-models/serbian/dspy-plm-augmented-cot-with-uncertainty/*_metrics.json",
        "results/large-language-models/slovenian/dspy-plm-augmented-cot-with-uncertainty-masked/*_metrics.json",
        "results/large-language-models/slovenian/dspy-plm-augmented-cot-with-uncertainty/*_metrics.json",
    ]
    for pattern in old_patterns:
        for path in sorted(ROOT.glob(pattern)):
            if "_shard-" in path.name:
                continue
            rows.append(llm_defer_row_from_metrics(path, parse_old_defer_metric_path(path)))

    rows.sort(
        key=lambda row: (
            row["Calibration"],
            row["_sort_language"],
            row["_sort_expert"],
            row["_sort_variant"],
            row["_sort_autorun"],
        )
    )
    return [{key: row[key] for key in LLM_DEFER_COLUMNS} for row in rows]


def as_float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return -math.inf


def source_dir(source: str) -> str:
    return str(Path(source).parent) if source else ""


def prediction_path_for_metric_source(source: str) -> Path | None:
    if not source:
        return None
    path = ROOT / source
    if path.name.endswith("_test_metrics.json"):
        pred_path = path.with_name(path.name.replace("_test_metrics.json", "_test_predictions.json"))
    elif path.name.endswith("_metrics.json"):
        pred_path = path.with_name(path.name.replace("_metrics.json", "_predictions.json"))
    else:
        return None
    return pred_path if pred_path.exists() else None


def build_best_llm_defer_rows(llm_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in llm_rows:
        grouped.setdefault((row["Language"], row["Expert"]), []).append(row)

    winners = []
    for (language, expert), rows in grouped.items():
        winner = sorted(
            rows,
            key=lambda row: (
                -as_float(row["F1"]),
                -as_float(row["QWK"]),
                -as_float(row["Accuracy"]),
                row["Calibration"],
                row["Prompt Variant"],
                row["Autorun"],
            ),
        )[0]
        winners.append(
            {
                "Language": language,
                "Expert": expert,
                "Best Calibration": winner["Calibration"],
                "Prompt Variant": winner["Prompt Variant"],
                "Autorun": winner["Autorun"],
                "Accuracy": winner["Accuracy"],
                "F1": winner["F1"],
                "QWK": winner["QWK"],
                "Source Dir": source_dir(winner["Source"]),
                "Source": winner["Source"],
            }
        )
    winners.sort(key=lambda row: (row["Language"], row["Expert"]))
    return winners


def build_best_llm_defer_seen_unseen_rows(best_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output = []
    for row in best_rows:
        language = row["Language"].lower()
        pred_path = prediction_path_for_metric_source(row["Source"])
        if pred_path is None:
            seen_metrics = {"f1_macro": math.nan, "qwk": math.nan}
            unseen_metrics = {"f1_macro": math.nan, "qwk": math.nan}
        else:
            records = load_json(pred_path)
            seen_records, unseen_records = split_seen_unseen(records, language)
            seen_metrics = metrics_for_record_subset(seen_records)
            unseen_metrics = metrics_for_record_subset(unseen_records)
        output.append(
            {
                "Language": row["Language"],
                "Expert": row["Expert"],
                "Best Calibration": row["Best Calibration"],
                "Prompt Variant": row["Prompt Variant"],
                "Autorun": row["Autorun"],
                "F1 (Seen)": "NA" if math.isnan(seen_metrics["f1_macro"]) else f"{seen_metrics['f1_macro'] * 100:.2f}",
                "F1 (Unseen)": "NA" if math.isnan(unseen_metrics["f1_macro"]) else f"{unseen_metrics['f1_macro'] * 100:.2f}",
                "QWK (Seen)": "NA" if math.isnan(seen_metrics["qwk"]) else f"{seen_metrics['qwk']:.3f}",
                "QWK (Unseen)": "NA" if math.isnan(unseen_metrics["qwk"]) else f"{unseen_metrics['qwk']:.3f}",
                "Source Dir": row["Source Dir"],
                "Source": row["Source"],
            }
        )
    return output


def metric_std(row: dict[str, Any], metric_name: str) -> float | None:
    metric = row.get("metrics", {}).get(metric_name, {})
    value = metric.get("std")
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return None
    return float(value)


def numeric_mean(row: dict[str, Any], metric_name: str) -> float | None:
    metric = row.get("metrics", {}).get(metric_name, {})
    value = metric.get("mean")
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return None
    return float(value)


def metric_display_value(
    row: dict[str, Any],
    metric_name: str,
    pct: bool,
    digits: int,
) -> float | None:
    value = numeric_mean(row, metric_name)
    if value is None:
        return None
    scale = 100 if pct else 1
    return round(value * scale, digits)


def choose_winner(
    indexed_rows: list[tuple[int, dict[str, Any]]],
    f1_metric: str,
    qwk_metric: str,
    pct: bool = True,
    digits: int = 2,
) -> dict[str, Any] | None:
    candidates = [
        (idx, row)
        for idx, row in indexed_rows
        if numeric_mean(row, f1_metric) is not None
    ]
    if not candidates:
        return None

    max_display_f1 = max(
        metric_display_value(row, f1_metric, pct, digits) for _, row in candidates
    )
    tied = [
        (idx, row)
        for idx, row in candidates
        if metric_display_value(row, f1_metric, pct, digits) == max_display_f1
    ]

    def rank_key(item: tuple[int, dict[str, Any]]) -> tuple[float, float, float, str]:
        _, row = item
        f1 = numeric_mean(row, f1_metric)
        qwk = numeric_mean(row, qwk_metric)
        f1_std = metric_std(row, f1_metric)
        return (
            -(f1 if f1 is not None else -math.inf),
            -(qwk if qwk is not None else -math.inf),
            f1_std if f1_std is not None else math.inf,
            row["strategy"],
        )

    winner_idx, winner_row = sorted(tied, key=rank_key)[0]
    return {
        "winner_index": winner_idx,
        "winner_key": winner_row["key"],
        "winner_group": winner_row["group"],
        "winner_strategy": winner_row["strategy"],
        "display_f1": max_display_f1,
        "tie_indices": [idx for idx, _ in tied],
        "tie_keys": [row["key"] for _, row in tied],
        "tie_break_used": len(tied) > 1,
        "tie_break_rule": "higher QWK, then lower F1 standard deviation",
    }


def add_highlight_role(
    cells: dict[str, dict[str, list[str]]],
    row_idx: int,
    column: str,
    role: str,
) -> None:
    row_key = str(row_idx)
    cells.setdefault(row_key, {}).setdefault(column, [])
    if role not in cells[row_key][column]:
        cells[row_key][column].append(role)


def build_highlighting(
    rows_by_language: dict[str, list[dict[str, Any]]],
    section: str,
) -> dict[str, Any]:
    if section == "overall":
        ranking_specs = [
            {
                "column": "F1",
                "f1_metric": "f1_macro",
                "qwk_metric": "qwk",
                "pct": True,
                "digits": 2,
            }
        ]
    elif section == "seen_unseen":
        ranking_specs = [
            {
                "column": "F1 (Seen)",
                "f1_metric": "f1_seen",
                "qwk_metric": "qwk_seen",
                "pct": True,
                "digits": 2,
            },
            {
                "column": "F1 (Unseen)",
                "f1_metric": "f1_unseen",
                "qwk_metric": "qwk_unseen",
                "pct": True,
                "digits": 2,
            },
        ]
    else:
        raise ValueError(f"Unknown highlight section: {section}")

    output: dict[str, Any] = {}
    for language, rows in rows_by_language.items():
        cells: dict[str, dict[str, list[str]]] = {}
        rankings: list[dict[str, Any]] = []
        indexed_rows = list(enumerate(rows))
        grouped: dict[str, list[tuple[int, dict[str, Any]]]] = {}
        for idx, row in indexed_rows:
            grouped.setdefault(row["group"], []).append((idx, row))

        for spec in ranking_specs:
            global_winner = choose_winner(
                indexed_rows,
                spec["f1_metric"],
                spec["qwk_metric"],
                spec["pct"],
                spec["digits"],
            )
            if global_winner:
                add_highlight_role(cells, global_winner["winner_index"], spec["column"], "global_winner")
                for tie_idx in global_winner["tie_indices"]:
                    if global_winner["tie_break_used"]:
                        add_highlight_role(cells, tie_idx, spec["column"], "tie_candidate")
                if global_winner["tie_break_used"]:
                    add_highlight_role(cells, global_winner["winner_index"], spec["column"], "tie_break_winner")
                rankings.append({"scope": "global", "metric_column": spec["column"], **global_winner})

            for group, group_rows in grouped.items():
                group_winner = choose_winner(
                    group_rows,
                    spec["f1_metric"],
                    spec["qwk_metric"],
                    spec["pct"],
                    spec["digits"],
                )
                if not group_winner:
                    continue
                add_highlight_role(cells, group_winner["winner_index"], spec["column"], "group_winner")
                for tie_idx in group_winner["tie_indices"]:
                    if group_winner["tie_break_used"]:
                        add_highlight_role(cells, tie_idx, spec["column"], "tie_candidate")
                if group_winner["tie_break_used"]:
                    add_highlight_role(cells, group_winner["winner_index"], spec["column"], "tie_break_winner")
                rankings.append({"scope": group, "metric_column": spec["column"], **group_winner})

        output[language] = {
            "cells": cells,
            "rankings": rankings,
            "legend": {
                "yellow": "Best displayed F1 within model family.",
                "red": "Best displayed F1 overall.",
                "blue_border": "Displayed-F1 tie resolved by higher QWK, then lower F1 standard deviation.",
            },
        }
    return output


def latex_escape(text: str) -> str:
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
    }
    return "".join(replacements.get(ch, ch) for ch in text)


def decorated_latex_value(
    row: dict[str, Any],
    metric_name: str,
    pct: bool,
    digits: int,
    group_max: float | None,
    global_max: float | None,
    group_size: int,
) -> str:
    value = numeric_mean(row, metric_name)
    text = format_stat(row["metrics"].get(metric_name, mean_std([])), pct, digits, latex=True)
    if value is None:
        return text
    is_group_max = (
        group_size > 1
        and group_max is not None
        and math.isclose(value, group_max, rel_tol=1e-12, abs_tol=1e-12)
    )
    is_global_max = global_max is not None and math.isclose(
        value, global_max, rel_tol=1e-12, abs_tol=1e-12
    )
    if is_group_max and is_global_max:
        text = r"\uline{\textbf{" + text + "}}"
    elif is_group_max:
        text = r"\uline{" + text + "}"
    elif is_global_max:
        text = r"\textbf{" + text + "}"
    return text


def make_latex_table(
    rows: list[dict[str, Any]],
    columns: list[tuple[str, str, bool, int]],
    language: str,
    table_kind: str,
) -> str:
    metric_names = [name for name, *_ in columns]
    global_max = {
        name: max(
            [v for v in (numeric_mean(row, name) for row in rows) if v is not None],
            default=None,
        )
        for name in metric_names
    }
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(row["group"], []).append(row)

    group_max: dict[tuple[str, str], float | None] = {}
    for group, group_rows in grouped.items():
        for name in metric_names:
            group_max[(group, name)] = max(
                [v for v in (numeric_mean(row, name) for row in group_rows) if v is not None],
                default=None,
            )

    if table_kind == "overall":
        header = (
            r"\begin{tabular}{@{}llrrrrrrrr@{}}"
            "\n"
            r"\toprule"
            "\n"
            r"\textbf{Method} & \textbf{Strategy} & \textbf{Accuracy} & \textbf{Precision} & "
            r"\textbf{Recall} & \textbf{F1} & \textbf{QWK} & \textbf{Negative F1} & "
            r"\textbf{Neutral F1} & \textbf{Positive F1} \\"
        )
    else:
        header = (
            r"\begin{tabular}{@{}llrrrr@{}}"
            "\n"
            r"\toprule"
            "\n"
            r"\textbf{Method} & \textbf{Strategy} & \textbf{F1 (Seen)} & "
            r"\textbf{F1 (Unseen)} & \textbf{QWK (Seen)} & \textbf{QWK (Unseen)} \\"
        )

    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\resizebox{\textwidth}{!}{%",
        header,
        r"\midrule",
    ]
    for group, group_rows in grouped.items():
        for i, row in enumerate(group_rows):
            group_label = latex_escape(group) if i == 0 else ""
            cells = [group_label, latex_escape(row["strategy"])]
            for name, _, pct, digits in columns:
                cells.append(
                    decorated_latex_value(
                        row,
                        name,
                        pct,
                        digits,
                        group_max[(group, name)],
                        global_max[name],
                        len(group_rows),
                    )
                )
            lines.append(" & ".join(cells) + r" \\")
        lines.append(r"\midrule")
    lines[-1] = r"\bottomrule"
    lines.extend(
        [
            r"\end{tabular}}",
            rf"\caption{{{table_kind.title()} results for the {language.title()} dataset.}}",
            rf"\label{{tab:{table_kind}-{language}-generated}}",
            r"\end{table}",
        ]
    )
    return "\n".join(lines)


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def build_inventory() -> list[dict[str, Any]]:
    inventory: list[dict[str, Any]] = []
    for root_name in ("results", "reviews"):
        root = ROOT / root_name
        if not root.exists():
            continue
        for summary_path in sorted(root.glob("**/test_metrics_summary.json")):
            parts = summary_path.relative_to(ROOT).parts
            if len(parts) < 4:
                continue
            language = summary_path.parent.name
            if language not in LANGUAGES:
                continue
            summary = load_json(summary_path)
            metrics = metric_values_from_summary(summary)
            inventory.append(
                {
                    "kind": "summary",
                    "path": rel(summary_path),
                    "language": language,
                    "f1": format_stat(metrics["f1_macro"], True, 2),
                    "qwk": format_stat(metrics["qwk"], False, 3),
                }
            )
        for metric_path in sorted(root.glob("**/*_metrics.json")):
            if "large-language-models" not in str(metric_path) and "/llms/" not in str(metric_path):
                continue
            metrics = metric_values_from_single(load_json(metric_path))
            language = "slovenian" if "/slovenian/" in str(metric_path) else "serbian"
            inventory.append(
                {
                    "kind": "llm",
                    "path": rel(metric_path),
                    "language": language,
                    "f1": format_stat(metrics["f1_macro"], True, 2),
                    "qwk": format_stat(metrics["qwk"], False, 3),
                }
            )
    return inventory


def normalized_source_map() -> list[dict[str, Any]]:
    fields = [
        "key",
        "group",
        "strategy",
        "kind",
        "dir_candidates",
        "prediction_candidates",
        "notes",
    ]
    normalized = []
    for spec in TABLE_SPECS:
        item = {field: spec.get(field, [] if field.endswith("_candidates") else "") for field in fields}
        normalized.append(item)
    return normalized


def notebook_cell(cell_type: str, source: str) -> dict[str, Any]:
    cell = {
        "cell_type": cell_type,
        "metadata": {},
        "source": source.splitlines(keepends=True),
    }
    if cell_type == "code":
        cell["execution_count"] = None
        cell["outputs"] = []
    return cell


def html_escape(text: Any) -> str:
    replacements = {
        "&": "&amp;",
        "<": "&lt;",
        ">": "&gt;",
        '"': "&quot;",
        "'": "&#39;",
    }
    return "".join(replacements.get(ch, ch) for ch in str(text))


def css_for_roles(roles: list[str]) -> str:
    styles = []
    if "global_winner" in roles:
        styles.append("background-color: #f6b3b3")
        styles.append("font-weight: 700")
    elif "group_winner" in roles:
        styles.append("background-color: #fff2a8")
        styles.append("font-weight: 700")
    elif "tie_candidate" in roles:
        styles.append("background-color: #e8f1ff")
    if "tie_candidate" in roles:
        styles.append("border: 2px solid #2563eb")
    if "tie_break_winner" in roles:
        styles.append("box-shadow: inset 0 0 0 2px #2563eb")
    return "; ".join(styles)


def display_rows_to_html_table(
    rows: list[dict[str, Any]],
    highlight: dict[str, Any],
) -> str:
    if not rows:
        return ""
    visible_rows = [{k: v for k, v in row.items() if k != "Source"} for row in rows]
    headers = list(visible_rows[0])
    cells = highlight.get("cells", {})
    lines = [
        '<table class="final-results">',
        "<thead>",
        "<tr>" + "".join(f"<th>{html_escape(header)}</th>" for header in headers) + "</tr>",
        "</thead>",
        "<tbody>",
    ]
    for row_idx, row in enumerate(visible_rows):
        lines.append("<tr>")
        for header in headers:
            roles = cells.get(str(row_idx), {}).get(header, [])
            style = css_for_roles(roles)
            style_attr = f' style="{style}"' if style else ""
            lines.append(f"<td{style_attr}>{html_escape(row[header])}</td>")
        lines.append("</tr>")
    lines.extend(["</tbody>", "</table>"])
    return "\n".join(lines)


def rankings_to_markdown(rankings: list[dict[str, Any]]) -> str:
    rows = []
    for item in rankings:
        if item["scope"] == "global" or item.get("tie_break_used"):
            rows.append(item)
    if not rows:
        return ""
    lines = [
        "| Scope | Column | Winner | Display F1 | Tie resolved? | Tie candidates |",
        "|---|---|---|---:|---|---|",
    ]
    for item in rows:
        tie_candidates = ", ".join(item.get("tie_keys", []))
        lines.append(
            "| {scope} | {column} | {winner} | {display_f1:.2f} | {tie} | {candidates} |".format(
                scope=html_escape(item["scope"]),
                column=html_escape(item["metric_column"]),
                winner=html_escape(f"{item['winner_group']} / {item['winner_strategy']}"),
                display_f1=float(item["display_f1"]),
                tie="yes" if item.get("tie_break_used") else "no",
                candidates=html_escape(tie_candidates),
            )
        )
    return "\n".join(lines)


def write_markdown(output: dict[str, Any]) -> None:
    lines = [
        "# Final Result Analysis",
        "",
        "Generated by `reviews/build_final_results_notebook.py`.",
        "",
        "Legend: <span style=\"background-color:#fff2a8;padding:2px 6px;\">yellow</span> = best displayed F1 within a model family; "
        "<span style=\"background-color:#f6b3b3;padding:2px 6px;\">red</span> = best displayed F1 overall; "
        "<span style=\"border:2px solid #2563eb;padding:2px 6px;\">blue border</span> = displayed-F1 tie resolved by higher QWK, then lower F1 standard deviation.",
        "",
        "<style>",
        "table.final-results { border-collapse: collapse; font-size: 0.9em; margin-bottom: 1.5rem; }",
        "table.final-results th, table.final-results td { border: 1px solid #d0d7de; padding: 4px 8px; }",
        "table.final-results th { background: #f6f8fa; }",
        "</style>",
    ]

    lines.extend(["", "## Overall Tables", ""])
    for language in LANGUAGES:
        lines.extend(
            [
                f"### {language.title()}",
                "",
                display_rows_to_html_table(
                    output["overall_display"][language],
                    output["highlighting"]["overall"][language],
                ),
                "",
            ]
        )
        ranking_md = rankings_to_markdown(output["highlighting"]["overall"][language]["rankings"])
        if ranking_md:
            lines.extend(["#### Tie/Overall Notes", "", ranking_md, ""])

    lines.extend(
        [
            "## Joint Scoring Tables",
            "",
            "These tables exclude prompting LLMs and automatic baselines. The aggregated score is the average of min-max normalized `F1 - SD` and `QWK - SD` within each language.",
            "",
        ]
    )
    for language in LANGUAGES:
        lines.extend(
            [
                f"### Overall: {language.title()}",
                "",
                display_rows_to_html_table(output["joint_scoring_display"][language], {"cells": {}}),
                "",
            ]
        )

    lines.extend(["", "## Seen vs Unseen Tables", ""])
    for language in LANGUAGES:
        lines.extend(
            [
                f"### {language.title()}",
                "",
                display_rows_to_html_table(
                    output["seen_unseen_display"][language],
                    output["highlighting"]["seen_unseen"][language],
                ),
                "",
            ]
        )
        ranking_md = rankings_to_markdown(output["highlighting"]["seen_unseen"][language]["rankings"])
        if ranking_md:
            lines.extend(["#### Tie/Overall Notes", "", ranking_md, ""])

    lines.extend(
        [
            "## LLM Learning To Defer With New Experts",
            "",
            "This section compares the old HAN+XLMR uncertainty-calibrated LLM setup against the new compact and rich calibrations over the new expert set. "
            "`new-calibration` is the compact prompt style, `new-rich-calibration` is the richer uncertainty-field prompt style, and `old-calibration` is the previous HAN+XLMR setup from `results/large-language-models`.",
            "",
            display_rows_to_html_table(output["llm_learning_to_defer_with_new_experts"], {"cells": {}}),
            "",
        ]
    )

    lines.extend(
        [
            "## Best LLM Learning To Defer Runs Per Expert",
            "",
            "For each language/expert pair, this selects the best run across `old-calibration`, `new-calibration`, and `new-rich-calibration`, across masked/unmasked prompt variants and medium/heavy autoruns where available. Selection uses highest macro F1, then higher QWK, then higher accuracy.",
            "",
            "### Overall Winners",
            "",
            display_rows_to_html_table(output["best_llm_learning_to_defer_by_expert"], {"cells": {}}),
            "",
            "### Seen vs Unseen for Overall Winners",
            "",
            display_rows_to_html_table(output["best_llm_learning_to_defer_seen_unseen_by_expert"], {"cells": {}}),
            "",
        ]
    )

    lines.extend(
        [
            "## Source Map",
            "",
            "The canonical machine-readable source map is `reviews/scratchpad/final_table_model_map.json`.",
            "",
        ]
    )
    MARKDOWN_PATH.write_text("\n".join(lines), encoding="utf-8")


def write_notebook() -> None:
    code_load = """\
import json
from pathlib import Path
import pandas as pd
from IPython.display import Markdown, display

pd.set_option("display.max_columns", None)
pd.set_option("display.max_colwidth", None)
pd.set_option("display.max_rows", None)

def find_repo_root(start=None):
    start = Path.cwd() if start is None else Path(start)
    for candidate in [start, *start.parents]:
        results_path = candidate / "reviews" / "scratchpad" / "final_table_results.json"
        if results_path.exists():
            return candidate
    raise FileNotFoundError(
        "Could not find reviews/scratchpad/final_table_results.json from "
        f"{start} or its parents. Run reviews/build_final_results_notebook.py first."
    )

ROOT = find_repo_root()
RESULTS_PATH = ROOT / "reviews" / "scratchpad" / "final_table_results.json"
with RESULTS_PATH.open("r", encoding="utf-8") as f:
    results = json.load(f)

def as_df(section, language):
    return pd.DataFrame(results[section][language])

def cell_style(roles):
    styles = []
    if "global_winner" in roles:
        styles.append("background-color: #f6b3b3")
        styles.append("font-weight: 700")
    elif "group_winner" in roles:
        styles.append("background-color: #fff2a8")
        styles.append("font-weight: 700")
    elif "tie_candidate" in roles:
        styles.append("background-color: #e8f1ff")
    if "tie_candidate" in roles:
        styles.append("border: 2px solid #2563eb")
    if "tie_break_winner" in roles:
        styles.append("box-shadow: inset 0 0 0 2px #2563eb")
    return "; ".join(styles)

def style_results_table(section, language):
    display_key = f"{section}_display"
    df = pd.DataFrame(results[display_key][language]).drop(columns=["Source"])
    cells = results["highlighting"][section][language]["cells"]
    styles = pd.DataFrame("", index=df.index, columns=df.columns)
    for row_idx, columns in cells.items():
        for column, roles in columns.items():
            if column in styles.columns:
                styles.loc[int(row_idx), column] = cell_style(roles)
    return df.style.apply(lambda _: styles, axis=None)

def show_highlight_legend():
    display(Markdown(
        "**Highlighting:** yellow = best displayed F1 within each model family; "
        "red = best displayed F1 overall; blue border = displayed-F1 tie resolved "
        "by higher QWK, then lower F1 standard deviation."
    ))

source_map = results["source_map"]
print(f"Loaded {RESULTS_PATH}")
	"""
    code_overall = """\
show_highlight_legend()
for language in ["slovenian", "serbian"]:
    print(f"\\n=== Overall: {language} ===")
    display(style_results_table("overall", language))
"""
    code_joint = """\
display(Markdown(
    "This excludes prompting LLMs and automatic baselines. "
    "`F1 - SD` and `QWK - SD` are conservative one-standard-deviation selection heuristics across the three runs, not confidence intervals. "
    "The aggregated score averages min-max normalized `F1 - SD` and `QWK - SD` within each language."
))
for language in ["slovenian", "serbian"]:
    print(f"\\n=== Overall: {language} ===")
    display(pd.DataFrame(results["joint_scoring_display"][language]).drop(columns=["Source"]))
"""
    code_seen = """\
show_highlight_legend()
for language in ["slovenian", "serbian"]:
    print(f"\\n=== Seen vs unseen: {language} ===")
    display(style_results_table("seen_unseen", language))
"""
    code_llm_defer = """\
display(Markdown(
    "Comparison of learning-to-defer LLM runs. "
    "`new-calibration` uses the compact current prompt, `new-rich-calibration` uses rich uncertainty fields, "
    "and `old-calibration` is the previous HAN+XLMR-only uncertainty setup."
))
llm_defer = pd.DataFrame(results["llm_learning_to_defer_with_new_experts"])
display(llm_defer)

for calibration in ["new-rich-calibration", "new-calibration", "old-calibration"]:
    subset = llm_defer[llm_defer["Calibration"] == calibration]
    if not subset.empty:
        print(f"\\n=== {calibration} ===")
        display(subset.sort_values(["Language", "F1", "QWK"], ascending=[True, False, False]))
"""
    code_highlights = """\
for section in ["overall", "seen_unseen"]:
    for language in ["slovenian", "serbian"]:
        rankings = pd.DataFrame(results["highlighting"][section][language]["rankings"])
        interesting = rankings[(rankings["scope"] == "global") | (rankings["tie_break_used"])]
        print(f"\\n=== {section}: {language} ===")
        display(interesting[[
            "scope", "metric_column", "winner_group", "winner_strategy",
            "display_f1", "tie_break_used", "tie_keys", "tie_break_rule"
        ]])
"""
    code_sources = """\
pd.DataFrame(source_map)[["key", "group", "strategy", "kind", "dir_candidates", "prediction_candidates", "notes"]].fillna("")
"""
    code_latex = """\
for name, latex in results["latex_tables"].items():
    print(f"\\n% --- {name} ---")
    print(latex)
"""
    code_inventory = """\
inventory = pd.read_csv(ROOT / "reviews" / "scratchpad" / "result_inventory.csv")
inventory.sort_values(["language", "f1"], ascending=[True, False]).head(80)
"""
    code_best_ltd = """\
display(Markdown(
    "Best run per language/expert across `old-calibration`, `new-calibration`, and `new-rich-calibration`, "
    "irrespective of prompt variant or autorun. Selection is highest macro F1, then higher QWK, then higher accuracy."
))
display(pd.DataFrame(results["best_llm_learning_to_defer_by_expert"]))

display(Markdown("Seen/unseen split for the same selected runs."))
display(pd.DataFrame(results["best_llm_learning_to_defer_seen_unseen_by_expert"]))
"""
    nb = {
        "cells": [
            notebook_cell(
                "markdown",
                "# Final Result Analysis\n\n"
                "Generated by `reviews/build_final_results_notebook.py`. "
                "The source map is explicit so paper rows do not accidentally absorb extra experiments.",
            ),
            notebook_cell("code", code_load),
            notebook_cell("markdown", "## Overall Tables"),
            notebook_cell("code", code_overall),
            notebook_cell("markdown", "## Joint Scoring Tables"),
            notebook_cell("code", code_joint),
            notebook_cell("markdown", "## Seen vs Unseen Tables"),
            notebook_cell("code", code_seen),
            notebook_cell("markdown", "## LLM Learning To Defer With New Experts"),
            notebook_cell("code", code_llm_defer),
            notebook_cell("markdown", "## Highlight Decisions"),
            notebook_cell("code", code_highlights),
            notebook_cell("markdown", "## Source Map"),
            notebook_cell("code", code_sources),
            notebook_cell("markdown", "## Generated LaTeX"),
            notebook_cell("code", code_latex),
            notebook_cell("markdown", "## Inventory"),
            notebook_cell("code", code_inventory),
            notebook_cell("markdown", "## Best LLM Learning To Defer Runs Per Expert"),
            notebook_cell("code", code_best_ltd),
        ],
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {"name": "python", "pygments_lexer": "ipython3"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    NOTEBOOK_PATH.write_text(json.dumps(nb, indent=2, ensure_ascii=False), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-notebook", action="store_true")
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    NOTEBOOK_PATH.parent.mkdir(parents=True, exist_ok=True)

    overall = {
        language: [build_overall_row(spec, language) for spec in TABLE_SPECS]
        for language in LANGUAGES
    }
    seen_unseen = {
        language: [build_seen_unseen_row(spec, language) for spec in TABLE_SPECS]
        for language in LANGUAGES
    }
    overall_display = {
        language: rows_to_display(rows, METRIC_COLUMNS)
        for language, rows in overall.items()
    }
    seen_unseen_display = {
        language: rows_to_display(rows, SEEN_UNSEEN_COLUMNS)
        for language, rows in seen_unseen.items()
    }
    expert_stability_display = {
        language: build_expert_stability_rows(rows)
        for language, rows in overall.items()
    }
    joint_scoring_display = {
        language: build_joint_scoring_rows(expert_stability_display[language])
        for language in LANGUAGES
    }
    highlighting = {
        "overall": build_highlighting(overall, "overall"),
        "seen_unseen": build_highlighting(seen_unseen, "seen_unseen"),
    }
    llm_defer_rows = build_llm_defer_rows()
    best_llm_defer_rows = build_best_llm_defer_rows(llm_defer_rows)
    best_llm_defer_seen_unseen_rows = build_best_llm_defer_seen_unseen_rows(best_llm_defer_rows)

    latex_tables: dict[str, str] = {}
    for language in LANGUAGES:
        latex_tables[f"overall_{language}"] = make_latex_table(
            overall[language], METRIC_COLUMNS, language, "overall"
        )
        latex_tables[f"seen_unseen_{language}"] = make_latex_table(
            seen_unseen[language], SEEN_UNSEEN_COLUMNS, language, "seen_unseen"
        )

    source_map = normalized_source_map()
    output = {
        "source_map": source_map,
        "overall": overall,
        "seen_unseen": seen_unseen,
        "overall_display": overall_display,
        "seen_unseen_display": seen_unseen_display,
        "expert_stability_display": expert_stability_display,
        "joint_scoring_display": joint_scoring_display,
        "llm_learning_to_defer_with_new_experts": llm_defer_rows,
        "best_llm_learning_to_defer_by_expert": best_llm_defer_rows,
        "best_llm_learning_to_defer_seen_unseen_by_expert": best_llm_defer_seen_unseen_rows,
        "highlighting": highlighting,
        "latex_tables": latex_tables,
    }

    (OUT_DIR / "final_table_model_map.json").write_text(
        json.dumps(source_map, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (OUT_DIR / "final_table_results.json").write_text(
        json.dumps(output, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (OUT_DIR / "final_tables.tex").write_text(
        "\n\n".join(latex_tables.values()), encoding="utf-8"
    )

    for language in LANGUAGES:
        write_csv(
            OUT_DIR / f"final_table_overall_{language}.csv",
            overall_display[language],
            list(overall_display[language][0].keys()),
        )
        write_csv(
            OUT_DIR / f"final_table_seen_unseen_{language}.csv",
            seen_unseen_display[language],
            list(seen_unseen_display[language][0].keys()),
        )
        write_csv(
            OUT_DIR / f"expert_stability_ranking_{language}.csv",
            expert_stability_display[language],
            EXPERT_STABILITY_COLUMNS,
        )
        write_csv(
            OUT_DIR / f"joint_scoring_{language}.csv",
            joint_scoring_display[language],
            JOINT_SCORING_COLUMNS,
        )
    write_csv(
        OUT_DIR / "llm_learning_to_defer_with_new_experts.csv",
        llm_defer_rows,
        LLM_DEFER_COLUMNS,
    )
    write_csv(
        OUT_DIR / "best_llm_learning_to_defer_by_expert.csv",
        best_llm_defer_rows,
        BEST_LLM_DEFER_COLUMNS,
    )
    write_csv(
        OUT_DIR / "best_llm_learning_to_defer_seen_unseen_by_expert.csv",
        best_llm_defer_seen_unseen_rows,
        BEST_LLM_DEFER_SEEN_UNSEEN_COLUMNS,
    )

    inventory = build_inventory()
    write_csv(
        OUT_DIR / "result_inventory.csv",
        inventory,
        ["kind", "path", "language", "f1", "qwk"],
    )

    if not args.skip_notebook:
        write_notebook()
    write_markdown(output)

    print(f"Wrote {OUT_DIR / 'final_table_results.json'}")
    print(f"Wrote {OUT_DIR / 'final_tables.tex'}")
    if not args.skip_notebook:
        print(f"Wrote {NOTEBOOK_PATH}")
    print(f"Wrote {MARKDOWN_PATH}")


if __name__ == "__main__":
    main()
