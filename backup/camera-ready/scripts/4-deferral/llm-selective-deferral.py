#!/usr/bin/env python3
"""DSPy selective-deferral optimizer/query pipeline.

This trains a gate over uncertainty examples rather than another full-test
sentiment classifier. The LLM sees a primary expert, auxiliary expert
agreement/disagreement, and article evidence, then chooses:

- keep_plm: keep the primary expert label
- override: replace it with the LLM sentiment
- abstain_uncertain: flag for manual/third-stage adjudication
"""

from __future__ import annotations

import argparse
import concurrent.futures as futures
import copy
import json
import math
import os
import random
import re
import threading
import time
from collections import Counter
from pathlib import Path
from typing import Any, Literal

import dspy
import numpy as np
from openai import OpenAI
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    cohen_kappa_score,
    precision_recall_fscore_support,
)
from sklearn.model_selection import train_test_split

from dspy.teleprompt import MIPROv2
from dspy_gemma_common import (
    EXPERT_DISPLAY_NAMES,
    INT_TO_STR_LABEL,
    LABELS_INT,
    MASK_TOKEN,
    OpenAICompletionLM,
    STR_TO_INT_LABEL,
    configure_openai_lm,
    default_sample_size,
    load_json,
    mask_aspect_in_text,
    maybe_clip_text,
    sanity_check_openai_chat_endpoint,
    strip_aspect_tags,
    to_jsonable,
    write_json,
)


ROOT = Path(os.environ.get("ABSA_RELEASE_ROOT", Path(__file__).resolve().parents[2])).resolve()
UNCERTAINTY_ROOT = ROOT / "reviews" / "uncertainty"
OUTPUT_ROOT = UNCERTAINTY_ROOT / "llm-selective-deferral"

DEFAULT_EXPERTS = {
    "slovenian": ["legacy_han_xlmr_masked", "longformer_masked", "slavic_specific_masked"],
    "serbian": ["legacy_han_xlmr_masked", "longformer_masked", "slavic_specific_masked", "mdeberta_masked"],
}

DEFAULT_PRIMARY = {
    "slovenian": "legacy_han_xlmr_masked",
    "serbian": "legacy_han_xlmr_masked",
}

LOCAL_EXPERT_DISPLAY_NAMES = {
    **EXPERT_DISPLAY_NAMES,
    "legacy_han_xlmr_masked": "HAN-XLMR Masked",
    "han_xlmr_masked": "HAN-XLMR Masked",
    "longformer_masked": "Longformer Masked",
    "mdeberta_masked": "mDeBERTa-v3 Masked",
    "slavic_specific_masked": "SloBERTa/BERTic Masked",
}

FALLBACK_SUCCESS = {
    "legacy_han_xlmr_masked": {
        "expert": "legacy_han_xlmr_masked",
        "json_key": "global-context-modelling/simplified-dart-xlmr",
        "probabilities_key": "global-context-modelling/simplified-dart-xlmr/probabilities",
        "uncertainty_key": "global-context-modelling/simplified-dart-xlmr/uncertainty",
    }
}

ACTION_KEEP = "keep_plm"
ACTION_OVERRIDE = "override"
ACTION_ABSTAIN = "abstain_uncertain"
ACTIONS = [ACTION_KEEP, ACTION_OVERRIDE, ACTION_ABSTAIN]
LABELS_STR = ["negative", "neutral", "positive"]


class SelectiveDeferralSignature(dspy.Signature):
    """Decide whether to keep an expert prediction, override it, or abstain.

    Use the primary expert when it is confident and supported by article
    evidence. Override only when the article evidence or auxiliary expert
    consensus clearly contradicts the primary expert. Abstain when the case is
    too ambiguous for an automatic override.
    """

    article: str = dspy.InputField(
        desc="The document text. The target aspect may be explicit or replaced by [MASK]."
    )
    aspect: str = dspy.InputField(desc="The target aspect or [MASK].")
    primary_expert: str = dspy.InputField(desc="Primary expert whose prediction is being audited.")
    primary_prediction: Literal["negative", "neutral", "positive"] = dspy.InputField(
        desc="Primary expert sentiment label."
    )
    primary_probabilities: str = dspy.InputField(
        desc="Primary expert probabilities for negative, neutral, positive."
    )
    primary_uncertainty: str = dspy.InputField(
        desc="Primary expert confidence, entropy, vote distribution, and threshold hints."
    )
    auxiliary_experts: str = dspy.InputField(
        desc="Other expert predictions, probabilities, uncertainty, and agreement summary."
    )
    routing_context: str = dspy.InputField(
        desc="Why this item was selected for possible deferral."
    )
    reasoning: str = dspy.OutputField(
        desc="Brief evidence-based reasoning. State whether evidence supports keeping, overriding, or abstaining."
    )
    action: Literal["keep_plm", "override", "abstain_uncertain"] = dspy.OutputField(
        desc="Decision: keep_plm, override, or abstain_uncertain."
    )
    sentiment: Literal["negative", "neutral", "positive"] = dspy.OutputField(
        desc="Final sentiment if action is override; otherwise repeat the primary label."
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["calibrate", "query"])
    parser.add_argument("--language", required=True, choices=["slovenian", "serbian"])
    parser.add_argument("--prompt-variant", required=True, choices=["masked", "unmasked"])
    parser.add_argument("--autorun", default="medium", choices=["light", "medium", "heavy"])
    parser.add_argument("--primary-expert", default=None)
    parser.add_argument("--experts", nargs="+", default=None)
    parser.add_argument("--uncertainty-root", default=str(UNCERTAINTY_ROOT))
    parser.add_argument("--output-root", default=str(OUTPUT_ROOT))
    parser.add_argument("--split-index", type=int, default=0)
    parser.add_argument("--train-size", type=int, default=None)
    parser.add_argument("--val-size", type=int, default=None)
    parser.add_argument("--balanced-sampling", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--prefer-balanced-splits", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--hard-gated-query", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--calibration-sampling",
        choices=["hard_biased", "low_confidence_stratified"],
        default="low_confidence_stratified",
    )
    parser.add_argument("--uncertain-pool-rate", type=float, default=0.10)
    parser.add_argument("--gate-rate", type=float, default=None)
    parser.add_argument("--hard-candidate-multiplier", type=float, default=3.0)
    parser.add_argument("--min-confidence", type=float, default=0.80)
    parser.add_argument("--max-article-chars", type=int, default=10000)
    parser.add_argument("--student-model", default="gemma27b")
    parser.add_argument("--teacher-model", default="qwen72b")
    parser.add_argument("--student-api-base", default="http://127.0.0.1:8000/v1")
    parser.add_argument("--teacher-api-base", default="http://127.0.0.1:8001/v1")
    parser.add_argument("--student-api-bases", default=None)
    parser.add_argument("--student-endpoint-type", choices=["chat", "completion"], default="chat")
    parser.add_argument("--teacher-endpoint-type", choices=["chat", "completion"], default="chat")
    parser.add_argument("--api-key", default=os.environ.get("OPENAI_API_KEY", "EMPTY"))
    parser.add_argument("--teacher-label", default="qwen-2.5-72b")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--top-k", type=int, default=None)
    parser.add_argument("--max-tokens", type=int, default=1024)
    parser.add_argument("--student-max-tokens", type=int, default=None)
    parser.add_argument("--teacher-max-tokens", type=int, default=None)
    parser.add_argument("--request-logprobs", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--top-logprobs", type=int, default=5)
    parser.add_argument("--num-retries", type=int, default=3)
    parser.add_argument("--miprov2-temp", type=float, default=1.0)
    parser.add_argument("--dspy-num-threads", type=int, default=8)
    parser.add_argument("--max-errors", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--eval-workers", type=int, default=4)
    parser.add_argument("--num-workers-per-endpoint", type=int, default=6)
    parser.add_argument("--num-workers-per-endpoints", default=None)
    parser.add_argument("--limit-items", type=int, default=None)
    parser.add_argument("--sample-items", type=int, default=None)
    parser.add_argument("--optimized-program", default=None)
    parser.add_argument("--skip-endpoint-check", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--disable-program-aware-proposer", action="store_true")
    parser.add_argument("--disable-data-aware-proposer", action="store_true")
    parser.add_argument("--disable-tip-aware-proposer", action="store_true")
    parser.add_argument("--disable-fewshot-aware-proposer", action="store_true")
    parser.add_argument("--view-data-batch-size", type=int, default=3)
    return parser.parse_args()


def variant_name(prompt_masked: bool) -> str:
    return "masked" if prompt_masked else "unmasked"


def run_name(language: str, prompt_masked: bool, autorun: str, max_tokens: int, teacher_label: str) -> str:
    return (
        f"selective-deferral-teacher-{teacher_label}-{max_tokens}-"
        f"{autorun}-{language}-{variant_name(prompt_masked)}"
    )


def base_output_dir(args: argparse.Namespace) -> Path:
    primary = args.primary_expert or DEFAULT_PRIMARY[args.language]
    return Path(args.output_root) / args.language / args.prompt_variant / primary / args.autorun


def gate_rate_suffix(rate: float) -> str:
    value = round(rate * 100, 4)
    if float(value).is_integer():
        text = str(int(value))
    else:
        text = str(value).replace(".", "p")
    return f"gate_rate_{text}"


def output_dir(args: argparse.Namespace) -> Path:
    path = base_output_dir(args)
    if args.mode == "query" and args.gate_rate is not None:
        path = path.parent / f"{path.name}_{gate_rate_suffix(args.gate_rate)}"
    return path


def program_filename(args: argparse.Namespace) -> str:
    return (
        f"optimized_program_{args.language}_selective-deferral_"
        f"{args.prompt_variant}_teacher_{args.teacher_label}_"
        f"autorun_{args.autorun}_temp_{args.miprov2_temp}.json"
    )


def calibration_paths(args: argparse.Namespace) -> dict[str, Path]:
    prompt_masked = args.prompt_variant == "masked"
    out_dir = output_dir(args)
    base_run_name = run_name(args.language, prompt_masked, args.autorun, args.max_tokens, args.teacher_label)
    return {
        "output_dir": out_dir,
        "program": out_dir / program_filename(args),
        "metadata": out_dir / f"{base_run_name}_calibration_metadata.json",
        "predictions": out_dir / f"{base_run_name}_calibration_predictions.json",
        "metrics": out_dir / f"{base_run_name}_calibration_metrics.json",
    }


def query_paths(args: argparse.Namespace) -> dict[str, Path]:
    prompt_masked = args.prompt_variant == "masked"
    out_dir = output_dir(args)
    program_dir = base_output_dir(args)
    program_path = Path(args.optimized_program) if args.optimized_program else program_dir / program_filename(args)
    base_run_name = run_name(args.language, prompt_masked, args.autorun, args.max_tokens, args.teacher_label)
    return {
        "output_dir": out_dir,
        "program": program_path,
        "metadata": out_dir / f"{base_run_name}_test_metadata.json",
        "predictions": out_dir / f"{base_run_name}_test_predictions.json",
        "metrics": out_dir / f"{base_run_name}_test_metrics.json",
    }


def print_path_status(paths: dict[str, Path]) -> None:
    for name, path in paths.items():
        status = "exists" if path.exists() else "missing"
        print(f"  {name}: {status} {path}", flush=True)


def success_path(root: Path, expert: str, language: str) -> Path:
    return root / expert / language / "_SUCCESS.json"


def records_path(root: Path, expert: str, language: str, split_name: str, prefer_balanced: bool = True) -> Path:
    base = root / expert / language
    if split_name == "test":
        candidates = []
        if prefer_balanced:
            candidates.append(base / f"{language}_test_balanced.json")
        candidates.append(base / f"{language}_test_complete.json")
        for path in candidates:
            if path.exists():
                return path
        return candidates[-1]
    index = split_name.rsplit("_", 1)[-1]
    candidates = []
    if prefer_balanced:
        candidates.append(base / f"{language}_train_val_balanced_{index}.json")
    candidates.append(base / f"{language}_train_val_complete_{index}.json")
    for path in candidates:
        if path.exists():
            return path
    return candidates[-1]


def load_success(root: Path, expert: str, language: str) -> dict[str, Any]:
    path = success_path(root, expert, language)
    if not path.exists() and expert in FALLBACK_SUCCESS:
        return dict(FALLBACK_SUCCESS[expert])
    if not path.exists():
        raise FileNotFoundError(f"Missing uncertainty success file: {path}")
    return load_json(path)


def load_split(
    root: Path,
    expert: str,
    language: str,
    split_name: str,
    subset: str | None,
    prefer_balanced: bool = True,
) -> list[dict[str, Any]]:
    path = records_path(root, expert, language, split_name, prefer_balanced)
    data = load_json(path)
    if split_name == "test":
        if isinstance(data, dict) and "test" in data:
            return data["test"]
        if isinstance(data, list):
            return data
    if subset and isinstance(data, dict):
        return data.get(subset, [])
    raise ValueError(f"Could not load {split_name}/{subset} from {path}")


def label_str(value: Any) -> str | None:
    if isinstance(value, str):
        value = value.strip().lower()
        value = re.sub(r"<\|.*?\|>", "", value).strip()
        value = value.strip(" .,:;!?'\"`")
        if value in STR_TO_INT_LABEL:
            return value
        for label in LABELS_STR:
            if re.search(rf"\b{label}\b", value):
                return label
        try:
            value = int(value)
        except ValueError:
            return None
    if value in INT_TO_STR_LABEL:
        return INT_TO_STR_LABEL[int(value)]
    return None


def label_int(value: Any) -> int | None:
    normalized = label_str(value)
    return STR_TO_INT_LABEL.get(normalized) if normalized else None


def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def expert_display_name(expert: str, language: str) -> str:
    if expert == "slavic_specific_masked":
        return "SloBERTa Masked" if language == "slovenian" else "BERTic Masked"
    return LOCAL_EXPERT_DISPLAY_NAMES.get(expert, expert)


def probability_text(probs: dict[str, Any]) -> str:
    return (
        f"negative={safe_float(probs.get('Negative')):.4f}, "
        f"neutral={safe_float(probs.get('Neutral')):.4f}, "
        f"positive={safe_float(probs.get('Positive')):.4f}"
    )


def uncertainty_text(unc: dict[str, Any]) -> str:
    dist = unc.get("prediction_distribution", {}) or {}
    total = int(safe_float(unc.get("total_mc_samples"), 0))
    votes = (
        f"negative={int(dist.get('-1', 0) or 0)}/{total}, "
        f"neutral={int(dist.get('0', 0) or 0)}/{total}, "
        f"positive={int(dist.get('1', 0) or 0)}/{total}"
    )
    return (
        f"confidence={safe_float(unc.get('confidence_score')):.3f}; "
        f"predictive_entropy={safe_float(unc.get('predictive_entropy')):.3f}; "
        f"expected_entropy={safe_float(unc.get('expected_entropy')):.3f}; "
        f"mutual_information={safe_float(unc.get('mutual_information')):.3f}; "
        f"votes: {votes}"
    )


def article_aspect(item: dict[str, Any], prompt_masked: bool, max_chars: int) -> tuple[str, str]:
    article = item.get("article", "") or ""
    aspect = item.get("aspect", "") or ""
    if prompt_masked:
        return maybe_clip_text(mask_aspect_in_text(article), max_chars), MASK_TOKEN
    return maybe_clip_text(strip_aspect_tags(article), max_chars), aspect


def build_combined_rows(
    root: Path,
    experts: list[str],
    primary_expert: str,
    language: str,
    split_name: str,
    subset: str | None,
    prefer_balanced: bool = True,
) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
    success = {expert: load_success(root, expert, language) for expert in experts}
    by_expert = {}
    for expert in experts:
        rows = load_split(root, expert, language, split_name, subset, prefer_balanced)
        by_expert[expert] = {str(item.get("uuid")): item for item in rows if item.get("uuid") is not None}
    common = sorted(set.intersection(*(set(rows) for rows in by_expert.values())))
    output = []
    for uuid in common:
        primary = by_expert[primary_expert][uuid]
        if label_int(primary.get("sentiment")) is None:
            continue
        expert_rows = {}
        valid = True
        for expert in experts:
            item = by_expert[expert][uuid]
            s = success[expert]
            pred = label_int(item.get(s["json_key"]))
            if pred is None:
                valid = False
                break
            probs = item.get(s["probabilities_key"], {}) or {}
            unc = item.get(s["uncertainty_key"], {}) or {}
            expert_rows[expert] = {
                "prediction_int": pred,
                "prediction_label": INT_TO_STR_LABEL[pred],
                "probabilities": probs,
                "uncertainty": unc,
                "confidence": safe_float(unc.get("confidence_score")),
                "entropy": safe_float(unc.get("predictive_entropy")),
            }
        if not valid:
            continue
        primary_pred = expert_rows[primary_expert]["prediction_int"]
        aux_preds = [row["prediction_int"] for expert, row in expert_rows.items() if expert != primary_expert]
        disagree = sum(1 for pred in aux_preds if pred != primary_pred)
        max_aux_vote = max(Counter(aux_preds).values()) if aux_preds else 0
        output.append(
            {
                "uuid": uuid,
                "language": language,
                "base_item": primary,
                "gold": int(primary["sentiment"]),
                "experts": expert_rows,
                "primary_pred": primary_pred,
                "primary_confidence": expert_rows[primary_expert]["confidence"],
                "primary_entropy": expert_rows[primary_expert]["entropy"],
                "num_aux_disagree": disagree,
                "num_aux": len(aux_preds),
                "max_aux_vote": max_aux_vote,
                "hard_score": hard_score(
                    expert_rows[primary_expert]["confidence"],
                    expert_rows[primary_expert]["entropy"],
                    disagree,
                    len(aux_preds),
                ),
            }
        )
    return output, success


def hard_score(confidence: float, entropy: float, num_disagree: int, num_aux: int) -> float:
    disagreement = num_disagree / num_aux if num_aux else 0.0
    return (1.0 - confidence) + 0.25 * entropy + 0.50 * disagreement


def is_gate_candidate(row: dict[str, Any], min_confidence: float) -> bool:
    return row["primary_confidence"] < min_confidence or row["num_aux_disagree"] > 0


def selection_summary(rows: list[dict[str, Any]], min_confidence: float) -> dict[str, Any]:
    hard = [row for row in rows if is_gate_candidate(row, min_confidence)]
    labels = Counter(row["gold"] for row in hard)
    return {
        "total": len(rows),
        "hard_gate_count": len(hard),
        "hard_gate_rate": (len(hard) / len(rows)) if rows else 0.0,
        "hard_gate_label_counts": {INT_TO_STR_LABEL[label]: labels.get(label, 0) for label in LABELS_INT},
    }


def query_gate_summary(rows: list[dict[str, Any]], args: argparse.Namespace) -> dict[str, Any]:
    routed = [row for row in rows if is_query_gate_candidate(row, args)]
    labels = Counter(row["gold"] for row in routed)
    summary = {
        "total": len(rows),
        "routed_to_llm_count": len(routed),
        "bypassed_count": len(rows) - len(routed),
        "routed_to_llm_rate": (len(routed) / len(rows)) if rows else 0.0,
        "routed_label_counts": {INT_TO_STR_LABEL[label]: labels.get(label, 0) for label in LABELS_INT},
    }
    if args.gate_rate is not None:
        summary["gate_policy"] = "lowest_confidence_rate"
        summary["gate_rate"] = args.gate_rate
    else:
        summary["gate_policy"] = "confidence_threshold_or_aux_disagreement"
        summary["min_confidence"] = args.min_confidence
    return summary


def confidence_band_uuids(rows: list[dict[str, Any]], rate: float | None) -> set[str]:
    if rate is None or rate <= 0 or not rows:
        return set()
    count = max(1, min(len(rows), int(math.ceil(len(rows) * rate))))
    selected = sorted(
        rows,
        key=lambda row: (
            row["primary_confidence"],
            -row["primary_entropy"],
            -row["num_aux_disagree"],
            -row["hard_score"],
        ),
    )[:count]
    return {row["uuid"] for row in selected}


def is_query_gate_candidate(row: dict[str, Any], args: argparse.Namespace) -> bool:
    gate_uuids = getattr(args, "query_gate_uuids", None)
    if gate_uuids is not None:
        return row["uuid"] in gate_uuids
    return is_gate_candidate(row, args.min_confidence)


def sample_hard_records(
    rows: list[dict[str, Any]],
    size: int,
    seed: int,
    hard_candidate_multiplier: float,
    min_confidence: float,
    balanced: bool = True,
) -> list[dict[str, Any]]:
    if size <= 0 or size >= len(rows):
        return list(rows)
    candidate_size = min(len(rows), max(size, int(round(size * hard_candidate_multiplier))))
    ranked = sorted(
        rows,
        key=lambda row: (
            row["primary_confidence"] >= min_confidence and row["num_aux_disagree"] == 0,
            -row["hard_score"],
        ),
    )
    candidates = ranked[:candidate_size]
    if balanced:
        rng = random.Random(seed)
        by_label = {label: [row for row in candidates if row["gold"] == label] for label in LABELS_INT}
        min_count = min(len(items) for items in by_label.values())
        per_class = min(min_count, max(1, size // len(LABELS_INT)))
        sampled: list[dict[str, Any]] = []
        for label in LABELS_INT:
            label_rows = by_label[label][:]
            rng.shuffle(label_rows)
            sampled.extend(label_rows[:per_class])
        remainder = size - len(sampled)
        if remainder > 0:
            used = {row["uuid"] for row in sampled}
            rest = [row for row in candidates if row["uuid"] not in used]
            rng.shuffle(rest)
            sampled.extend(rest[:remainder])
        rng.shuffle(sampled)
        return sampled[:size]
    labels = [row["gold"] for row in candidates]
    counts = Counter(labels)
    if len(counts) < 2 or min(counts.values()) < 2 or size >= len(candidates):
        rng = random.Random(seed)
        rng.shuffle(candidates)
        return candidates[:size]
    _, sampled = train_test_split(candidates, test_size=size, stratify=labels, random_state=seed)
    return list(sampled)


def sample_low_confidence_stratified_records(
    rows: list[dict[str, Any]],
    size: int,
    seed: int,
    pool_rate: float,
    balanced: bool = True,
) -> list[dict[str, Any]]:
    if size <= 0 or size >= len(rows):
        return list(rows)
    pool_size = min(len(rows), max(size, int(math.ceil(len(rows) * pool_rate))))
    candidates = sorted(
        rows,
        key=lambda row: (
            row["primary_confidence"],
            -row["primary_entropy"],
            -row["num_aux_disagree"],
            -row["hard_score"],
        ),
    )[:pool_size]
    rng = random.Random(seed)
    if balanced:
        by_label = {
            label: sorted(
                [row for row in candidates if row["gold"] == label],
                key=lambda row: (row["primary_confidence"], -row["primary_entropy"], -row["hard_score"]),
            )
            for label in LABELS_INT
        }
        per_class = max(1, size // len(LABELS_INT))
        sampled: list[dict[str, Any]] = []
        for label in LABELS_INT:
            label_rows = by_label[label][:]
            sampled.extend(label_rows[:per_class])
        remainder = size - len(sampled)
        if remainder > 0:
            used = {row["uuid"] for row in sampled}
            rest = [row for row in candidates if row["uuid"] not in used]
            sampled.extend(rest[:remainder])
        rng.shuffle(sampled)
        return sampled[:size]
    labels = [row["gold"] for row in candidates]
    counts = Counter(labels)
    if len(counts) < 2 or min(counts.values()) < 2 or size >= len(candidates):
        rng.shuffle(candidates)
        return candidates[:size]
    _, sampled = train_test_split(candidates, test_size=size, stratify=labels, random_state=seed)
    return list(sampled)


def sample_calibration_records(
    rows: list[dict[str, Any]],
    size: int,
    seed: int,
    args: argparse.Namespace,
) -> list[dict[str, Any]]:
    if args.calibration_sampling == "low_confidence_stratified":
        return sample_low_confidence_stratified_records(
            rows,
            size,
            seed,
            args.uncertain_pool_rate,
            args.balanced_sampling,
        )
    return sample_hard_records(
        rows,
        size,
        seed,
        args.hard_candidate_multiplier,
        args.min_confidence,
        args.balanced_sampling,
    )


def action_label(row: dict[str, Any], primary_expert: str) -> str:
    if row["primary_pred"] == row["gold"]:
        return ACTION_KEEP
    return ACTION_OVERRIDE


def example_from_row(
    row: dict[str, Any],
    experts: list[str],
    primary_expert: str,
    prompt_masked: bool,
    max_article_chars: int,
    include_label: bool,
) -> dspy.Example | None:
    item = row["base_item"]
    article, aspect = article_aspect(item, prompt_masked, max_article_chars)
    if not article or not aspect:
        return None
    primary = row["experts"][primary_expert]
    aux_lines = []
    for expert in experts:
        if expert == primary_expert:
            continue
        info = row["experts"][expert]
        aux_lines.append(
            f"{expert_display_name(expert, row['language'])}: "
            f"prediction={info['prediction_label']}; "
            f"probabilities=({probability_text(info['probabilities'])}); "
            f"uncertainty=({uncertainty_text(info['uncertainty'])})"
        )
    args = {
        "article": article,
        "aspect": aspect,
        "primary_expert": expert_display_name(primary_expert, row["language"]),
        "primary_prediction": primary["prediction_label"],
        "primary_probabilities": probability_text(primary["probabilities"]),
        "primary_uncertainty": uncertainty_text(primary["uncertainty"]),
        "auxiliary_experts": "\n".join(aux_lines) if aux_lines else "No auxiliary expert available.",
        "routing_context": (
            f"Selected for possible deferral because primary_confidence={row['primary_confidence']:.3f}, "
            f"primary_entropy={row['primary_entropy']:.3f}, "
            f"auxiliary_disagreement={row['num_aux_disagree']}/{row['num_aux']}, "
            f"hard_score={row['hard_score']:.3f}. "
            "Choose keep_plm when primary is supported; override only with clear evidence; "
            "abstain_uncertain when evidence is insufficient for automatic override."
        ),
    }
    if include_label:
        action = action_label(row, primary_expert)
        args["action"] = action
        args["sentiment"] = INT_TO_STR_LABEL[row["gold"]] if action == ACTION_OVERRIDE else primary["prediction_label"]
    return dspy.Example(**args).with_inputs(
        "article",
        "aspect",
        "primary_expert",
        "primary_prediction",
        "primary_probabilities",
        "primary_uncertainty",
        "auxiliary_experts",
        "routing_context",
    )


def prepare_examples(
    rows: list[dict[str, Any]],
    experts: list[str],
    primary_expert: str,
    prompt_masked: bool,
    max_article_chars: int,
    include_label: bool,
) -> tuple[list[dspy.Example], list[dict[str, Any]]]:
    examples = []
    kept_rows = []
    for row in rows:
        example = example_from_row(row, experts, primary_expert, prompt_masked, max_article_chars, include_label)
        if example is not None:
            examples.append(example)
            kept_rows.append(row)
    return examples, kept_rows


def build_program() -> dspy.Module:
    return dspy.ChainOfThought(SelectiveDeferralSignature)


def normalize_action(value: Any) -> str:
    if isinstance(value, str):
        cleaned = value.strip().lower().replace("-", "_").replace(" ", "_")
        cleaned = re.sub(r"<\|.*?\|>", "", cleaned).strip("_.:,;!?")
        if cleaned in {"keep", "keep_plm", "keep_primary", "keep_expert"}:
            return ACTION_KEEP
        if cleaned in {"override", "replace", "change"}:
            return ACTION_OVERRIDE
        if cleaned in {"abstain", "uncertain", "abstain_uncertain", "defer_uncertain"}:
            return ACTION_ABSTAIN
        if "abstain" in cleaned or "uncertain" in cleaned:
            return ACTION_ABSTAIN
        if "override" in cleaned or "replace" in cleaned:
            return ACTION_OVERRIDE
        if "keep" in cleaned:
            return ACTION_KEEP
    return ACTION_ABSTAIN


def final_prediction(primary_pred: int, action: str, sentiment: str | None) -> int:
    if action == ACTION_OVERRIDE and sentiment in STR_TO_INT_LABEL:
        return STR_TO_INT_LABEL[sentiment]
    return primary_pred


def selective_metric(example: dspy.Example, pred: dspy.Prediction, trace=None) -> bool:
    action = normalize_action(getattr(pred, "action", ""))
    sentiment = label_str(getattr(pred, "sentiment", None))
    expected_action = normalize_action(getattr(example, "action", ""))
    expected_sentiment = label_str(getattr(example, "sentiment", None))
    if action != expected_action:
        return False
    if expected_action == ACTION_OVERRIDE:
        return sentiment == expected_sentiment
    return True


def optimize_program(
    trainset: list[dspy.Example],
    valset: list[dspy.Example],
    program_path: Path,
    student_lm: dspy.LM,
    teacher_lm: dspy.LM,
    args: argparse.Namespace,
) -> dspy.Module:
    dspy.settings.configure(lm=student_lm)
    optimizer = MIPROv2(
        metric=selective_metric,
        prompt_model=teacher_lm,
        task_model=student_lm,
        max_bootstrapped_demos=0,
        max_labeled_demos=0,
        auto=args.autorun,
        num_threads=args.dspy_num_threads,
        max_errors=args.max_errors,
        init_temperature=args.miprov2_temp,
        verbose=True,
        seed=args.seed,
    )
    optimized = optimizer.compile(
        student=build_program(),
        trainset=trainset,
        valset=valset,
        requires_permission_to_run=False,
        seed=args.seed,
        program_aware_proposer=not args.disable_program_aware_proposer,
        data_aware_proposer=not args.disable_data_aware_proposer,
        tip_aware_proposer=not args.disable_tip_aware_proposer,
        fewshot_aware_proposer=not args.disable_fewshot_aware_proposer,
        view_data_batch_size=args.view_data_batch_size,
    )
    program_path.parent.mkdir(parents=True, exist_ok=True)
    optimized.save(str(program_path))
    return optimized


def load_program(path: Path) -> dspy.Module:
    program = build_program()
    program.load(str(path))
    return program


def latest_lm_logprob_snapshot(lm: dspy.LM) -> dict[str, Any] | None:
    history = getattr(lm, "history", None)
    if not history:
        return None
    last = history[-1]

    def find_logprobs(value: Any, path: str, seen: set[int]) -> tuple[str, Any] | None:
        if value is None:
            return None
        value_id = id(value)
        if value_id in seen:
            return None
        seen.add(value_id)
        if hasattr(value, "logprobs") and getattr(value, "logprobs") is not None:
            return path + ".logprobs", getattr(value, "logprobs")
        if hasattr(value, "choices"):
            for index, choice in enumerate(getattr(value, "choices") or []):
                result = find_logprobs(choice, f"{path}.choices[{index}]", seen)
                if result:
                    return result
        if isinstance(value, dict):
            for key, child in value.items():
                if key == "logprobs" and child is not None:
                    return f"{path}.{key}", child
                result = find_logprobs(child, f"{path}.{key}", seen)
                if result:
                    return result
        elif isinstance(value, (list, tuple)):
            for index, child in enumerate(value):
                result = find_logprobs(child, f"{path}[{index}]", seen)
                if result:
                    return result
        return None

    result = find_logprobs(last, "history[-1]", set())
    if not result:
        text = repr(last)
        if "logprob" not in text.lower():
            return None
        return {"source": "history[-1].repr_fallback", "repr": text[:4000]}
    source, payload = result
    return {
        "source": source,
        "type": type(payload).__name__,
        "repr": repr(payload)[:8000],
    }


def run_program_on_rows(
    program: dspy.Module,
    examples: list[dspy.Example],
    rows: list[dict[str, Any]],
    num_workers: int,
    lm: dspy.LM | None = None,
    progress_label: str | None = None,
) -> list[dict[str, Any]]:
    lock = threading.Lock()
    started = time.time()
    counter = {"done": 0}

    def run_one(index: int) -> dict[str, Any]:
        example = examples[index]
        row = rows[index]
        detail = {
            "status": "failed",
            "reasoning": None,
            "raw_prediction_object_str": None,
            "raw_action": None,
            "raw_sentiment": None,
            "lm_logprob_snapshot": None,
        }
        action = ACTION_ABSTAIN
        sentiment = INT_TO_STR_LABEL[row["primary_pred"]]
        try:
            pred = program(**example.inputs())
            detail["raw_prediction_object_str"] = str(pred)
            detail["reasoning"] = str(getattr(pred, "reasoning", ""))
            detail["raw_action"] = getattr(pred, "action", None)
            detail["raw_sentiment"] = getattr(pred, "sentiment", None)
            detail["lm_logprob_snapshot"] = latest_lm_logprob_snapshot(lm) if lm is not None else None
            action = normalize_action(detail["raw_action"])
            sentiment = label_str(detail["raw_sentiment"]) or sentiment
            detail["status"] = "success"
        except Exception as exc:
            detail["status"] = "exception"
            detail["raw_prediction_object_str"] = f"Exception: {exc}"
        pred_int = final_prediction(row["primary_pred"], action, sentiment)
        result = {
            "uuid": row["uuid"],
            "ground_truth_int": row["gold"],
            "primary_prediction_int": row["primary_pred"],
            "primary_prediction_label": INT_TO_STR_LABEL[row["primary_pred"]],
            "prediction_int": pred_int,
            "prediction_label": INT_TO_STR_LABEL[pred_int],
            "action": action,
            "override_sentiment": sentiment,
            "primary_confidence": row["primary_confidence"],
            "primary_entropy": row["primary_entropy"],
            "num_aux_disagree": row["num_aux_disagree"],
            "num_aux": row["num_aux"],
            "hard_score": row["hard_score"],
            "llm_called": True,
            "gate_decision": "llm_called",
            "status": detail["status"],
            "dspy_query_details": [detail],
        }
        with lock:
            counter["done"] += 1
            done = counter["done"]
            if done == 1 or done == len(examples) or done % 25 == 0:
                elapsed = time.time() - started
                rate = done / elapsed if elapsed > 0 else 0.0
                eta = (len(examples) - done) / rate if rate > 0 else 0.0
                prefix = f"[{progress_label}] " if progress_label else ""
                print(f"{prefix}[progress] {done}/{len(examples)} elapsed={elapsed/60:.1f}m eta={eta/60:.1f}m", flush=True)
        return result

    if num_workers <= 1:
        return [run_one(index) for index in range(len(examples))]
    with futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
        return list(executor.map(run_one, range(len(examples))))


def bypass_result(row: dict[str, Any], min_confidence: float) -> dict[str, Any]:
    detail = {
        "status": "bypassed_hard_gate",
        "reasoning": (
            f"Kept primary prediction because primary_confidence={row['primary_confidence']:.3f} "
            f">= {min_confidence:.3f} and auxiliary_disagreement={row['num_aux_disagree']}/{row['num_aux']}."
        ),
        "raw_prediction_object_str": None,
        "raw_action": ACTION_KEEP,
        "raw_sentiment": INT_TO_STR_LABEL[row["primary_pred"]],
        "lm_logprob_snapshot": None,
    }
    return {
        "uuid": row["uuid"],
        "ground_truth_int": row["gold"],
        "primary_prediction_int": row["primary_pred"],
        "primary_prediction_label": INT_TO_STR_LABEL[row["primary_pred"]],
        "prediction_int": row["primary_pred"],
        "prediction_label": INT_TO_STR_LABEL[row["primary_pred"]],
        "action": ACTION_KEEP,
        "override_sentiment": INT_TO_STR_LABEL[row["primary_pred"]],
        "primary_confidence": row["primary_confidence"],
        "primary_entropy": row["primary_entropy"],
        "num_aux_disagree": row["num_aux_disagree"],
        "num_aux": row["num_aux"],
        "hard_score": row["hard_score"],
        "llm_called": False,
        "gate_decision": "kept_primary_without_llm",
        "status": "success",
        "dspy_query_details": [detail],
    }


def calculate_metrics(results: list[dict[str, Any]]) -> dict[str, Any]:
    successful = [row for row in results if row.get("status") == "success"]
    y_true = [int(row["ground_truth_int"]) for row in successful]
    y_pred = [int(row["prediction_int"]) for row in successful]
    y_primary = [int(row["primary_prediction_int"]) for row in successful]
    if not y_true:
        return {"error": "No successful predictions.", "num_samples_evaluated": 0}
    p_macro, r_macro, f1_macro, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=LABELS_INT, average="macro", zero_division=0
    )
    primary_f1 = precision_recall_fscore_support(
        y_true, y_primary, labels=LABELS_INT, average="macro", zero_division=0
    )[2]
    actions = Counter(row["action"] for row in successful)
    llm_called = sum(1 for row in successful if row.get("llm_called", True))
    corrections = sum(
        row["primary_prediction_int"] != row["ground_truth_int"] and row["prediction_int"] == row["ground_truth_int"]
        for row in successful
    )
    degradations = sum(
        row["primary_prediction_int"] == row["ground_truth_int"] and row["prediction_int"] != row["ground_truth_int"]
        for row in successful
    )
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision_macro": float(p_macro),
        "recall_macro": float(r_macro),
        "f1_macro": float(f1_macro),
        "qwk": float(cohen_kappa_score(y_true, y_pred, labels=LABELS_INT, weights="quadratic")),
        "primary_f1_macro": float(primary_f1),
        "primary_qwk": float(cohen_kappa_score(y_true, y_primary, labels=LABELS_INT, weights="quadratic")),
        "num_results_total": len(results),
        "num_successful": len(successful),
        "num_samples_evaluated": len(y_true),
        "action_counts": dict(actions),
        "override_rate": actions.get(ACTION_OVERRIDE, 0) / len(successful),
        "abstain_rate": actions.get(ACTION_ABSTAIN, 0) / len(successful),
        "num_llm_called": llm_called,
        "num_bypassed": len(successful) - llm_called,
        "llm_call_rate": llm_called / len(successful),
        "corrections": int(corrections),
        "degradations": int(degradations),
        "net_corrections": int(corrections - degradations),
        "per_class_report": classification_report(
            y_true,
            y_pred,
            labels=LABELS_INT,
            target_names=[INT_TO_STR_LABEL[label] for label in LABELS_INT],
            output_dict=True,
            zero_division=0,
        ),
    }


def attach_predictions(rows: list[dict[str, Any]], results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_uuid = {row["uuid"]: row for row in results}
    output = []
    for row in rows:
        item = copy.deepcopy(row["base_item"])
        result = by_uuid.get(row["uuid"])
        item["selective_deferral"] = result
        if result is None:
            item["prediction"] = row["primary_pred"]
            item["processing_status"] = "missing"
        else:
            item["prediction"] = result["prediction_int"]
            item["prediction_label"] = result["prediction_label"]
            item["selective_deferral_action"] = result["action"]
            item["processing_status"] = result["status"]
        output.append(item)
    return output


def resolve_experts(args: argparse.Namespace) -> tuple[list[str], str]:
    experts = args.experts or DEFAULT_EXPERTS[args.language]
    primary = args.primary_expert or DEFAULT_PRIMARY[args.language]
    if primary not in experts:
        experts = [primary, *experts]
    return list(dict.fromkeys(experts)), primary


def configure_selective_lm(
    model_name: str,
    api_base: str,
    api_key: str,
    temperature: float,
    top_p: float,
    max_tokens: int,
    num_retries: int,
    request_logprobs: bool,
    top_logprobs: int,
    endpoint_type: str,
    top_k: int | None,
) -> dspy.LM:
    if not api_base.endswith("/v1"):
        api_base = api_base.rstrip("/") + "/v1"
    model_type = "text" if endpoint_type == "completion" else "chat"
    lm_cls = OpenAICompletionLM if endpoint_type == "completion" else dspy.LM
    kwargs: dict[str, Any] = {}
    if top_k is not None and top_k > 0:
        kwargs["extra_body"] = {"top_k": top_k}
    if request_logprobs:
        if endpoint_type == "completion":
            kwargs.update({"logprobs": top_logprobs})
        else:
            kwargs.update({"logprobs": True, "top_logprobs": top_logprobs})
    return lm_cls(
        model=f"openai/{model_name}",
        api_base=api_base,
        api_key=api_key,
        model_type=model_type,
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
        num_retries=num_retries,
        cache=False,
        **kwargs,
    )


def make_lm(args: argparse.Namespace, api_base: str, max_tokens: int) -> dspy.LM:
    return configure_selective_lm(
        args.student_model,
        api_base,
        args.api_key,
        args.temperature,
        args.top_p,
        max_tokens,
        args.num_retries,
        args.request_logprobs,
        args.top_logprobs,
        args.student_endpoint_type,
        args.top_k,
    )


def calibration_mode(args: argparse.Namespace) -> None:
    started = time.time()
    root = Path(args.uncertainty_root)
    paths = calibration_paths(args)
    out_dir = paths["output_dir"]
    out_dir.mkdir(parents=True, exist_ok=True)
    experts, primary = resolve_experts(args)
    prompt_masked = args.prompt_variant == "masked"
    base_run_name = run_name(args.language, prompt_masked, args.autorun, args.max_tokens, args.teacher_label)
    program_path = paths["program"]
    metadata_path = paths["metadata"]
    predictions_path = paths["predictions"]
    metrics_path = paths["metrics"]
    print("--- LLM Selective Deferral Calibration Preflight ---", flush=True)
    print(f"Language: {args.language}", flush=True)
    print(f"Prompt variant: {args.prompt_variant}", flush=True)
    print(f"Primary expert: {primary}", flush=True)
    print(f"Experts: {', '.join(experts)}", flush=True)
    print(f"Force: {args.force}", flush=True)
    print_path_status(paths)
    if args.dry_run:
        if program_path.exists() and not args.force:
            print("DRY RUN decision: SKIP calibration because optimized program already exists.", flush=True)
        elif program_path.exists() and args.force:
            print("DRY RUN decision: RUN calibration and overwrite the optimized program because --force is set.", flush=True)
        else:
            print("DRY RUN decision: RUN calibration because optimized program is missing.", flush=True)
        return
    if program_path.exists() and not args.force:
        print(
            "Skipping calibration because optimized program already exists. "
            "Use FORCE=1/--force to overwrite it.",
            flush=True,
        )
        if not metrics_path.exists() or not predictions_path.exists():
            print(
                "Note: current-name calibration metrics/predictions are missing, but the optimized "
                "program is preserved to avoid accidental prompt overwrite.",
                flush=True,
            )
        return
    if program_path.exists() and args.force:
        print(f"FORCE=1: overwriting existing optimized program: {program_path}", flush=True)

    train_rows, success = build_combined_rows(
        root,
        experts,
        primary,
        args.language,
        f"train_val_{args.split_index}",
        "train",
        args.prefer_balanced_splits,
    )
    val_rows, _ = build_combined_rows(
        root,
        experts,
        primary,
        args.language,
        f"train_val_{args.split_index}",
        "val",
        args.prefer_balanced_splits,
    )
    sample_size = default_sample_size(args.autorun)
    train_size = args.train_size or sample_size
    val_size = args.val_size or sample_size
    train_sample = sample_calibration_records(train_rows, train_size, args.seed, args)
    val_sample = sample_calibration_records(val_rows, val_size, args.seed + 1, args)
    trainset, train_sample = prepare_examples(train_sample, experts, primary, prompt_masked, args.max_article_chars, True)
    valset, val_sample = prepare_examples(val_sample, experts, primary, prompt_masked, args.max_article_chars, True)
    if not trainset or not valset:
        raise RuntimeError("Prepared empty trainset or valset.")
    train_summary = selection_summary(train_rows, args.min_confidence)
    val_summary = selection_summary(val_rows, args.min_confidence)
    print("--- LLM Selective Deferral Calibration ---", flush=True)
    print(f"Language: {args.language}", flush=True)
    print(f"Prompt variant: {args.prompt_variant}", flush=True)
    print(f"Primary expert: {primary}", flush=True)
    print(f"Experts: {', '.join(experts)}", flush=True)
    print(f"Calibration sampling: {args.calibration_sampling}", flush=True)
    if args.calibration_sampling == "low_confidence_stratified":
        print(f"Uncertain pool rate: {args.uncertain_pool_rate:.1%}", flush=True)
    print(
        f"Train rows: {len(train_rows)}; hard gate candidates: "
        f"{train_summary['hard_gate_count']} ({train_summary['hard_gate_rate']:.1%}); "
        f"examples passed to DSPy: {len(trainset)}",
        flush=True,
    )
    print(
        f"Val rows: {len(val_rows)}; hard gate candidates: "
        f"{val_summary['hard_gate_count']} ({val_summary['hard_gate_rate']:.1%}); "
        f"examples passed to DSPy: {len(valset)}",
        flush=True,
    )
    print(
        "Calibration sampling is class-stratified; query mode applies the strict gate by default.",
        flush=True,
    )

    if not args.skip_endpoint_check:
        sanity_check_openai_chat_endpoint(
            args.student_api_base,
            args.api_key,
            args.student_model,
            "student",
            endpoint_type=args.student_endpoint_type,
        )
        sanity_check_openai_chat_endpoint(
            args.teacher_api_base,
            args.api_key,
            args.teacher_model,
            "teacher",
            endpoint_type=args.teacher_endpoint_type,
        )

    student_lm = configure_selective_lm(
        args.student_model,
        args.student_api_base,
        args.api_key,
        args.temperature,
        args.top_p,
        args.student_max_tokens or args.max_tokens,
        args.num_retries,
        args.request_logprobs,
        args.top_logprobs,
        args.student_endpoint_type,
        args.top_k,
    )
    teacher_lm = configure_openai_lm(
        args.teacher_model,
        args.teacher_api_base,
        args.api_key,
        args.temperature,
        args.top_p,
        args.teacher_max_tokens or args.max_tokens,
        args.num_retries,
        cache=False,
        endpoint_type=args.teacher_endpoint_type,
        top_k=args.top_k,
    )
    optimized = optimize_program(trainset, valset, program_path, student_lm, teacher_lm, args)
    dspy.settings.configure(lm=student_lm)
    results = run_program_on_rows(optimized, valset, val_sample, args.eval_workers, lm=student_lm, progress_label="calibration")
    metrics = calculate_metrics(results)
    metrics.update(
        {
            "phase": "calibration_validation",
            "evaluation_scope": "sampled_calibration_validation_subset",
            "run_name": base_run_name,
            "language": args.language,
            "prompt_variant": args.prompt_variant,
            "autorun": args.autorun,
            "primary_expert": primary,
            "experts": experts,
            "elapsed_seconds": time.time() - started,
        }
    )
    write_json(predictions_path, attach_predictions(val_sample, results))
    write_json(metrics_path, metrics)
    write_json(
        metadata_path,
        {
            "language": args.language,
            "prompt_variant": args.prompt_variant,
            "autorun": args.autorun,
            "primary_expert": primary,
            "experts": experts,
            "train_rows_available": len(train_rows),
            "val_rows_available": len(val_rows),
            "train_selection_summary": train_summary,
            "val_selection_summary": val_summary,
            "train_examples": len(trainset),
            "val_examples": len(valset),
            "hard_candidate_multiplier": args.hard_candidate_multiplier,
            "calibration_sampling": args.calibration_sampling,
            "uncertain_pool_rate": args.uncertain_pool_rate,
            "min_confidence": args.min_confidence,
            "balanced_sampling": args.balanced_sampling,
            "hard_gated_query": args.hard_gated_query,
            "prefer_balanced_splits": args.prefer_balanced_splits,
            "optimized_program_path": str(program_path),
            "success_by_expert": success,
        },
    )
    print(f"Wrote {metrics_path}", flush=True)


def shard_rows(rows: list[dict[str, Any]], num_shards: int, shard_index: int) -> list[dict[str, Any]]:
    return [row for idx, row in enumerate(rows) if idx % num_shards == shard_index]


def parse_endpoint_workers(args: argparse.Namespace) -> list[int]:
    if not args.num_workers_per_endpoints:
        return [args.num_workers_per_endpoint for _ in args.api_bases]
    workers = [int(item.strip()) for item in args.num_workers_per_endpoints.split(",") if item.strip()]
    if len(workers) != len(args.api_bases):
        raise ValueError(
            "NUM_WORKERS_PER_ENDPOINTS must have the same number of comma-separated values as STUDENT_API_BASES "
            f"({len(workers)} workers for {len(args.api_bases)} endpoints)."
        )
    if any(worker <= 0 for worker in workers):
        raise ValueError("All endpoint worker counts must be positive.")
    return workers


def run_query_shard(
    args: argparse.Namespace,
    shard_index: int,
    api_base: str,
    num_workers: int,
    rows: list[dict[str, Any]],
    experts: list[str],
    primary: str,
    program_path: Path,
    out_dir: Path,
    base_run_name: str,
) -> list[dict[str, Any]]:
    prompt_masked = args.prompt_variant == "masked"
    shard = shard_rows(rows, len(args.api_bases), shard_index)
    if not shard:
        return []
    shard_name = f"{base_run_name}_shard-{shard_index:02d}-of-{len(args.api_bases):02d}"
    shard_predictions = out_dir / f"{shard_name}_test_predictions.json"
    shard_metrics = out_dir / f"{shard_name}_test_metrics.json"
    if shard_predictions.exists() and shard_metrics.exists() and not args.force:
        saved = load_json(shard_predictions)
        results = []
        for item in saved:
            result = item.get("selective_deferral")
            if result:
                results.append(result)
        return results
    if args.hard_gated_query:
        llm_rows = [row for row in shard if is_query_gate_candidate(row, args)]
        bypass_rows = [row for row in shard if not is_query_gate_candidate(row, args)]
    else:
        llm_rows = list(shard)
        bypass_rows = []
    bypass_results = [bypass_result(row, args.min_confidence) for row in bypass_rows]
    if not llm_rows:
        write_json(shard_predictions, attach_predictions(shard, bypass_results))
        write_json(shard_metrics, calculate_metrics(bypass_results))
        return bypass_results
    if not args.skip_endpoint_check:
        sanity_check_openai_chat_endpoint(
            api_base,
            args.api_key,
            args.student_model,
            f"student shard {shard_index}",
            endpoint_type=args.student_endpoint_type,
        )
    examples, llm_rows = prepare_examples(llm_rows, experts, primary, prompt_masked, args.max_article_chars, False)
    lm = make_lm(args, api_base, args.max_tokens)
    dspy.settings.configure(lm=lm)
    program = load_program(program_path)
    llm_results = run_program_on_rows(
        program,
        examples,
        llm_rows,
        num_workers,
        lm=lm,
        progress_label=f"shard {shard_index}",
    )
    results = [*bypass_results, *llm_results]
    results.sort(key=lambda item: str(item.get("uuid")))
    metrics = calculate_metrics(results)
    write_json(shard_predictions, attach_predictions(shard, results))
    write_json(shard_metrics, metrics)
    return results


def query_mode(args: argparse.Namespace) -> None:
    started = time.time()
    root = Path(args.uncertainty_root)
    paths = query_paths(args)
    out_dir = paths["output_dir"]
    out_dir.mkdir(parents=True, exist_ok=True)
    experts, primary = resolve_experts(args)
    prompt_masked = args.prompt_variant == "masked"
    program_path = paths["program"]
    base_run_name = run_name(args.language, prompt_masked, args.autorun, args.max_tokens, args.teacher_label)
    predictions_path = paths["predictions"]
    metrics_path = paths["metrics"]
    metadata_path = paths["metadata"]
    print("--- LLM Selective Deferral Query Preflight ---", flush=True)
    print(f"Language: {args.language}", flush=True)
    print(f"Prompt variant: {args.prompt_variant}", flush=True)
    print(f"Primary expert: {primary}", flush=True)
    print(f"Experts: {', '.join(experts)}", flush=True)
    print(f"Gate rate: {args.gate_rate}", flush=True)
    print(f"Force: {args.force}", flush=True)
    print_path_status(paths)
    if args.dry_run:
        if not program_path.exists():
            print("DRY RUN decision: ERROR, optimized program is missing.", flush=True)
        elif predictions_path.exists() and metrics_path.exists() and not args.force:
            print("DRY RUN decision: SKIP query because final test predictions and metrics already exist.", flush=True)
        elif (predictions_path.exists() or metrics_path.exists()) and args.force:
            print("DRY RUN decision: RUN query and overwrite existing final test outputs because --force is set.", flush=True)
        else:
            print("DRY RUN decision: RUN query.", flush=True)
        return
    if not program_path.exists():
        raise FileNotFoundError(f"Missing optimized program: {program_path}")
    if predictions_path.exists() and metrics_path.exists() and not args.force:
        print(f"Skipping existing query outputs: {metrics_path}", flush=True)
        return
    if (predictions_path.exists() or metrics_path.exists()) and args.force:
        print(f"FORCE=1: overwriting query outputs under: {out_dir}", flush=True)
    rows, success = build_combined_rows(
        root,
        experts,
        primary,
        args.language,
        "test",
        None,
        args.prefer_balanced_splits,
    )
    if args.sample_items:
        rows = sample_hard_records(
            rows,
            args.sample_items,
            args.seed,
            args.hard_candidate_multiplier,
            args.min_confidence,
            args.balanced_sampling,
        )
    if args.limit_items:
        rows = rows[: args.limit_items]
    args.api_bases = [item.strip() for item in (args.student_api_bases or args.student_api_base).split(",") if item.strip()]
    args.api_worker_counts = parse_endpoint_workers(args)
    if args.gate_rate is not None:
        args.query_gate_uuids = confidence_band_uuids(rows, args.gate_rate)
    else:
        args.query_gate_uuids = None
    gate_summary = query_gate_summary(rows, args)
    print("--- LLM Selective Deferral Query ---", flush=True)
    print(f"Language: {args.language}", flush=True)
    print(f"Prompt variant: {args.prompt_variant}", flush=True)
    print(f"Primary expert: {primary}", flush=True)
    print(f"Experts: {', '.join(experts)}", flush=True)
    print(f"Records: {len(rows)}", flush=True)
    print(
        "Student endpoints: "
        + ", ".join(
            f"{api_base} ({workers} workers)"
            for api_base, workers in zip(args.api_bases, args.api_worker_counts, strict=False)
        ),
        flush=True,
    )
    if args.hard_gated_query:
        if args.gate_rate is not None:
            print(
                f"Hard-gated LLM calls: {gate_summary['routed_to_llm_count']}/{len(rows)} "
                f"({gate_summary['routed_to_llm_rate']:.1%}) "
                f"using lowest-confidence gate rate {args.gate_rate:.1%}",
                flush=True,
            )
        else:
            print(
                f"Hard-gated LLM calls: {gate_summary['routed_to_llm_count']}/{len(rows)} "
                f"({gate_summary['routed_to_llm_rate']:.1%}) "
                f"using confidence < {args.min_confidence:.2f} or auxiliary disagreement",
                flush=True,
            )
    else:
        print("Hard-gated LLM calls: disabled; every record is sent to the LLM", flush=True)
    print(f"Program: {program_path}", flush=True)
    all_results = []
    with futures.ProcessPoolExecutor(max_workers=len(args.api_bases)) as executor:
        future_to_index = {
            executor.submit(
                run_query_shard,
                args,
                shard_index,
                api_base,
                args.api_worker_counts[shard_index],
                rows,
                experts,
                primary,
                program_path,
                out_dir,
                base_run_name,
            ): shard_index
            for shard_index, api_base in enumerate(args.api_bases)
        }
        for future in futures.as_completed(future_to_index):
            all_results.extend(future.result())
    all_results.sort(key=lambda item: str(item.get("uuid")))
    rows_by_uuid = {row["uuid"]: row for row in rows}
    ordered_rows = [rows_by_uuid[row["uuid"]] for row in all_results if row.get("uuid") in rows_by_uuid]
    metrics = calculate_metrics(all_results)
    metrics.update(
        {
            "phase": "test",
            "evaluation_scope": "full_test_set_with_selective_llm_routing",
            "run_name": base_run_name,
            "language": args.language,
            "prompt_variant": args.prompt_variant,
            "autorun": args.autorun,
            "primary_expert": primary,
            "experts": experts,
            "student_model": args.student_model,
            "student_api_bases": args.api_bases,
            "student_api_worker_counts": args.api_worker_counts,
            "optimized_program_path": str(program_path),
            "hard_gated_query": args.hard_gated_query,
            "gate_rate": args.gate_rate,
            "query_gate_summary": gate_summary,
            "min_confidence": args.min_confidence,
            "elapsed_seconds": time.time() - started,
        }
    )
    llm_call_rate = metrics.get("llm_call_rate")
    llm_call_rate_text = f"{llm_call_rate:.3f}" if isinstance(llm_call_rate, (int, float)) else "n/a"
    print(
        "Query metrics: "
        f"evaluated={metrics.get('num_samples_evaluated')} "
        f"records_loaded={len(rows)} "
        f"routed_to_llm={metrics.get('num_llm_called')} "
        f"bypassed={metrics.get('num_bypassed')} "
        f"llm_call_rate={llm_call_rate_text}",
        flush=True,
    )
    write_json(predictions_path, attach_predictions(ordered_rows, all_results))
    write_json(metrics_path, metrics)
    write_json(
        metadata_path,
        {
            "language": args.language,
            "prompt_variant": args.prompt_variant,
            "autorun": args.autorun,
            "primary_expert": primary,
            "experts": experts,
            "num_records_loaded": len(rows),
            "num_results": len(all_results),
            "student_api_worker_counts": args.api_worker_counts,
            "threshold_selection_summary": selection_summary(rows, args.min_confidence),
            "query_gate_summary": gate_summary,
            "hard_gated_query": args.hard_gated_query,
            "gate_rate": args.gate_rate,
            "min_confidence": args.min_confidence,
            "optimized_program_path": str(program_path),
            "success_by_expert": success,
        },
    )
    print(f"Wrote {metrics_path}", flush=True)


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    if args.mode == "calibrate":
        calibration_mode(args)
    else:
        query_mode(args)


if __name__ == "__main__":
    main()
