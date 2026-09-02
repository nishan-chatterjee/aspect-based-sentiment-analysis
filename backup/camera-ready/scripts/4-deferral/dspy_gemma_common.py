#!/usr/bin/env python3
"""Shared DSPy helpers for Gemma/Qwen uncertainty calibration experiments."""

from __future__ import annotations

import concurrent.futures as futures
import copy
import datetime as dt
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

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    cohen_kappa_score,
    precision_recall_fscore_support,
)
from sklearn.model_selection import train_test_split

import dspy
from dspy.teleprompt import MIPROv2


ROOT_DIR = Path(os.environ.get("ABSA_RELEASE_ROOT", Path(__file__).resolve().parents[2])).resolve()
UNCERTAINTY_ROOT = ROOT_DIR / "reviews" / "uncertainty"
DEFAULT_OUTPUT_ROOT = UNCERTAINTY_ROOT / "llm-dspy-calibration-cot"
MASK_TOKEN = "[MASK]"
INT_TO_STR_LABEL = {-1: "negative", 0: "neutral", 1: "positive"}
STR_TO_INT_LABEL = {value: key for key, value in INT_TO_STR_LABEL.items()}
LABELS_INT = [-1, 0, 1]
LABELS_STR = ["negative", "neutral", "positive"]
NEUTRAL_STR = "neutral"

DEFAULT_EXPERTS_BY_LANGUAGE = {
    "slovenian": ["han_xlmr_masked", "longformer_masked", "slavic_specific_masked"],
    "serbian": ["longformer_masked", "mdeberta_masked", "slavic_specific_masked"],
}

EXPERT_DISPLAY_NAMES = {
    "han_xlmr_masked": "HAN + XLMR masked",
    "longformer_masked": "Longformer masked",
    "mdeberta_masked": "mDeBERTa-v3 masked",
    "slavic_specific_masked": "SloBERTa/BERTic masked",
}

AUTORUN_SAMPLE_SIZES = {
    "light": 100,
    "medium": 300,
    "heavy": 1000,
}

PLM_CALIBRATION_INFO = {
    "slovenian": "General Note: The PLM is generally well-calibrated on this language but can be slightly overconfident.",
    "serbian": "General Note: The PLM is known to be significantly overconfident on this language; its confidence scores are often higher than its actual accuracy.",
}


class ExpertUncertaintyCOTSignature(dspy.Signature):
    """Use the article, aspect, and an expert model's uncertainty-aware prediction to infer sentiment.

    Reason carefully from the article evidence. Treat the expert prediction as useful but fallible:
    high confidence and low uncertainty make it more trustworthy; high uncertainty or divided
    votes mean the article evidence should dominate. End with one label: negative, neutral,
    or positive.
    """

    article: str = dspy.InputField(
        desc="The article text. The aspect may be explicit or replaced by [MASK]."
    )
    aspect: str = dspy.InputField(
        desc="The target aspect, or [MASK] when the aspect is hidden."
    )
    expert_name: str = dspy.InputField(desc="The expert model family that produced the suggestion.")
    expert_suggestion: Literal["negative", "neutral", "positive"] = dspy.InputField(
        desc="The expert model's sentiment suggestion."
    )
    expert_probabilities: str = dspy.InputField(
        desc="The expert model's mean probabilities for Negative, Neutral, and Positive."
    )
    expert_uncertainty: str = dspy.InputField(
        desc="Compact confidence, entropy, mutual information, and MC vote summary."
    )
    reasoning: str = dspy.OutputField(
        desc="Brief reasoning that weighs article evidence against expert uncertainty."
    )
    sentiment: Literal["negative", "neutral", "positive"] = dspy.OutputField(
        desc="Final sentiment label."
    )


class LegacyPLMUncertaintyCOTSignature(dspy.Signature):
    """Given an article, aspect, a PLM suggestion, and its uncertainty metrics:
1. Provide step-by-step reasoning, explicitly considering the PLM suggestion and its detailed confidence, entropy, and known calibration biases.
2. Conclude with the final sentiment ('negative', 'neutral', 'positive')."""

    article: str = dspy.InputField(
        desc="The full text of the article. The 'aspect' may be explicitly named or replaced with a generic '[MASK]' placeholder."
    )
    aspect: str = dspy.InputField(
        desc="The specific aspect. This will either be the specific aspect phrase or the generic placeholder '[MASK]' if the aspect's name has been hidden in the article."
    )
    plm_suggestion: Literal["negative", "neutral", "positive"] = dspy.InputField(
        desc="Sentiment suggestion from a prior model."
    )
    plm_confidence_score: str = dspy.InputField(
        desc="The PLM's confidence in its suggestion (e.g., 'Confidence: 60.0%'). This reflects the proportion of internal model votes for the suggestion."
    )
    plm_predictive_entropy: str = dspy.InputField(
        desc="A measure of the PLM's internal disagreement. Higher values (e.g., >1.0) mean more uncertainty. (e.g., 'Predictive Entropy: 0.989')."
    )
    plm_calibration_fact: str = dspy.InputField(
        desc="A general fact about the PLM's typical calibration behavior for this language (e.g., whether it tends to be overconfident)."
    )
    reasoning: str = dspy.OutputField(
        desc="Step-by-step reasoning, incorporating the PLM suggestion and all provided uncertainty metrics."
    )
    sentiment: Literal["negative", "neutral", "positive"] = dspy.OutputField(desc="Final sentiment.")


class RichExpertUncertaintyCOTSignature(dspy.Signature):
    """Given an article, aspect, an expert model suggestion, probabilities, and detailed uncertainty metrics:
1. Provide step-by-step reasoning, explicitly considering article evidence, expert identity, expert probabilities, confidence, entropy, epistemic uncertainty, vote distribution, and known calibration behavior.
2. Trust the expert more when confidence is high, probabilities are concentrated, entropy and mutual information are low, and votes agree.
3. Rely more on the article evidence when the expert is uncertain, votes are divided, or the article clearly contradicts the expert.
4. Conclude with the final sentiment ('negative', 'neutral', 'positive')."""

    article: str = dspy.InputField(
        desc="The full text of the article. The aspect may be explicitly named or replaced with a generic '[MASK]' placeholder."
    )
    aspect: str = dspy.InputField(
        desc="The specific aspect. This will either be the specific aspect phrase or the generic placeholder '[MASK]' if the aspect's name has been hidden in the article."
    )
    expert_name: str = dspy.InputField(desc="The expert model family that produced the suggestion.")
    expert_suggestion: Literal["negative", "neutral", "positive"] = dspy.InputField(
        desc="Sentiment suggestion from the expert model."
    )
    expert_probabilities: str = dspy.InputField(
        desc="The expert model's mean probabilities for negative, neutral, and positive sentiment."
    )
    expert_confidence_score: str = dspy.InputField(
        desc="The expert model's confidence in its suggested sentiment."
    )
    expert_predictive_entropy: str = dspy.InputField(
        desc="Predictive entropy of the averaged expert distribution; higher values indicate more uncertainty."
    )
    expert_expected_entropy: str = dspy.InputField(
        desc="Expected entropy across stochastic expert samples; higher values indicate uncertainty within individual samples."
    )
    expert_mutual_information: str = dspy.InputField(
        desc="Epistemic uncertainty estimated as predictive entropy minus expected entropy; higher values indicate model uncertainty."
    )
    expert_vote_distribution: str = dspy.InputField(
        desc="Distribution of stochastic expert votes across negative, neutral, and positive labels."
    )
    expert_calibration_fact: str = dspy.InputField(
        desc="A general note about calibration behavior for this language and how to use the uncertainty fields."
    )
    reasoning: str = dspy.OutputField(
        desc="Step-by-step reasoning that weighs article evidence against the expert suggestion and uncertainty metrics."
    )
    sentiment: Literal["negative", "neutral", "positive"] = dspy.OutputField(desc="Final sentiment.")


class OpenAICompletionLM(dspy.LM):
    """DSPy LM wrapper for OpenAI-compatible /v1/completions endpoints.

    DSPy may pass chat/structured-output kwargs such as response_format at call
    time. llama.cpp's completions endpoint rejects those kwargs, so strip them
    for text-completion servers.
    """

    UNSUPPORTED_COMPLETION_KWARGS = {
        "response_format",
        "tools",
        "tool_choice",
        "parallel_tool_calls",
    }

    def forward(self, prompt: str | None = None, messages: list[dict[str, Any]] | None = None, **kwargs):
        for key in self.UNSUPPORTED_COMPLETION_KWARGS:
            kwargs.pop(key, None)
        return super().forward(prompt=prompt, messages=messages, **kwargs)


def configure_openai_lm(
    model_name: str,
    api_base: str,
    api_key: str,
    temperature: float,
    top_p: float,
    max_tokens: int,
    num_retries: int,
    cache: bool = False,
    endpoint_type: str = "chat",
    top_k: int | None = None,
) -> dspy.LM:
    if not api_base.endswith("/v1"):
        api_base = api_base.rstrip("/") + "/v1"
    model_type = "text" if endpoint_type == "completion" else "chat"
    lm_cls = OpenAICompletionLM if endpoint_type == "completion" else dspy.LM
    kwargs: dict[str, Any] = {}
    if top_k is not None and top_k > 0:
        kwargs["extra_body"] = {"top_k": top_k}
    return lm_cls(
        model=f"openai/{model_name}",
        api_base=api_base,
        api_key=api_key,
        model_type=model_type,
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
        num_retries=num_retries,
        cache=cache,
        **kwargs,
    )


def sanity_check_openai_chat_endpoint(
    api_base: str,
    api_key: str,
    model_name: str,
    role: str,
    timeout: float = 60.0,
    endpoint_type: str = "chat",
) -> None:
    try:
        from openai import OpenAI
    except Exception as exc:
        raise RuntimeError(f"Could not import openai package for endpoint check: {exc}") from exc
    if not api_base.endswith("/v1"):
        api_base = api_base.rstrip("/") + "/v1"
    client = OpenAI(base_url=api_base, api_key=api_key, timeout=timeout)
    try:
        if endpoint_type == "completion":
            response = client.completions.create(
                model=model_name,
                prompt="Reply with exactly: ok\n",
                max_tokens=8,
                temperature=0,
            )
            content = response.choices[0].text if response.choices else ""
        else:
            response = client.chat.completions.create(
                model=model_name,
                messages=[{"role": "user", "content": "Reply with exactly: ok"}],
                max_tokens=8,
                temperature=0,
            )
            content = response.choices[0].message.content if response.choices else ""
    except Exception as exc:
        message = str(exc)
        if "chat template" in message.lower():
            raise RuntimeError(
                f"{role} endpoint {api_base} for model {model_name!r} is missing a chat template. "
                "For Qwen2.5 served by vLLM, restart vLLM with "
                "--chat-template reviews/scratchpad/qwen2_5_instruct_chat_template.jinja "
                "or provide a tokenizer directory containing tokenizer_config.json with chat_template."
            ) from exc
        raise RuntimeError(
            f"{role} endpoint sanity check failed for {api_base} model {model_name!r} "
            f"using {endpoint_type!r}: {exc}"
        ) from exc
    if not content:
        raise RuntimeError(f"{role} endpoint returned an empty sanity-check response using {endpoint_type!r}.")


def mask_aspect_in_text(article_text: str) -> str:
    if not isinstance(article_text, str):
        return ""
    return re.sub(r"<aspect>.*?</aspect>", MASK_TOKEN, article_text, flags=re.DOTALL)


def strip_aspect_tags(article_text: str) -> str:
    if not isinstance(article_text, str):
        return ""
    return article_text.replace("<aspect>", "").replace("</aspect>", "")


def maybe_clip_text(text: str, max_chars: int | None) -> str:
    if max_chars is None or max_chars <= 0 or len(text) <= max_chars:
        return text
    half = max_chars // 2
    return text[:half].rstrip() + "\n\n[...]\n\n" + text[-half:].lstrip()


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(to_jsonable(data), f, indent=2, ensure_ascii=False)


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
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (dt.datetime, dt.date)):
        return value.isoformat()
    return value


def load_success(expert: str, language: str, uncertainty_root: Path) -> dict[str, Any]:
    path = uncertainty_root / expert / language / "_SUCCESS.json"
    if not path.exists():
        raise FileNotFoundError(
            f"Missing uncertainty success file: {path}. "
            "Run reviews/sh-jobs-uncertainty.sh for this expert/language first."
        )
    return load_json(path)


def uncertainty_data_path(expert: str, language: str, split_name: str, uncertainty_root: Path) -> Path:
    base = uncertainty_root / expert / language
    if split_name == "test":
        return base / f"{language}_test_complete.json"
    if split_name.startswith("train_val_"):
        index = split_name.rsplit("_", 1)[-1]
        return base / f"{language}_train_val_complete_{index}.json"
    raise ValueError(f"Unsupported split_name: {split_name}")


def load_train_val_records(
    expert: str,
    language: str,
    uncertainty_root: Path,
    split_index: int = 0,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    path = uncertainty_data_path(expert, language, f"train_val_{split_index}", uncertainty_root)
    data = load_json(path)
    return data.get("train", []), data.get("val", [])


def load_test_records(expert: str, language: str, uncertainty_root: Path) -> list[dict[str, Any]]:
    path = uncertainty_data_path(expert, language, "test", uncertainty_root)
    data = load_json(path)
    if isinstance(data, dict) and "test" in data:
        return data["test"]
    if isinstance(data, dict) and "records" in data:
        return data["records"]
    if isinstance(data, list):
        return data
    raise ValueError(f"Could not locate test records in {path}")


def normalize_sentiment_label(value: Any) -> str | None:
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in STR_TO_INT_LABEL:
            return lowered
        try:
            value = int(value)
        except ValueError:
            return None
    if value in INT_TO_STR_LABEL:
        return INT_TO_STR_LABEL[int(value)]
    return None


def expert_key_for(success: dict[str, Any]) -> tuple[str, str, str]:
    json_key = success["json_key"]
    return json_key, success["probabilities_key"], success["uncertainty_key"]


def format_probabilities(probabilities: dict[str, Any]) -> str:
    return (
        "Negative={neg:.4f}, Neutral={neu:.4f}, Positive={pos:.4f}".format(
            neg=float(probabilities.get("Negative", 0.0) or 0.0),
            neu=float(probabilities.get("Neutral", 0.0) or 0.0),
            pos=float(probabilities.get("Positive", 0.0) or 0.0),
        )
    )


def format_vote_distribution(uncertainty: dict[str, Any]) -> str:
    distribution = uncertainty.get("prediction_distribution", {})
    total = float(uncertainty.get("total_mc_samples", 0) or 0)
    parts = []
    for raw_label, name in [("-1", "negative"), ("0", "neutral"), ("1", "positive")]:
        count = int(distribution.get(raw_label, 0) or 0)
        pct = count / total if total else 0.0
        parts.append(f"{name}: {count}/{int(total)} ({pct:.1%})")
    return "; ".join(parts)


def format_compact_uncertainty(uncertainty: dict[str, Any]) -> str:
    distribution = uncertainty.get("prediction_distribution", {}) or {}
    total = int(float(uncertainty.get("total_mc_samples", 0) or 0))
    counts = [
        int(distribution.get("-1", 0) or 0),
        int(distribution.get("0", 0) or 0),
        int(distribution.get("1", 0) or 0),
    ]
    return (
        "conf={conf:.3f}; vote_conf={vote_conf:.3f}; "
        "entropy={entropy:.3f}; expected_entropy={expected_entropy:.3f}; "
        "mutual_info={mutual_info:.3f}; votes_neg_neu_pos={votes}/{total}. "
        "Trust more when confidence is high and entropy/MI are low."
    ).format(
        conf=float(uncertainty.get("confidence_score", 0.0) or 0.0),
        vote_conf=float(uncertainty.get("vote_confidence", 0.0) or 0.0),
        entropy=float(uncertainty.get("predictive_entropy", 0.0) or 0.0),
        expected_entropy=float(uncertainty.get("expected_entropy", 0.0) or 0.0),
        mutual_info=float(uncertainty.get("mutual_information", 0.0) or 0.0),
        votes="/".join(str(count) for count in counts),
        total=total,
    )


def legacy_uncertainty_args(
    expert_prediction: str,
    uncertainty: dict[str, Any],
    language: str,
) -> dict[str, str]:
    confidence = float(uncertainty.get("confidence_score", 0.0) or 0.0)
    entropy = float(uncertainty.get("predictive_entropy", 0.0) or 0.0)
    return {
        "plm_suggestion": expert_prediction,
        "plm_confidence_score": f"Confidence in this suggestion is {confidence:.1%}.",
        "plm_predictive_entropy": (
            f"The predictive entropy is {entropy:.3f}. "
            "A higher value (e.g., > 1.0) indicates more internal disagreement/uncertainty in the PLM."
        ),
        "plm_calibration_fact": PLM_CALIBRATION_INFO.get(
            language,
            "General Note: The PLM's calibration behavior for this language is unknown.",
        ),
    }


def rich_uncertainty_args(
    expert_name: str,
    expert_prediction: str,
    probabilities: dict[str, Any],
    uncertainty: dict[str, Any],
    language: str,
) -> dict[str, str]:
    confidence = float(uncertainty.get("confidence_score", 0.0) or 0.0)
    vote_confidence = float(uncertainty.get("vote_confidence", 0.0) or 0.0)
    predictive_entropy = float(uncertainty.get("predictive_entropy", 0.0) or 0.0)
    expected_entropy = float(uncertainty.get("expected_entropy", 0.0) or 0.0)
    mutual_information = float(uncertainty.get("mutual_information", 0.0) or 0.0)
    return {
        "expert_name": expert_name,
        "expert_suggestion": expert_prediction,
        "expert_probabilities": (
            "Expert mean probabilities: "
            f"Negative={float(probabilities.get('Negative', 0.0) or 0.0):.4f}, "
            f"Neutral={float(probabilities.get('Neutral', 0.0) or 0.0):.4f}, "
            f"Positive={float(probabilities.get('Positive', 0.0) or 0.0):.4f}. "
            "More concentrated probabilities indicate a stronger expert preference."
        ),
        "expert_confidence_score": (
            f"Confidence in the expert suggestion is {confidence:.1%}; "
            f"MC vote confidence is {vote_confidence:.1%}."
        ),
        "expert_predictive_entropy": (
            f"Predictive entropy is {predictive_entropy:.3f}. "
            "Higher values indicate more uncertainty in the averaged expert distribution."
        ),
        "expert_expected_entropy": (
            f"Expected entropy is {expected_entropy:.3f}. "
            "Higher values indicate that individual stochastic expert samples are themselves uncertain."
        ),
        "expert_mutual_information": (
            f"Mutual information is {mutual_information:.3f}. "
            "Higher values indicate epistemic/model uncertainty across stochastic expert samples."
        ),
        "expert_vote_distribution": (
            "MC vote distribution across labels: "
            f"{format_vote_distribution(uncertainty)}. "
            "Divided votes mean the article evidence should receive more weight."
        ),
        "expert_calibration_fact": (
            f"{PLM_CALIBRATION_INFO.get(language, 'General Note: calibration behavior for this language is unknown.')} "
            "Use high confidence, concentrated probabilities, low entropy, low mutual information, and consistent votes as support for the expert; override the expert when article evidence clearly disagrees."
        ),
    }


def example_from_item(
    item: dict[str, Any],
    success: dict[str, Any],
    language: str,
    prompt_masked: bool,
    max_article_chars: int | None,
    include_label: bool,
    prompt_style: str = "current",
) -> dspy.Example | None:
    json_key, probabilities_key, uncertainty_key = expert_key_for(success)
    sentiment = normalize_sentiment_label(item.get("sentiment"))
    if include_label and sentiment is None:
        return None

    article = item.get("article", "") or ""
    aspect = item.get("aspect", "") or ""
    if not article or not aspect:
        return None
    if prompt_masked:
        article = mask_aspect_in_text(article)
        aspect = MASK_TOKEN
    elif prompt_style != "legacy":
        article = strip_aspect_tags(article)
    article = maybe_clip_text(article, max_article_chars)

    expert_prediction = normalize_sentiment_label(item.get(json_key)) or NEUTRAL_STR
    probabilities = item.get(probabilities_key, {}) or {}
    uncertainty = item.get(uncertainty_key, {}) or {}

    expert_name = EXPERT_DISPLAY_NAMES.get(success["expert"], success["expert"])
    if prompt_style == "legacy":
        args = {
            "article": article,
            "aspect": aspect,
            **legacy_uncertainty_args(expert_prediction, uncertainty, language),
        }
        input_keys = [
            "article",
            "aspect",
            "plm_suggestion",
            "plm_confidence_score",
            "plm_predictive_entropy",
            "plm_calibration_fact",
        ]
    elif prompt_style == "rich":
        args = {
            "article": article,
            "aspect": aspect,
            **rich_uncertainty_args(expert_name, expert_prediction, probabilities, uncertainty, language),
        }
        input_keys = [
            "article",
            "aspect",
            "expert_name",
            "expert_suggestion",
            "expert_probabilities",
            "expert_confidence_score",
            "expert_predictive_entropy",
            "expert_expected_entropy",
            "expert_mutual_information",
            "expert_vote_distribution",
            "expert_calibration_fact",
        ]
    else:
        args = {
            "article": article,
            "aspect": aspect,
            "expert_name": expert_name,
            "expert_suggestion": expert_prediction,
            "expert_probabilities": format_probabilities(probabilities),
            "expert_uncertainty": format_compact_uncertainty(uncertainty),
        }
        input_keys = [
            "article",
            "aspect",
            "expert_name",
            "expert_suggestion",
            "expert_probabilities",
            "expert_uncertainty",
        ]
    if include_label:
        args["sentiment"] = sentiment

    return dspy.Example(**args).with_inputs(*input_keys)


def prepare_examples(
    records: list[dict[str, Any]],
    success: dict[str, Any],
    language: str,
    prompt_masked: bool,
    max_article_chars: int | None,
    include_label: bool = True,
    prompt_style: str = "current",
) -> list[dspy.Example]:
    examples = []
    for item in records:
        example = example_from_item(
            item,
            success,
            language,
            prompt_masked,
            max_article_chars,
            include_label,
            prompt_style,
        )
        if example is not None:
            examples.append(example)
    return examples


def stratified_sample_records(
    records: list[dict[str, Any]],
    size: int,
    seed: int,
) -> list[dict[str, Any]]:
    if size <= 0 or size >= len(records):
        return list(records)
    labels = [normalize_sentiment_label(item.get("sentiment")) for item in records]
    valid_indices = [idx for idx, label in enumerate(labels) if label is not None]
    valid_records = [records[idx] for idx in valid_indices]
    valid_labels = [labels[idx] for idx in valid_indices]
    if size >= len(valid_records):
        return valid_records
    counts = Counter(valid_labels)
    if len(counts) < 2 or min(counts.values()) < 2:
        rng = random.Random(seed)
        sample = valid_records[:]
        rng.shuffle(sample)
        return sample[:size]
    _, sampled = train_test_split(
        valid_records,
        test_size=size,
        stratify=valid_labels,
        random_state=seed,
    )
    return list(sampled)


def validate_sentiment(example: dspy.Example, pred: dspy.Prediction, trace=None) -> bool:
    return getattr(pred, "sentiment", "").strip().lower() == example.sentiment.strip().lower()


def build_program(prompt_style: str = "current") -> dspy.Module:
    if prompt_style == "legacy":
        signature = LegacyPLMUncertaintyCOTSignature
    elif prompt_style == "rich":
        signature = RichExpertUncertaintyCOTSignature
    else:
        signature = ExpertUncertaintyCOTSignature
    return dspy.ChainOfThought(signature)


def optimize_program(
    trainset: list[dspy.Example],
    valset: list[dspy.Example],
    output_program_path: Path,
    student_lm: dspy.LM,
    teacher_lm: dspy.LM,
    autorun: str,
    init_temperature: float,
    num_threads: int,
    max_errors: int,
    seed: int,
    program_aware_proposer: bool = True,
    data_aware_proposer: bool = False,
    tip_aware_proposer: bool = True,
    fewshot_aware_proposer: bool = False,
    view_data_batch_size: int = 3,
    prompt_style: str = "current",
) -> dspy.Module:
    dspy.settings.configure(lm=student_lm)
    optimizer = MIPROv2(
        metric=validate_sentiment,
        prompt_model=teacher_lm,
        task_model=student_lm,
        max_bootstrapped_demos=0,
        max_labeled_demos=0,
        auto=autorun,
        num_threads=num_threads,
        max_errors=max_errors,
        init_temperature=init_temperature,
        verbose=True,
        seed=seed,
    )
    program = build_program(prompt_style)
    optimized = optimizer.compile(
        student=program,
        trainset=trainset,
        valset=valset,
        requires_permission_to_run=False,
        seed=seed,
        program_aware_proposer=program_aware_proposer,
        data_aware_proposer=data_aware_proposer,
        tip_aware_proposer=tip_aware_proposer,
        fewshot_aware_proposer=fewshot_aware_proposer,
        view_data_batch_size=view_data_batch_size,
    )
    output_program_path.parent.mkdir(parents=True, exist_ok=True)
    optimized.save(str(output_program_path))
    return optimized


def load_program(path: Path, prompt_style: str = "current") -> dspy.Module:
    program = build_program(prompt_style)
    program.load(str(path))
    return program


def run_program_on_examples(
    program: dspy.Module,
    examples: list[dspy.Example],
    records: list[dict[str, Any]],
    num_queries: int,
    num_workers: int,
    progress_label: str | None = None,
) -> list[dict[str, Any]]:
    lock = threading.Lock()
    total = len(examples)
    started = time.time()
    counter = {"done": 0}

    def run_one(index: int) -> dict[str, Any]:
        example = examples[index]
        record = records[index]
        predictions = []
        query_details = []
        for query_idx in range(num_queries):
            detail = {
                "query_index": query_idx,
                "status": "failed",
                "predicted_sentiment": None,
                "reasoning": None,
                "raw_prediction_object_str": None,
            }
            try:
                pred = program(**example.inputs())
                raw_sentiment = getattr(pred, "sentiment", None)
                detail["raw_prediction_object_str"] = str(pred)
                detail["reasoning"] = str(getattr(pred, "reasoning", ""))
                if isinstance(raw_sentiment, str) and raw_sentiment.strip().lower() in STR_TO_INT_LABEL:
                    normalized = raw_sentiment.strip().lower()
                    predictions.append(normalized)
                    detail["status"] = "success"
                    detail["predicted_sentiment"] = normalized
                else:
                    detail["status"] = "invalid_label"
                    detail["predicted_sentiment"] = raw_sentiment
            except Exception as exc:
                detail["status"] = "exception"
                detail["raw_prediction_object_str"] = f"Exception: {exc}"
            query_details.append(detail)
        final_label = mode_label(predictions)
        result = {
            "uuid": record.get("uuid"),
            "ground_truth_int": int(record["sentiment"]) if record.get("sentiment") is not None else None,
            "prediction_int": STR_TO_INT_LABEL.get(final_label),
            "prediction_label": final_label,
            "status": "success" if final_label in STR_TO_INT_LABEL else "failed",
            "dspy_query_details": query_details,
        }
        with lock:
            counter["done"] += 1
            done = counter["done"]
            if done == 1 or done == total or done % 25 == 0:
                elapsed = time.time() - started
                rate = done / elapsed if elapsed > 0 else 0
                eta = (total - done) / rate if rate > 0 else 0
                prefix = f"[{progress_label}] " if progress_label else ""
                print(
                    f"{prefix}[progress] predictions {done}/{total} elapsed={elapsed/60:.1f}m eta={eta/60:.1f}m",
                    flush=True,
                )
        return result

    if num_workers <= 1:
        return [run_one(idx) for idx in range(len(examples))]
    with futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
        return list(executor.map(run_one, range(len(examples))))


def mode_label(predictions: list[str]) -> str:
    valid = [p for p in predictions if p in STR_TO_INT_LABEL]
    if not valid:
        return NEUTRAL_STR
    counts = Counter(valid)
    max_count = max(counts.values())
    modes = [label for label, count in counts.items() if count == max_count]
    if len(modes) == 1:
        return modes[0]
    return NEUTRAL_STR if NEUTRAL_STR in modes else sorted(modes)[0]


def calculate_metrics(results: list[dict[str, Any]]) -> dict[str, Any]:
    y_true = []
    y_pred = []
    for item in results:
        if item.get("status") == "success" and item.get("ground_truth_int") is not None and item.get("prediction_int") is not None:
            y_true.append(int(item["ground_truth_int"]))
            y_pred.append(int(item["prediction_int"]))
    if not y_true:
        return {
            "error": "No successful predictions available for evaluation.",
            "num_samples_evaluated": 0,
        }
    p_macro, r_macro, f1_macro, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", labels=LABELS_INT, zero_division=0
    )
    p_micro, r_micro, f1_micro, _ = precision_recall_fscore_support(
        y_true, y_pred, average="micro", labels=LABELS_INT, zero_division=0
    )
    p_weighted, r_weighted, f1_weighted, _ = precision_recall_fscore_support(
        y_true, y_pred, average="weighted", labels=LABELS_INT, zero_division=0
    )
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision_macro": float(p_macro),
        "recall_macro": float(r_macro),
        "f1_macro": float(f1_macro),
        "precision_micro": float(p_micro),
        "recall_micro": float(r_micro),
        "f1_micro": float(f1_micro),
        "precision_weighted": float(p_weighted),
        "recall_weighted": float(r_weighted),
        "f1_weighted": float(f1_weighted),
        "qwk": float(cohen_kappa_score(y_true, y_pred, weights="quadratic", labels=LABELS_INT)),
        "num_samples_evaluated": len(y_true),
        "per_class_report": classification_report(
            y_true,
            y_pred,
            labels=LABELS_INT,
            target_names=[INT_TO_STR_LABEL[label] for label in LABELS_INT],
            output_dict=True,
            zero_division=0,
        ),
    }


def attach_predictions(records: list[dict[str, Any]], results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_uuid = {item["uuid"]: item for item in results}
    output = []
    for record in records:
        item = copy.deepcopy(record)
        result = by_uuid.get(item.get("uuid"))
        if result is None:
            item["prediction"] = None
            item["processing_status"] = "missing"
        else:
            item["prediction"] = result.get("prediction_int")
            item["prediction_label"] = result.get("prediction_label")
            item["processing_status"] = result.get("status")
            item["dspy_query_details"] = result.get("dspy_query_details", [])
        output.append(item)
    return output


def variant_name(prompt_masked: bool) -> str:
    return "masked" if prompt_masked else "unmasked"


def run_name(language: str, prompt_masked: bool, autorun: str, max_tokens: int, teacher_label: str) -> str:
    return (
        f"dspy-plm-augmented-cot-teacher-{teacher_label}-{max_tokens}-"
        f"{autorun}-uncertainty-{language}-{variant_name(prompt_masked)}"
    )


def optimized_program_filename(
    language: str,
    prompt_masked: bool,
    teacher_label: str,
    autorun: str,
    miprov2_temp: float,
) -> str:
    return (
        f"optimized_program_{language}_dspy-plm-augmented-cot-with-uncertainty_"
        f"{variant_name(prompt_masked)}_teacher_{teacher_label}_"
        f"autorun_{autorun}_temp_{miprov2_temp}.json"
    )


def experiment_dir(
    output_root: Path,
    expert: str,
    language: str,
    prompt_masked: bool,
    autorun: str,
) -> Path:
    return output_root / expert / language / variant_name(prompt_masked) / autorun


def default_sample_size(autorun: str) -> int:
    return AUTORUN_SAMPLE_SIZES[autorun]


def resolve_task_matrix(task_filter: str | None = None) -> list[tuple[str, str]]:
    tasks = []
    for language, experts in DEFAULT_EXPERTS_BY_LANGUAGE.items():
        for expert in experts:
            tasks.append((expert, language))
    if not task_filter or task_filter == "all":
        return tasks
    if task_filter in {"slovenian", "serbian"}:
        return [(expert, language) for expert, language in tasks if language == task_filter]
    if ":" in task_filter:
        expert_filter, language_filter = task_filter.split(":", 1)
        return [
            (expert, language)
            for expert, language in tasks
            if expert == expert_filter and language == language_filter
        ]
    return [(expert, language) for expert, language in tasks if expert == task_filter]
