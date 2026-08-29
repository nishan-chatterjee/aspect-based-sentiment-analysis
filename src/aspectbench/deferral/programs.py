"""Stable DSPy signature and OpenAI-compatible endpoint configuration."""

from __future__ import annotations

from pathlib import Path
import os
from typing import Any


SIGNATURE_INSTRUCTIONS = """You are a selective-deference router for aspect-based sentiment analysis.
Use the article, tagged aspect, primary model prediction, calibrated probabilities,
uncertainty, and optional auxiliary predictions. Return concise reasoning, then one
action: keep_plm, override, or abstain_uncertain. Sentiment must be negative,
neutral, or positive. Never invent facts not present in the supplied record."""


def require_dspy():
    # Local endpoints do not need LiteLLM's periodically downloaded price map.
    os.environ.setdefault("LITELLM_LOCAL_MODEL_COST_MAP", "True")
    try:
        import dspy
    except ImportError as exc:
        raise RuntimeError(
            "DSPy is unavailable. Activate the 'absa' or 'vllm' conda environment."
        ) from exc
    return dspy


def build_program():
    dspy = require_dspy()

    class SelectiveDeferralSignature(dspy.Signature):
        """Selectively keep, override, or abstain from a PLM sentiment prediction."""

        article: str = dspy.InputField(desc="Article with <aspect>...</aspect> markers.")
        aspect: str = dspy.InputField(desc="Aspect being classified.")
        primary_expert: str = dspy.InputField(desc="Primary PLM name.")
        primary_prediction: str = dspy.InputField(desc="Primary sentiment label.")
        primary_probabilities: str = dspy.InputField(desc="Calibrated class probabilities.")
        primary_uncertainty: str = dspy.InputField(desc="Uncertainty statistics.")
        auxiliary_experts: str = dspy.InputField(desc="Other model predictions, if any.")
        routing_context: str = dspy.InputField(desc="Dataset and routing policy context.")
        reasoning: str = dspy.OutputField(desc="Brief evidence-based routing rationale.")
        action: str = dspy.OutputField(
            desc="Exactly one of keep_plm, override, abstain_uncertain."
        )
        sentiment: str = dspy.OutputField(
            desc="Exactly one of negative, neutral, positive."
        )

    SelectiveDeferralSignature.__doc__ = SIGNATURE_INSTRUCTIONS
    return dspy.ChainOfThought(SelectiveDeferralSignature)


def load_or_build_program(path: str | Path | None = None):
    program = build_program()
    if path is not None:
        program.load(str(path))
    return program


def configure_lm(
    *,
    model: str,
    api_base: str,
    api_key: str = "local",
    model_type: str = "chat",
    temperature: float = 0.0,
    max_tokens: int = 256,
    cache: bool = False,
):
    dspy = require_dspy()
    model_name = model if "/" in model else f"openai/{model}"
    lm = dspy.LM(
        model=model_name,
        api_base=api_base.rstrip("/") + "/v1",
        api_key=api_key,
        model_type=model_type,
        temperature=temperature,
        max_tokens=max_tokens,
        cache=cache,
        num_retries=3,
    )
    dspy.configure(lm=lm)
    return lm


def field(prediction: Any, name: str, default: str = "") -> str:
    if hasattr(prediction, name):
        return str(getattr(prediction, name))
    if isinstance(prediction, dict):
        return str(prediction.get(name, default))
    return default
