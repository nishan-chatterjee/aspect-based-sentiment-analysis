"""Stable DSPy signature and OpenAI-compatible endpoint configuration."""

from __future__ import annotations

from pathlib import Path
import os
import json
from typing import Any

from ..registry import normalize_language, resolve_model


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


def program_path(
    *,
    source: str,
    model: str,
    dataset: str,
    variant: str,
    program_root: str | Path = "selective-deferral-programs",
    run_id: str | None = None,
) -> Path:
    """Resolve immutable packaged programs or local user-optimized programs."""

    language = normalize_language(dataset)
    canonical_model = resolve_model(model, language=language, variant=variant).name
    root = Path(program_root)
    if source == "precalibrated":
        if run_id is not None:
            raise ValueError("--program-run-id is only valid with --program-source optimized.")
        path = root / "precalibrated" / canonical_model / language / variant / "program.json"
    elif source == "optimized":
        if not run_id:
            raise ValueError("--program-run-id is required with --program-source optimized.")
        path = root / "optimized" / canonical_model / language / variant / run_id / "program.json"
    else:
        raise ValueError("program source must be 'precalibrated' or 'optimized'")
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def optimized_program_dir(
    *,
    model: str,
    dataset: str,
    variant: str,
    run_id: str,
    program_root: str | Path = "selective-deferral-programs",
) -> Path:
    language = normalize_language(dataset)
    canonical_model = resolve_model(model, language=language, variant=variant).name
    return Path(program_root) / "optimized" / canonical_model / language / variant / run_id


def validate_program_metadata(
    path: str | Path,
    *,
    model: str,
    dataset: str,
    variant: str,
    allow_mismatch: bool = False,
) -> dict[str, Any] | None:
    program = Path(path)
    metadata_path = program.parent / "metadata.json"
    if not metadata_path.is_file():
        if allow_mismatch:
            return None
        raise FileNotFoundError(
            f"Program metadata is required for compatibility checks: {metadata_path}"
        )
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    language = normalize_language(dataset)
    canonical_model = resolve_model(model, language=language, variant=variant).name
    observed = {
        "model": metadata.get("model", metadata.get("primary_model")),
        "dataset": metadata.get("language", metadata.get("dataset")),
        "variant": metadata.get("prompt_variant", metadata.get("variant")),
    }
    expected = {"model": canonical_model, "dataset": language, "variant": variant}
    mismatches = {
        key: {"expected": expected[key], "observed": observed[key]}
        for key in expected
        if observed[key] != expected[key]
    }
    if mismatches and not allow_mismatch:
        raise ValueError(f"DSPy program metadata mismatch: {mismatches}")
    return metadata


def configure_lm(
    *,
    model: str,
    api_base: str,
    api_key: str = "local",
    model_type: str = "chat",
    temperature: float = 0.0,
    max_tokens: int = 256,
    cache: bool = False,
    configure: bool = True,
):
    dspy = require_dspy()
    model_name = model if "/" in model else f"openai/{model}"
    normalized_api_base = api_base.rstrip("/")
    if not normalized_api_base.endswith("/v1"):
        normalized_api_base += "/v1"
    lm = dspy.LM(
        model=model_name,
        api_base=normalized_api_base,
        api_key=api_key,
        model_type=model_type,
        temperature=temperature,
        max_tokens=max_tokens,
        cache=cache,
        num_retries=3,
    )
    if configure:
        dspy.configure(lm=lm)
    return lm


def field(prediction: Any, name: str, default: str = "") -> str:
    if hasattr(prediction, name):
        return str(getattr(prediction, name))
    if isinstance(prediction, dict):
        return str(prediction.get(name, default))
    return default
