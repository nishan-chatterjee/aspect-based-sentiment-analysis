"""Canonical model registry and one/few/all selection."""

from __future__ import annotations

from collections.abc import Iterable

from .models.base import ModelSpec
from .models.bertic import MODEL_SPEC as BERTIC
from .models.bge_m3_mlp import MODEL_SPEC as BGE_M3_MLP
from .models.han_xlmr import MODEL_SPEC as HAN_XLMR
from .models.longformer import MODEL_SPEC as LONGFORMER
from .models.mdeberta import MODEL_SPEC as MDEBERTA
from .models.mt5 import MODEL_SPEC as MT5
from .models.sloberta import MODEL_SPEC as SLOBERTA
from .models.xlmr import MODEL_SPEC as XLMR


_SPECS = (XLMR, HAN_XLMR, LONGFORMER, MDEBERTA, MT5, BERTIC, SLOBERTA, BGE_M3_MLP)
MODEL_REGISTRY: dict[str, ModelSpec] = {spec.name: spec for spec in _SPECS}

LANGUAGE_ALIASES = {
    "hbs": "hbs",
    "serbian": "hbs",
    "serbo-croatian": "hbs",
    "sl": "sl",
    "slovenian": "sl",
    "slovene": "sl",
}

MODEL_ALIASES = {
    "han_xlmr": "han-xlmr",
    "mdeberta": "mdeberta-v3",
    "mdeberta_v3": "mdeberta-v3",
    "bge_m3_mlp": "bge-m3-mlp",
}


def normalize_language(language: str) -> str:
    try:
        return LANGUAGE_ALIASES[language.strip().lower()]
    except KeyError as exc:
        raise ValueError(f"Unknown language {language!r}; use 'hbs' or 'sl'.") from exc


def available_models(language: str | None = None) -> tuple[str, ...]:
    if language is None:
        return tuple(MODEL_REGISTRY)
    canonical = normalize_language(language)
    return tuple(name for name, spec in MODEL_REGISTRY.items() if canonical in spec.languages)


def normalize_model_name(name: str, language: str | None = None) -> str:
    normalized = name.strip().lower().replace("_", "-")
    normalized = MODEL_ALIASES.get(name.strip().lower(), normalized)
    if normalized == "slavic-specific":
        if language is None:
            raise ValueError("The 'slavic-specific' alias requires --language hbs or sl.")
        normalized = "bertic" if normalize_language(language) == "hbs" else "sloberta"
    return normalized


def resolve_model(name: str, language: str | None = None, variant: str | None = None) -> ModelSpec:
    canonical_language = normalize_language(language) if language is not None else None
    canonical_name = normalize_model_name(name, canonical_language)
    try:
        spec = MODEL_REGISTRY[canonical_name]
    except KeyError as exc:
        choices = ", ".join(available_models(canonical_language))
        raise ValueError(f"Unknown model {name!r}; choices: {choices}") from exc
    if canonical_language is not None and not spec.supports(canonical_language, variant):
        raise ValueError(
            f"{spec.name} does not support language={canonical_language!r}, variant={variant!r}."
        )
    return spec


def select_models(
    names: Iterable[str], language: str | None = None, variant: str | None = None
) -> tuple[ModelSpec, ...]:
    requested = tuple(names)
    if not requested:
        raise ValueError("Select at least one model or use 'all'.")
    if "all" in {name.strip().lower() for name in requested}:
        if len(requested) != 1:
            raise ValueError("Use 'all' alone, not together with explicit model names.")
        requested = available_models(language)
    output: list[ModelSpec] = []
    seen: set[str] = set()
    for name in requested:
        spec = resolve_model(name, language=language, variant=variant)
        if spec.name not in seen:
            output.append(spec)
            seen.add(spec.name)
    return tuple(output)
