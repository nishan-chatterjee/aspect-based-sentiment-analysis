"""Bridge canonical AspectBench names to the validated Hugging Face release engine."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
from types import ModuleType
from typing import Any

from ..registry import normalize_language, resolve_model


def release_scripts_dir(repository_root: str | Path) -> Path:
    path = Path(repository_root).resolve() / "huggingface" / "scripts"
    if not (path / "inference.py").is_file():
        raise FileNotFoundError(f"Hugging Face inference engine not found under {path}")
    return path


def _load_module(path: Path, name: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def load_release_modules(repository_root: str | Path) -> tuple[ModuleType, ModuleType]:
    scripts = release_scripts_dir(repository_root)
    registry_name = "aspectbench_hf_model_registry"
    registry = sys.modules.get(registry_name) or _load_module(
        scripts / "model_registry.py", registry_name
    )
    # inference.py uses the historical absolute import name.
    previous = sys.modules.get("model_registry")
    sys.modules["model_registry"] = registry
    try:
        inference_name = "aspectbench_hf_inference"
        inference = sys.modules.get(inference_name) or _load_module(
            scripts / "inference.py", inference_name
        )
    finally:
        if previous is None:
            sys.modules.pop("model_registry", None)
        else:
            sys.modules["model_registry"] = previous
    return inference, registry


def release_coordinates(model: str, language: str) -> tuple[str, str]:
    canonical_language = normalize_language(language)
    spec = resolve_model(model, language=canonical_language)
    release_model = spec.huggingface_dir
    release_language = "slovenian" if canonical_language == "sl" else "hbs"
    return release_model, release_language


def checkpoint_status(
    repository_root: str | Path,
    model_root: str | Path,
    model: str,
    language: str,
    variant: str,
) -> dict[str, Any]:
    _, registry = load_release_modules(repository_root)
    release_model, release_language = release_coordinates(model, language)
    selection = registry.CHECKPOINTS[(release_model, release_language, variant)]
    weight = registry.weight_path(model_root, release_model, release_language, variant)
    return {
        "available": bool(selection["available"] and weight.is_file()),
        "registry_available": bool(selection["available"]),
        "weight_file": str(weight),
        "reason": selection.get("unavailable_reason") if not selection["available"] else (
            None if weight.is_file() else f"Checkpoint file is missing: {weight}"
        ),
        "release_model": release_model,
        "release_language": release_language,
    }


def create_engine(
    *,
    repository_root: str | Path,
    model_root: str | Path,
    base_model_root: str | Path | None,
    model: str,
    language: str,
    variant: str,
    device: str,
):
    inference, _ = load_release_modules(repository_root)
    release_model, release_language = release_coordinates(model, language)
    return inference.InferenceEngine(
        model_name=release_model,
        language=release_language,
        mode=variant,
        model_root=model_root,
        base_model_root=base_model_root,
        device=device,
    )
