"""Canonical model, checkpoint-selection, and Hugging Face repository registry."""

from __future__ import annotations

from pathlib import Path
from typing import Any


NAMESPACE = "nishan-chatterjee"
COLLECTION_SLUG = (
    "nishan-chatterjee/aspect-based-sentiment-analysis-6a9016a6d9cab7b093f122d3"
)
TOOLKIT_REPO = f"{NAMESPACE}/aspect-based-sentiment-analysis"
LANGUAGES = ("hbs", "slovenian")
MODES = ("masked", "unmasked")


MODEL_SPECS: dict[str, dict[str, Any]] = {
    "xlmr": {
        "display_name": "XLM-R",
        "backend": "encoder",
        "base_model": "FacebookAI/xlm-roberta-base",
        "local_base_dir": "xlm-roberta-base",
        "max_length": 512,
        "hf_repo": f"{NAMESPACE}/aspectbench-xlmr",
    },
    "han-xlmr": {
        "display_name": "HAN-XLM-R",
        "backend": "han",
        "base_model": "FacebookAI/xlm-roberta-base",
        "local_base_dir": "xlm-roberta-base",
        "max_length": 96,
        "max_sentences": 128,
        "hf_repo": f"{NAMESPACE}/aspectbench-han-xlmr",
    },
    "longformer": {
        "display_name": "XLM-R Longformer",
        "backend": "encoder",
        "base_model": "markussagen/xlm-roberta-longformer-base-4096",
        "local_base_dir": "markussagen_xlm-roberta-longformer-base-4096",
        "max_length": 4096,
        "global_attention": True,
        "hf_repo": f"{NAMESPACE}/aspectbench-longformer",
    },
    "mdeberta-v3": {
        "display_name": "mDeBERTa-v3",
        "backend": "encoder",
        "base_model": "microsoft/mdeberta-v3-base",
        "local_base_dir": "microsoft_mdeberta-v3-base",
        "max_length": 512,
        "hf_repo": f"{NAMESPACE}/aspectbench-mdeberta-v3",
    },
    "mt5": {
        "display_name": "mT5",
        "backend": "mt5",
        "base_model": "google/mt5-base",
        "local_base_dir": "google_mt5-base",
        "max_length": 512,
        "max_target_length": 8,
        "hf_repo": f"{NAMESPACE}/aspectbench-mt5",
    },
    "slavic-specific": {
        "display_name": "BERTić / SloBERTa",
        "backend": "encoder",
        "base_model": {
            "hbs": "classla/bcms-bertic",
            "slovenian": "EMBEDDIA/sloberta",
        },
        "local_base_dir": {
            "hbs": "classla_bcms-bertic",
            "slovenian": "EMBEDDIA_sloberta",
        },
        "max_length": 512,
        "hf_repo": f"{NAMESPACE}/aspectbench-slavic-specific",
    },
    "bge-m3-mlp": {
        "display_name": "BGE-M3 dense + MLP",
        "backend": "bge",
        "base_model": "BAAI/bge-m3",
        "local_base_dir": "embeddings/bge-m3",
        "max_length": 8192,
        "hf_repo": f"{NAMESPACE}/aspectbench-bge-m3-mlp",
    },
}


def _entry(
    source: str | None,
    run: int | None,
    validation_macro_f1: float | None,
    *,
    reason: str | None = None,
    expected_source: str | None = None,
) -> dict[str, Any]:
    return {
        "source": source,
        "run": run,
        "validation_macro_f1": validation_macro_f1,
        "available": source is not None,
        "unavailable_reason": reason,
        "expected_source": expected_source,
    }


CHECKPOINTS: dict[tuple[str, str, str], dict[str, Any]] = {
    ("xlmr", "hbs", "masked"): _entry(
        "additional-tasks/xlmr_truncated/masked/serbian/best_model_0.pt",
        0,
        0.9247234996073129,
    ),
    ("xlmr", "slovenian", "masked"): _entry(
        "additional-tasks/xlmr_truncated/masked/slovenian/best_model_2.pt",
        2,
        0.8357596148205779,
    ),
    ("xlmr", "hbs", "unmasked"): _entry(
        None,
        1,
        0.9261041473729551,
        reason=(
            "Validation metrics identify the best unmasked run, but its trained "
            "checkpoint file is absent from the source tree. Retraining or archive "
            "recovery is required."
        ),
        expected_source="models/xlmr/no_summary/serbian/best_model_1.pt",
    ),
    ("xlmr", "slovenian", "unmasked"): _entry(
        None,
        2,
        0.8286127750538895,
        reason=(
            "Validation metrics identify the best unmasked run, but its trained "
            "checkpoint file is absent from the source tree. Retraining or archive "
            "recovery is required."
        ),
        expected_source="models/xlmr/no_summary/slovenian/best_model_2.pt",
    ),
    ("han-xlmr", "hbs", "masked"): _entry(
        "results/global-context-modelling/simplified-dart-xlmr/serbian/best_model_2.pt",
        2,
        0.9137775530809181,
    ),
    ("han-xlmr", "slovenian", "masked"): _entry(
        "results/global-context-modelling/simplified-dart-xlmr/slovenian/best_model_1.pt",
        1,
        0.8108233024053296,
    ),
    ("han-xlmr", "hbs", "unmasked"): _entry(
        None,
        2,
        0.9106720097631231,
        reason=(
            "Validation metrics identify the best aspect-marker run, but its trained "
            "checkpoint file is absent from the source tree. Retraining or archive "
            "recovery is required."
        ),
        expected_source=(
            "models/global-context-modelling/with-aspect-markers/serbian/best_model_2.pt"
        ),
    ),
    ("han-xlmr", "slovenian", "unmasked"): _entry(
        None,
        2,
        0.8127192586145947,
        reason=(
            "Validation metrics identify the best aspect-marker run, but its trained "
            "checkpoint file is absent from the source tree. Retraining or archive "
            "recovery is required."
        ),
        expected_source=(
            "models/global-context-modelling/with-aspect-markers/slovenian/best_model_2.pt"
        ),
    ),
    ("longformer", "hbs", "masked"): _entry(
        "reviews/longformer/masked/serbian/best_model_0.pt", 0, 0.9235562204487023
    ),
    ("longformer", "slovenian", "masked"): _entry(
        "reviews/longformer/masked/slovenian/best_model_1.pt", 1, 0.8349811751435684
    ),
    ("longformer", "hbs", "unmasked"): _entry(
        "reviews/longformer/unmasked/serbian/best_model_0.pt", 0, 0.9237049934972151
    ),
    ("longformer", "slovenian", "unmasked"): _entry(
        "reviews/longformer/unmasked/slovenian/best_model_1.pt", 1, 0.8250958292470165
    ),
    ("mdeberta-v3", "hbs", "masked"): _entry(
        "reviews/mdeberta/masked/serbian/best_model_0.pt", 0, 0.9150760776163819
    ),
    ("mdeberta-v3", "slovenian", "masked"): _entry(
        "reviews/mdeberta/masked/slovenian/best_model_1.pt", 1, 0.8242447008523016
    ),
    ("mdeberta-v3", "hbs", "unmasked"): _entry(
        "reviews/mdeberta/unmasked/serbian/best_model_0.pt", 0, 0.9190600266461267
    ),
    ("mdeberta-v3", "slovenian", "unmasked"): _entry(
        "reviews/mdeberta/unmasked/slovenian/best_model_1.pt", 1, 0.8223988954252537
    ),
    ("mt5", "hbs", "masked"): _entry(
        "reviews/mt5/masked/serbian/best_model_0.pt", 0, 0.889948707182005
    ),
    ("mt5", "slovenian", "masked"): _entry(
        "reviews/mt5/masked/slovenian/best_model_1.pt", 1, 0.8012255203557931
    ),
    ("mt5", "hbs", "unmasked"): _entry(
        "reviews/mt5/unmasked/serbian/best_model_1.pt", 1, 0.8942548569348862
    ),
    ("mt5", "slovenian", "unmasked"): _entry(
        "reviews/mt5/unmasked/slovenian/best_model_2.pt", 2, 0.8070351391836906
    ),
    ("slavic-specific", "hbs", "masked"): _entry(
        "reviews/slavic_specific/masked/serbian/best_model_1.pt", 1, 0.9151353276491324
    ),
    ("slavic-specific", "slovenian", "masked"): _entry(
        "reviews/slavic_specific/masked/slovenian/best_model_2.pt", 2, 0.8545257213567837
    ),
    ("slavic-specific", "hbs", "unmasked"): _entry(
        "reviews/slavic_specific/unmasked/serbian/best_model_0.pt", 0, 0.9250658853407349
    ),
    ("slavic-specific", "slovenian", "unmasked"): _entry(
        "reviews/slavic_specific/unmasked/slovenian/best_model_2.pt", 2, 0.8638678818708625
    ),
    ("bge-m3-mlp", "hbs", "masked"): _entry(
        None,
        2,
        0.8840815245920188,
        reason="Validation metrics exist, but the selected trained dense MLP head file is absent.",
        expected_source="models/bge-m3_mlp/masked/serbian/best_model_2.pt",
    ),
    ("bge-m3-mlp", "slovenian", "masked"): _entry(
        None,
        1,
        0.7514188081139693,
        reason="Validation metrics exist, but the selected trained dense MLP head file is absent.",
        expected_source="models/bge-m3_mlp/masked/slovenian/best_model_1.pt",
    ),
    ("bge-m3-mlp", "hbs", "unmasked"): _entry(
        None,
        1,
        0.8894579185247774,
        reason="Validation metrics exist, but the selected trained dense MLP head file is absent.",
        expected_source="models/bge-m3_mlp/whole/serbian/best_model_1.pt",
    ),
    ("bge-m3-mlp", "slovenian", "unmasked"): _entry(
        None,
        1,
        0.7393257242124599,
        reason="Validation metrics exist, but the selected trained dense MLP head file is absent.",
        expected_source="models/bge-m3_mlp/whole/slovenian/best_model_1.pt",
    ),
}


def _apply_local_release_metadata() -> None:
    """Expose newly retrained slots before their metadata reaches the remote."""

    models_root = Path(__file__).resolve().parents[1] / "models"
    for model_name in MODEL_SPECS:
        path = models_root / model_name / "availability.json"
        if not path.is_file():
            continue
        try:
            import json

            entries = json.loads(path.read_text(encoding="utf-8"))["entries"]
        except (KeyError, OSError, TypeError, ValueError):
            continue
        for entry in entries:
            key = (model_name, entry.get("language"), entry.get("mode"))
            weight = models_root / model_name / str(entry.get("weight_path", "")).split(
                f"{model_name}/", 1
            )[-1]
            if key not in CHECKPOINTS or not entry.get("available") or not weight.is_file():
                continue
            CHECKPOINTS[key].update(
                available=True,
                unavailable_reason=None,
                validation_macro_f1=entry.get("validation_macro_f1"),
                run=entry.get("selected_split", CHECKPOINTS[key].get("run")),
            )


_apply_local_release_metadata()


def model_spec(model_name: str, language: str) -> dict[str, Any]:
    spec = dict(MODEL_SPECS[model_name])
    for key in ("base_model", "local_base_dir"):
        if isinstance(spec[key], dict):
            spec[key] = spec[key][language]
    return spec


def weight_path(model_root: str | Path, model_name: str, language: str, mode: str) -> Path:
    return Path(model_root) / model_name / language / f"{mode}.pt"


def all_slots():
    for model_name in MODEL_SPECS:
        for language in LANGUAGES:
            for mode in MODES:
                yield model_name, language, mode, CHECKPOINTS[(model_name, language, mode)]
