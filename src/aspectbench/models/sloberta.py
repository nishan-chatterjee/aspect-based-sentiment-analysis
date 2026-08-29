"""SloBERTa adapter metadata for Slovenian."""

from .base import ModelSpec

MODEL_SPEC = ModelSpec(
    name="sloberta",
    display_name="SloBERTa",
    backend="encoder",
    languages=("sl",),
    variants=("masked", "unmasked"),
    base_model="EMBEDDIA/sloberta",
    local_base_dir="EMBEDDIA_sloberta",
    huggingface_dir="slavic-specific",
    max_length=512,
)
