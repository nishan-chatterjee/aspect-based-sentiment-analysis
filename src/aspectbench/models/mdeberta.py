"""mDeBERTa-v3 adapter metadata."""

from .base import ModelSpec

MODEL_SPEC = ModelSpec(
    name="mdeberta-v3",
    display_name="mDeBERTa-v3",
    backend="encoder",
    languages=("hbs", "sl"),
    variants=("masked", "unmasked"),
    base_model="microsoft/mdeberta-v3-base",
    local_base_dir="microsoft_mdeberta-v3-base",
    huggingface_dir="mdeberta-v3",
    max_length=512,
)
