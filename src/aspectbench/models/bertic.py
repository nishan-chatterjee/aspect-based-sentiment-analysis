"""BERTić adapter metadata for HBS."""

from .base import ModelSpec

MODEL_SPEC = ModelSpec(
    name="bertic",
    display_name="BERTić",
    backend="encoder",
    languages=("hbs",),
    variants=("masked", "unmasked"),
    base_model="classla/bcms-bertic",
    local_base_dir="classla_bcms-bertic",
    huggingface_dir="slavic-specific",
    max_length=512,
)
