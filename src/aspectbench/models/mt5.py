"""mT5 text-to-text classification adapter metadata."""

from .base import ModelSpec

MODEL_SPEC = ModelSpec(
    name="mt5",
    display_name="mT5",
    backend="text-to-text",
    languages=("hbs", "sl"),
    variants=("masked", "unmasked"),
    base_model="google/mt5-base",
    local_base_dir="google_mt5-base",
    huggingface_dir="mt5",
    max_length=512,
    extra=(("max_target_length", 8), ("scoring", "candidate-label-conditional-loss")),
)
