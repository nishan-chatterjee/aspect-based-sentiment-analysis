"""Hierarchical-attention XLM-R adapter metadata."""

from .base import ModelSpec

MODEL_SPEC = ModelSpec(
    name="han-xlmr",
    display_name="HAN-XLM-R",
    backend="hierarchical-encoder",
    languages=("hbs", "sl"),
    variants=("masked", "unmasked"),
    base_model="FacebookAI/xlm-roberta-base",
    local_base_dir="xlm-roberta-base",
    huggingface_dir="han-xlmr",
    max_length=96,
    extra=(("max_sentences", 128), ("interaction_layers", 2), ("interaction_heads", 8)),
)
