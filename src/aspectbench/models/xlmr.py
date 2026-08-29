"""XLM-R adapter metadata; training/inference logic is migrated here."""

from .base import ModelSpec

MODEL_SPEC = ModelSpec(
    name="xlmr",
    display_name="XLM-R",
    backend="encoder",
    languages=("hbs", "sl"),
    variants=("masked", "unmasked"),
    base_model="FacebookAI/xlm-roberta-base",
    local_base_dir="xlm-roberta-base",
    huggingface_dir="xlmr",
    max_length=512,
    extra=(("published_masked_lineage", "truncated"),),
)
