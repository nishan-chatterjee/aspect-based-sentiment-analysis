"""XLM-R Longformer adapter metadata."""

from .base import ModelSpec

MODEL_SPEC = ModelSpec(
    name="longformer",
    display_name="XLM-R Longformer",
    backend="long-document-encoder",
    languages=("hbs", "sl"),
    variants=("masked", "unmasked"),
    base_model="markussagen/xlm-roberta-longformer-base-4096",
    local_base_dir="markussagen_xlm-roberta-longformer-base-4096",
    huggingface_dir="longformer",
    max_length=4096,
    extra=(("global_attention", True),),
)
