"""BGE-M3 dense-embedding MLP adapter metadata."""

from .base import ModelSpec

MODEL_SPEC = ModelSpec(
    name="bge-m3-mlp",
    display_name="BGE-M3 dense + MLP",
    backend="embedding-mlp",
    languages=("hbs", "sl"),
    variants=("masked", "unmasked"),
    base_model="BAAI/bge-m3",
    local_base_dir="embeddings/bge-m3",
    huggingface_dir="bge-m3-mlp",
    max_length=8192,
    extra=(("released_head_available", False),),
)
