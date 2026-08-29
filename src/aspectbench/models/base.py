"""Shared contracts for architecture-specific model adapters."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import re
from typing import Any


_KEBAB_CASE = re.compile(r"^[a-z0-9]+(?:-[a-z0-9]+)*$")


@dataclass(frozen=True)
class ModelSpec:
    """Stable registry metadata used by inference and training orchestration."""

    name: str
    display_name: str
    backend: str
    languages: tuple[str, ...]
    variants: tuple[str, ...]
    base_model: str
    local_base_dir: str
    huggingface_dir: str
    max_length: int
    trainable: bool = True
    extra: tuple[tuple[str, Any], ...] = ()

    def __post_init__(self) -> None:
        if not _KEBAB_CASE.fullmatch(self.name):
            raise ValueError(f"Model name must be kebab-case: {self.name!r}")
        if not self.languages or any(value not in {"hbs", "sl"} for value in self.languages):
            raise ValueError(f"Unsupported languages for {self.name}: {self.languages}")
        if not self.variants or any(value not in {"masked", "unmasked"} for value in self.variants):
            raise ValueError(f"Unsupported variants for {self.name}: {self.variants}")

    def supports(self, language: str, variant: str | None = None) -> bool:
        return language in self.languages and (variant is None or variant in self.variants)

    def to_dict(self) -> dict[str, Any]:
        output = asdict(self)
        output["extra"] = dict(self.extra)
        return output
