#!/usr/bin/env python3
"""Portable filesystem contract for camera-ready entry points."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


def _path_from_env(name: str, fallback: Path) -> Path:
    value = os.environ.get(name)
    return Path(value).expanduser().resolve() if value else fallback.resolve()


@dataclass(frozen=True)
class ReleasePaths:
    root: Path
    data: Path
    models: Path
    results: Path
    reviews: Path
    source_root: Path | None

    @classmethod
    def from_environment(cls) -> "ReleasePaths":
        default_root = Path(__file__).resolve().parents[1]
        root = _path_from_env("ABSA_RELEASE_ROOT", default_root)
        source_value = os.environ.get("ABSA_SOURCE_ROOT")
        return cls(
            root=root,
            data=_path_from_env("ABSA_DATA_DIR", root / "data"),
            models=_path_from_env("ABSA_MODEL_DIR", root / "models"),
            results=_path_from_env("ABSA_RESULTS_DIR", root / "results"),
            reviews=_path_from_env("ABSA_REVIEWS_DIR", root / "reviews"),
            source_root=(
                Path(source_value).expanduser().resolve() if source_value else None
            ),
        )


PATHS = ReleasePaths.from_environment()
