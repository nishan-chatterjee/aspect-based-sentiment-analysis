"""Stable classification metrics for the original AspectBench label space."""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence
import math
import warnings
from typing import Any

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    cohen_kappa_score,
    confusion_matrix,
    precision_recall_fscore_support,
)


LABELS = (-1, 0, 1)
LABEL_NAMES = {-1: "negative", 0: "neutral", 1: "positive"}


def normalize_label(value: Any) -> int:
    """Return an original label (`-1`, `0`, `1`) without remapping class IDs."""

    if isinstance(value, bool):
        raise ValueError(f"Boolean is not a sentiment label: {value!r}")
    try:
        label = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Invalid sentiment label: {value!r}") from exc
    if label not in LABELS:
        raise ValueError(f"Sentiment label must be one of {LABELS}: {value!r}")
    return label


def _validated_labels(y_true: Sequence[Any], y_pred: Sequence[Any]) -> tuple[list[int], list[int]]:
    if len(y_true) != len(y_pred):
        raise ValueError(f"Gold/prediction length mismatch: {len(y_true)} != {len(y_pred)}")
    if not y_true:
        raise ValueError("At least one scored record is required.")
    return [normalize_label(value) for value in y_true], [normalize_label(value) for value in y_pred]


def quadratic_weighted_kappa(y_true: Sequence[int], y_pred: Sequence[int]) -> float | None:
    """Compute QWK, returning `None` when the denominator is undefined."""

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        value = float(cohen_kappa_score(y_true, y_pred, labels=LABELS, weights="quadratic"))
    return value if math.isfinite(value) else None


def imbalance_diagnostics(y_true: Sequence[int]) -> dict[str, Any]:
    counts = Counter(y_true)
    support = {LABEL_NAMES[label]: int(counts.get(label, 0)) for label in LABELS}
    total = len(y_true)
    proportions = {name: count / total for name, count in support.items()}
    nonzero = [count for count in support.values() if count]
    majority_name, majority_support = max(support.items(), key=lambda item: item[1])
    return {
        "support": support,
        "proportions": proportions,
        "absent_classes": [name for name, count in support.items() if count == 0],
        "majority_class": majority_name,
        "majority_fraction": majority_support / total,
        "max_to_min_nonzero_support_ratio": (
            max(nonzero) / min(nonzero) if len(nonzero) > 1 else 1.0
        ),
    }


def classification_metrics(y_true: Sequence[Any], y_pred: Sequence[Any]) -> dict[str, Any]:
    """Score predictions with fixed labels so minority/absent classes stay visible."""

    gold, predicted = _validated_labels(y_true, y_pred)
    p_macro, r_macro, f1_macro, _ = precision_recall_fscore_support(
        gold, predicted, labels=LABELS, average="macro", zero_division=0
    )
    p_weighted, r_weighted, f1_weighted, _ = precision_recall_fscore_support(
        gold, predicted, labels=LABELS, average="weighted", zero_division=0
    )
    per_precision, per_recall, per_f1, per_support = precision_recall_fscore_support(
        gold, predicted, labels=LABELS, average=None, zero_division=0
    )
    per_class = {
        LABEL_NAMES[label]: {
            "label": label,
            "precision": float(per_precision[index]),
            "recall": float(per_recall[index]),
            "f1": float(per_f1[index]),
            "support": int(per_support[index]),
        }
        for index, label in enumerate(LABELS)
    }
    return {
        "n": len(gold),
        "accuracy": float(accuracy_score(gold, predicted)),
        "precision_macro": float(p_macro),
        "recall_macro": float(r_macro),
        "f1_macro": float(f1_macro),
        "precision_weighted": float(p_weighted),
        "recall_weighted": float(r_weighted),
        "f1_weighted": float(f1_weighted),
        "qwk": quadratic_weighted_kappa(gold, predicted),
        "per_class": per_class,
        "confusion_matrix": confusion_matrix(gold, predicted, labels=LABELS).astype(int).tolist(),
        "imbalance": imbalance_diagnostics(gold),
    }
