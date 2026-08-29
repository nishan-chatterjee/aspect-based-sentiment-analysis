"""Record-level reporting grouped by target aspect and seen/unseen status."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from statistics import fmean
from typing import Any

from .metrics import LABEL_NAMES, classification_metrics, normalize_label


def normalize_aspect(value: Any) -> str:
    return " ".join(str(value or "").strip().casefold().split())


def seen_aspects_from_records(records: Iterable[Mapping[str, Any]], aspect_key: str = "aspect") -> set[str]:
    aspects = {normalize_aspect(record.get(aspect_key)) for record in records}
    aspects.discard("")
    return aspects


def _labels_from_records(
    records: Iterable[Mapping[str, Any]], gold_key: str, prediction_key: str
) -> tuple[list[int], list[int]]:
    gold: list[int] = []
    predictions: list[int] = []
    for index, record in enumerate(records):
        if gold_key not in record or prediction_key not in record:
            raise ValueError(
                f"Record {index} needs {gold_key!r} and {prediction_key!r} fields."
            )
        gold.append(normalize_label(record[gold_key]))
        predictions.append(normalize_label(record[prediction_key]))
    return gold, predictions


def score_records(
    records: Sequence[Mapping[str, Any]],
    gold_key: str = "sentiment",
    prediction_key: str = "prediction",
) -> dict[str, Any]:
    gold, predictions = _labels_from_records(records, gold_key, prediction_key)
    return classification_metrics(gold, predictions)


def per_aspect_report(
    records: Sequence[Mapping[str, Any]],
    aspect_key: str = "aspect",
    gold_key: str = "sentiment",
    prediction_key: str = "prediction",
) -> dict[str, Any]:
    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    display_names: dict[str, str] = {}
    for index, record in enumerate(records):
        aspect = normalize_aspect(record.get(aspect_key))
        if not aspect:
            raise ValueError(f"Record {index} has no non-empty {aspect_key!r} field.")
        grouped[aspect].append(record)
        display_names.setdefault(aspect, str(record[aspect_key]).strip())

    aspects: dict[str, Any] = {}
    for key in sorted(grouped):
        aspects[key] = {
            "aspect": display_names[key],
            **score_records(grouped[key], gold_key=gold_key, prediction_key=prediction_key),
        }

    def numeric_mean(values: Iterable[float | None]) -> float | None:
        clean = [float(value) for value in values if value is not None]
        return fmean(clean) if clean else None

    summary = {
        "n_aspects": len(aspects),
        "f1_macro_across_aspects": numeric_mean(row["f1_macro"] for row in aspects.values()),
        "qwk_macro_across_aspects": numeric_mean(row["qwk"] for row in aspects.values()),
        "qwk_defined_aspects": sum(row["qwk"] is not None for row in aspects.values()),
        "per_class_f1_macro_across_aspects": {
            name: numeric_mean(row["per_class"][name]["f1"] for row in aspects.values())
            for name in LABEL_NAMES.values()
        },
    }
    return {"summary": summary, "aspects": aspects}


def seen_unseen_report(
    records: Sequence[Mapping[str, Any]],
    seen_aspects: set[str],
    aspect_key: str = "aspect",
    gold_key: str = "sentiment",
    prediction_key: str = "prediction",
) -> dict[str, Any]:
    canonical_seen = {normalize_aspect(value) for value in seen_aspects}
    buckets: dict[str, list[Mapping[str, Any]]] = {"seen": [], "unseen": []}
    for record in records:
        bucket = "seen" if normalize_aspect(record.get(aspect_key)) in canonical_seen else "unseen"
        buckets[bucket].append(record)

    output: dict[str, Any] = {"seen_aspect_count": len(canonical_seen)}
    for name, rows in buckets.items():
        output[name] = (
            score_records(rows, gold_key=gold_key, prediction_key=prediction_key)
            if rows
            else {"n": 0, "available": False}
        )
    return output


def build_evaluation_report(
    records: Sequence[Mapping[str, Any]],
    training_records: Sequence[Mapping[str, Any]] | None = None,
    *,
    aspect_key: str = "aspect",
    gold_key: str = "sentiment",
    prediction_key: str = "prediction",
) -> dict[str, Any]:
    """Build the stable report consumed by the CLI and analysis notebooks."""

    output = {
        "schema_version": 1,
        "fields": {
            "aspect": aspect_key,
            "gold": gold_key,
            "prediction": prediction_key,
        },
        "overall": score_records(records, gold_key=gold_key, prediction_key=prediction_key),
        "by_aspect": per_aspect_report(
            records,
            aspect_key=aspect_key,
            gold_key=gold_key,
            prediction_key=prediction_key,
        ),
    }
    if training_records is not None:
        output["seen_unseen"] = seen_unseen_report(
            records,
            seen_aspects_from_records(training_records, aspect_key=aspect_key),
            aspect_key=aspect_key,
            gold_key=gold_key,
            prediction_key=prediction_key,
        )
    return output
