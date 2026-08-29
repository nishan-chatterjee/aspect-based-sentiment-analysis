"""Prepare compact, sortable records for local qualitative review."""

from __future__ import annotations

from collections import Counter
from typing import Any, Sequence

from ..evaluation.metrics import normalize_label


def _confidence(record: dict[str, Any]) -> float | None:
    uncertainty = record.get("uncertainty", {})
    if isinstance(uncertainty, dict) and uncertainty.get("confidence") is not None:
        return float(uncertainty["confidence"])
    return None


def build_error_review(
    records: Sequence[dict[str, Any]],
    *,
    include_correct: bool = False,
    limit: int | None = None,
) -> dict[str, Any]:
    """Sort errors by confident-wrong first and summarize error transitions."""

    rows: list[dict[str, Any]] = []
    transitions: Counter[str] = Counter()
    for index, record in enumerate(records):
        gold_value = record.get("sentiment", record.get("gold_sentiment"))
        prediction_value = record.get("prediction", record.get("predicted_sentiment"))
        if gold_value is None or prediction_value is None:
            raise ValueError(f"Record {index} needs gold sentiment and prediction.")
        gold = normalize_label(gold_value)
        prediction = normalize_label(prediction_value)
        correct = gold == prediction
        if correct and not include_correct:
            continue
        transitions[f"{gold}->{prediction}"] += 1
        uncertainty = record.get("uncertainty", {})
        rows.append(
            {
                "record_id": record.get("record_id", record.get("uuid", str(index))),
                "model": record.get("model"),
                "aspect": record.get("aspect"),
                "sentiment": gold,
                "prediction": prediction,
                "correct": correct,
                "confidence": _confidence(record),
                "predictive_entropy_bits": (
                    uncertainty.get("predictive_entropy_bits")
                    if isinstance(uncertainty, dict)
                    else None
                ),
                "action": record.get("action"),
                "article": record.get("article", record.get("input_article")),
            }
        )
    rows.sort(
        key=lambda row: (
            row["correct"],
            -(row["confidence"] if row["confidence"] is not None else -1.0),
            str(row["record_id"]),
        )
    )
    if limit is not None:
        if limit < 1:
            raise ValueError("limit must be at least 1")
        rows = rows[:limit]
    return {
        "schema_version": 1,
        "input_records": len(records),
        "review_records": len(rows),
        "include_correct": include_correct,
        "error_transitions": dict(sorted(transitions.items())),
        "records": rows,
    }
