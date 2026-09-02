"""Deterministic majority- and confidence-vote inference exports."""

from __future__ import annotations

from collections import Counter
from datetime import datetime
import json
from pathlib import Path
import re
from typing import Any, Iterable, Sequence

from ..registry import normalize_language
from ..runtime.runs import atomic_json


SENTIMENT_NAMES = {-1: "negative", 0: "neutral", 1: "positive"}
_SAFE_FILENAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")


def _confidence(row: dict[str, Any]) -> float:
    uncertainty = row.get("uncertainty") or {}
    return float(uncertainty.get("confidence", 0.0))


def _label_probability(row: dict[str, Any], label: int) -> float:
    probabilities = row.get("probabilities") or {}
    for key, value in probabilities.items():
        if str(key).strip().split(maxsplit=1)[0] == str(label):
            return float(value)
    return 0.0


def _expert(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "model": row["model"],
        "variant": row.get("variant"),
        "prediction": int(row["prediction"]),
        "prediction_name": row.get(
            "prediction_name", SENTIMENT_NAMES[int(row["prediction"])]
        ),
        "confidence": _confidence(row),
        "probabilities": row.get("probabilities", {}),
        "uncertainty": row.get("uncertainty", {}),
    }


def build_ensemble_rows(rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    """Group expert predictions by record and add two transparent aggregations."""

    grouped: dict[str, list[dict[str, Any]]] = {}
    order: list[str] = []
    seen_experts: set[tuple[str, str, str]] = set()
    for row in rows:
        identifier = str(row.get("record_id", "")).strip()
        if not identifier:
            raise ValueError("Every inference row must contain record_id.")
        prediction = int(row["prediction"])
        if prediction not in SENTIMENT_NAMES:
            raise ValueError(f"Unsupported prediction {prediction}; expected -1, 0, or 1.")
        expert_key = (identifier, str(row.get("model", "")), str(row.get("variant", "")))
        if expert_key in seen_experts:
            raise ValueError(
                "Duplicate expert prediction for "
                f"record/model/variant {expert_key!r}."
            )
        seen_experts.add(expert_key)
        if identifier not in grouped:
            grouped[identifier] = []
            order.append(identifier)
        grouped[identifier].append(row)

    output: list[dict[str, Any]] = []
    for identifier in order:
        candidates = grouped[identifier]
        first = candidates[0]
        counts = Counter(int(row["prediction"]) for row in candidates)
        largest = max(counts.values())
        tied_labels = sorted(label for label, count in counts.items() if count == largest)
        majority_label = max(
            tied_labels,
            key=lambda label: (
                sum(_label_probability(row, label) for row in candidates) / len(candidates),
                -abs(label),
                -label,
            ),
        )
        selected = sorted(
            candidates,
            key=lambda row: (
                -_confidence(row),
                str(row.get("model", "")),
                str(row.get("variant", "")),
            ),
        )[0]
        selected_label = int(selected["prediction"])
        output.append(
            {
                "record_id": identifier,
                "language": first.get("language"),
                "article": first.get("article"),
                "aspect": first.get("aspect"),
                "sentiment": first.get("sentiment"),
                "majority_prediction": majority_label,
                "confidence_prediction": selected_label,
                "experts": [_expert(row) for row in candidates],
                "majority_vote": {
                    "prediction": majority_label,
                    "prediction_name": SENTIMENT_NAMES[majority_label],
                    "vote_counts": {
                        str(label): int(counts.get(label, 0)) for label in (-1, 0, 1)
                    },
                    "tied": len(tied_labels) > 1,
                    "tie_break": (
                        "mean_probability_then_neutral_negative_positive"
                        if len(tied_labels) > 1
                        else None
                    ),
                },
                "confidence_vote": {
                    "prediction": selected_label,
                    "prediction_name": SENTIMENT_NAMES[selected_label],
                    "selected_model": selected["model"],
                    "selected_variant": selected.get("variant"),
                    "confidence": _confidence(selected),
                },
            }
        )
    return output


def resolve_output_filename(
    value: str, *, seed: int, now: datetime | None = None
) -> str:
    """Resolve a stable name, a seed-prefixed name, or a minute timestamp."""

    value = value.strip()
    if value == "timestamp":
        moment = now or datetime.now().astimezone()
        return f"{moment.strftime('%Y-%m-%d-%H%M')}-predictions.json"
    if value == "seed":
        return f"seed-{seed}-predictions.json"
    if value.endswith(".json"):
        value = value[:-5]
    if not value or not _SAFE_FILENAME.fullmatch(value) or value in {".", ".."}:
        raise ValueError(
            "--filename must be 'predictions', 'timestamp', 'seed', or a safe file stem."
        )
    return f"{value}.json"


def publish_inference_outputs(
    rows: Sequence[dict[str, Any]],
    *,
    dataset: str,
    run_id: str,
    output_root: str | Path = "outputs",
    filename: str = "predictions",
    seed: int = 42,
    now: datetime | None = None,
) -> tuple[Path, Path]:
    """Write detailed expert rows and record-level ensemble decisions."""

    language = normalize_language(dataset)
    name = resolve_output_filename(filename, seed=seed, now=now)
    directory = Path(output_root) / "inference" / language / run_id
    raw_path = directory / name
    ensemble_path = directory / f"{Path(name).stem}-ensemble.json"
    atomic_json(raw_path, list(rows))
    atomic_json(ensemble_path, build_ensemble_rows(rows))
    return raw_path, ensemble_path


def load_prediction_files(paths: Iterable[str | Path]) -> list[dict[str, Any]]:
    """Load and concatenate the raw outputs of independently scheduled experts."""

    rows: list[dict[str, Any]] = []
    for value in paths:
        path = Path(value)
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, list):
            raise ValueError(f"Expected a JSON list in {path}.")
        rows.extend(payload)
    return rows
