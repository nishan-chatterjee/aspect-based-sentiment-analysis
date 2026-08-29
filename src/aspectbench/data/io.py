"""Load records from the JSON shapes used by AspectBench releases."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


RECORD_ARRAY_KEYS = ("records", "train", "val", "test")


def records_from_payload(payload: Any, keys: tuple[str, ...] = RECORD_ARRAY_KEYS) -> list[dict[str, Any]]:
    if isinstance(payload, list):
        records = payload
    elif isinstance(payload, dict) and all(key in payload for key in ("article", "aspect")):
        records = [payload]
    elif isinstance(payload, dict):
        records = []
        for key in keys:
            value = payload.get(key)
            if isinstance(value, list):
                records.extend(value)
    else:
        raise ValueError("JSON must contain a record, a record list, or named record arrays.")
    if not records:
        raise ValueError("JSON contains no records in the requested arrays.")
    if any(not isinstance(record, dict) for record in records):
        raise ValueError("Every record must be a JSON object.")
    return list(records)


def load_records(path: str | Path, keys: tuple[str, ...] = RECORD_ARRAY_KEYS) -> list[dict[str, Any]]:
    input_path = Path(path)
    if input_path.suffix.lower() == ".jsonl":
        with input_path.open(encoding="utf-8") as handle:
            payload = [json.loads(line) for line in handle if line.strip()]
    else:
        with input_path.open(encoding="utf-8") as handle:
            payload = json.load(handle)
    return records_from_payload(payload, keys=keys)
