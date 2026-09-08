#!/usr/bin/env python3
"""Build aggregate-only local dataset manifests without exposing records or names."""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path


def file_digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def summarize(path: Path) -> dict[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    splits = payload if isinstance(payload, dict) else {"records": payload}
    result = {}
    for name, rows in splits.items():
        if not isinstance(rows, list):
            continue
        labels = Counter(str(row.get("sentiment")) for row in rows)
        result[name] = {
            "records": len(rows),
            "label_counts": dict(sorted(labels.items())),
            "unique_articles": len({str(row.get("article", "")) for row in rows}),
            "unique_aspects": len({str(row.get("aspect", "")) for row in rows}),
        }
    return {"file": path.name, "size_bytes": path.stat().st_size, "sha256": file_digest(path), "splits": result}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    args = parser.parse_args()
    for dataset, pattern in (("hbs", "hbs_*.json"), ("sl", "slovene_*.json")):
        folder = args.data_root / dataset
        files = [path for path in sorted(folder.glob(pattern)) if path.name != "statistics.json"]
        if not files:
            raise FileNotFoundError(f"No dataset files found under {folder}")
        report = {
            "schema_version": 1,
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "dataset": dataset,
            "aggregate_only": True,
            "contains_records_or_aspect_names": False,
            "files": [summarize(path) for path in files],
        }
        output = folder / "statistics.json"
        temporary = output.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(report, indent=2) + "\n")
        os.replace(temporary, output)
        print(output.resolve())


if __name__ == "__main__":
    main()
