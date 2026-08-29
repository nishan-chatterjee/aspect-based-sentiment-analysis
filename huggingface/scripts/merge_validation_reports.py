#!/usr/bin/env python3
"""Merge per-family validation reports produced by interactive GPU workers."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-reports", type=int, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    paths = sorted(args.input_dir.glob("*.json"))
    if len(paths) != args.expected_reports:
        raise SystemExit(
            f"Expected {args.expected_reports} shard reports, found {len(paths)} "
            f"in {args.input_dir}."
        )

    merged: dict[str, Any] = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "execution": "interactive_multi_gpu",
        "shard_reports": [str(path) for path in paths],
        "matrix_slots": 0,
        "passed": 0,
        "failed": 0,
        "skipped": 0,
        "unavailable": 0,
        "results": [],
    }
    missing_model_languages: set[tuple[str, str]] = set()
    for path in paths:
        payload = json.loads(path.read_text(encoding="utf-8"))
        for key in ("matrix_slots", "passed", "failed", "skipped", "unavailable"):
            merged[key] += int(payload.get(key, 0))
        merged["results"].extend(payload.get("results", []))
        for result in payload.get("results", []):
            if result.get("status") == "skipped":
                missing_model_languages.add((result["model"], result["language"]))

    merged["missing_model_language_combinations"] = len(missing_model_languages)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(merged, indent=2) + "\n", encoding="utf-8")
    print(
        f"MERGED REPORT: passed={merged['passed']} failed={merged['failed']} "
        f"skipped_slots={merged['skipped']} "
        f"missing_model_languages={merged['missing_model_language_combinations']} "
        f"path={args.output}",
        flush=True,
    )
    if merged["failed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
