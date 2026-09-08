#!/usr/bin/env python3
"""Verify all public-program smoke outputs and record explicit unavailable slots."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path


MODELS = {
    "hbs": ("xlmr", "han-xlmr", "longformer", "mdeberta-v3", "mt5", "bertic", "bge-m3-mlp"),
    "sl": ("xlmr", "han-xlmr", "longformer", "mdeberta-v3", "mt5", "sloberta", "bge-m3-mlp"),
}
BEST = {
    "hbs": {"xlmr": "masked", "han-xlmr": "masked"},
    "sl": {"mt5": "unmasked", "sloberta": "unmasked"},
}


def best_variant(dataset: str, model: str) -> str:
    return BEST.get(dataset, {}).get(model, "unmasked" if dataset == "hbs" else "masked")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--run-root", type=Path, default=Path("models/_runs"))
    parser.add_argument("--program-root", type=Path, default=Path("selective-deferral-programs"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rows = []
    for dataset, models in MODELS.items():
        for model in models:
            plm_variant = best_variant(dataset, model)
            for prompt_variant in ("masked", "unmasked"):
                program = args.program_root / "precalibrated" / model / dataset / prompt_variant / "program.json"
                row = {"dataset": dataset, "model": model, "plm_variant": plm_variant, "prompt_variant": prompt_variant}
                if not program.is_file():
                    rows.append({**row, "status": "skipped", "reason": "no public precalibrated program"})
                    continue
                run = f"{args.run_id}-{dataset}-{model}-{plm_variant}-{prompt_variant}"
                prediction_path = args.run_root / "dspy-inference" / run / "predictions.json"
                if not prediction_path.is_file():
                    rows.append({**row, "status": "failed", "reason": "predictions.json missing", "run_id": run})
                    continue
                predictions = json.loads(prediction_path.read_text())
                valid = bool(predictions) and all(
                    item.get("deferred") is True
                    and item.get("dspy_status") == "complete"
                    and item.get("action") in {"keep_plm", "override", "abstain_uncertain"}
                    and item.get("prediction") in {-1, 0, 1}
                    for item in predictions
                )
                rows.append({**row, "status": "passed" if valid else "failed", "run_id": run, "records": len(predictions)})
    counts = {status: sum(row["status"] == status for row in rows) for status in ("passed", "failed", "skipped")}
    payload = {"schema_version": 1, "run_id": args.run_id, "counts": counts, "rows": rows}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n")
    os.replace(temporary, args.output)
    print(json.dumps(counts))
    if counts["failed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
