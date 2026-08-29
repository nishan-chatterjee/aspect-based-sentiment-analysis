#!/usr/bin/env python3
"""Validate every model/language/mode slot with single and batched inference."""

from __future__ import annotations

import argparse
import gc
import json
import math
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch

from inference import InferenceEngine, load_records
from model_registry import CHECKPOINTS, LANGUAGES, MODES, MODEL_SPECS, weight_path


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_ROOT = SCRIPT_DIR.parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-root", type=Path, default=DEFAULT_ROOT / "models")
    parser.add_argument("--base-model-root", type=Path)
    parser.add_argument("--examples-root", type=Path, default=DEFAULT_ROOT / "examples")
    parser.add_argument("--output", type=Path, default=DEFAULT_ROOT / "validation-report.json")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=10)
    parser.add_argument("--mc-passes", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--model", action="append", choices=sorted(MODEL_SPECS))
    parser.add_argument(
        "--require-complete-matrix",
        action="store_true",
        help="Fail for documented missing checkpoint slots as well as inference failures.",
    )
    return parser.parse_args()


def validate_prediction(row: dict[str, Any], expected: dict[str, str]) -> None:
    for key, value in expected.items():
        if row.get(key) != value:
            raise AssertionError(f"{key}: expected {value!r}, got {row.get(key)!r}")
    if row["predicted_sentiment"] not in (-1, 0, 1):
        raise AssertionError("Prediction is outside {-1, 0, 1}.")
    probabilities = row["class_probabilities"]
    if set(probabilities) != {"-1 (negative)", "0 (neutral)", "1 (positive)"}:
        raise AssertionError("Probability labels are incorrect.")
    if not math.isclose(sum(probabilities.values()), 1.0, rel_tol=0, abs_tol=1e-5):
        raise AssertionError("Class probabilities do not sum to 1.")
    uncertainty = row["uncertainty_across_classes"]
    if not 0.0 <= uncertainty["confidence"] <= 1.0:
        raise AssertionError("Confidence is outside [0, 1].")
    if expected["mode"] == "masked" and row["aspect_used"] != "[ASPECT]":
        raise AssertionError("Masked inference exposed the aspect text to the model.")


def main() -> None:
    args = parse_args()
    if args.batch_size < 2:
        raise ValueError("--batch-size must be at least 2 to exercise batched inference.")
    selected = set(args.model or MODEL_SPECS)
    examples = {
        "hbs": load_records(args.examples_root / "hbs-tagged-examples.json"),
        "slovenian": load_records(args.examples_root / "sl-tagged-synthetic-examples.json"),
    }
    report: dict[str, Any] = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "matrix_slots": 0,
        "passed": 0,
        "failed": 0,
        "skipped": 0,
        "unavailable": 0,
        "results": [],
    }
    missing_model_languages: set[tuple[str, str]] = set()
    for model_name in MODEL_SPECS:
        if model_name not in selected:
            continue
        for language in LANGUAGES:
            for mode in MODES:
                report["matrix_slots"] += 1
                selection = CHECKPOINTS[(model_name, language, mode)]
                result: dict[str, Any] = {
                    "model": model_name,
                    "language": language,
                    "mode": mode,
                    "weight": str(weight_path(args.model_root, model_name, language, mode)),
                }
                label = f"{model_name}/{language}/{mode}"
                if not selection["available"]:
                    result.update(
                        status="skipped", reason=selection["unavailable_reason"]
                    )
                    report["skipped"] += 1
                    report["unavailable"] += 1
                    missing_model_languages.add((model_name, language))
                    report["results"].append(result)
                    print(f"SKIP {label}: {result['reason']}", flush=True)
                    continue
                stage = "model loading"
                try:
                    engine = InferenceEngine(
                        model_name=model_name,
                        language=language,
                        mode=mode,
                        model_root=args.model_root,
                        base_model_root=args.base_model_root,
                        device=args.device,
                    )
                    expected = {"model": model_name, "language": language, "mode": mode}
                    stage = "single inference"
                    single = engine.predict(
                        examples[language][0],
                        mc_passes=args.mc_passes,
                        seed=args.seed,
                    )
                    validate_prediction(single, expected)
                    stage = "batched inference"
                    batch_input = examples[language][: args.batch_size]
                    batch = engine.predict_batch(
                        batch_input,
                        batch_size=args.batch_size,
                        mc_passes=args.mc_passes,
                        seed=args.seed,
                    )
                    if len(batch) != len(batch_input):
                        raise AssertionError("Batch output length does not match input length.")
                    for row in batch:
                        validate_prediction(row, expected)
                    result.update(
                        status="passed",
                        single_status="completed",
                        batch_status="completed",
                        single_prediction=single["predicted_sentiment"],
                        batch_predictions=[row["predicted_sentiment"] for row in batch],
                        single_output=single,
                        batch_outputs=batch,
                    )
                    report["passed"] += 1
                    print(
                        f"PASS {label}: single=completed, batch=completed",
                        flush=True,
                    )
                    del engine
                except Exception as exc:
                    result.update(
                        status="failed",
                        failure_stage=stage,
                        error=f"{type(exc).__name__}: {exc}",
                        traceback=traceback.format_exc(),
                    )
                    report["failed"] += 1
                    print(f"FAIL {label} during {stage}: {result['error']}", flush=True)
                report["results"].append(result)
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    report["missing_model_language_combinations"] = len(missing_model_languages)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(
        f"Summary: {report['passed']} passed, {report['failed']} failed, "
        f"{report['skipped']} skipped slots across "
        f"{report['missing_model_language_combinations']} model-language combinations; "
        f"report={args.output}",
        flush=True,
    )
    if report["failed"] or (args.require_complete_matrix and report["unavailable"]):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
