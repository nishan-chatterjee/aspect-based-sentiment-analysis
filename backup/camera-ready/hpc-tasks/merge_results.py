#!/usr/bin/env python3
"""Rebuild notebook-compatible test_metrics_summary.json files after HPC arrays.

Parallel single-run jobs safely write disjoint files such as best_model_0.pt,
training_metrics_0.json, and test_predictions_0.json. The old per-process
test_metrics_summary.json files may race, so this script recomputes the final
summary from predictions after all array tasks finish.
"""

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    cohen_kappa_score,
    precision_recall_fscore_support,
)


APPROACHES = ("longformer", "mdeberta", "mt5", "slavic_specific")
LANGUAGES = ("slovenian", "serbian")
VARIANTS = ("unmasked", "masked")
RUN_INDICES = (0, 1, 2)
FIXED_ORIGINAL_LABELS = [-1, 0, 1]
FIXED_MAPPED_LABELS = [0, 1, 2]
TARGET_NAMES = ["Negative (0)", "Neutral (1)", "Positive (2)"]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", default="reviews")
    parser.add_argument(
        "--approaches",
        nargs="+",
        default=list(APPROACHES),
        choices=list(APPROACHES),
    )
    parser.add_argument("--strict", action="store_true")
    return parser.parse_args()


def map_label(label):
    return int(label) + 1


def metrics_from_predictions(prediction_path):
    with prediction_path.open("r", encoding="utf-8") as f:
        rows = json.load(f)

    y_true_original = [int(row["sentiment"]) for row in rows]
    y_pred_original = [int(row["prediction"]) for row in rows]
    y_true = [map_label(v) for v in y_true_original]
    y_pred = [map_label(v) for v in y_pred_original]

    precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=FIXED_MAPPED_LABELS,
        average="macro",
        zero_division=0,
    )
    precision_micro, recall_micro, f1_micro, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=FIXED_MAPPED_LABELS,
        average="micro",
        zero_division=0,
    )
    precision_weighted, recall_weighted, f1_weighted, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=FIXED_MAPPED_LABELS,
        average="weighted",
        zero_division=0,
    )
    report_dict = classification_report(
        y_true,
        y_pred,
        labels=FIXED_MAPPED_LABELS,
        target_names=TARGET_NAMES,
        zero_division=0,
        output_dict=True,
    )
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision_macro": precision_macro,
        "recall_macro": recall_macro,
        "f1_macro": f1_macro,
        "precision_micro": precision_micro,
        "recall_micro": recall_micro,
        "f1_micro": f1_micro,
        "precision_weighted": precision_weighted,
        "recall_weighted": recall_weighted,
        "f1_weighted": f1_weighted,
        "qwk": cohen_kappa_score(
            y_true, y_pred, labels=FIXED_MAPPED_LABELS, weights="quadratic"
        ),
        "per_class_report": report_dict,
    }


def average_performance(run_metrics):
    if not run_metrics:
        return {"error": "No model predictions found."}

    def vals(key):
        return [m[key] for m in run_metrics if m.get(key) is not None]

    f1 = vals("f1_macro")
    acc = vals("accuracy")
    qwk = vals("qwk")
    return {
        "f1_macro_mean": float(np.mean(f1)) if f1 else 0.0,
        "f1_macro_std": float(np.std(f1)) if len(f1) > 1 else 0.0,
        "num_models_f1": len(f1),
        "accuracy_mean": float(np.mean(acc)) if acc else 0.0,
        "accuracy_std": float(np.std(acc)) if len(acc) > 1 else 0.0,
        "num_models_accuracy": len(acc),
        "qwk_mean": float(np.mean(qwk)) if qwk else 0.0,
        "qwk_std": float(np.std(qwk)) if len(qwk) > 1 else 0.0,
        "num_models_qwk": len(qwk),
    }


def main():
    args = parse_args()
    output_root = Path(args.output_root)
    missing = []

    for approach in args.approaches:
        for variant in VARIANTS:
            for language in LANGUAGES:
                result_dir = output_root / approach / variant / language
                result_dir.mkdir(parents=True, exist_ok=True)
                summary = {}
                successful = []

                for run_index in RUN_INDICES:
                    prediction_path = result_dir / ("test_predictions_%s.json" % run_index)
                    model_path = result_dir / ("best_model_%s.pt" % run_index)
                    key = "model_%s" % run_index

                    if not prediction_path.exists():
                        message = "Missing %s" % prediction_path
                        missing.append(message)
                        summary[key] = {"error": message}
                        continue

                    metrics = metrics_from_predictions(prediction_path)
                    run_result = {
                        "model_run_index": run_index,
                        "model_path": str(model_path),
                        "test_loss": None,
                        **metrics,
                    }
                    summary[key] = run_result
                    successful.append(metrics)

                summary["average_performance"] = average_performance(successful)
                summary_path = result_dir / "test_metrics_summary.json"
                with summary_path.open("w", encoding="utf-8") as f:
                    json.dump(summary, f, indent=4, ensure_ascii=False)
                print("Wrote %s" % summary_path)

    if missing:
        print("\nMissing prediction files:")
        for item in missing:
            print("  - %s" % item)
        if args.strict:
            raise SystemExit(1)


if __name__ == "__main__":
    main()
