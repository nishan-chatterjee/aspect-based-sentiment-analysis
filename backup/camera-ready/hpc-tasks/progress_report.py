#!/usr/bin/env python3
"""Small progress table for ABSA additional comparison baseline runs."""

import argparse
import json
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", default="reviews")
    return parser.parse_args()


def read_json(path):
    try:
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def main():
    args = parse_args()
    root = Path(args.output_root)
    metric_files = sorted(root.glob("*/*/*/training_metrics_*.json"))
    if not metric_files:
        print("No training_metrics_*.json files found under %s" % root)
        return

    rows = []
    for path in metric_files:
        data = read_json(path)
        if not data:
            continue
        evals = data.get("eval_metrics", [])
        last_eval = evals[-1] if evals else {}
        best_f1 = max((row.get("f1_macro", 0.0) for row in evals), default=0.0)
        parts = path.parts
        rows.append(
            {
                "approach": parts[-4],
                "variant": parts[-3],
                "language": parts[-2],
                "run": str(data.get("run_index", path.stem.rsplit("_", 1)[-1])),
                "epochs": str(len(evals)),
                "last_f1": "%.4f" % last_eval.get("f1_macro", 0.0),
                "best_f1": "%.4f" % best_f1,
                "last_qwk": "%.4f" % last_eval.get("qwk", 0.0),
            }
        )

    header = ["approach", "variant", "language", "run", "epochs", "last_f1", "best_f1", "last_qwk"]
    widths = {col: max(len(col), *(len(row[col]) for row in rows)) for col in header}
    print("  ".join(col.ljust(widths[col]) for col in header))
    print("  ".join("-" * widths[col] for col in header))
    for row in rows:
        print("  ".join(row[col].ljust(widths[col]) for col in header))


if __name__ == "__main__":
    main()
