#!/usr/bin/env python3
"""Select the best validation split and activate it after a training grid."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from aspectbench.registry import select_models  # noqa: E402
from aspectbench.training.runner import _activate_checkpoint  # noqa: E402


def atomic_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
    os.replace(temporary, path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, choices=("hbs", "sl"))
    parser.add_argument("--models", nargs="+", required=True)
    parser.add_argument("--variants", nargs="+", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--output-model-root", type=Path, default=Path("models"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    selections = []
    for spec in select_models(args.models, language=args.dataset):
        for variant in args.variants:
            reports = []
            pattern = (
                args.output_model_root
                / spec.name
                / args.dataset
                / variant
                / args.run_id
                / spec.name
                / variant
            )
            for path in sorted(pattern.glob("split-*/training-report.json")):
                report = json.loads(path.read_text())
                if report.get("status") == "complete":
                    reports.append((path, report))
            if not reports:
                selections.append({"model": spec.name, "variant": variant, "status": "missing"})
                continue
            report_path, best = max(reports, key=lambda item: item[1]["best_validation_macro_f1"])
            checkpoint = Path(best["best_checkpoint"])
            active = _activate_checkpoint(
                checkpoint,
                output_model_root=args.output_model_root,
                model=spec.name,
                language=args.dataset,
                variant=variant,
            )
            selections.append(
                {
                    "model": spec.name,
                    "variant": variant,
                    "status": "selected",
                    "split": int(report_path.parent.name.split("-")[-1]),
                    "validation_macro_f1": best["best_validation_macro_f1"],
                    "checkpoint": str(checkpoint.resolve()),
                    "active_checkpoint": str(active.resolve()),
                }
            )
    atomic_json(args.output, {"schema_version": 1, "run_id": args.run_id, "selections": selections})
    print(args.output.resolve())
    if any(row["status"] != "selected" for row in selections):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
