#!/usr/bin/env python3
"""Select the best calibrated DSPy setting per expert/language."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


TASKS = [
    ("han_xlmr_masked", "slovenian"),
    ("longformer_masked", "slovenian"),
    ("slavic_specific_masked", "slovenian"),
    ("longformer_masked", "serbian"),
    ("mdeberta_masked", "serbian"),
    ("slavic_specific_masked", "serbian"),
]


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def metric_path(
    output_root: Path,
    expert: str,
    language: str,
    prompt_variant: str,
    autorun: str,
    max_tokens: int,
) -> Path:
    return (
        output_root
        / expert
        / language
        / prompt_variant
        / autorun
        / (
            f"dspy-plm-augmented-cot-teacher-qwen-{max_tokens}-{autorun}-"
            f"uncertainty-{language}-{prompt_variant}_calibration_metrics.json"
        )
    )


def program_path(
    output_root: Path,
    expert: str,
    language: str,
    prompt_variant: str,
    autorun: str,
    teacher_label: str,
    miprov2_temp: float,
) -> Path:
    return (
        output_root
        / expert
        / language
        / prompt_variant
        / autorun
        / (
            f"optimized_program_{language}_dspy-plm-augmented-cot-with-uncertainty_"
            f"{prompt_variant}_teacher_{teacher_label}_autorun_{autorun}_temp_{miprov2_temp}.json"
        )
    )


def score_row(row: dict[str, Any]) -> tuple[float, float, float, int, int]:
    autorun_rank = {"heavy": 1, "medium": 0, "light": -1}
    prompt_rank = {"masked": 1, "unmasked": 0}
    return (
        float(row.get("f1_macro", float("-inf"))),
        float(row.get("qwk", float("-inf"))),
        float(row.get("accuracy", float("-inf"))),
        autorun_rank.get(str(row.get("autorun")), -1),
        prompt_rank.get(str(row.get("prompt_variant")), -1),
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", default="reviews/uncertainty/llm-dspy-calibration-cot")
    parser.add_argument("--max-tokens", type=int, default=1024)
    parser.add_argument("--teacher-label", default="qwen-2.5-72b")
    parser.add_argument("--miprov2-temp", type=float, default=1.0)
    parser.add_argument("--autoruns", nargs="+", default=["medium", "heavy"])
    parser.add_argument("--prompt-variants", nargs="+", default=["unmasked", "masked"])
    parser.add_argument("--selection-json", default=None)
    parser.add_argument("--selection-tsv", default=None)
    parser.add_argument("--allow-missing", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_root = Path(args.output_root)
    selection_json = Path(args.selection_json) if args.selection_json else output_root / "best_calibrations.json"
    selection_tsv = Path(args.selection_tsv) if args.selection_tsv else output_root / "best_calibrations.tsv"

    selected = []
    missing = []
    for expert, language in TASKS:
        candidates = []
        for prompt_variant in args.prompt_variants:
            for autorun in args.autoruns:
                metrics_file = metric_path(
                    output_root,
                    expert,
                    language,
                    prompt_variant,
                    autorun,
                    args.max_tokens,
                )
                program_file = program_path(
                    output_root,
                    expert,
                    language,
                    prompt_variant,
                    autorun,
                    args.teacher_label,
                    args.miprov2_temp,
                )
                if not metrics_file.exists() or not program_file.exists():
                    missing.append(
                        {
                            "expert": expert,
                            "language": language,
                            "prompt_variant": prompt_variant,
                            "autorun": autorun,
                            "metrics_path": str(metrics_file),
                            "program_path": str(program_file),
                        }
                    )
                    continue
                metrics = load_json(metrics_file)
                candidates.append(
                    {
                        "expert": expert,
                        "language": language,
                        "prompt_variant": prompt_variant,
                        "autorun": autorun,
                        "f1_macro": metrics.get("f1_macro"),
                        "qwk": metrics.get("qwk"),
                        "accuracy": metrics.get("accuracy"),
                        "num_samples_evaluated": metrics.get("num_samples_evaluated"),
                        "metrics_path": str(metrics_file),
                        "program_path": str(program_file),
                    }
                )
        if not candidates:
            if args.allow_missing:
                print(
                    f"Skipping {expert}/{language}: no completed calibration candidates at max_tokens={args.max_tokens}.",
                    flush=True,
                )
                continue
            raise FileNotFoundError(
                f"No completed calibration candidates found for {expert}/{language} at max_tokens={args.max_tokens}."
            )
        candidates.sort(key=score_row, reverse=True)
        best = dict(candidates[0])
        best["ranked_candidates"] = candidates
        selected.append(best)

    if missing and not args.allow_missing:
        print(f"Warning: {len(missing)} calibration candidates are missing.", flush=True)
        for item in missing[:12]:
            print(
                "  missing {expert}/{language}/{prompt_variant}/{autorun}".format(**item),
                flush=True,
            )
        if len(missing) > 12:
            print(f"  ... and {len(missing) - 12} more", flush=True)

    write_json(selection_json, {"selected": selected, "missing": missing})
    selection_tsv.parent.mkdir(parents=True, exist_ok=True)
    with selection_tsv.open("w", encoding="utf-8") as f:
        f.write("expert\tlanguage\tprompt_variant\tautorun\tf1_macro\tqwk\taccuracy\tnum_samples_evaluated\n")
        for row in selected:
            f.write(
                "{expert}\t{language}\t{prompt_variant}\t{autorun}\t{f1_macro}\t{qwk}\t{accuracy}\t{num_samples_evaluated}\n".format(
                    **row
                )
            )

    print(f"Wrote {selection_json}", flush=True)
    print(f"Wrote {selection_tsv}", flush=True)
    for row in selected:
        print(
            "{expert}/{language}: {prompt_variant} {autorun} "
            "f1={f1_macro} qwk={qwk} acc={accuracy}".format(**row),
            flush=True,
        )


if __name__ == "__main__":
    main()
