#!/usr/bin/env python3
"""Run calibrated DSPy Gemma programs on complete uncertainty-annotated test sets."""

from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

import dspy

from dspy_gemma_common import (
    DEFAULT_OUTPUT_ROOT,
    UNCERTAINTY_ROOT,
    attach_predictions,
    calculate_metrics,
    configure_openai_lm,
    experiment_dir,
    load_program,
    load_success,
    load_test_records,
    optimized_program_filename,
    prepare_examples,
    run_name,
    run_program_on_examples,
    sanity_check_openai_chat_endpoint,
    stratified_sample_records,
    write_json,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--expert", required=True)
    parser.add_argument("--language", required=True, choices=["slovenian", "serbian"])
    parser.add_argument("--prompt-variant", required=True, choices=["masked", "unmasked"])
    parser.add_argument("--prompt-style", default="current", choices=["current", "legacy", "rich"])
    parser.add_argument("--autorun", required=True, choices=["light", "medium", "heavy"])
    parser.add_argument("--uncertainty-root", default=str(UNCERTAINTY_ROOT))
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--optimized-program", default=None)
    parser.add_argument("--max-article-chars", type=int, default=10000)
    parser.add_argument("--student-model", default="gemma27b")
    parser.add_argument("--student-api-base", default="http://127.0.0.1:8001/v1")
    parser.add_argument("--api-key", default=os.environ.get("OPENAI_API_KEY", "EMPTY"))
    parser.add_argument("--teacher-label", default="qwen-2.5-72b")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--max-tokens", type=int, default=1024)
    parser.add_argument("--num-retries", type=int, default=3)
    parser.add_argument("--miprov2-temp", type=float, default=1.0)
    parser.add_argument("--num-queries", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--limit-items", type=int, default=None)
    parser.add_argument("--sample-items", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--skip-endpoint-check", action="store_true")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    started_at = time.time()
    prompt_masked = args.prompt_variant == "masked"
    output_root = Path(args.output_root)
    uncertainty_root = Path(args.uncertainty_root)
    exp_dir = experiment_dir(output_root, args.expert, args.language, prompt_masked, args.autorun)
    exp_dir.mkdir(parents=True, exist_ok=True)
    base_run_name = run_name(
        args.language,
        prompt_masked,
        args.autorun,
        args.max_tokens,
        "qwen",
    )
    program_path = (
        Path(args.optimized_program)
        if args.optimized_program
        else exp_dir
        / optimized_program_filename(
            args.language,
            prompt_masked,
            args.teacher_label,
            args.autorun,
            args.miprov2_temp,
        )
    )
    predictions_path = exp_dir / f"{base_run_name}_test_predictions.json"
    metrics_path = exp_dir / f"{base_run_name}_test_metrics.json"
    metadata_path = exp_dir / f"{base_run_name}_test_metadata.json"

    if predictions_path.exists() and metrics_path.exists() and not args.force:
        print(f"Skipping existing test outputs: {metrics_path}", flush=True)
        return
    if not program_path.exists():
        raise FileNotFoundError(f"Missing optimized program: {program_path}")

    print("--- DSPy Calibrated Gemma Test Query ---", flush=True)
    print(f"Expert: {args.expert}", flush=True)
    print(f"Language: {args.language}", flush=True)
    print(f"Prompt variant: {args.prompt_variant}", flush=True)
    print(f"Autorun: {args.autorun}", flush=True)
    print(f"Program path: {program_path}", flush=True)
    print(f"Output dir: {exp_dir}", flush=True)

    success = load_success(args.expert, args.language, uncertainty_root)
    records = load_test_records(args.expert, args.language, uncertainty_root)
    if args.sample_items:
        records = stratified_sample_records(records, args.sample_items, args.seed)
    if args.limit_items:
        records = records[: args.limit_items]
    examples = prepare_examples(
        records,
        success,
        args.language,
        prompt_masked,
        args.max_article_chars,
        include_label=True,
        prompt_style=args.prompt_style,
    )
    if not examples:
        raise RuntimeError("Prepared zero test examples.")

    if not args.skip_endpoint_check:
        sanity_check_openai_chat_endpoint(
            args.student_api_base,
            args.api_key,
            args.student_model,
            "student",
        )

    student_lm = configure_openai_lm(
        args.student_model,
        args.student_api_base,
        args.api_key,
        args.temperature,
        args.top_p,
        args.max_tokens,
        args.num_retries,
        cache=False,
    )
    dspy.settings.configure(lm=student_lm)
    program = load_program(program_path, args.prompt_style)
    results = run_program_on_examples(
        program,
        examples,
        records[: len(examples)],
        args.num_queries,
        args.num_workers,
    )
    metrics = calculate_metrics(results)
    metrics.update(
        {
            "phase": "test",
            "run_name": base_run_name,
            "expert": args.expert,
            "language": args.language,
            "prompt_variant": args.prompt_variant,
            "prompt_style": args.prompt_style,
            "autorun": args.autorun,
            "student_model": args.student_model,
            "student_api_base": args.student_api_base,
            "max_tokens": args.max_tokens,
            "num_queries": args.num_queries,
            "num_workers": args.num_workers,
            "optimized_program_path": str(program_path),
            "elapsed_seconds": time.time() - started_at,
        }
    )
    metadata = {
        "expert": args.expert,
        "language": args.language,
        "prompt_variant": args.prompt_variant,
        "prompt_style": args.prompt_style,
        "autorun": args.autorun,
        "student_model": args.student_model,
        "student_api_base": args.student_api_base,
        "teacher_label": args.teacher_label,
        "max_tokens": args.max_tokens,
        "num_queries": args.num_queries,
        "num_workers": args.num_workers,
        "max_article_chars": args.max_article_chars,
        "num_records_loaded": len(records),
        "num_examples": len(examples),
        "optimized_program_path": str(program_path),
        "uncertainty_success": success,
    }
    write_json(predictions_path, attach_predictions(records[: len(examples)], results))
    write_json(metrics_path, metrics)
    write_json(metadata_path, metadata)
    print(f"Test query complete in {(time.time() - started_at) / 60:.1f}m", flush=True)
    print(f"Wrote {metrics_path}", flush=True)


if __name__ == "__main__":
    main()
