#!/usr/bin/env python3
"""Optimize DSPy prompts for Gemma using Qwen as MIPROv2 teacher."""

from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

from dspy_gemma_common import (
    DEFAULT_OUTPUT_ROOT,
    UNCERTAINTY_ROOT,
    attach_predictions,
    calculate_metrics,
    configure_openai_lm,
    default_sample_size,
    experiment_dir,
    load_program,
    load_success,
    load_train_val_records,
    optimize_program,
    optimized_program_filename,
    prepare_examples,
    run_name,
    run_program_on_examples,
    sanity_check_openai_chat_endpoint,
    stratified_sample_records,
    variant_name,
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
    parser.add_argument("--split-index", type=int, default=0)
    parser.add_argument("--train-size", type=int, default=None)
    parser.add_argument("--val-size", type=int, default=None)
    parser.add_argument("--max-article-chars", type=int, default=10000)
    parser.add_argument("--student-model", default="gemma27b")
    parser.add_argument("--teacher-model", default="qwen72b")
    parser.add_argument("--student-api-base", default="http://127.0.0.1:8001/v1")
    parser.add_argument("--teacher-api-base", default="http://127.0.0.1:8000/v1")
    parser.add_argument("--api-key", default=os.environ.get("OPENAI_API_KEY", "EMPTY"))
    parser.add_argument("--teacher-label", default="qwen-2.5-72b")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--max-tokens", type=int, default=1024)
    parser.add_argument("--student-max-tokens", type=int, default=None)
    parser.add_argument("--teacher-max-tokens", type=int, default=None)
    parser.add_argument("--num-retries", type=int, default=3)
    parser.add_argument("--miprov2-temp", type=float, default=1.0)
    parser.add_argument("--dspy-num-threads", type=int, default=8)
    parser.add_argument("--max-errors", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-queries", type=int, default=1)
    parser.add_argument("--eval-workers", type=int, default=4)
    parser.add_argument("--disable-program-aware-proposer", action="store_true")
    parser.add_argument("--disable-data-aware-proposer", action="store_true")
    parser.add_argument("--disable-tip-aware-proposer", action="store_true")
    parser.add_argument("--disable-fewshot-aware-proposer", action="store_true")
    parser.add_argument("--view-data-batch-size", type=int, default=3)
    parser.add_argument("--skip-calibration-eval", action="store_true")
    parser.add_argument("--skip-endpoint-check", action="store_true")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    started_at = time.time()
    prompt_masked = args.prompt_variant == "masked"
    student_max_tokens = args.student_max_tokens or args.max_tokens
    teacher_max_tokens = args.teacher_max_tokens or args.max_tokens
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
    program_path = exp_dir / optimized_program_filename(
        args.language,
        prompt_masked,
        args.teacher_label,
        args.autorun,
        args.miprov2_temp,
    )
    metadata_path = exp_dir / f"{base_run_name}_calibration_metadata.json"
    predictions_path = exp_dir / f"{base_run_name}_calibration_predictions.json"
    metrics_path = exp_dir / f"{base_run_name}_calibration_metrics.json"

    print("--- DSPy Gemma Calibration ---", flush=True)
    print(f"Expert: {args.expert}", flush=True)
    print(f"Language: {args.language}", flush=True)
    print(f"Prompt variant: {args.prompt_variant}", flush=True)
    print(f"Autorun: {args.autorun}", flush=True)
    print(f"Output dir: {exp_dir}", flush=True)
    print(f"Program path: {program_path}", flush=True)

    if (
        program_path.exists()
        and metrics_path.exists()
        and predictions_path.exists()
        and not args.force
        and not args.skip_calibration_eval
    ):
        print(f"Skipping existing calibration outputs: {metrics_path}", flush=True)
        return

    success = load_success(args.expert, args.language, uncertainty_root)
    train_records, val_records = load_train_val_records(
        args.expert,
        args.language,
        uncertainty_root,
        split_index=args.split_index,
    )
    sample_default = default_sample_size(args.autorun)
    train_size = args.train_size or sample_default
    val_size = args.val_size or sample_default
    train_sample = stratified_sample_records(train_records, train_size, args.seed)
    val_sample = stratified_sample_records(val_records, val_size, args.seed + 1)
    trainset = prepare_examples(
        train_sample,
        success,
        args.language,
        prompt_masked,
        args.max_article_chars,
        include_label=True,
        prompt_style=args.prompt_style,
    )
    valset = prepare_examples(
        val_sample,
        success,
        args.language,
        prompt_masked,
        args.max_article_chars,
        include_label=True,
        prompt_style=args.prompt_style,
    )
    if not trainset or not valset:
        raise RuntimeError("Prepared empty DSPy trainset or valset.")

    if not args.skip_endpoint_check:
        sanity_check_openai_chat_endpoint(
            args.student_api_base,
            args.api_key,
            args.student_model,
            "student",
        )
        sanity_check_openai_chat_endpoint(
            args.teacher_api_base,
            args.api_key,
            args.teacher_model,
            "teacher",
        )

    student_lm = configure_openai_lm(
        args.student_model,
        args.student_api_base,
        args.api_key,
        args.temperature,
        args.top_p,
        student_max_tokens,
        args.num_retries,
        cache=False,
    )
    teacher_lm = configure_openai_lm(
        args.teacher_model,
        args.teacher_api_base,
        args.api_key,
        args.temperature,
        args.top_p,
        teacher_max_tokens,
        args.num_retries,
        cache=False,
    )

    if program_path.exists() and not args.force:
        print(f"Loading existing optimized program: {program_path}", flush=True)
        optimized_program = load_program(program_path, args.prompt_style)
    else:
        optimized_program = optimize_program(
            trainset,
            valset,
            program_path,
            student_lm,
            teacher_lm,
            args.autorun,
            args.miprov2_temp,
            args.dspy_num_threads,
            args.max_errors,
            args.seed,
            program_aware_proposer=not args.disable_program_aware_proposer,
            data_aware_proposer=not args.disable_data_aware_proposer,
            tip_aware_proposer=not args.disable_tip_aware_proposer,
            fewshot_aware_proposer=not args.disable_fewshot_aware_proposer,
            view_data_batch_size=args.view_data_batch_size,
            prompt_style=args.prompt_style,
        )

    metadata = {
        "expert": args.expert,
        "language": args.language,
        "prompt_variant": args.prompt_variant,
        "prompt_style": args.prompt_style,
        "autorun": args.autorun,
        "student_model": args.student_model,
        "teacher_model": args.teacher_model,
        "student_api_base": args.student_api_base,
        "teacher_api_base": args.teacher_api_base,
        "teacher_label": args.teacher_label,
        "max_tokens": args.max_tokens,
        "student_max_tokens": student_max_tokens,
        "teacher_max_tokens": teacher_max_tokens,
        "miprov2_temp": args.miprov2_temp,
        "dspy_num_threads": args.dspy_num_threads,
        "max_errors": args.max_errors,
        "seed": args.seed,
        "program_aware_proposer": not args.disable_program_aware_proposer,
        "data_aware_proposer": not args.disable_data_aware_proposer,
        "tip_aware_proposer": not args.disable_tip_aware_proposer,
        "fewshot_aware_proposer": not args.disable_fewshot_aware_proposer,
        "view_data_batch_size": args.view_data_batch_size,
        "split_index": args.split_index,
        "train_size_requested": train_size,
        "val_size_requested": val_size,
        "train_examples": len(trainset),
        "val_examples": len(valset),
        "max_article_chars": args.max_article_chars,
        "optimized_program_path": str(program_path),
        "uncertainty_success": success,
    }
    write_json(metadata_path, metadata)

    if not args.skip_calibration_eval:
        print("Running calibration validation predictions...", flush=True)
        results = run_program_on_examples(
            optimized_program,
            valset,
            val_sample[: len(valset)],
            args.num_queries,
            args.eval_workers,
        )
        metrics = calculate_metrics(results)
        metrics.update(
            {
                "phase": "calibration_validation",
                "run_name": base_run_name,
                "metadata_path": str(metadata_path),
                "optimized_program_path": str(program_path),
                "elapsed_seconds": time.time() - started_at,
            }
        )
        write_json(predictions_path, attach_predictions(val_sample[: len(valset)], results))
        write_json(metrics_path, metrics)

    print(f"Calibration complete in {(time.time() - started_at) / 60:.1f}m", flush=True)


if __name__ == "__main__":
    main()
