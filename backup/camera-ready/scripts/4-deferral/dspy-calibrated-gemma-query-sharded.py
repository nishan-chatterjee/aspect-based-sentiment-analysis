#!/usr/bin/env python3
"""Run one calibrated DSPy Gemma test query task sharded across multiple endpoints."""

from __future__ import annotations

import argparse
import concurrent.futures as futures
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
    parser.add_argument("--student-api-bases", required=True)
    parser.add_argument("--api-key", default=os.environ.get("OPENAI_API_KEY", "EMPTY"))
    parser.add_argument("--teacher-label", default="qwen-2.5-72b")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--max-tokens", type=int, default=1024)
    parser.add_argument("--num-retries", type=int, default=3)
    parser.add_argument("--miprov2-temp", type=float, default=1.0)
    parser.add_argument("--num-queries", type=int, default=1)
    parser.add_argument("--num-workers-per-endpoint", type=int, default=6)
    parser.add_argument("--limit-items", type=int, default=None)
    parser.add_argument("--sample-items", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--skip-endpoint-check", action="store_true")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def shard_items(items: list, num_shards: int, shard_index: int) -> list:
    return [item for index, item in enumerate(items) if index % num_shards == shard_index]


def run_shard(
    args: argparse.Namespace,
    shard_index: int,
    api_base: str,
    records: list[dict],
    success: dict,
    program_path: Path,
    exp_dir: Path,
    base_run_name: str,
) -> list[dict]:
    started_at = time.time()
    prompt_masked = args.prompt_variant == "masked"
    shard_records = shard_items(records, len(args.api_bases), shard_index)
    if not shard_records:
        return []

    shard_name = f"{base_run_name}_shard-{shard_index:02d}-of-{len(args.api_bases):02d}"
    shard_predictions_path = exp_dir / f"{shard_name}_test_predictions.json"
    shard_metrics_path = exp_dir / f"{shard_name}_test_metrics.json"
    shard_metadata_path = exp_dir / f"{shard_name}_test_metadata.json"

    if shard_predictions_path.exists() and shard_metrics_path.exists() and not args.force:
        print(f"[shard {shard_index}] loading existing shard outputs: {shard_metrics_path}", flush=True)
        # Reconstruct minimal result rows from saved predictions for merging.
        from dspy_gemma_common import load_json

        saved = load_json(shard_predictions_path)
        results = []
        for item in saved:
            results.append(
                {
                    "uuid": item.get("uuid"),
                    "ground_truth_int": int(item["sentiment"]) if item.get("sentiment") is not None else None,
                    "prediction_int": item.get("prediction"),
                    "prediction_label": item.get("prediction_label"),
                    "status": item.get("processing_status"),
                    "dspy_query_details": item.get("dspy_query_details", []),
                }
            )
        return results

    if not args.skip_endpoint_check:
        sanity_check_openai_chat_endpoint(api_base, args.api_key, args.student_model, f"student shard {shard_index}")

    examples = prepare_examples(
        shard_records,
        success,
        args.language,
        prompt_masked,
        args.max_article_chars,
        include_label=True,
        prompt_style=args.prompt_style,
    )
    if not examples:
        return []

    print(
        f"[shard {shard_index}] endpoint={api_base} records={len(shard_records)} examples={len(examples)}",
        flush=True,
    )
    student_lm = configure_openai_lm(
        args.student_model,
        api_base,
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
        shard_records[: len(examples)],
        args.num_queries,
        args.num_workers_per_endpoint,
        progress_label=f"shard {shard_index}",
    )
    shard_metrics = calculate_metrics(results)
    shard_metrics.update(
        {
            "phase": "test_shard",
            "run_name": shard_name,
            "expert": args.expert,
            "language": args.language,
            "prompt_variant": args.prompt_variant,
            "prompt_style": args.prompt_style,
            "autorun": args.autorun,
            "student_model": args.student_model,
            "student_api_base": api_base,
            "shard_index": shard_index,
            "num_shards": len(args.api_bases),
            "elapsed_seconds": time.time() - started_at,
        }
    )
    write_json(shard_predictions_path, attach_predictions(shard_records[: len(examples)], results))
    write_json(shard_metrics_path, shard_metrics)
    write_json(
        shard_metadata_path,
        {
            "expert": args.expert,
            "language": args.language,
            "prompt_variant": args.prompt_variant,
            "prompt_style": args.prompt_style,
            "autorun": args.autorun,
            "student_api_base": api_base,
            "num_records_loaded": len(records),
            "num_shard_records": len(shard_records),
            "num_examples": len(examples),
            "optimized_program_path": str(program_path),
        },
    )
    print(f"[shard {shard_index}] complete in {(time.time() - started_at) / 60:.1f}m", flush=True)
    return results


def main() -> None:
    args = parse_args()
    args.api_bases = [item.strip() for item in args.student_api_bases.split(",") if item.strip()]
    if not args.api_bases:
        raise RuntimeError("--student-api-bases is empty")

    started_at = time.time()
    prompt_masked = args.prompt_variant == "masked"
    output_root = Path(args.output_root)
    uncertainty_root = Path(args.uncertainty_root)
    exp_dir = experiment_dir(output_root, args.expert, args.language, prompt_masked, args.autorun)
    exp_dir.mkdir(parents=True, exist_ok=True)
    base_run_name = run_name(args.language, prompt_masked, args.autorun, args.max_tokens, "qwen")
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
        print(f"Skipping existing merged outputs: {metrics_path}", flush=True)
        return
    if not program_path.exists():
        raise FileNotFoundError(f"Missing optimized program: {program_path}")

    success = load_success(args.expert, args.language, uncertainty_root)
    records = load_test_records(args.expert, args.language, uncertainty_root)
    if args.sample_items:
        records = stratified_sample_records(records, args.sample_items, args.seed)
    if args.limit_items:
        records = records[: args.limit_items]

    print("--- DSPy Sharded Calibrated Gemma Test Query ---", flush=True)
    print(f"Task: {args.expert}/{args.language}/{args.prompt_variant}/{args.autorun}", flush=True)
    print(f"Prompt style: {args.prompt_style}", flush=True)
    print(f"Program path: {program_path}", flush=True)
    print(f"Endpoints: {', '.join(args.api_bases)}", flush=True)
    print(f"Records: {len(records)}", flush=True)

    all_results = []
    # Use processes, not threads: DSPy's active LM is global process state, so
    # thread-based shards can race and route traffic through the last configured
    # endpoint. One process per endpoint keeps each shard pinned to its API base.
    with futures.ProcessPoolExecutor(max_workers=len(args.api_bases)) as executor:
        future_to_index = {
            executor.submit(
                run_shard,
                args,
                shard_index,
                api_base,
                records,
                success,
                program_path,
                exp_dir,
                base_run_name,
            ): shard_index
            for shard_index, api_base in enumerate(args.api_bases)
        }
        for future in futures.as_completed(future_to_index):
            all_results.extend(future.result())

    all_results.sort(key=lambda item: str(item.get("uuid")))
    records_by_uuid = {str(item.get("uuid")): item for item in records}
    ordered_records = [records_by_uuid[str(item.get("uuid"))] for item in all_results if str(item.get("uuid")) in records_by_uuid]
    metrics = calculate_metrics(all_results)
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
            "student_api_bases": args.api_bases,
            "max_tokens": args.max_tokens,
            "num_queries": args.num_queries,
            "num_workers_per_endpoint": args.num_workers_per_endpoint,
            "optimized_program_path": str(program_path),
            "elapsed_seconds": time.time() - started_at,
        }
    )
    write_json(predictions_path, attach_predictions(ordered_records, all_results))
    write_json(metrics_path, metrics)
    write_json(
        metadata_path,
        {
            "expert": args.expert,
            "language": args.language,
            "prompt_variant": args.prompt_variant,
            "prompt_style": args.prompt_style,
            "autorun": args.autorun,
            "student_api_bases": args.api_bases,
            "max_tokens": args.max_tokens,
            "num_queries": args.num_queries,
            "num_workers_per_endpoint": args.num_workers_per_endpoint,
            "max_article_chars": args.max_article_chars,
            "num_records_loaded": len(records),
            "num_results": len(all_results),
            "optimized_program_path": str(program_path),
            "uncertainty_success": success,
        },
    )
    print(f"Merged sharded query complete in {(time.time() - started_at) / 60:.1f}m", flush=True)
    print(f"Wrote {metrics_path}", flush=True)


if __name__ == "__main__":
    main()
