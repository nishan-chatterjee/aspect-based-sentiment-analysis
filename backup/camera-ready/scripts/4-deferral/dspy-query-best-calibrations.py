#!/usr/bin/env python3
"""Run calibrated Gemma test queries for the best setting per expert/language."""

from __future__ import annotations

import argparse
import concurrent.futures as futures
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--selection-json", default="reviews/uncertainty/llm-dspy-calibration-cot/best_calibrations.json")
    parser.add_argument("--python-bin", default=sys.executable)
    parser.add_argument("--output-root", default="reviews/uncertainty/llm-dspy-calibration-cot")
    parser.add_argument("--uncertainty-root", default="reviews/uncertainty")
    parser.add_argument("--log-dir", default=None)
    parser.add_argument("--student-model", default="gemma27b")
    parser.add_argument("--student-api-bases", default="http://127.0.0.1:8001/v1")
    parser.add_argument("--prompt-style", default="current", choices=["current", "legacy", "rich"])
    parser.add_argument("--api-key", default=os.environ.get("OPENAI_API_KEY", "EMPTY"))
    parser.add_argument("--teacher-label", default="qwen-2.5-72b")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--max-tokens", type=int, default=1024)
    parser.add_argument("--miprov2-temp", type=float, default=1.0)
    parser.add_argument("--max-article-chars", type=int, default=10000)
    parser.add_argument("--num-queries", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--max-parallel", type=int, default=4)
    parser.add_argument("--limit-items", type=int, default=None)
    parser.add_argument("--sample-items", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--skip-endpoint-check", action="store_true")
    return parser.parse_args()


def run_one(args: argparse.Namespace, row: dict[str, Any], task_index: int, api_bases: list[str]) -> int:
    api_base = api_bases[task_index % len(api_bases)]
    log_path = None
    if args.log_dir:
        log_dir = Path(args.log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)
        log_path = log_dir / (
            "bestquery_{idx}_{expert}_{language}_{prompt_variant}_{autorun}.log".format(
                idx=task_index,
                **row,
            )
        )
    cmd = [
        args.python_bin,
        "scripts/4-deferral/dspy-calibrated-gemma-query.py",
        "--expert",
        row["expert"],
        "--language",
        row["language"],
        "--prompt-variant",
        row["prompt_variant"],
        "--prompt-style",
        args.prompt_style,
        "--autorun",
        row["autorun"],
        "--uncertainty-root",
        args.uncertainty_root,
        "--output-root",
        args.output_root,
        "--optimized-program",
        row["program_path"],
        "--student-model",
        args.student_model,
        "--student-api-base",
        api_base,
        "--api-key",
        args.api_key,
        "--teacher-label",
        args.teacher_label,
        "--temperature",
        str(args.temperature),
        "--top-p",
        str(args.top_p),
        "--max-tokens",
        str(args.max_tokens),
        "--miprov2-temp",
        str(args.miprov2_temp),
        "--max-article-chars",
        str(args.max_article_chars),
        "--num-queries",
        str(args.num_queries),
        "--num-workers",
        str(args.num_workers),
        "--seed",
        str(args.seed),
    ]
    if args.limit_items is not None:
        cmd.extend(["--limit-items", str(args.limit_items)])
    if args.sample_items is not None:
        cmd.extend(["--sample-items", str(args.sample_items)])
    if args.force:
        cmd.append("--force")
    if args.skip_endpoint_check:
        cmd.append("--skip-endpoint-check")

    print(
        "Starting {idx}: {expert}/{language}/{prompt_variant}/{autorun} on {api_base}; log={log}".format(
            idx=task_index,
            api_base=api_base,
            log=str(log_path) if log_path else "stdout",
            **row,
        ),
        flush=True,
    )
    if log_path:
        with log_path.open("w", encoding="utf-8") as log_file:
            log_file.write("Command:")
            for part in cmd:
                log_file.write(f" {part!r}")
            log_file.write("\n")
            log_file.flush()
            completed = subprocess.run(cmd, check=False, stdout=log_file, stderr=subprocess.STDOUT)
    else:
        completed = subprocess.run(cmd, check=False)
    if completed.returncode == 0:
        print(
            "Completed {idx}: {expert}/{language}/{prompt_variant}/{autorun}".format(
                idx=task_index,
                **row,
            ),
            flush=True,
        )
    else:
        print(
            "FAILED {idx}: {expert}/{language}/{prompt_variant}/{autorun}".format(
                idx=task_index,
                **row,
            ),
            flush=True,
        )
    return completed.returncode


def main() -> None:
    args = parse_args()
    selection = load_json(Path(args.selection_json))
    selected = selection.get("selected", [])
    if not selected:
        raise RuntimeError(f"No selected calibration rows in {args.selection_json}")
    api_bases = [item.strip() for item in args.student_api_bases.split(",") if item.strip()]
    if not api_bases:
        raise RuntimeError("--student-api-bases is empty")

    print(f"Running {len(selected)} best calibration test-query jobs.", flush=True)
    failures = 0
    with futures.ThreadPoolExecutor(max_workers=args.max_parallel) as executor:
        future_to_index = {
            executor.submit(run_one, args, row, index, api_bases): index
            for index, row in enumerate(selected)
        }
        for future in futures.as_completed(future_to_index):
            failures += 1 if future.result() != 0 else 0
    print(f"Best-calibration query jobs finished; failures={failures}", flush=True)
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
