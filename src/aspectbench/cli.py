"""Command-line entry points for the refactored AspectBench package."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any, Sequence

from .data import load_records
from .evaluation import build_evaluation_report
from .registry import select_models


def _input_records(args: argparse.Namespace, *, labeled: bool = False) -> list[dict[str, Any]]:
    if getattr(args, "input", None):
        records = load_records(args.input)
    elif getattr(args, "input_doc", None):
        record: dict[str, Any] = {"article": args.input_doc}
        if getattr(args, "aspect", None):
            record["aspect"] = args.aspect
        if getattr(args, "sentiment", None) is not None:
            record["sentiment"] = args.sentiment
        records = [record]
    else:
        raise ValueError("Supply either --input FILE or --input-doc TEXT.")
    if getattr(args, "limit", None) is not None:
        records = records[: args.limit]
    if labeled and any(record.get("sentiment") not in (-1, 0, 1) for record in records):
        raise ValueError("This command requires sentiment -1, 0, or 1 on every record.")
    return records


def _infer_command(args: argparse.Namespace) -> int:
    from .inference import build_ensemble_rows, publish_inference_outputs, run_inference

    records = _input_records(args)
    output = run_inference(
        records,
        models=args.models,
        language=args.dataset,
        variant=args.variant,
        repository_root=args.repository_root,
        model_root=args.model_root,
        base_model_root=args.base_model_root,
        run_root=args.run_root,
        run_id=args.run_id,
        device=args.device,
        batch_size=args.batch_size,
        shard_size=args.shard_size,
        mc_passes=args.mc_passes,
        seed=args.seed,
        resume=args.resume,
        skip_unavailable=args.skip_unavailable,
    )
    rows = json.loads(output.read_text(encoding="utf-8"))
    if args.publish:
        raw_path, ensemble_path = publish_inference_outputs(
            rows,
            dataset=args.dataset,
            run_id=args.run_id,
            output_root=args.output_root,
            filename=args.filename,
            seed=args.seed,
        )
        if args.input_doc:
            ensemble = build_ensemble_rows(rows)
            if ensemble:
                summary = {
                    "experts": [
                        {
                            key: expert[key]
                            for key in (
                                "model",
                                "variant",
                                "prediction",
                                "prediction_name",
                                "confidence",
                            )
                        }
                        for expert in ensemble[0]["experts"]
                    ],
                    "majority_vote": ensemble[0]["majority_vote"],
                    "confidence_vote": ensemble[0]["confidence_vote"],
                }
                _json_dump(summary, None)
        print(f"Detailed predictions: {raw_path.resolve()}")
        print(f"Ensemble predictions: {ensemble_path.resolve()}")
    print(f"Resumable run state: {output.resolve()}")
    return 0


def _aggregate_inference_command(args: argparse.Namespace) -> int:
    from .inference import load_prediction_files, publish_inference_outputs

    rows = load_prediction_files(args.predictions)
    if not rows:
        raise ValueError("No expert predictions were available to aggregate.")
    raw_path, ensemble_path = publish_inference_outputs(
        rows,
        dataset=args.dataset,
        run_id=args.run_id,
        output_root=args.output_root,
        filename=args.filename,
        seed=args.seed,
    )
    print(f"Detailed predictions: {raw_path.resolve()}")
    print(f"Ensemble predictions: {ensemble_path.resolve()}")
    return 0


def _train_smoke_command(args: argparse.Namespace) -> int:
    from .training import run_training_smoke

    output = run_training_smoke(
        _input_records(args, labeled=True),
        models=args.models,
        language=args.dataset,
        variant=args.variant,
        repository_root=args.repository_root,
        pretrained_model_root=args.model_root,
        output_model_root=args.output_model_root,
        base_model_root=args.base_model_root,
        run_root=args.run_root,
        run_id=args.run_id,
        device=args.device,
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        reload_check=args.reload_check,
        resume=args.resume,
        skip_unavailable=args.skip_unavailable,
    )
    print(output.resolve())
    return 0


def _train_command(args: argparse.Namespace) -> int:
    from .training import run_training

    training_records = load_records(args.train_input, keys=("train",))
    validation_records = load_records(args.val_input, keys=("val", "validation"))
    uncertainty_sets: dict[str, list[dict[str, Any]]] = {
        "validation": validation_records
    }
    for value in args.uncertainty_input:
        if "=" not in value:
            raise ValueError("--uncertainty-input must use NAME=PATH.")
        name, path = value.split("=", 1)
        if not name.strip() or not path.strip():
            raise ValueError("--uncertainty-input must use non-empty NAME=PATH.")
        uncertainty_sets[name.strip()] = load_records(path.strip())
    output = run_training(
        training_records,
        validation_records,
        uncertainty_sets=uncertainty_sets,
        models=args.models,
        language=args.dataset,
        variant=args.variant,
        repository_root=args.repository_root,
        pretrained_model_root=args.model_root,
        output_model_root=args.output_model_root,
        base_model_root=args.base_model_root,
        run_root=args.run_root,
        run_id=args.run_id,
        device=args.device,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        max_steps=args.max_steps,
        mc_passes=args.mc_passes,
        shard_size=args.shard_size,
        seed=args.seed,
        resume=args.resume,
        skip_unavailable=args.skip_unavailable,
    )
    print(output.resolve())
    return 0


def _defer_query_command(args: argparse.Namespace) -> int:
    from .deferral.query import run_deferral
    from .deferral.programs import program_path, validate_program_metadata

    prompt_variant = args.prompt_variant or args.variant
    resolved_program = Path(args.program) if args.program else program_path(
        source=args.program_source,
        model=args.primary_model,
        dataset=args.dataset,
        variant=prompt_variant,
        program_root=args.program_root,
        run_id=args.program_run_id,
    )
    validate_program_metadata(
        resolved_program,
        model=args.primary_model,
        dataset=args.dataset,
        variant=prompt_variant,
        allow_mismatch=args.allow_program_mismatch,
    )

    output = run_deferral(
        _input_records(args),
        models=args.models,
        primary_model=args.primary_model,
        language=args.dataset,
        variant=args.variant,
        prompt_variant=prompt_variant,
        repository_root=args.repository_root,
        pretrained_model_root=args.model_root,
        base_model_root=args.base_model_root,
        run_root=args.run_root,
        run_id=args.run_id,
        endpoint_model=args.endpoint_model,
        api_base=args.api_base,
        api_bases=(
            [value.strip() for value in args.api_bases.split(",") if value.strip()]
            if args.api_bases else None
        ),
        num_workers_per_endpoint=(
            [int(value.strip()) for value in args.num_workers_per_endpoint.split(",")]
            if args.num_workers_per_endpoint else None
        ),
        api_key=args.api_key,
        model_type=args.model_type,
        program_path=resolved_program,
        device=args.device,
        mc_passes=args.mc_passes,
        batch_size=args.batch_size,
        shard_size=args.shard_size,
        gate_rate=args.gate_rate,
        resume=args.resume,
        retry_failed=args.retry_failed,
    )
    print(output.resolve())
    return 0


def _defer_optimize_command(args: argparse.Namespace) -> int:
    from .deferral.optimize import optimize_program
    from .inference import run_inference
    from .registry import normalize_language, resolve_model

    language = normalize_language(args.dataset)
    primary_model = resolve_model(
        args.primary_model, language=language, variant=args.variant
    ).name
    prompt_variant = args.prompt_variant or args.variant

    def predict(path: str, split: str) -> list[dict[str, Any]]:
        records = load_records(
            path, keys=("train",) if split == "train" else ("val", "validation")
        )
        output = run_inference(
            records,
            models=args.models or [args.primary_model],
            language=language,
            variant=args.variant,
            repository_root=args.repository_root,
            model_root=args.model_root,
            base_model_root=args.base_model_root,
            run_root=args.run_root,
            run_id=f"{args.run_id}-{split}-plm",
            device=args.device,
            batch_size=args.batch_size,
            shard_size=args.shard_size,
            mc_passes=args.mc_passes,
            resume=args.resume,
        )
        rows = json.loads(output.read_text(encoding="utf-8"))
        by_record: dict[str, dict[str, dict[str, Any]]] = {}
        for row in rows:
            by_record.setdefault(row["record_id"], {})[row["model"]] = row
        selected = []
        for model_rows in by_record.values():
            if primary_model not in model_rows:
                continue
            primary = dict(model_rows[primary_model])
            primary["auxiliary_experts"] = {
                name: {
                    "prediction": row["prediction"],
                    "probabilities": row["probabilities"],
                    "uncertainty": row["uncertainty"],
                }
                for name, row in model_rows.items()
                if name != primary_model
            }
            selected.append(primary)
        if len(selected) != len(records):
            raise ValueError(
                f"Primary model {primary_model!r} has {len(selected)}/{len(records)} {split} predictions."
            )
        return selected

    output = optimize_program(
        predict(args.train_input, "train"),
        predict(args.val_input, "validation"),
        endpoint_model=args.endpoint_model,
        api_base=args.api_base,
        teacher_endpoint_model=args.teacher_endpoint_model,
        teacher_api_base=args.teacher_api_base,
        teacher_api_key=args.teacher_api_key,
        api_key=args.api_key,
        model_type=args.model_type,
        run_root=args.run_root,
        run_id=args.run_id,
        primary_model=primary_model,
        dataset=language,
        variant=prompt_variant,
        program_root=args.program_root,
        auto=args.auto,
        seed=args.seed,
        resume=args.resume,
    )
    print(output.resolve())
    return 0


def _progress_command(args: argparse.Namespace) -> int:
    path = Path(args.path)
    if path.is_dir():
        path = path / "progress.json"
    if not path.is_file():
        raise FileNotFoundError(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    _json_dump(payload, None)
    return 0


def _qualitative_command(args: argparse.Namespace) -> int:
    from .analysis import build_error_review

    report = build_error_review(
        load_records(args.predictions),
        include_correct=args.include_correct,
        limit=args.limit,
    )
    _json_dump(report, args.output)
    return 0


def _json_dump(payload: Any, path: str | Path | None) -> None:
    rendered = json.dumps(payload, ensure_ascii=False, indent=2) + "\n"
    if path is None:
        print(rendered, end="")
        return
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(rendered, encoding="utf-8")
    print(f"Wrote {output.resolve()}")


def _models_command(args: argparse.Namespace) -> int:
    specs = select_models(args.models, language=args.language, variant=args.variant)
    payload = [spec.to_dict() for spec in specs]
    if args.json:
        _json_dump(payload, None)
        return 0
    for spec in specs:
        languages = ",".join(spec.languages)
        variants = ",".join(spec.variants)
        print(f"{spec.name:14} {languages:6} {variants:17} {spec.display_name}")
    return 0


def _score_command(args: argparse.Namespace) -> int:
    predictions = load_records(args.predictions)
    training_records = None
    if args.train_data:
        training_records = []
        for path in args.train_data:
            training_records.extend(load_records(path, keys=("train", "val")))
    report = build_evaluation_report(
        predictions,
        training_records=training_records,
        aspect_key=args.aspect_key,
        gold_key=args.gold_key,
        prediction_key=args.prediction_key,
    )
    _json_dump(report, args.output)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="aspectbench")
    subparsers = parser.add_subparsers(dest="command", required=True)

    models = subparsers.add_parser("models", help="Inspect or resolve model adapters.")
    models.add_argument("--models", nargs="+", default=["all"])
    models.add_argument("--language", default=None)
    models.add_argument("--variant", choices=("masked", "unmasked"), default=None)
    models.add_argument("--json", action="store_true")
    models.set_defaults(handler=_models_command)

    score = subparsers.add_parser("score", help="Build an evaluation report from predictions.")
    score.add_argument("--predictions", required=True)
    score.add_argument("--train-data", nargs="*", default=None)
    score.add_argument("--output", default=None)
    score.add_argument("--aspect-key", default="aspect")
    score.add_argument("--gold-key", default="sentiment")
    score.add_argument("--prediction-key", default="prediction")
    score.set_defaults(handler=_score_command)

    def add_input(command: argparse.ArgumentParser, *, allow_sentiment: bool = False) -> None:
        source = command.add_mutually_exclusive_group(required=True)
        source.add_argument("--input", help="JSON/JSONL record file.")
        source.add_argument("--input-doc", help="One tagged article string.")
        command.add_argument("--aspect", default=None, help="Optional explicit target aspect.")
        if allow_sentiment:
            command.add_argument("--sentiment", type=int, choices=(-1, 0, 1), default=None)
        command.add_argument("--limit", type=int, default=None)

    def add_model_runtime(command: argparse.ArgumentParser) -> None:
        command.add_argument("--models", nargs="+", default=["all"])
        command.add_argument("--dataset", required=True, help="hbs or sl")
        command.add_argument("--variant", choices=("masked", "unmasked"), default="masked")
        command.add_argument("--repository-root", default=".")
        command.add_argument("--model-root", default="huggingface/models")
        command.add_argument("--base-model-root", default=None)
        command.add_argument("--run-root", default="models/_runs")
        command.add_argument("--run-id", required=True)
        command.add_argument("--device", default="auto")
        command.add_argument("--batch-size", type=int, default=8)
        command.add_argument("--shard-size", type=int, default=64)
        command.add_argument("--mc-passes", type=int, default=8)
        command.add_argument("--seed", type=int, default=42)
        command.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)

    infer = subparsers.add_parser("infer", help="Predict with one/few/all released models.")
    add_input(infer)
    add_model_runtime(infer)
    infer.add_argument(
        "--skip-unavailable",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Defaults to true for --models all and false for explicit models.",
    )
    infer.add_argument("--output-root", default="outputs")
    infer.add_argument(
        "--filename",
        default="predictions",
        help="File stem, 'timestamp', or 'seed' (JSON is added automatically).",
    )
    infer.add_argument(
        "--publish", action=argparse.BooleanOptionalAction, default=True,
        help="Publish user-facing detailed and ensemble files under --output-root.",
    )
    infer.set_defaults(handler=_infer_command)

    aggregate = subparsers.add_parser(
        "aggregate-inference",
        help="Combine independently scheduled expert predictions into ensemble output.",
    )
    aggregate.add_argument("--predictions", nargs="+", required=True)
    aggregate.add_argument("--dataset", required=True, help="hbs or sl")
    aggregate.add_argument("--run-id", required=True)
    aggregate.add_argument("--output-root", default="outputs")
    aggregate.add_argument("--filename", default="predictions")
    aggregate.add_argument("--seed", type=int, default=42)
    aggregate.set_defaults(handler=_aggregate_inference_command)

    train = subparsers.add_parser(
        "train-smoke", help="Run one optimizer update and checkpoint/reload verification."
    )
    add_input(train, allow_sentiment=True)
    add_model_runtime(train)
    train.add_argument("--output-model-root", default="models")
    train.add_argument("--learning-rate", type=float, default=1e-5)
    train.add_argument(
        "--reload-check", action=argparse.BooleanOptionalAction, default=True
    )
    train.add_argument(
        "--skip-unavailable", action=argparse.BooleanOptionalAction, default=None
    )
    train.set_defaults(handler=_train_smoke_command)

    full_train = subparsers.add_parser(
        "train", help="Run resumable training and automatic MC uncertainty inference."
    )
    full_train.add_argument("--train-input", required=True)
    full_train.add_argument("--val-input", required=True)
    full_train.add_argument(
        "--uncertainty-input",
        action="append",
        default=[],
        metavar="NAME=PATH",
        help="Additional labeled/unlabeled split to predict with MC dropout after training.",
    )
    add_model_runtime(full_train)
    full_train.add_argument("--output-model-root", default="models")
    full_train.add_argument("--epochs", type=int, default=3)
    full_train.add_argument("--learning-rate", type=float, default=2e-5)
    full_train.add_argument("--weight-decay", type=float, default=0.01)
    full_train.add_argument("--gradient-accumulation-steps", type=int, default=1)
    full_train.add_argument("--max-steps", type=int, default=None)
    full_train.add_argument(
        "--skip-unavailable", action=argparse.BooleanOptionalAction, default=None
    )
    full_train.set_defaults(handler=_train_command)

    query = subparsers.add_parser(
        "defer-query", help="Run PLM uncertainty gating and a DSPy selective-deferral program."
    )
    add_input(query)
    add_model_runtime(query)
    query.add_argument("--primary-model", required=True)
    query.add_argument("--endpoint-model", required=True)
    query.add_argument("--api-base", default="http://127.0.0.1:8000")
    query.add_argument("--api-bases", default=None, help="Comma-separated query endpoints.")
    query.add_argument(
        "--num-workers-per-endpoint", default=None,
        help="Comma-separated positive worker counts matching --api-bases.",
    )
    query.add_argument("--api-key", default="local")
    query.add_argument("--model-type", choices=("chat", "text"), default="chat")
    query.add_argument("--program", default=None)
    query.add_argument(
        "--prompt-variant", choices=("masked", "unmasked"), default=None,
        help="DSPy prompt variant; defaults to the PLM --variant for compatibility.",
    )
    query.add_argument(
        "--program-source",
        choices=("precalibrated", "optimized"),
        default="precalibrated",
        help="Resolve an audited packaged program or a local user-optimized program.",
    )
    query.add_argument("--program-root", default="selective-deferral-programs")
    query.add_argument("--program-run-id", default=None)
    query.add_argument("--allow-program-mismatch", action="store_true")
    query.add_argument("--gate-rate", type=float, default=0.25)
    query.add_argument("--retry-failed", action="store_true")
    query.set_defaults(handler=_defer_query_command)

    optimize = subparsers.add_parser(
        "defer-optimize", help="Optimize and save a reusable DSPy program with MIPROv2."
    )
    optimize.add_argument("--train-input", required=True)
    optimize.add_argument("--val-input", required=True)
    optimize.add_argument("--primary-model", required=True)
    optimize.add_argument(
        "--models", nargs="+", default=None,
        help="Primary plus auxiliary PLMs; defaults to the primary only and accepts all.",
    )
    optimize.add_argument("--dataset", required=True)
    optimize.add_argument("--variant", choices=("masked", "unmasked"), default="masked")
    optimize.add_argument(
        "--prompt-variant", choices=("masked", "unmasked"), default=None,
        help="Program variant, independent of the fine-tuned PLM checkpoint variant.",
    )
    optimize.add_argument("--repository-root", default=".")
    optimize.add_argument("--model-root", default="huggingface/models")
    optimize.add_argument("--base-model-root", default=None)
    optimize.add_argument("--run-root", default="models/_runs")
    optimize.add_argument("--run-id", required=True)
    optimize.add_argument("--program-root", default="selective-deferral-programs")
    optimize.add_argument("--device", default="auto")
    optimize.add_argument("--batch-size", type=int, default=8)
    optimize.add_argument("--shard-size", type=int, default=64)
    optimize.add_argument("--mc-passes", type=int, default=8)
    optimize.add_argument("--endpoint-model", required=True)
    optimize.add_argument("--api-base", default="http://127.0.0.1:8000")
    optimize.add_argument("--api-key", default="local")
    optimize.add_argument("--teacher-endpoint-model", default=None)
    optimize.add_argument("--teacher-api-base", default=None)
    optimize.add_argument("--teacher-api-key", default=None)
    optimize.add_argument("--model-type", choices=("chat", "text"), default="chat")
    optimize.add_argument("--auto", choices=("light", "medium", "heavy"), default="light")
    optimize.add_argument("--seed", type=int, default=42)
    optimize.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    optimize.set_defaults(handler=_defer_optimize_command)

    progress = subparsers.add_parser("progress", help="Print a run's progress JSON.")
    progress.add_argument("path", help="Run directory or progress.json path.")
    progress.set_defaults(handler=_progress_command)

    qualitative = subparsers.add_parser(
        "qualitative", help="Export high-confidence errors for local article review."
    )
    qualitative.add_argument("--predictions", required=True)
    qualitative.add_argument("--output", default=None)
    qualitative.add_argument("--limit", type=int, default=None)
    qualitative.add_argument("--include-correct", action="store_true")
    qualitative.set_defaults(handler=_qualitative_command)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return int(args.handler(args))
    except (OSError, RuntimeError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
