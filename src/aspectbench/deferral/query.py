"""Query a reusable DSPy deferral program over PLM predictions."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Sequence

from ..inference.runner import run_inference
from ..registry import normalize_language, resolve_model
from ..runtime.runs import RunLayout, atomic_json, chunks
from .programs import configure_lm, field, load_or_build_program


LABEL_NAMES = {-1: "negative", 0: "neutral", 1: "positive"}
NAME_LABELS = {value: key for key, value in LABEL_NAMES.items()}
VALID_ACTIONS = {"keep_plm", "override", "abstain_uncertain"}


def _confidence(row: dict[str, Any]) -> float:
    return float(row["uncertainty"]["confidence"])


def _select_gated(rows: Sequence[dict[str, Any]], gate_rate: float) -> set[str]:
    if not 0.0 <= gate_rate <= 1.0:
        raise ValueError("gate_rate must be between 0 and 1")
    count = int(round(len(rows) * gate_rate))
    if gate_rate > 0 and count == 0 and rows:
        count = 1
    ordered = sorted(rows, key=lambda row: (_confidence(row), row["record_id"]))
    return {row["record_id"] for row in ordered[:count]}


def _prediction_index(rows: Sequence[dict[str, Any]]) -> dict[str, dict[str, dict[str, Any]]]:
    output: dict[str, dict[str, dict[str, Any]]] = {}
    for row in rows:
        output.setdefault(row["record_id"], {})[row["model"]] = row
    return output


def run_deferral(
    records: Sequence[dict[str, Any]],
    *,
    models: Sequence[str],
    primary_model: str,
    language: str,
    variant: str,
    repository_root: str | Path,
    pretrained_model_root: str | Path,
    run_root: str | Path,
    run_id: str,
    endpoint_model: str,
    api_base: str,
    program_path: str | Path | None = None,
    api_key: str = "local",
    model_type: str = "chat",
    base_model_root: str | Path | None = None,
    device: str = "auto",
    mc_passes: int = 8,
    batch_size: int = 8,
    shard_size: int = 32,
    gate_rate: float = 0.25,
    resume: bool = True,
    retry_failed: bool = False,
) -> Path:
    language = normalize_language(language)
    primary_model = resolve_model(primary_model, language=language, variant=variant).name
    inference_path = run_inference(
        records,
        models=models,
        language=language,
        variant=variant,
        repository_root=repository_root,
        model_root=pretrained_model_root,
        base_model_root=base_model_root,
        run_root=run_root,
        run_id=f"{run_id}-plm",
        device=device,
        batch_size=batch_size,
        shard_size=shard_size,
        mc_passes=mc_passes,
        resume=resume,
    )
    predictions = json.loads(inference_path.read_text(encoding="utf-8"))
    index = _prediction_index(predictions)
    primary_rows = [
        model_rows[primary_model]
        for model_rows in index.values()
        if primary_model in model_rows
    ]
    if len(primary_rows) != len(records):
        raise ValueError(
            f"Primary model {primary_model!r} has {len(primary_rows)}/{len(records)} predictions."
        )
    gated = _select_gated(primary_rows, gate_rate)
    layout = RunLayout("dspy-inference", run_id, run_root=run_root, resume=resume)
    logger = layout.logger("dspy-inference")
    layout.write_manifest(
        {
            "models": list(models),
            "primary_model": primary_model,
            "language": language,
            "variant": variant,
            "endpoint_model": endpoint_model,
            "api_base": api_base,
            "program": str(program_path) if program_path else "base-signature",
            "gate_rate": gate_rate,
            "gated_records": len(gated),
        }
    )
    namespace = f"{primary_model}-{language}-{variant}"
    completed = layout.completed_ids(namespace, include_failed=not retry_failed)
    pending = [row for row in primary_rows if row["record_id"] not in completed]
    configure_lm(
        model=endpoint_model,
        api_base=api_base,
        api_key=api_key,
        model_type=model_type,
    )
    program = load_or_build_program(program_path)
    next_shard = len(list((layout.shards / namespace).glob("shard-*.json")))
    processed = len(completed)
    for shard_index, batch in enumerate(chunks(pending, shard_size), start=next_shard):
        outputs: list[dict[str, Any]] = []
        for primary in batch:
            record_models = index[primary["record_id"]]
            if primary["record_id"] not in gated:
                outputs.append(
                    {
                        **primary,
                        "base_prediction": primary["prediction"],
                        "action": "keep_plm",
                        "deferred": False,
                        "dspy_status": "not-gated",
                    }
                )
                continue
            auxiliaries = {
                name: {
                    "prediction": row["prediction"],
                    "probabilities": row["probabilities"],
                    "uncertainty": row["uncertainty"],
                }
                for name, row in record_models.items()
                if name != primary_model
            }
            try:
                response = program(
                    article=primary["article"],
                    aspect=primary["aspect"],
                    primary_expert=primary_model,
                    primary_prediction=LABEL_NAMES[primary["prediction"]],
                    primary_probabilities=json.dumps(primary["probabilities"], sort_keys=True),
                    primary_uncertainty=json.dumps(primary["uncertainty"], sort_keys=True),
                    auxiliary_experts=json.dumps(auxiliaries, sort_keys=True),
                    routing_context=f"language={language}; variant={variant}",
                )
                action = field(response, "action").strip().lower()
                sentiment_name = field(response, "sentiment").strip().lower()
                if action not in VALID_ACTIONS or sentiment_name not in NAME_LABELS:
                    raise ValueError(
                        f"Invalid DSPy output action={action!r}, sentiment={sentiment_name!r}"
                    )
                prediction = (
                    primary["prediction"] if action == "keep_plm" else NAME_LABELS[sentiment_name]
                )
                outputs.append(
                    {
                        **primary,
                        "base_prediction": primary["prediction"],
                        "prediction": prediction,
                        "action": action,
                        "deferred": True,
                        "reasoning": field(response, "reasoning"),
                        "dspy_status": "complete",
                    }
                )
            except Exception as exc:
                logger.exception("DSPy query failed for %s", primary["record_id"])
                outputs.append(
                    {
                        **primary,
                        "base_prediction": primary["prediction"],
                        "action": "keep_plm",
                        "deferred": True,
                        "dspy_status": "failed-fallback",
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                )
        layout.write_shard(namespace, shard_index, outputs)
        processed += len(outputs)
        layout.update_progress(status="running", completed_records=processed, total_records=len(records))
    output = layout.root / "predictions.json"
    atomic_json(output, layout.collect_latest(namespace))
    layout.update_progress(status="complete", completed_records=len(records), output=str(output))
    return output
