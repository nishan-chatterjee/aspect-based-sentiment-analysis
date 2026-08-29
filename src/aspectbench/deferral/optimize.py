"""Lightweight, resumable MIPROv2 prompt optimization for selective deferral."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Sequence

from ..runtime.runs import RunLayout, atomic_json
from .programs import build_program, configure_lm, field, require_dspy


LABEL_NAMES = {-1: "negative", 0: "neutral", 1: "positive"}


def _example(dspy: Any, row: dict[str, Any]):
    gold_name = LABEL_NAMES[int(row["sentiment"])]
    primary_name = LABEL_NAMES[int(row["prediction"])]
    action = "keep_plm" if gold_name == primary_name else "override"
    return dspy.Example(
        article=row["article"],
        aspect=row["aspect"],
        primary_expert=row["model"],
        primary_prediction=primary_name,
        primary_probabilities=json.dumps(row["probabilities"], sort_keys=True),
        primary_uncertainty=json.dumps(row["uncertainty"], sort_keys=True),
        auxiliary_experts=json.dumps(row.get("auxiliary_experts", {}), sort_keys=True),
        routing_context=f"language={row['language']}; variant={row['variant']}",
        reasoning="Use the labeled calibration decision.",
        action=action,
        sentiment=gold_name,
    ).with_inputs(
        "article",
        "aspect",
        "primary_expert",
        "primary_prediction",
        "primary_probabilities",
        "primary_uncertainty",
        "auxiliary_experts",
        "routing_context",
    )


def _metric(example: Any, prediction: Any, trace: Any = None) -> float:
    action = field(prediction, "action").strip().lower()
    sentiment = field(prediction, "sentiment").strip().lower()
    return float(action == str(example.action).lower() and sentiment == str(example.sentiment).lower())


def optimize_program(
    training_predictions: Sequence[dict[str, Any]],
    validation_predictions: Sequence[dict[str, Any]],
    *,
    endpoint_model: str,
    api_base: str,
    run_root: str | Path,
    run_id: str,
    api_key: str = "local",
    model_type: str = "chat",
    auto: str = "light",
    seed: int = 42,
    resume: bool = True,
) -> Path:
    if not training_predictions or not validation_predictions:
        raise ValueError("Optimization requires non-empty labeled train and validation predictions.")
    for row in [*training_predictions, *validation_predictions]:
        if row.get("sentiment") not in (-1, 0, 1):
            raise ValueError("Every optimization record needs a gold sentiment label.")
    layout = RunLayout("dspy-optimization", run_id, run_root=run_root, resume=resume)
    logger = layout.logger("dspy-optimization")
    output = layout.root / "optimized-program.json"
    if resume and output.is_file():
        logger.info("Program already exists; resuming without recompilation: %s", output)
        return output
    dspy = require_dspy()
    lm = configure_lm(
        model=endpoint_model,
        api_base=api_base,
        api_key=api_key,
        model_type=model_type,
    )
    trainset = [_example(dspy, row) for row in training_predictions]
    valset = [_example(dspy, row) for row in validation_predictions]
    layout.write_manifest(
        {
            "endpoint_model": endpoint_model,
            "api_base": api_base,
            "auto": auto,
            "seed": seed,
            "train_records": len(trainset),
            "validation_records": len(valset),
        }
    )
    layout.update_progress(status="optimizing")
    optimizer = dspy.MIPROv2(
        metric=_metric,
        prompt_model=lm,
        task_model=lm,
        max_bootstrapped_demos=0,
        max_labeled_demos=0,
        auto=auto,
    )
    compiled = optimizer.compile(
        student=build_program(),
        trainset=trainset,
        valset=valset,
        seed=seed,
        requires_permission_to_run=False,
    )
    compiled.save(str(output))
    atomic_json(
        layout.root / "optimization-summary.json",
        {
            "program": str(output),
            "train_records": len(trainset),
            "validation_records": len(valset),
            "auto": auto,
        },
    )
    layout.update_progress(status="complete", program=str(output))
    logger.info("Saved reusable program to %s", output)
    return output
