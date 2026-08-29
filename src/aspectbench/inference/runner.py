"""One/few/all inference orchestration with restart-safe shards."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

from ..registry import normalize_language, select_models
from ..runtime.runs import RunLayout, atomic_json, chunks, record_id
from .hf_bridge import checkpoint_status, create_engine


def _normalize_output(
    prediction: dict[str, Any], source: dict[str, Any], model: str
) -> dict[str, Any]:
    return {
        "record_id": record_id(source),
        "model": model,
        "language": "sl" if prediction["language"] == "slovenian" else "hbs",
        "variant": prediction["mode"],
        "article": prediction["input_article"],
        "aspect": source.get("aspect") or prediction["tagged_aspects"][0],
        "sentiment": prediction.get("gold_sentiment"),
        "prediction": prediction["predicted_sentiment"],
        "prediction_name": prediction["predicted_sentiment_name"],
        "probabilities": prediction["class_probabilities"],
        "uncertainty": prediction["uncertainty_across_classes"],
        "inference": prediction["inference"],
    }


def run_inference(
    records: Sequence[dict[str, Any]],
    *,
    models: Sequence[str],
    language: str,
    variant: str,
    repository_root: str | Path,
    model_root: str | Path,
    base_model_root: str | Path | None = None,
    run_root: str | Path = "models/_runs",
    run_id: str = "inference",
    device: str = "auto",
    batch_size: int = 8,
    shard_size: int = 64,
    mc_passes: int = 8,
    seed: int = 42,
    resume: bool = True,
    skip_unavailable: bool | None = None,
) -> Path:
    if not records:
        raise ValueError("Inference needs at least one input record.")
    language = normalize_language(language)
    specs = select_models(models, language=language, variant=variant)
    requested_all = len(models) == 1 and models[0].lower() == "all"
    if skip_unavailable is None:
        skip_unavailable = requested_all
    layout = RunLayout("inference", run_id, run_root=run_root, resume=resume)
    logger = layout.logger("inference")
    statuses: dict[str, Any] = {}
    layout.write_manifest(
        {
            "models": [spec.name for spec in specs],
            "language": language,
            "variant": variant,
            "record_count": len(records),
            "mc_passes": mc_passes,
            "model_root": str(Path(model_root).resolve()),
        }
    )
    for model_index, spec in enumerate(specs):
        namespace = f"{spec.name}-{language}-{variant}"
        status = checkpoint_status(
            repository_root, model_root, spec.name, language, variant
        )
        statuses[spec.name] = status
        if not status["available"]:
            logger.warning("Skipping %s: %s", spec.name, status["reason"])
            statuses[spec.name]["state"] = "skipped"
            if not skip_unavailable:
                layout.update_progress(status="failed", models=statuses)
                raise FileNotFoundError(status["reason"])
            continue
        completed = layout.completed_ids(namespace)
        pending = [row for row in records if record_id(row) not in completed]
        logger.info(
            "%s: %d complete, %d pending", spec.name, len(completed), len(pending)
        )
        if pending:
            engine = create_engine(
                repository_root=repository_root,
                model_root=model_root,
                base_model_root=base_model_root,
                model=spec.name,
                language=language,
                variant=variant,
                device=device,
            )
            next_shard = len(list((layout.shards / namespace).glob("shard-*.json")))
            for offset, batch in enumerate(chunks(pending, shard_size)):
                predicted = engine.predict_batch(
                    batch, batch_size=batch_size, mc_passes=mc_passes, seed=seed
                )
                rows = [
                    _normalize_output(output, source, spec.name)
                    for output, source in zip(predicted, batch, strict=True)
                ]
                path = layout.write_shard(namespace, next_shard + offset, rows)
                completed.update(row["record_id"] for row in rows)
                logger.info("Wrote %s (%d records)", path, len(rows))
                layout.update_progress(
                    status="running",
                    current_model=spec.name,
                    completed_records=len(completed),
                    total_records=len(records),
                    models=statuses,
                )
            del engine
        statuses[spec.name]["state"] = "complete"
        statuses[spec.name]["records"] = len(layout.collect(namespace))
        layout.update_progress(
            status="running", completed_models=model_index + 1, models=statuses
        )
    merged: list[dict[str, Any]] = []
    for spec in specs:
        merged.extend(layout.collect(f"{spec.name}-{language}-{variant}"))
    output = layout.root / "predictions.json"
    atomic_json(output, merged)
    layout.update_progress(status="complete", records_written=len(merged), models=statuses)
    logger.info("Complete: %s", output)
    return output
