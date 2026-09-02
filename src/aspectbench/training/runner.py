"""Full resumable fine-tuning followed by MC-dropout uncertainty inference."""

from __future__ import annotations

import json
import os
from pathlib import Path
import random
import shutil
from typing import Any, Sequence

from ..evaluation.metrics import classification_metrics
from ..inference.hf_bridge import checkpoint_status, create_engine, release_coordinates
from ..inference.runner import _normalize_output
from ..registry import normalize_language, select_models
from ..runtime.runs import RunLayout, atomic_json, chunks, record_id
from .smoke import _loss_for_batch


def _evaluate(engine: Any, records: Sequence[dict[str, Any]], batch_size: int) -> dict[str, Any]:
    predictions = engine.predict_batch(records, batch_size=batch_size, mc_passes=0)
    gold = [record["sentiment"] for record in records]
    predicted = [row["predicted_sentiment"] for row in predictions]
    return classification_metrics(gold, predicted)


def _uncertainty_export(
    engine: Any,
    records: Sequence[dict[str, Any]],
    *,
    model: str,
    output: Path,
    batch_size: int,
    shard_size: int,
    mc_passes: int,
    seed: int,
) -> None:
    output.mkdir(parents=True, exist_ok=True)
    all_rows: list[dict[str, Any]] = []
    for index, batch in enumerate(chunks(records, shard_size)):
        predictions = engine.predict_batch(
            batch, batch_size=batch_size, mc_passes=mc_passes, seed=seed + index
        )
        rows = [
            _normalize_output(prediction, source, model)
            for prediction, source in zip(predictions, batch, strict=True)
        ]
        atomic_json(output / f"shard-{index:06d}.json", rows)
        all_rows.extend(rows)
    atomic_json(output / "predictions-with-uncertainty.json", all_rows)
    atomic_json(
        output / "_SUCCESS.json",
        {"records": len(all_rows), "mc_passes": mc_passes, "complete": True},
    )


def _activate_checkpoint(
    checkpoint: Path,
    *,
    output_model_root: str | Path,
    model: str,
    language: str,
    variant: str,
) -> Path:
    """Publish a stable local pointer consumable through ``--model-root``."""

    release_model, release_language = release_coordinates(model, language)
    active = (
        Path(output_model_root)
        / "_active"
        / release_model
        / release_language
        / f"{variant}.pt"
    )
    active.parent.mkdir(parents=True, exist_ok=True)
    temporary = active.with_name(f".{active.name}.tmp")
    temporary.unlink(missing_ok=True)
    try:
        temporary.symlink_to(os.path.relpath(checkpoint.resolve(), active.parent.resolve()))
    except OSError:
        try:
            os.link(checkpoint, temporary)
        except OSError:
            shutil.copy2(checkpoint, temporary)
    os.replace(temporary, active)
    atomic_json(
        active.with_suffix(".metadata.json"),
        {
            "model": model,
            "language": language,
            "variant": variant,
            "checkpoint": str(checkpoint.resolve()),
        },
    )
    return active


def run_training(
    training_records: Sequence[dict[str, Any]],
    validation_records: Sequence[dict[str, Any]],
    *,
    uncertainty_sets: dict[str, Sequence[dict[str, Any]]] | None,
    models: Sequence[str],
    language: str,
    variant: str,
    repository_root: str | Path,
    pretrained_model_root: str | Path,
    output_model_root: str | Path = "models",
    base_model_root: str | Path | None = None,
    run_root: str | Path = "models/_runs",
    run_id: str,
    device: str = "auto",
    epochs: int = 3,
    batch_size: int = 4,
    learning_rate: float = 2e-5,
    weight_decay: float = 0.01,
    gradient_accumulation_steps: int = 1,
    max_steps: int | None = None,
    mc_passes: int = 10,
    shard_size: int = 64,
    seed: int = 42,
    resume: bool = True,
    skip_unavailable: bool | None = None,
) -> Path:
    import torch

    if not training_records or not validation_records:
        raise ValueError("Training and validation records must both be non-empty.")
    for name, rows in (("train", training_records), ("validation", validation_records)):
        if any(row.get("sentiment") not in (-1, 0, 1) for row in rows):
            raise ValueError(f"Every {name} record needs sentiment -1, 0, or 1.")
    if epochs < 1 or batch_size < 1 or gradient_accumulation_steps < 1:
        raise ValueError("epochs, batch_size, and gradient accumulation must be positive.")
    language = normalize_language(language)
    specs = select_models(models, language=language, variant=variant)
    requested_all = any(str(value).strip().lower() == "all" for value in models)
    if skip_unavailable is None:
        skip_unavailable = requested_all
    uncertainty_sets = uncertainty_sets or {"validation": validation_records}
    layout = RunLayout("training", run_id, run_root=run_root, resume=resume)
    logger = layout.logger("training")
    layout.write_manifest(
        {
            "models": [spec.name for spec in specs],
            "language": language,
            "variant": variant,
            "train_records": len(training_records),
            "validation_records": len(validation_records),
            "epochs": epochs,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "mc_passes_after_training": mc_passes,
        }
    )
    reports: list[dict[str, Any]] = []
    for model_index, spec in enumerate(specs):
        status = checkpoint_status(
            repository_root, pretrained_model_root, spec.name, language, variant
        )
        if not status["available"]:
            report = {"model": spec.name, "status": "skipped", "reason": status["reason"]}
            reports.append(report)
            logger.warning("Skipping %s: %s", spec.name, status["reason"])
            if not skip_unavailable:
                raise FileNotFoundError(status["reason"])
            continue
        output_dir = Path(output_model_root) / spec.name / language / variant / run_id
        output_dir.mkdir(parents=True, exist_ok=True)
        best_path = output_dir / "best-model.pt"
        last_path = output_dir / "last-training-state.pt"
        report_path = output_dir / "training-report.json"
        engine = create_engine(
            repository_root=repository_root,
            model_root=pretrained_model_root,
            base_model_root=base_model_root,
            model=spec.name,
            language=language,
            variant=variant,
            device=device,
        )
        parameters = [parameter for parameter in engine.model.parameters() if parameter.requires_grad]
        optimizer = torch.optim.AdamW(
            parameters, lr=learning_rate, weight_decay=weight_decay
        )
        start_epoch = 0
        global_step = 0
        best_f1 = -1.0
        history: list[dict[str, Any]] = []
        if resume and last_path.is_file():
            try:
                state = torch.load(last_path, map_location="cpu", weights_only=False)
            except TypeError:
                state = torch.load(last_path, map_location="cpu")
            engine.model.load_state_dict(state["model"], strict=True)
            optimizer.load_state_dict(state["optimizer"])
            start_epoch = int(state["epoch"]) + 1
            global_step = int(state["global_step"])
            best_f1 = float(state["best_f1"])
            history = list(state.get("history", []))
            logger.info("Resumed %s at epoch %d, step %d", spec.name, start_epoch, global_step)
        stop = False
        for epoch in range(start_epoch, epochs):
            indices = list(range(len(training_records)))
            random.Random(seed + epoch).shuffle(indices)
            ordered = [training_records[index] for index in indices]
            engine.model.train()
            optimizer.zero_grad(set_to_none=True)
            losses: list[float] = []
            for batch_index, batch in enumerate(chunks(ordered, batch_size)):
                loss = _loss_for_batch(engine, batch) / gradient_accumulation_steps
                loss.backward()
                losses.append(float(loss.detach().cpu().item()) * gradient_accumulation_steps)
                if (batch_index + 1) % gradient_accumulation_steps == 0 or (
                    batch_index + 1
                ) * batch_size >= len(ordered):
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)
                    global_step += 1
                    if max_steps is not None and global_step >= max_steps:
                        stop = True
                        break
            engine.model.eval()
            metrics = _evaluate(engine, validation_records, batch_size)
            epoch_report = {
                "epoch": epoch,
                "global_step": global_step,
                "mean_training_loss": sum(losses) / len(losses),
                "validation": metrics,
            }
            history.append(epoch_report)
            if float(metrics["f1_macro"]) > best_f1:
                best_f1 = float(metrics["f1_macro"])
                torch.save(engine.model.state_dict(), best_path)
            torch.save(
                {
                    "model": engine.model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "epoch": epoch,
                    "global_step": global_step,
                    "best_f1": best_f1,
                    "history": history,
                },
                last_path,
            )
            atomic_json(report_path, {"status": "running", "history": history})
            layout.update_progress(
                status="training",
                current_model=spec.name,
                epoch=epoch,
                global_step=global_step,
                completed_models=model_index,
                total_models=len(specs),
            )
            logger.info(
                "%s epoch %d: loss %.6f, validation Macro-F1 %.6f",
                spec.name,
                epoch,
                epoch_report["mean_training_loss"],
                metrics["f1_macro"],
            )
            if stop:
                break
        if not best_path.is_file():
            raise RuntimeError(f"Training produced no best checkpoint for {spec.name}.")
        try:
            best_state = torch.load(best_path, map_location="cpu", weights_only=True)
        except TypeError:
            best_state = torch.load(best_path, map_location="cpu")
        engine.model.load_state_dict(best_state, strict=True)
        engine.model.eval()
        active_checkpoint = _activate_checkpoint(
            best_path,
            output_model_root=output_model_root,
            model=spec.name,
            language=language,
            variant=variant,
        )
        uncertainty_outputs: dict[str, str] = {}
        for set_name, rows in uncertainty_sets.items():
            target = output_dir / "uncertainty" / set_name
            _uncertainty_export(
                engine,
                rows,
                model=spec.name,
                output=target,
                batch_size=batch_size,
                shard_size=shard_size,
                mc_passes=mc_passes,
                seed=seed,
            )
            uncertainty_outputs[set_name] = str(target.resolve())
        report = {
            "schema_version": 1,
            "model": spec.name,
            "language": language,
            "variant": variant,
            "status": "complete",
            "best_validation_macro_f1": best_f1,
            "global_step": global_step,
            "best_checkpoint": str(best_path.resolve()),
            "active_checkpoint": str(active_checkpoint.resolve()),
            "last_training_state": str(last_path.resolve()),
            "uncertainty_outputs": uncertainty_outputs,
            "history": history,
        }
        atomic_json(report_path, report)
        reports.append(report)
        del engine, optimizer, best_state
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    summary = layout.root / "training-report.json"
    atomic_json(summary, reports)
    layout.update_progress(status="complete", reports=reports)
    return summary
