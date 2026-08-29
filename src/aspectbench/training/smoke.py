"""Run one real optimizer update and verify a restartable checkpoint."""

from __future__ import annotations

import json
import math
from pathlib import Path
import sys
from typing import Any, Sequence

from ..inference.hf_bridge import (
    checkpoint_status,
    create_engine,
)
from ..registry import normalize_language, select_models
from ..runtime.runs import RunLayout, atomic_json


LABEL_TO_ID = {-1: 0, 0: 1, 1: 2}
LABEL_TEXT = {-1: "negative", 0: "neutral", 1: "positive"}


def _loss_for_batch(engine: Any, records: Sequence[dict[str, Any]]):
    import torch
    import torch.nn.functional as functional

    inference = sys.modules[engine.__class__.__module__]
    prepared = [inference.prepare_record(record, engine.mode) for record in records]
    labels = torch.tensor(
        [LABEL_TO_ID[record["sentiment"]] for record in records],
        dtype=torch.long,
        device=engine.device,
    )
    if engine.backend == "encoder":
        return engine.model(**engine._encoder_inputs(prepared), labels=labels).loss
    if engine.backend == "han":
        logits = engine.model(
            **engine._han_inputs(prepared),
            aspect_target_token_id=torch.tensor(
                [engine.aspect_token_id], device=engine.device
            ),
        )
        return functional.cross_entropy(logits, labels)
    if engine.backend == "mt5":
        inputs = engine._mt5_sources(prepared)
        targets = engine.tokenizer(
            [LABEL_TEXT[record["sentiment"]] for record in records],
            padding=True,
            truncation=True,
            max_length=engine.spec["max_target_length"],
            return_tensors="pt",
        )["input_ids"].to(engine.device)
        targets[targets == engine.tokenizer.pad_token_id] = -100
        return engine.model(**inputs, labels=targets, use_cache=False).loss
    embeddings = engine._bge_inputs(prepared)
    return functional.cross_entropy(engine.model(embeddings), labels)


def _one_update(engine: Any, records: Sequence[dict[str, Any]], learning_rate: float):
    import torch

    engine.model.train()
    parameters = [parameter for parameter in engine.model.parameters() if parameter.requires_grad]
    if not parameters:
        raise RuntimeError("Model exposes no trainable parameters.")
    optimizer = torch.optim.AdamW(parameters, lr=learning_rate)
    optimizer.zero_grad(set_to_none=True)
    loss = _loss_for_batch(engine, records)
    if not torch.isfinite(loss):
        raise RuntimeError(f"Non-finite smoke-test loss: {loss.item()}")
    loss.backward()
    gradient_parameters = sum(parameter.grad is not None for parameter in parameters)
    probe = next(
        (
            parameter
            for parameter in reversed(parameters)
            if parameter.grad is not None
            and parameter.numel()
            and bool(torch.any(parameter.grad != 0).item())
        ),
        None,
    )
    if probe is None:
        raise RuntimeError("Backward pass produced no nonzero parameter gradients.")
    before = probe.detach().float().cpu().clone()
    optimizer.step()
    delta = float((probe.detach().float().cpu() - before).abs().max().item())
    engine.model.eval()
    if gradient_parameters == 0:
        raise RuntimeError("Backward pass produced no parameter gradients.")
    if delta == 0.0:
        raise RuntimeError("Optimizer step did not change the probed parameter.")
    return float(loss.detach().cpu().item()), gradient_parameters, delta


def run_training_smoke(
    records: Sequence[dict[str, Any]],
    *,
    models: Sequence[str],
    language: str,
    variant: str,
    repository_root: str | Path,
    pretrained_model_root: str | Path,
    output_model_root: str | Path = "models",
    base_model_root: str | Path | None = None,
    run_root: str | Path = "models/_runs",
    run_id: str = "training-smoke",
    device: str = "auto",
    learning_rate: float = 1e-5,
    batch_size: int = 1,
    reload_check: bool = True,
    resume: bool = True,
    skip_unavailable: bool | None = None,
) -> Path:
    import torch

    if not records:
        raise ValueError("Training smoke test needs at least one labeled record.")
    for index, record in enumerate(records):
        if record.get("sentiment") not in LABEL_TO_ID:
            raise ValueError(f"Record {index} needs sentiment -1, 0, or 1.")
    if batch_size < 1:
        raise ValueError("batch_size must be at least 1")
    batch = list(records[:batch_size])
    language = normalize_language(language)
    specs = select_models(models, language=language, variant=variant)
    requested_all = len(models) == 1 and models[0].lower() == "all"
    if skip_unavailable is None:
        skip_unavailable = requested_all
    layout = RunLayout("training-smoke", run_id, run_root=run_root, resume=resume)
    logger = layout.logger("training-smoke")
    results: list[dict[str, Any]] = []
    layout.write_manifest(
        {
            "models": [spec.name for spec in specs],
            "language": language,
            "variant": variant,
            "batch_size": len(batch),
            "learning_rate": learning_rate,
        }
    )
    for index, spec in enumerate(specs):
        checkpoint_dir = (
            Path(output_model_root)
            / spec.name
            / language
            / variant
            / "smoke"
            / run_id
        )
        checkpoint = checkpoint_dir / "best-model.pt"
        report_path = checkpoint_dir / "smoke-report.json"
        if resume and checkpoint.is_file() and report_path.is_file():
            report = json.loads(report_path.read_text(encoding="utf-8"))
            results.append(report)
            logger.info("Resuming %s from completed checkpoint %s", spec.name, checkpoint)
            continue
        status = checkpoint_status(
            repository_root, pretrained_model_root, spec.name, language, variant
        )
        if not status["available"]:
            report = {"model": spec.name, "status": "skipped", "reason": status["reason"]}
            results.append(report)
            logger.warning("Skipping %s: %s", spec.name, status["reason"])
            if not skip_unavailable:
                raise FileNotFoundError(status["reason"])
            continue
        engine = create_engine(
            repository_root=repository_root,
            model_root=pretrained_model_root,
            base_model_root=base_model_root,
            model=spec.name,
            language=language,
            variant=variant,
            device=device,
        )
        loss, gradient_parameters, max_parameter_delta = _one_update(
            engine, batch, learning_rate
        )
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        torch.save(engine.model.state_dict(), checkpoint)
        saved_bytes = checkpoint.stat().st_size
        del engine
        reload_prediction = None
        if reload_check:
            reloaded = create_engine(
                repository_root=repository_root,
                model_root=pretrained_model_root,
                base_model_root=base_model_root,
                model=spec.name,
                language=language,
                variant=variant,
                device=device,
            )
            try:
                state = torch.load(checkpoint, map_location="cpu", weights_only=True)
            except TypeError:
                state = torch.load(checkpoint, map_location="cpu")
            reloaded.model.load_state_dict(state, strict=True)
            reload_prediction = reloaded.predict(batch[0], mc_passes=0)[
                "predicted_sentiment"
            ]
            del reloaded, state
        report = {
            "schema_version": 1,
            "model": spec.name,
            "language": language,
            "variant": variant,
            "status": "complete",
            "optimizer_steps": 1,
            "loss": loss,
            "gradient_parameter_count": gradient_parameters,
            "max_probe_parameter_delta": max_parameter_delta,
            "weight_update_observed": max_parameter_delta > 0.0,
            "checkpoint": str(checkpoint.resolve()),
            "checkpoint_bytes": saved_bytes,
            "reload_check": reload_check,
            "reload_prediction": reload_prediction,
        }
        if not math.isfinite(loss):
            raise RuntimeError("Smoke loss is non-finite after update.")
        atomic_json(report_path, report)
        results.append(report)
        logger.info("%s update and checkpoint reload complete", spec.name)
        layout.update_progress(
            status="running", completed_models=index + 1, total_models=len(specs)
        )
    summary = layout.root / "training-smoke-report.json"
    atomic_json(summary, results)
    layout.update_progress(status="complete", results=results)
    return summary
