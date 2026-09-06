"""Resumable recovery of the four missing XLM-R and HAN-XLM-R release heads."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from datetime import datetime, timezone
import gc
import hashlib
import json
from pathlib import Path
import random
import re
from statistics import fmean, pstdev
from types import SimpleNamespace
from typing import Any

import numpy as np

from ..evaluation import build_evaluation_report
from ..evaluation.metrics import classification_metrics
from ..inference.hf_bridge import load_release_modules
from ..runtime.runs import atomic_json, record_id


MODEL_NAME = "xlm-roberta-base"
DEFAULT_XLMR_REVISION = "e73636d4f797dec63c3081bb6ed5c7b0bb3f2089"
FAMILIES = ("xlmr", "han-xlmr")
RELEASE_LANGUAGES = {"hbs": "hbs", "sl": "slovenian"}
DATA_FILENAMES = {
    "hbs": ("hbs_train_val_{split}.json", "hbs_test.json"),
    "sl": ("slovene_train_val_{split}.json", "slovene_test.json"),
}
PAPER_REFERENCE = {
    ("xlmr", "hbs"): {
        "validation_macro_f1": 0.9261041473729551,
        "selected_split": 1,
        "test_mean": {
            "accuracy": 0.8541683458262809,
            "f1_macro": 0.8084361977930362,
            "qwk": 0.7886420082689276,
        },
    },
    ("xlmr", "sl"): {
        "validation_macro_f1": 0.8286127750538895,
        "selected_split": 2,
        "test_mean": {
            "accuracy": 0.9028423772609818,
            "f1_macro": 0.7002955394338332,
            "qwk": 0.6589650358494209,
        },
    },
    ("han-xlmr", "hbs"): {
        "validation_macro_f1": 0.9106720097631231,
        "selected_split": 2,
        "test_mean": {
            "accuracy": 0.8386126111603666,
            "f1_macro": 0.7949805701223812,
            "qwk": 0.7642809455173875,
        },
    },
    ("han-xlmr", "sl"): {
        "validation_macro_f1": 0.8127192586145947,
        "selected_split": 2,
        "test_mean": {
            "accuracy": 0.892764857881137,
            "f1_macro": 0.701052551589123,
            "qwk": 0.6401006415597948,
        },
    },
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _fingerprint(payload: dict[str, Any]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _file_fingerprint(path: Path) -> dict[str, Any]:
    stat = path.stat()
    return {"path": str(path.resolve()), "size": stat.st_size, "mtime_ns": stat.st_mtime_ns}


def _atomic_torch_save(payload: Any, path: Path) -> None:
    import torch

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def _torch_load(path: Path, *, device: str = "cpu", weights_only: bool = False) -> Any:
    import torch

    try:
        return torch.load(path, map_location=device, weights_only=weights_only)
    except TypeError:
        return torch.load(path, map_location=device)


def _load_rows(path: Path, key: str) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = payload.get(key) if isinstance(payload, dict) else None
    if not isinstance(rows, list) or not rows:
        raise ValueError(f"{path} needs a non-empty {key!r} list.")
    for row in rows:
        if row.get("sentiment") not in (-1, 0, 1):
            raise ValueError(f"Every row in {path} needs sentiment -1, 0, or 1.")
        if "<aspect>" not in str(row.get("article", "")):
            raise ValueError(f"Every row in {path} needs an <aspect> target.")
    return rows


def data_paths(
    data_root: str | Path, dataset: str, split_indices: Sequence[int]
) -> tuple[list[Path], Path]:
    if dataset not in DATA_FILENAMES:
        raise ValueError("dataset must be hbs or sl")
    train_pattern, test_name = DATA_FILENAMES[dataset]
    root = Path(data_root) / dataset
    split_paths = [root / train_pattern.format(split=index) for index in split_indices]
    test_path = root / test_name
    missing = [path for path in [*split_paths, test_path] if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing data files: " + ", ".join(map(str, missing)))
    return split_paths, test_path


def _resolve_base_path(
    *, base_model: str, base_model_root: str | Path | None
) -> str:
    if base_model_root:
        candidate = Path(base_model_root) / "xlm-roberta-base"
        if candidate.is_dir():
            return str(candidate)
    return base_model


def build_training_engine(
    *,
    repository_root: Path,
    family: str,
    dataset: str,
    base_model: str,
    base_model_root: str | Path | None,
    revision: str | None,
    device: str,
    max_length: int,
    max_sentences: int,
    interaction_layers: int,
    interaction_heads: int,
    aggregation_heads: int,
    final_mlp_hidden_dim: int,
    dropout: float,
    require_spacy: bool,
) -> Any:
    import torch
    from transformers import AutoConfig, AutoModel, AutoTokenizer

    if family not in FAMILIES:
        raise ValueError(f"family must be one of {FAMILIES}")
    release_language = RELEASE_LANGUAGES[dataset]
    inference, registry = load_release_modules(repository_root)
    base_path = _resolve_base_path(
        base_model=base_model, base_model_root=base_model_root
    )
    local_only = Path(base_path).is_dir()
    load_kwargs: dict[str, Any] = {"local_files_only": local_only}
    if revision and not local_only:
        load_kwargs["revision"] = revision
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            base_path, use_fast=True, fix_mistral_regex=True, **load_kwargs
        )
    except TypeError:
        tokenizer = AutoTokenizer.from_pretrained(base_path, use_fast=True, **load_kwargs)
    config = AutoConfig.from_pretrained(base_path, **load_kwargs)
    pretrained = AutoModel.from_pretrained(base_path, **load_kwargs)
    if family == "xlmr":
        model = inference.XLMRTruncatedClassifier(config, base_model=pretrained)
        backend = "encoder"
        aspect_token_id = None
        spacy_nlp = None
        spec = {"backend": backend, "max_length": max_length}
    else:
        tokenizer.add_special_tokens(
            {"additional_special_tokens": [inference.HAN_ASPECT_TOKEN]}
        )
        inference._resize_embeddings(pretrained, len(tokenizer))
        model = inference.SimplifiedDARTModel(
            config,
            len(tokenizer),
            interaction_layers=interaction_layers,
            interaction_heads=interaction_heads,
            aggregation_heads=aggregation_heads,
            max_sentences=max_sentences,
            final_mlp_hidden_dim=final_mlp_hidden_dim,
            dropout_rate=dropout,
            base_model=pretrained,
        )
        backend = "han"
        aspect_token_id = tokenizer.convert_tokens_to_ids(inference.HAN_ASPECT_TOKEN)
        spacy_nlp = inference._load_spacy(release_language)
        if require_spacy and spacy_nlp is None:
            raise RuntimeError(
                "HAN paper recovery requires hr_core_news_sm/sl_core_news_sm; "
                "install the missing spaCy model or pass --no-require-spacy explicitly."
            )
        spec = {
            "backend": backend,
            "max_length": max_length,
            "max_sentences": max_sentences,
        }
    torch_device = torch.device(device)
    model = model.to(torch_device)
    engine = inference.InferenceEngine.__new__(inference.InferenceEngine)
    engine.device = torch_device
    engine.model_name = family
    engine.language = release_language
    engine.mode = "unmasked"
    engine.spec = spec
    engine.base_path = base_path
    engine.backend = backend
    engine.tokenizer = tokenizer
    engine.spacy_nlp = spacy_nlp
    engine.aspect_token_id = aspect_token_id
    engine.model = model
    return engine


def _logits(engine: Any, records: Sequence[dict[str, Any]]):
    import torch

    inference = __import__(engine.__class__.__module__, fromlist=["prepare_record"])
    prepared = [inference.prepare_record(record, "unmasked") for record in records]
    if engine.backend == "encoder":
        output = engine.model(**engine._encoder_inputs(prepared))
        return output if torch.is_tensor(output) else output.logits
    return engine.model(
        **engine._han_inputs(prepared),
        aspect_target_token_id=torch.tensor(
            [engine.aspect_token_id], dtype=torch.long, device=engine.device
        ),
    )


def _evaluate(
    engine: Any, records: Sequence[dict[str, Any]], *, batch_size: int
) -> tuple[dict[str, Any], list[int], list[list[float]]]:
    import torch
    import torch.nn.functional as functional

    engine.model.eval()
    predictions: list[int] = []
    probabilities: list[list[float]] = []
    total_loss = 0.0
    with torch.inference_mode():
        for start in range(0, len(records), batch_size):
            batch = records[start : start + batch_size]
            labels = torch.tensor(
                [int(row["sentiment"]) + 1 for row in batch],
                dtype=torch.long,
                device=engine.device,
            )
            logits = _logits(engine, batch)
            total_loss += float(
                functional.cross_entropy(logits, labels, reduction="sum").cpu()
            )
            probs = torch.softmax(logits, dim=-1).float().cpu()
            probabilities.extend(probs.tolist())
            predictions.extend((probs.argmax(dim=-1) - 1).tolist())
    gold = [int(row["sentiment"]) for row in records]
    metrics = classification_metrics(gold, predictions)
    metrics["loss"] = total_loss / len(records)
    return metrics, predictions, probabilities


def _make_scaler(torch: Any, enabled: bool):
    try:
        return torch.amp.GradScaler("cuda", enabled=enabled)
    except TypeError:
        return torch.cuda.amp.GradScaler(enabled=enabled)


def _last_state(
    *,
    model: Any,
    optimizer: Any,
    scheduler: Any,
    scaler: Any,
    epoch_index: int,
    next_batch_index: int,
    global_step: int,
    best_f1: float,
    history: list[dict[str, Any]],
    epoch_loss_sum: float,
    epoch_seen: int,
) -> dict[str, Any]:
    return {
        "epoch_index": epoch_index,
        "next_batch_index": next_batch_index,
        "global_step": global_step,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "scaler_state_dict": scaler.state_dict(),
        "best_validation_macro_f1": best_f1,
        "history": history,
        "epoch_loss_sum": epoch_loss_sum,
        "epoch_seen": epoch_seen,
    }


def train_split(
    train_records: Sequence[dict[str, Any]],
    validation_records: Sequence[dict[str, Any]],
    test_records: Sequence[dict[str, Any]],
    *,
    repository_root: Path,
    family: str,
    dataset: str,
    split_index: int,
    output_dir: Path,
    base_model: str,
    base_model_root: str | Path | None,
    revision: str | None,
    device: str,
    epochs: int,
    batch_size: int,
    accumulation_steps: int,
    learning_rate: float,
    weight_decay: float,
    max_length: int,
    max_sentences: int,
    interaction_layers: int,
    interaction_heads: int,
    aggregation_heads: int,
    final_mlp_hidden_dim: int,
    dropout: float,
    precision: str,
    checkpoint_every_steps: int,
    seed: int,
    resume: bool,
    require_spacy: bool,
) -> dict[str, Any]:
    import torch
    import torch.nn.functional as functional

    if min(epochs, batch_size, accumulation_steps, checkpoint_every_steps) < 1:
        raise ValueError("epochs, batch sizes, and checkpoint interval must be positive")
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / "training-report.json"
    success_path = output_dir / "_SUCCESS.json"
    if success_path.is_file() and resume:
        report = json.loads(report_path.read_text(encoding="utf-8"))
        if int(report["epochs_requested"]) == epochs:
            print(f"{family}/{dataset}/split-{split_index}: already complete", flush=True)
            return report
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable.")
    if precision not in ("float32", "float16", "bfloat16"):
        raise ValueError("precision must be float32, float16, or bfloat16")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    engine = build_training_engine(
        repository_root=repository_root,
        family=family,
        dataset=dataset,
        base_model=base_model,
        base_model_root=base_model_root,
        revision=revision,
        device=device,
        max_length=max_length,
        max_sentences=max_sentences,
        interaction_layers=interaction_layers,
        interaction_heads=interaction_heads,
        aggregation_heads=aggregation_heads,
        final_mlp_hidden_dim=final_mlp_hidden_dim,
        dropout=dropout,
        require_spacy=require_spacy,
    )
    optimizer = torch.optim.AdamW(
        engine.model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )
    patience = 1 if family == "han-xlmr" else 2
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.2, patience=patience
    )
    amp_enabled = engine.device.type == "cuda" and precision != "float32"
    amp_dtype = torch.float16 if precision == "float16" else torch.bfloat16
    scaler = _make_scaler(torch, amp_enabled and precision == "float16")
    best_path = output_dir / "best-model.pt"
    last_path = output_dir / "last-training-state.pt"
    epoch_index = 0
    next_batch_index = 0
    global_step = 0
    best_f1 = -1.0
    history: list[dict[str, Any]] = []
    epoch_loss_sum = 0.0
    epoch_seen = 0
    if last_path.is_file() and resume:
        state = _torch_load(last_path, device=str(engine.device))
        engine.model.load_state_dict(state["model_state_dict"], strict=True)
        optimizer.load_state_dict(state["optimizer_state_dict"])
        scheduler.load_state_dict(state["scheduler_state_dict"])
        if state.get("scaler_state_dict"):
            scaler.load_state_dict(state["scaler_state_dict"])
        epoch_index = int(state["epoch_index"])
        next_batch_index = int(state["next_batch_index"])
        global_step = int(state["global_step"])
        best_f1 = float(state["best_validation_macro_f1"])
        history = list(state["history"])
        epoch_loss_sum = float(state.get("epoch_loss_sum", 0.0))
        epoch_seen = int(state.get("epoch_seen", 0))
        print(
            f"{family}/{dataset}/split-{split_index}: resume epoch "
            f"{epoch_index + 1}, batch {next_batch_index}",
            flush=True,
        )

    while epoch_index < epochs:
        generator = torch.Generator().manual_seed(seed + epoch_index)
        order = torch.randperm(len(train_records), generator=generator).tolist()
        batches = [
            order[start : start + batch_size]
            for start in range(0, len(order), batch_size)
        ]
        engine.model.train()
        optimizer.zero_grad(set_to_none=True)
        for batch_index in range(next_batch_index, len(batches)):
            rows = [train_records[index] for index in batches[batch_index]]
            labels = torch.tensor(
                [int(row["sentiment"]) + 1 for row in rows],
                dtype=torch.long,
                device=engine.device,
            )
            with torch.autocast(
                device_type=engine.device.type,
                dtype=amp_dtype,
                enabled=amp_enabled,
            ):
                loss = functional.cross_entropy(_logits(engine, rows), labels)
                scaled_loss = loss / accumulation_steps
            if not torch.isfinite(loss):
                raise RuntimeError(
                    f"Non-finite loss at epoch {epoch_index + 1}, batch {batch_index}."
                )
            scaler.scale(scaled_loss).backward()
            epoch_loss_sum += float(loss.detach().cpu()) * len(rows)
            epoch_seen += len(rows)
            update = (batch_index + 1) % accumulation_steps == 0 or (
                batch_index + 1 == len(batches)
            )
            if not update:
                continue
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(engine.model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            global_step += 1
            next_batch_index = batch_index + 1
            if global_step % checkpoint_every_steps == 0:
                _atomic_torch_save(
                    _last_state(
                        model=engine.model,
                        optimizer=optimizer,
                        scheduler=scheduler,
                        scaler=scaler,
                        epoch_index=epoch_index,
                        next_batch_index=next_batch_index,
                        global_step=global_step,
                        best_f1=best_f1,
                        history=history,
                        epoch_loss_sum=epoch_loss_sum,
                        epoch_seen=epoch_seen,
                    ),
                    last_path,
                )
                atomic_json(
                    report_path,
                    {
                        "status": "training",
                        "family": family,
                        "dataset": dataset,
                        "split_index": split_index,
                        "epoch": epoch_index + 1,
                        "next_batch": next_batch_index,
                        "global_step": global_step,
                        "best_validation_macro_f1": best_f1,
                    },
                )
        validation, _, _ = _evaluate(
            engine,
            validation_records,
            batch_size=max(1, batch_size),
        )
        scheduler.step(validation["loss"])
        row = {
            "epoch": epoch_index + 1,
            "training_loss": epoch_loss_sum / epoch_seen,
            "learning_rate": float(optimizer.param_groups[0]["lr"]),
            "validation": validation,
        }
        history.append(row)
        if float(validation["f1_macro"]) > best_f1:
            best_f1 = float(validation["f1_macro"])
            _atomic_torch_save(
                {
                    "epoch": epoch_index + 1,
                    "model_state_dict": engine.model.state_dict(),
                    "validation": validation,
                    "split_index": split_index,
                    "family": family,
                },
                best_path,
            )
        print(
            f"{family}/{dataset}/split-{split_index} epoch {epoch_index + 1}/{epochs}: "
            f"loss={row['training_loss']:.6f}, "
            f"val_macro_f1={validation['f1_macro']:.6f}, val_qwk={validation['qwk']}",
            flush=True,
        )
        epoch_index += 1
        next_batch_index = 0
        epoch_loss_sum = 0.0
        epoch_seen = 0
        _atomic_torch_save(
            _last_state(
                model=engine.model,
                optimizer=optimizer,
                scheduler=scheduler,
                scaler=scaler,
                epoch_index=epoch_index,
                next_batch_index=0,
                global_step=global_step,
                best_f1=best_f1,
                history=history,
                epoch_loss_sum=0.0,
                epoch_seen=0,
            ),
            last_path,
        )
        atomic_json(
            report_path,
            {
                "status": "training",
                "family": family,
                "dataset": dataset,
                "split_index": split_index,
                "epochs_requested": epochs,
                "completed_epochs": epoch_index,
                "best_validation_macro_f1": best_f1,
                "history": history,
            },
        )

    best = _torch_load(best_path, device=str(engine.device))
    engine.model.load_state_dict(best["model_state_dict"], strict=True)
    test_metrics, predictions, probabilities = _evaluate(
        engine, test_records, batch_size=max(1, batch_size)
    )
    prediction_rows = [
        {
            "record_id": record_id(source),
            "aspect": source.get("aspect"),
            "sentiment": source["sentiment"],
            "prediction": prediction,
            "probabilities": {
                "-1 (negative)": probs[0],
                "0 (neutral)": probs[1],
                "1 (positive)": probs[2],
            },
        }
        for source, prediction, probs in zip(
            test_records, predictions, probabilities, strict=True
        )
    ]
    atomic_json(output_dir / "test-predictions.json", prediction_rows)
    evaluation = build_evaluation_report(
        prediction_rows, training_records=train_records
    )
    report = {
        "schema_version": 1,
        "status": "complete",
        "family": family,
        "dataset": dataset,
        "variant": "unmasked",
        "split_index": split_index,
        "seed": seed,
        "epochs_requested": epochs,
        "best_epoch": int(best["epoch"]),
        "best_validation_macro_f1": float(best["validation"]["f1_macro"]),
        "validation_at_best": best["validation"],
        "test": test_metrics,
        "seen_unseen": evaluation["seen_unseen"],
        "best_checkpoint": str(best_path.resolve()),
        "last_training_state": str(last_path.resolve()),
        "history": history,
    }
    atomic_json(report_path, report)
    atomic_json(success_path, {"complete": True, "updated_at": utc_now()})
    del engine, optimizer, scheduler, scaler
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return report


def _metric_summary(reports: Sequence[dict[str, Any]], field: str) -> dict[str, Any]:
    values = [report["test"][field] for report in reports]
    finite = [float(value) for value in values if value is not None]
    return {
        "mean": fmean(finite) if finite else None,
        "std": pstdev(finite) if finite else None,
        "values": values,
    }


def _validate_state_keys(family: str, state: dict[str, Any]) -> None:
    required = (
        {"xlmr.embeddings.word_embeddings.weight", "classifier.weight", "classifier.bias"}
        if family == "xlmr"
        else {
            "base_model.embeddings.word_embeddings.weight",
            "global_aggregation_attention.in_proj_weight",
            "classifier.3.weight",
        }
    )
    missing = required.difference(state)
    if missing:
        raise ValueError(f"{family} selected checkpoint has incompatible keys: {sorted(missing)}")


def select_release_checkpoint(
    reports: Sequence[dict[str, Any]],
    *,
    family: str,
    dataset: str,
    family_root: Path,
    difference_threshold: float,
) -> dict[str, Any]:
    if len(reports) != 3:
        raise ValueError("Release selection requires exactly three split reports.")
    selected = max(reports, key=lambda row: float(row["best_validation_macro_f1"]))
    checkpoint = _torch_load(Path(selected["best_checkpoint"]))
    state = checkpoint["model_state_dict"]
    _validate_state_keys(family, state)
    release_language = RELEASE_LANGUAGES[dataset]
    release_path = family_root / release_language / "unmasked.pt"
    _atomic_torch_save(state, release_path)
    verified = _torch_load(release_path, weights_only=True)
    _validate_state_keys(family, verified)
    aggregate = {
        field: _metric_summary(reports, field)
        for field in ("accuracy", "precision_macro", "recall_macro", "f1_macro", "qwk")
    }
    paper = PAPER_REFERENCE[(family, dataset)]
    deltas = {
        field: float(aggregate[field]["mean"]) - float(reference)
        for field, reference in paper["test_mean"].items()
    }
    return {
        "schema_version": 1,
        "model": family,
        "dataset": dataset,
        "release_language": release_language,
        "variant": "unmasked",
        "selection_rule": "highest validation Macro-F1 among three fixed splits; test metrics unused",
        "selected_split": int(selected["split_index"]),
        "selected_validation_macro_f1": float(selected["best_validation_macro_f1"]),
        "selected_checkpoint": selected["best_checkpoint"],
        "release_checkpoint": str(release_path.resolve()),
        "release_checkpoint_sha256": _sha256(release_path),
        "test_across_three_splits": aggregate,
        "paper_reference": paper,
        "comparison_to_paper": {
            "absolute_difference_threshold": difference_threshold,
            "deltas_new_minus_paper": deltas,
            "material_difference": any(
                abs(value) > difference_threshold for value in deltas.values()
            ),
        },
        "completed_at": utc_now(),
    }


def run_training(args: argparse.Namespace) -> Path:
    split_indices = tuple(args.split_indices)
    if split_indices != (0, 1, 2):
        raise ValueError("Release recovery requires --split-indices 0 1 2.")
    split_paths, test_path = data_paths(args.data_root, args.dataset, split_indices)
    family_root = Path(args.output_root) / args.family
    release_language = RELEASE_LANGUAGES[args.dataset]
    run_root = (
        family_root / "training" / "runs" / args.run_id / release_language / "unmasked"
    )
    config = {
        "schema_version": 1,
        "family": args.family,
        "dataset": args.dataset,
        "variant": "unmasked",
        "base_model": args.base_model,
        "base_model_root": str(Path(args.base_model_root).resolve()) if args.base_model_root else None,
        "revision": args.revision,
        "split_indices": list(split_indices),
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "accumulation_steps": args.accumulation_steps,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "max_length": args.max_length,
        "max_sentences": args.max_sentences,
        "interaction_layers": args.interaction_layers,
        "interaction_heads": args.interaction_heads,
        "aggregation_heads": args.aggregation_heads,
        "final_mlp_hidden_dim": args.final_mlp_hidden_dim,
        "dropout": args.dropout,
        "precision": args.precision,
        "seed": args.seed,
        "data_files": [_file_fingerprint(path) for path in [*split_paths, test_path]],
    }
    config["configuration_sha256"] = _fingerprint(config)
    manifest_path = run_root / "manifest.json"
    if manifest_path.is_file():
        existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        if existing.get("configuration_sha256") != config["configuration_sha256"]:
            raise ValueError(
                f"Training configuration changed for {run_root}; use a new RUN_ID."
            )
        if not args.resume:
            raise FileExistsError(run_root)
    else:
        atomic_json(manifest_path, {**config, "created_at": utc_now()})
    test_records = _load_rows(test_path, "test")
    reports = []
    for split_index, split_path in zip(split_indices, split_paths, strict=True):
        payload = json.loads(split_path.read_text(encoding="utf-8"))
        reports.append(
            train_split(
                payload["train"],
                payload["val"],
                test_records,
                repository_root=Path(args.repository_root).resolve(),
                family=args.family,
                dataset=args.dataset,
                split_index=split_index,
                output_dir=run_root / f"split-{split_index}",
                base_model=args.base_model,
                base_model_root=args.base_model_root,
                revision=args.revision,
                device=args.device,
                epochs=args.epochs,
                batch_size=args.batch_size,
                accumulation_steps=args.accumulation_steps,
                learning_rate=args.learning_rate,
                weight_decay=args.weight_decay,
                max_length=args.max_length,
                max_sentences=args.max_sentences,
                interaction_layers=args.interaction_layers,
                interaction_heads=args.interaction_heads,
                aggregation_heads=args.aggregation_heads,
                final_mlp_hidden_dim=args.final_mlp_hidden_dim,
                dropout=args.dropout,
                precision=args.precision,
                checkpoint_every_steps=args.checkpoint_every_steps,
                seed=args.seed + split_index,
                resume=args.resume,
                require_spacy=args.require_spacy,
            )
        )
        gc.collect()
    selection = select_release_checkpoint(
        reports,
        family=args.family,
        dataset=args.dataset,
        family_root=family_root,
        difference_threshold=args.difference_threshold,
    )
    selection_path = run_root / "selection.json"
    atomic_json(selection_path, selection)
    atomic_json(run_root / "_SUCCESS.json", {"complete": True, "updated_at": utc_now()})
    print(f"Promoted selected checkpoint to {selection['release_checkpoint']}", flush=True)
    return selection_path


def _update_model_card(
    family_root: Path, selections: dict[str, dict[str, Any]]
) -> None:
    path = family_root / "README.md"
    text = path.read_text(encoding="utf-8")
    available = json.loads((family_root / "availability.json").read_text())["entries"]
    count = sum(bool(row["available"]) for row in available)
    text = re.sub(r"This repository contains \d/4 language-mode", f"This repository contains {count}/4 language-mode", text)
    for language in ("hbs", "slovenian"):
        selection = selections.get(language)
        if selection is None:
            continue
        score = selection["selected_validation_macro_f1"]
        text = re.sub(
            rf"^\| {language} \| unmasked \|.*$",
            f"| {language} | unmasked | Available (retrained) | {score:.4f} |",
            text,
            flags=re.MULTILINE,
        )
    marker = "## Recovery provenance"
    rows = [
        "| Language | Selected split | Validation Macro-F1 | Mean test Macro-F1 | Mean test QWK |",
        "|---|---:|---:|---:|---:|",
    ]
    for language in ("hbs", "slovenian"):
        selection = selections.get(language)
        if selection:
            rows.append(
                f"| {language} | {selection['selected_split']} | "
                f"{selection['selected_validation_macro_f1']:.4f} | "
                f"{selection['test_across_three_splits']['f1_macro']['mean']:.4f} | "
                f"{selection['test_across_three_splits']['qwk']['mean']:.4f} |"
            )
    section = (
        f"{marker}\n\nThe missing unmasked heads were retrained over all three fixed splits and "
        "selected only by validation Macro-F1. Optimizer state, logs, and row-level "
        "outputs are excluded from this model repository.\n\n"
        + "\n".join(rows)
        + "\n\n"
    )
    if marker in text:
        text = re.sub(
            rf"{re.escape(marker)}.*?(?=^## )",
            section,
            text,
            flags=re.MULTILINE | re.DOTALL,
        )
    else:
        text = text.replace("## Getting started", section + "## Getting started")
    path.write_text(text, encoding="utf-8")


def finalize_release(
    *, repository_root: Path, run_id: str, require_complete: bool = True
) -> Path:
    model_root = repository_root / "huggingface" / "models"
    selections: dict[tuple[str, str], dict[str, Any]] = {}
    missing = []
    for family in FAMILIES:
        for dataset, language in RELEASE_LANGUAGES.items():
            path = (
                model_root
                / family
                / "training"
                / "runs"
                / run_id
                / language
                / "unmasked"
                / "selection.json"
            )
            if not path.is_file():
                missing.append(str(path))
                continue
            selection = json.loads(path.read_text(encoding="utf-8"))
            checkpoint = model_root / family / language / "unmasked.pt"
            if not checkpoint.is_file() or _sha256(checkpoint) != selection["release_checkpoint_sha256"]:
                raise ValueError(f"Release checkpoint mismatch for {family}/{language}.")
            selections[(family, language)] = selection
    if missing and require_complete:
        raise FileNotFoundError("Missing completed grid selections: " + ", ".join(missing))

    manifest_path = model_root / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    for family in FAMILIES:
        availability_path = model_root / family / "availability.json"
        availability = json.loads(availability_path.read_text(encoding="utf-8"))
        family_selections: dict[str, dict[str, Any]] = {}
        for entry in availability["entries"]:
            if entry["mode"] != "unmasked":
                continue
            selection = selections.get((family, entry["language"]))
            if selection is None:
                continue
            family_selections[entry["language"]] = selection
            entry.update(
                available=True,
                unavailable_reason=None,
                status="available_retrained",
                validation_macro_f1=selection["selected_validation_macro_f1"],
                selected_split=selection["selected_split"],
                sha256=selection["release_checkpoint_sha256"],
                test_macro_f1_mean=selection["test_across_three_splits"]["f1_macro"]["mean"],
                test_qwk_mean=selection["test_across_three_splits"]["qwk"]["mean"],
            )
        atomic_json(availability_path, availability)
        _update_model_card(model_root / family, family_selections)

    for entry in manifest["entries"]:
        selection = selections.get((entry["model"], entry["language"]))
        if selection is None or entry["mode"] != "unmasked":
            continue
        checkpoint = Path(selection["release_checkpoint"])
        entry.update(
            available=True,
            unavailable_reason=None,
            status="available_retrained",
            validation_macro_f1=selection["selected_validation_macro_f1"],
            run=selection["selected_split"],
            sha256=selection["release_checkpoint_sha256"],
            size=checkpoint.stat().st_size,
        )
    manifest["available_slots"] = sum(bool(entry["available"]) for entry in manifest["entries"])
    manifest["unavailable_slots"] = manifest["expected_slots"] - manifest["available_slots"]
    atomic_json(manifest_path, manifest)
    comparison = {
        "schema_version": 1,
        "run_id": run_id,
        "selection_rule": "highest validation Macro-F1; test results compared only after selection",
        "results": list(selections.values()),
        "any_material_difference": any(
            row["comparison_to_paper"]["material_difference"]
            for row in selections.values()
        ),
        "generated_at": utc_now(),
    }
    output = model_root / "_recovery" / "runs" / run_id / "comparison-to-paper.json"
    atomic_json(output, comparison)
    print(f"Updated XLM-R/HAN release metadata; comparison: {output}")
    return output


def build_train_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository-root", default=".")
    parser.add_argument("--family", choices=FAMILIES, required=True)
    parser.add_argument("--dataset", choices=("hbs", "sl"), required=True)
    parser.add_argument("--split-indices", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--data-root", default="data")
    parser.add_argument("--output-root", default="huggingface/models")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--base-model", default=MODEL_NAME)
    parser.add_argument("--base-model-root")
    parser.add_argument("--revision", default=DEFAULT_XLMR_REVISION)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, required=True)
    parser.add_argument("--accumulation-steps", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, required=True)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--max-length", type=int, required=True)
    parser.add_argument("--max-sentences", type=int, default=128)
    parser.add_argument("--interaction-layers", type=int, default=2)
    parser.add_argument("--interaction-heads", type=int, default=8)
    parser.add_argument("--aggregation-heads", type=int, default=4)
    parser.add_argument("--final-mlp-hidden-dim", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--precision", choices=("float32", "float16", "bfloat16"), default="float16")
    parser.add_argument("--checkpoint-every-steps", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--difference-threshold", type=float, default=0.03)
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--require-spacy", action=argparse.BooleanOptionalAction, default=True)
    return parser


def train_main(argv: Sequence[str] | None = None) -> int:
    args = build_train_parser().parse_args(argv)
    try:
        run_training(args)
    except (OSError, RuntimeError, ValueError) as exc:
        print(f"error: {exc}")
        return 2
    return 0


def finalize_main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Finalize the four missing transformer heads.")
    parser.add_argument("--repository-root", default=".")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--require-complete", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args(argv)
    try:
        finalize_release(
            repository_root=Path(args.repository_root).resolve(),
            run_id=args.run_id,
            require_complete=args.require_complete,
        )
    except (OSError, RuntimeError, ValueError) as exc:
        print(f"error: {exc}")
        return 2
    return 0
