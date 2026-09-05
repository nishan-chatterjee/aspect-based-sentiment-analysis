"""Resumable BGE-M3 dense-embedding + MLP retraining and release promotion."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from datetime import datetime, timezone
import gc
import hashlib
import json
import math
from pathlib import Path
import re
from statistics import fmean, pstdev
from typing import Any

import numpy as np

from ..evaluation import build_evaluation_report
from ..evaluation.metrics import classification_metrics
from ..runtime.runs import atomic_json, record_id


MODEL_NAME = "BAAI/bge-m3"
DEFAULT_BGE_REVISION = "5617a9f61b028005a4858fdac845db406aefb181"
EMBEDDING_DIMENSION = 1024
TRANSFORM_VERSION = "historical-bge-m3-mlp-v1"
ASPECT_PATTERN = re.compile(r"<aspect>(.*?)</aspect>", flags=re.DOTALL)
RELEASE_LANGUAGES = {"hbs": "hbs", "sl": "slovenian"}
DATA_FILENAMES = {
    "hbs": ("hbs_train_val_{split}.json", "hbs_test.json"),
    "sl": ("slovene_train_val_{split}.json", "slovene_test.json"),
}
PAPER_REFERENCE = {
    ("hbs", "masked"): {
        "validation_macro_f1": 0.8840815245920188,
        "selected_split": 2,
        "test_mean": {"accuracy": 0.8405, "f1_macro": 0.7893, "qwk": 0.761},
    },
    ("hbs", "unmasked"): {
        "validation_macro_f1": 0.8894579185247774,
        "selected_split": 1,
        "test_mean": {"accuracy": 0.8395, "f1_macro": 0.7876, "qwk": 0.759},
    },
    ("sl", "masked"): {
        "validation_macro_f1": 0.7514188081139693,
        "selected_split": 1,
        "test_mean": {"accuracy": 0.9039, "f1_macro": 0.6721, "qwk": 0.653},
    },
    ("sl", "unmasked"): {
        "validation_macro_f1": 0.7393257242124599,
        "selected_split": 1,
        "test_mean": {"accuracy": 0.9017, "f1_macro": 0.6839, "qwk": 0.640},
    },
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def transform_article(article: str, variant: str) -> str:
    """Reproduce the historical inputs consumed by the released BGE engine."""

    if not isinstance(article, str) or not article.strip():
        raise ValueError("Every record needs a non-empty article.")
    matches = ASPECT_PATTERN.findall(article)
    if not matches:
        raise ValueError("Every article must contain at least one <aspect> tag.")
    if variant == "masked":
        return f"{ASPECT_PATTERN.sub('[ASPECT_MENTION]', article).strip()} [ASPECT_NAME]"
    if variant == "unmasked":
        return article.replace("<aspect>", "").replace("</aspect>", "").strip()
    raise ValueError("variant must be masked or unmasked")


def _load_json(path: Path, key: str) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = payload.get(key) if isinstance(payload, dict) else None
    if not isinstance(rows, list) or not rows:
        raise ValueError(f"{path} needs a non-empty {key!r} list.")
    for row in rows:
        if row.get("sentiment") not in (-1, 0, 1):
            raise ValueError(f"Every record in {path} needs sentiment -1, 0, or 1.")
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


def _file_fingerprint(path: Path) -> dict[str, Any]:
    stat = path.stat()
    return {
        "path": str(path.resolve()),
        "size": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
    }


def _configuration_fingerprint(payload: dict[str, Any]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _atomic_npz(path: Path, *, ids: list[str], embeddings: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("wb") as handle:
        np.savez(handle, ids=np.asarray(ids), embeddings=embeddings)
    temporary.replace(path)


def _unique_records(rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    unique: dict[str, dict[str, Any]] = {}
    for row in rows:
        identifier = record_id(row)
        existing = unique.get(identifier)
        if existing is not None and (
            existing.get("article") != row.get("article")
            or existing.get("sentiment") != row.get("sentiment")
        ):
            raise ValueError(f"Conflicting records share identifier {identifier!r}.")
        unique.setdefault(identifier, row)
    return list(unique.values())


def _load_encoder(
    base_model: str,
    *,
    revision: str | None,
    device: str,
    max_length: int,
    precision: str,
):
    from sentence_transformers import SentenceTransformer

    kwargs: dict[str, Any] = {"device": device}
    if revision:
        kwargs["revision"] = revision
    encoder = SentenceTransformer(base_model, **kwargs)
    encoder.max_seq_length = max_length
    if precision == "float16":
        encoder.half()
    elif precision == "bfloat16":
        encoder.bfloat16()
    elif precision != "float32":
        raise ValueError("embedding precision must be float32, float16, or bfloat16")
    return encoder


def build_embedding_cache(
    records: Sequence[dict[str, Any]],
    *,
    dataset: str,
    variant: str,
    cache_dir: Path,
    base_model: str,
    revision: str | None,
    device: str,
    max_length: int,
    precision: str,
    batch_size: int,
    shard_size: int,
    data_fingerprint: list[dict[str, Any]],
    resume: bool,
) -> None:
    if batch_size < 1 or shard_size < 1:
        raise ValueError("embedding batch and shard sizes must be positive")
    records = _unique_records(records)
    identifiers = [record_id(row) for row in records]
    config = {
        "schema_version": 1,
        "dataset": dataset,
        "variant": variant,
        "transform_version": TRANSFORM_VERSION,
        "base_model": base_model,
        "revision": revision,
        "max_length": max_length,
        "precision": precision,
        "shard_size": shard_size,
        "embedding_dimension": EMBEDDING_DIMENSION,
        "record_count": len(records),
        "record_ids_sha256": hashlib.sha256("\n".join(identifiers).encode()).hexdigest(),
        "data_files": data_fingerprint,
    }
    config["configuration_sha256"] = _configuration_fingerprint(config)
    manifest_path = cache_dir / "manifest.json"
    if manifest_path.is_file():
        existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        if existing.get("configuration_sha256") != config["configuration_sha256"]:
            raise ValueError(
                f"Embedding cache configuration changed at {cache_dir}; use another "
                "--cache-root or remove that specific cache directory."
            )
        if not resume:
            raise FileExistsError(f"Embedding cache already exists: {cache_dir}")
    else:
        cache_dir.mkdir(parents=True, exist_ok=True)
        atomic_json(manifest_path, {**config, "created_at": utc_now()})

    shard_specs: list[tuple[int, list[dict[str, Any]], list[str], Path]] = []
    for start in range(0, len(records), shard_size):
        index = start // shard_size
        shard_records = list(records[start : start + shard_size])
        shard_ids = identifiers[start : start + shard_size]
        path = cache_dir / "shards" / f"shard-{index:06d}.npz"
        if path.is_file():
            with np.load(path, allow_pickle=False) as payload:
                stored_ids = payload["ids"].astype(str).tolist()
                shape = payload["embeddings"].shape
            if stored_ids != shard_ids or shape != (len(shard_ids), EMBEDDING_DIMENSION):
                raise ValueError(f"Invalid or stale embedding shard: {path}")
            continue
        shard_specs.append((index, shard_records, shard_ids, path))

    if not shard_specs:
        atomic_json(cache_dir / "progress.json", {"status": "complete", "shards": math.ceil(len(records) / shard_size), "records": len(records), "updated_at": utc_now()})
        return

    encoder = _load_encoder(
        base_model,
        revision=revision,
        device=device,
        max_length=max_length,
        precision=precision,
    )
    try:
        completed = math.ceil(len(records) / shard_size) - len(shard_specs)
        for index, shard_records, shard_ids, path in shard_specs:
            texts = [transform_article(row["article"], variant) for row in shard_records]
            order = sorted(range(len(texts)), key=lambda item: len(texts[item]))
            sorted_embeddings = encoder.encode(
                [texts[item] for item in order],
                batch_size=batch_size,
                show_progress_bar=False,
                convert_to_numpy=True,
                normalize_embeddings=True,
            )
            sorted_embeddings = np.asarray(sorted_embeddings, dtype=np.float32)
            if sorted_embeddings.shape != (len(texts), EMBEDDING_DIMENSION):
                raise ValueError(
                    f"BGE returned {sorted_embeddings.shape}; expected "
                    f"({len(texts)}, {EMBEDDING_DIMENSION})."
                )
            embeddings = np.empty_like(sorted_embeddings)
            embeddings[order] = sorted_embeddings
            _atomic_npz(path, ids=shard_ids, embeddings=embeddings)
            completed += 1
            atomic_json(
                cache_dir / "progress.json",
                {
                    "status": "running",
                    "completed_shards": completed,
                    "total_shards": math.ceil(len(records) / shard_size),
                    "last_shard": index,
                    "updated_at": utc_now(),
                },
            )
            print(
                f"[{dataset}/{variant}] embedded shard {index + 1}/"
                f"{math.ceil(len(records) / shard_size)}",
                flush=True,
            )
    finally:
        del encoder
        gc.collect()
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass
    atomic_json(
        cache_dir / "progress.json",
        {
            "status": "complete",
            "shards": math.ceil(len(records) / shard_size),
            "records": len(records),
            "updated_at": utc_now(),
        },
    )


def load_embedding_cache(cache_dir: Path) -> tuple[np.ndarray, dict[str, int]]:
    manifest = json.loads((cache_dir / "manifest.json").read_text(encoding="utf-8"))
    arrays: list[np.ndarray] = []
    identifiers: list[str] = []
    for path in sorted((cache_dir / "shards").glob("shard-*.npz")):
        with np.load(path, allow_pickle=False) as payload:
            identifiers.extend(payload["ids"].astype(str).tolist())
            arrays.append(np.asarray(payload["embeddings"], dtype=np.float32))
    if not arrays:
        raise FileNotFoundError(f"No embedding shards found in {cache_dir}")
    embeddings = np.concatenate(arrays, axis=0)
    if len(identifiers) != manifest["record_count"] or embeddings.shape != (
        manifest["record_count"],
        EMBEDDING_DIMENSION,
    ):
        raise ValueError(f"Embedding cache is incomplete: {cache_dir}")
    lookup = {identifier: index for index, identifier in enumerate(identifiers)}
    if len(lookup) != len(identifiers):
        raise ValueError(f"Embedding cache contains duplicate identifiers: {cache_dir}")
    return embeddings, lookup


def _torch_load(path: Path, *, weights_only: bool = False) -> Any:
    import torch

    try:
        return torch.load(path, map_location="cpu", weights_only=weights_only)
    except TypeError:
        return torch.load(path, map_location="cpu")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_torch_save(payload: Any, path: Path) -> None:
    import torch

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def _make_model(hidden_dim1: int, hidden_dim2: int, dropout: float):
    import torch.nn as nn

    if hidden_dim1 < 1 or hidden_dim2 < 1 or not 0 <= dropout < 1:
        raise ValueError("MLP dimensions must be positive and dropout must be in [0, 1).")
    return nn.Sequential(
        nn.Linear(EMBEDDING_DIMENSION, hidden_dim1),
        nn.ReLU(),
        nn.Dropout(dropout),
        nn.Linear(hidden_dim1, hidden_dim2),
        nn.ReLU(),
        nn.Dropout(dropout),
        nn.Linear(hidden_dim2, 3),
    )


def _tensors_for_records(
    records: Sequence[dict[str, Any]], embeddings: np.ndarray, lookup: dict[str, int]
):
    import torch

    try:
        indices = [lookup[record_id(row)] for row in records]
    except KeyError as exc:
        raise ValueError(f"Record {exc.args[0]!r} is absent from the embedding cache.") from exc
    features = torch.from_numpy(np.asarray(embeddings[indices], dtype=np.float32))
    labels = torch.tensor([int(row["sentiment"]) + 1 for row in records], dtype=torch.long)
    return features, labels


def _evaluate_model(model: Any, features: Any, labels: Any, *, device: Any, batch_size: int):
    import torch
    import torch.nn.functional as functional

    model.eval()
    predictions: list[int] = []
    probabilities: list[list[float]] = []
    total_loss = 0.0
    with torch.inference_mode():
        for start in range(0, len(labels), batch_size):
            batch_features = features[start : start + batch_size].to(
                device, non_blocking=True
            )
            batch_labels = labels[start : start + batch_size].to(
                device, non_blocking=True
            )
            logits = model(batch_features)
            total_loss += float(
                functional.cross_entropy(logits, batch_labels, reduction="sum").cpu()
            )
            probs = torch.softmax(logits, dim=-1).cpu()
            probabilities.extend(probs.tolist())
            predictions.extend((probs.argmax(dim=-1) - 1).tolist())
    gold = (labels - 1).tolist()
    metrics = classification_metrics(gold, predictions)
    metrics["loss"] = total_loss / len(labels)
    return metrics, predictions, probabilities


def train_split(
    train_records: Sequence[dict[str, Any]],
    validation_records: Sequence[dict[str, Any]],
    test_records: Sequence[dict[str, Any]],
    *,
    embeddings: np.ndarray,
    lookup: dict[str, int],
    split_index: int,
    output_dir: Path,
    device: str,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    weight_decay: float,
    hidden_dim1: int,
    hidden_dim2: int,
    dropout: float,
    seed: int,
    resume: bool,
) -> dict[str, Any]:
    import torch
    import torch.nn.functional as functional
    from torch.utils.data import DataLoader, TensorDataset

    if epochs < 1 or batch_size < 1:
        raise ValueError("epochs and MLP batch size must be positive")
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / "training-report.json"
    success_path = output_dir / "_SUCCESS.json"
    if success_path.is_file() and resume:
        report = json.loads(report_path.read_text(encoding="utf-8"))
        if int(report["epochs_requested"]) == epochs:
            print(f"Split {split_index}: already complete; reusing {report_path}", flush=True)
            return report

    train_features, train_labels = _tensors_for_records(train_records, embeddings, lookup)
    val_features, val_labels = _tensors_for_records(validation_records, embeddings, lookup)
    test_features, test_labels = _tensors_for_records(test_records, embeddings, lookup)
    torch_device = torch.device(device)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    model = _make_model(hidden_dim1, hidden_dim2, dropout).to(torch_device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.2, patience=2
    )
    best_path = output_dir / "best-model.pt"
    last_path = output_dir / "last-training-state.pt"
    start_epoch = 0
    best_f1 = -1.0
    history: list[dict[str, Any]] = []
    if last_path.is_file() and resume:
        state = _torch_load(last_path)
        model.load_state_dict(state["model_state_dict"], strict=True)
        optimizer.load_state_dict(state["optimizer_state_dict"])
        scheduler.load_state_dict(state["scheduler_state_dict"])
        start_epoch = int(state["epoch"]) + 1
        best_f1 = float(state["best_validation_macro_f1"])
        history = list(state["history"])
        print(f"Split {split_index}: resuming at epoch {start_epoch + 1}", flush=True)

    dataset = TensorDataset(train_features, train_labels)
    for epoch in range(start_epoch, epochs):
        torch.manual_seed(seed + epoch)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed + epoch)
        generator = torch.Generator().manual_seed(seed + epoch)
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            generator=generator,
            num_workers=0,
            pin_memory=torch_device.type == "cuda",
        )
        model.train()
        total_loss = 0.0
        seen = 0
        for batch_features, batch_labels in loader:
            batch_features = batch_features.to(torch_device, non_blocking=True)
            batch_labels = batch_labels.to(torch_device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            logits = model(batch_features)
            loss = functional.cross_entropy(logits, batch_labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += float(loss.detach().cpu()) * len(batch_labels)
            seen += len(batch_labels)
        validation, _, _ = _evaluate_model(
            model,
            val_features,
            val_labels,
            device=torch_device,
            batch_size=batch_size * 2,
        )
        scheduler.step(validation["loss"])
        epoch_row = {
            "epoch": epoch + 1,
            "training_loss": total_loss / seen,
            "learning_rate": float(optimizer.param_groups[0]["lr"]),
            "validation": validation,
        }
        history.append(epoch_row)
        if float(validation["f1_macro"]) > best_f1:
            best_f1 = float(validation["f1_macro"])
            _atomic_torch_save(
                {
                    "epoch": epoch + 1,
                    "model_state_dict": model.state_dict(),
                    "validation": validation,
                    "split_index": split_index,
                    "architecture": {
                        "input_dim": EMBEDDING_DIMENSION,
                        "hidden_dim1": hidden_dim1,
                        "hidden_dim2": hidden_dim2,
                        "dropout": dropout,
                    },
                },
                best_path,
            )
        _atomic_torch_save(
            {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "best_validation_macro_f1": best_f1,
                "history": history,
            },
            last_path,
        )
        atomic_json(
            report_path,
            {
                "status": "training",
                "split_index": split_index,
                "epochs_requested": epochs,
                "best_validation_macro_f1": best_f1,
                "history": history,
            },
        )
        print(
            f"Split {split_index}, epoch {epoch + 1}/{epochs}: "
            f"loss={total_loss / seen:.6f}, val_macro_f1={validation['f1_macro']:.6f}, "
            f"val_qwk={validation['qwk']}",
            flush=True,
        )

    best = _torch_load(best_path)
    model.load_state_dict(best["model_state_dict"], strict=True)
    test_metrics, predictions, probabilities = _evaluate_model(
        model,
        test_features,
        test_labels,
        device=torch_device,
        batch_size=batch_size * 2,
    )
    prediction_rows = []
    for row, prediction, probs in zip(
        test_records, predictions, probabilities, strict=True
    ):
        prediction_rows.append(
            {
                "record_id": record_id(row),
                "aspect": row.get("aspect"),
                "sentiment": row["sentiment"],
                "prediction": prediction,
                "probabilities": {
                    "-1 (negative)": probs[0],
                    "0 (neutral)": probs[1],
                    "1 (positive)": probs[2],
                },
            }
        )
    atomic_json(output_dir / "test-predictions.json", prediction_rows)
    evaluation = build_evaluation_report(
        prediction_rows,
        training_records=train_records,
    )
    report = {
        "schema_version": 1,
        "status": "complete",
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
    del model, optimizer, scheduler, dataset
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return report


def _mean_std(reports: Sequence[dict[str, Any]], field: str) -> dict[str, float]:
    values = [float(report["test"][field]) for report in reports]
    return {"mean": fmean(values), "std": pstdev(values), "values": values}


def select_release_checkpoint(
    reports: Sequence[dict[str, Any]],
    *,
    dataset: str,
    variant: str,
    family_root: Path,
    difference_threshold: float,
) -> dict[str, Any]:
    if len(reports) != 3:
        raise ValueError("Release selection requires exactly three split reports.")
    selected = max(reports, key=lambda row: float(row["best_validation_macro_f1"]))
    best_checkpoint = Path(selected["best_checkpoint"])
    checkpoint = _torch_load(best_checkpoint)
    state = checkpoint["model_state_dict"]
    release_language = RELEASE_LANGUAGES[dataset]
    release_path = family_root / release_language / f"{variant}.pt"
    _atomic_torch_save(state, release_path)
    verified = _torch_load(release_path, weights_only=True)
    model = _make_model(
        int(checkpoint["architecture"]["hidden_dim1"]),
        int(checkpoint["architecture"]["hidden_dim2"]),
        float(checkpoint["architecture"]["dropout"]),
    )
    cleaned = {key.removeprefix("classifier."): value for key, value in verified.items()}
    model.load_state_dict(cleaned, strict=True)

    aggregate = {
        field: _mean_std(reports, field)
        for field in ("accuracy", "precision_macro", "recall_macro", "f1_macro", "qwk")
    }
    paper = PAPER_REFERENCE[(dataset, variant)]
    deltas = {
        field: aggregate[field]["mean"] - float(reference)
        for field, reference in paper["test_mean"].items()
    }
    comparison = {
        "absolute_difference_threshold": difference_threshold,
        "deltas_new_minus_paper": deltas,
        "material_difference": any(abs(value) > difference_threshold for value in deltas.values()),
    }
    return {
        "schema_version": 1,
        "model": "bge-m3-mlp",
        "dataset": dataset,
        "release_language": release_language,
        "variant": variant,
        "selection_rule": "highest validation Macro-F1 among the three fixed train/validation splits; test metrics unused",
        "selected_split": int(selected["split_index"]),
        "selected_validation_macro_f1": float(selected["best_validation_macro_f1"]),
        "selected_checkpoint": str(best_checkpoint.resolve()),
        "release_checkpoint": str(release_path.resolve()),
        "release_checkpoint_sha256": _sha256(release_path),
        "test_across_three_splits": aggregate,
        "paper_reference": paper,
        "comparison_to_paper": comparison,
        "completed_at": utc_now(),
    }


def run_training(args: argparse.Namespace) -> Path:
    dataset = args.dataset
    variant = args.variant
    split_indices = tuple(args.split_indices)
    if split_indices != (0, 1, 2):
        raise ValueError("The release recovery run requires --split-indices 0 1 2.")
    split_paths, test_path = data_paths(args.data_root, dataset, split_indices)
    first_payload = json.loads(split_paths[0].read_text(encoding="utf-8"))
    universe = _unique_records([*first_payload["train"], *first_payload["val"]])
    universe_ids = {record_id(row) for row in universe}
    for path in split_paths[1:]:
        payload = json.loads(path.read_text(encoding="utf-8"))
        current = {record_id(row) for row in [*payload["train"], *payload["val"]]}
        if current != universe_ids:
            raise ValueError(f"Train/validation universe differs in {path}.")
        del payload, current
    test_records = _load_json(test_path, "test")
    cache_records = _unique_records([*universe, *test_records])
    family_root = Path(args.output_root)
    cache_dir = Path(args.cache_root) / dataset / variant
    fingerprints = [_file_fingerprint(path) for path in [*split_paths, test_path]]
    build_embedding_cache(
        cache_records,
        dataset=dataset,
        variant=variant,
        cache_dir=cache_dir,
        base_model=args.base_model,
        revision=args.revision,
        device=args.device,
        max_length=args.max_length,
        precision=args.embedding_precision,
        batch_size=args.embedding_batch_size,
        shard_size=args.embedding_shard_size,
        data_fingerprint=fingerprints,
        resume=args.resume,
    )
    del cache_records, universe, first_payload
    gc.collect()
    embeddings, lookup = load_embedding_cache(cache_dir)
    run_root = family_root / "training" / "runs" / args.run_id / RELEASE_LANGUAGES[dataset] / variant
    config = {
        "schema_version": 1,
        "model": "bge-m3-mlp",
        "base_model": args.base_model,
        "revision": args.revision,
        "dataset": dataset,
        "variant": variant,
        "split_indices": list(split_indices),
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "hidden_dim1": args.hidden_dim1,
        "hidden_dim2": args.hidden_dim2,
        "dropout": args.dropout,
        "seed": args.seed,
        "embedding_cache": str(cache_dir.resolve()),
    }
    config["configuration_sha256"] = _configuration_fingerprint(config)
    manifest_path = run_root / "manifest.json"
    if manifest_path.is_file():
        existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        if existing.get("configuration_sha256") != config["configuration_sha256"]:
            raise ValueError(f"Training configuration changed for existing run: {run_root}")
        if not args.resume:
            raise FileExistsError(run_root)
    else:
        atomic_json(manifest_path, {**config, "created_at": utc_now()})

    reports = []
    for split_index, path in zip(split_indices, split_paths, strict=True):
        payload = json.loads(path.read_text(encoding="utf-8"))
        train_records = payload["train"]
        validation_records = payload["val"]
        report = train_split(
            train_records,
            validation_records,
            test_records,
            embeddings=embeddings,
            lookup=lookup,
            split_index=split_index,
            output_dir=run_root / f"split-{split_index}",
            device=args.device,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            hidden_dim1=args.hidden_dim1,
            hidden_dim2=args.hidden_dim2,
            dropout=args.dropout,
            seed=args.seed + split_index,
            resume=args.resume,
        )
        reports.append(report)
        del payload, train_records, validation_records
        gc.collect()
    selection = select_release_checkpoint(
        reports,
        dataset=dataset,
        variant=variant,
        family_root=family_root,
        difference_threshold=args.difference_threshold,
    )
    selection_path = run_root / "selection.json"
    atomic_json(selection_path, selection)
    atomic_json(run_root / "_SUCCESS.json", {"complete": True, "updated_at": utc_now()})
    print(f"Promoted selected checkpoint to {selection['release_checkpoint']}", flush=True)
    print(f"Selection report: {selection_path.resolve()}", flush=True)
    return selection_path


def _format_score(value: float | None) -> str:
    return "—" if value is None else f"{value:.4f}"


def finalize_release(
    *, repository_root: Path, run_id: str, require_complete: bool = True
) -> Path:
    family_root = repository_root / "huggingface" / "models" / "bge-m3-mlp"
    selections: dict[tuple[str, str], dict[str, Any]] = {}
    missing = []
    for dataset, release_language in RELEASE_LANGUAGES.items():
        for variant in ("masked", "unmasked"):
            path = (
                family_root
                / "training"
                / "runs"
                / run_id
                / release_language
                / variant
                / "selection.json"
            )
            if not path.is_file():
                missing.append(str(path))
                continue
            selection = json.loads(path.read_text(encoding="utf-8"))
            checkpoint = family_root / release_language / f"{variant}.pt"
            if not checkpoint.is_file() or _sha256(checkpoint) != selection["release_checkpoint_sha256"]:
                raise ValueError(f"Release checkpoint mismatch for {dataset}/{variant}.")
            selections[(dataset, variant)] = selection
    if missing and require_complete:
        raise FileNotFoundError("Missing completed grid selections: " + ", ".join(missing))

    availability_path = family_root / "availability.json"
    availability = json.loads(availability_path.read_text(encoding="utf-8"))
    for entry in availability["entries"]:
        dataset = "sl" if entry["language"] == "slovenian" else "hbs"
        selection = selections.get((dataset, entry["mode"]))
        if selection is None:
            continue
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

    manifest_path = repository_root / "huggingface" / "models" / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    for entry in manifest["entries"]:
        if entry["model"] != "bge-m3-mlp":
            continue
        dataset = "sl" if entry["language"] == "slovenian" else "hbs"
        selection = selections.get((dataset, entry["mode"]))
        if selection is None:
            continue
        entry.update(
            available=True,
            unavailable_reason=None,
            status="available_retrained",
            validation_macro_f1=selection["selected_validation_macro_f1"],
            run=selection["selected_split"],
            sha256=selection["release_checkpoint_sha256"],
            size=Path(selection["release_checkpoint"]).stat().st_size,
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
            row["comparison_to_paper"]["material_difference"] for row in selections.values()
        ),
        "generated_at": utc_now(),
    }
    comparison_path = family_root / "training" / "runs" / run_id / "comparison-to-paper.json"
    atomic_json(comparison_path, comparison)

    rows = []
    for dataset, release_language in RELEASE_LANGUAGES.items():
        for variant in ("masked", "unmasked"):
            selection = selections.get((dataset, variant))
            if selection is None:
                rows.append(f"| {release_language} | {variant} | Missing | — | — | — |")
                continue
            rows.append(
                f"| {release_language} | {variant} | Available (retrained) | "
                f"{_format_score(selection['selected_validation_macro_f1'])} | "
                f"{_format_score(selection['test_across_three_splits']['f1_macro']['mean'])} | "
                f"{_format_score(selection['test_across_three_splits']['qwk']['mean'])} |"
            )
    card = f"""---
library_name: pytorch
tags:
- aspect-based-sentiment-analysis
- south-slavic
- text-classification
license: other
---

# AspectBench BGE-M3 dense + MLP

Selected model-only heads for normalized 1024-dimensional `BAAI/bge-m3`
document embeddings. Each released head is the best validation Macro-F1 result
among three fixed train/validation splits; test results were not used for
selection. The shared inference toolkit reconstructs the 1024→512→256→3 MLP.

| Language | Mode | Status | Selected validation Macro-F1 | Mean test Macro-F1 (3 splits) | Mean test QWK (3 splits) |
|---|---|---|---:|---:|---:|
{chr(10).join(rows)}

Masked training reproduces the historical implementation used for the paper:
tagged mentions become `[ASPECT_MENTION]` and `[ASPECT_NAME]` is appended.
Unmasked training removes the literal XML-like aspect tags. Complete metrics,
per-class results, seen/unseen reports, seeds, and paper deltas are retained in
the private ignored training run before upload.

## Use

Download this repository beneath the toolkit at
`huggingface/models/bge-m3-mlp/`, then run `aspectbench infer --models
bge-m3-mlp ...`. The checkpoint files contain tensors only—no optimizer state,
dataset rows, or cached embeddings.

See the shared toolkit at
[`nishan-chatterjee/aspect-based-sentiment-analysis`](https://huggingface.co/nishan-chatterjee/aspect-based-sentiment-analysis)
for input examples, uncertainty output, and validation commands.
"""
    (family_root / "README.md").write_text(card, encoding="utf-8")
    print(f"Updated release metadata and model card; comparison: {comparison_path}")
    return comparison_path


def build_train_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=("hbs", "sl"), required=True)
    parser.add_argument("--variant", choices=("masked", "unmasked"), required=True)
    parser.add_argument("--split-indices", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--data-root", default="data")
    parser.add_argument("--output-root", default="huggingface/models/bge-m3-mlp")
    parser.add_argument("--cache-root", default="huggingface/models/bge-m3-mlp/training/cache")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--base-model", default=MODEL_NAME)
    parser.add_argument("--revision", default=DEFAULT_BGE_REVISION)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-length", type=int, default=8192)
    parser.add_argument("--embedding-precision", choices=("float32", "float16", "bfloat16"), default="float32")
    parser.add_argument("--embedding-batch-size", type=int, default=8)
    parser.add_argument("--embedding-shard-size", type=int, default=512)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--hidden-dim1", type=int, default=512)
    parser.add_argument("--hidden-dim2", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--difference-threshold", type=float, default=0.03)
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
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
    parser = argparse.ArgumentParser(description="Finalize all four BGE release heads.")
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
