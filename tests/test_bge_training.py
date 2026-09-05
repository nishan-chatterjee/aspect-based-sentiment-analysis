import json
import hashlib

import numpy as np
import torch

from aspectbench.training.bge_m3 import (
    EMBEDDING_DIMENSION,
    _make_model,
    build_embedding_cache,
    load_embedding_cache,
    select_release_checkpoint,
    finalize_release,
    train_split,
    transform_article,
)


def test_bge_text_transforms_match_release_inference():
    article = "Poziv za <aspect>Primer Grupu</aspect> je uspeo."

    assert transform_article(article, "masked") == (
        "Poziv za [ASPECT_MENTION] je uspeo. [ASPECT_NAME]"
    )
    assert transform_article(article, "unmasked") == "Poziv za Primer Grupu je uspeo."


def test_embedding_cache_is_resumable_and_restores_record_order(tmp_path, monkeypatch):
    class Encoder:
        def encode(self, texts, **kwargs):
            rows = []
            for text in texts:
                value = 1.0 if "Kratak" in text else 2.0
                rows.append(np.full(EMBEDDING_DIMENSION, value, dtype=np.float32))
            return np.stack(rows)

    calls = []

    def load_encoder(*args, **kwargs):
        calls.append(True)
        return Encoder()

    monkeypatch.setattr("aspectbench.training.bge_m3._load_encoder", load_encoder)
    records = [
        {
            "uuid": "short",
            "article": "Kratak <aspect>tekst</aspect>.",
            "sentiment": 0,
        },
        {
            "uuid": "long",
            "article": "Znatno duži <aspect>primer dokumenta</aspect> za proveru.",
            "sentiment": 1,
        },
    ]
    cache = tmp_path / "cache"
    arguments = dict(
        records=records,
        dataset="hbs",
        variant="unmasked",
        cache_dir=cache,
        base_model="fake",
        revision="test",
        device="cpu",
        max_length=8192,
        precision="float32",
        batch_size=2,
        shard_size=2,
        data_fingerprint=[],
        resume=True,
    )

    build_embedding_cache(**arguments)
    build_embedding_cache(**arguments)
    embeddings, lookup = load_embedding_cache(cache)

    assert len(calls) == 1
    assert embeddings[lookup["short"], 0] == 1.0
    assert embeddings[lookup["long"], 0] == 2.0
    assert json.loads((cache / "progress.json").read_text())["status"] == "complete"


def test_release_selection_uses_validation_not_test(tmp_path):
    reports = []
    validation_scores = [0.70, 0.80, 0.75]
    test_scores = [0.95, 0.72, 0.80]
    for index, (validation, test) in enumerate(zip(validation_scores, test_scores)):
        model = _make_model(512, 256, 0.3)
        checkpoint = tmp_path / f"split-{index}.pt"
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "architecture": {
                    "hidden_dim1": 512,
                    "hidden_dim2": 256,
                    "dropout": 0.3,
                },
            },
            checkpoint,
        )
        reports.append(
            {
                "split_index": index,
                "best_validation_macro_f1": validation,
                "best_checkpoint": str(checkpoint),
                "test": {
                    "accuracy": test,
                    "precision_macro": test,
                    "recall_macro": test,
                    "f1_macro": test,
                    "qwk": test,
                },
            }
        )

    selection = select_release_checkpoint(
        reports,
        dataset="hbs",
        variant="masked",
        family_root=tmp_path / "family",
        difference_threshold=0.03,
    )

    assert selection["selected_split"] == 1
    assert selection["selected_validation_macro_f1"] == 0.80
    state = torch.load(
        tmp_path / "family" / "hbs" / "masked.pt",
        map_location="cpu",
        weights_only=True,
    )
    assert set(state) == {
        "0.weight",
        "0.bias",
        "3.weight",
        "3.bias",
        "6.weight",
        "6.bias",
    }


def test_one_epoch_training_writes_reloadable_split_report(tmp_path):
    rows = []
    for index in range(24):
        label = (-1, 0, 1)[index % 3]
        rows.append(
            {
                "uuid": f"row-{index}",
                "article": f"Dokument <aspect>entitet-{index % 4}</aspect>.",
                "aspect": f"entitet-{index % 4}",
                "sentiment": label,
            }
        )
    generator = np.random.default_rng(42)
    embeddings = generator.normal(size=(24, EMBEDDING_DIMENSION)).astype(np.float32)
    lookup = {row["uuid"]: index for index, row in enumerate(rows)}

    report = train_split(
        rows[:12],
        rows[12:18],
        rows[18:],
        embeddings=embeddings,
        lookup=lookup,
        split_index=0,
        output_dir=tmp_path / "split-0",
        device="cpu",
        epochs=1,
        batch_size=4,
        learning_rate=1e-4,
        weight_decay=0.01,
        hidden_dim1=512,
        hidden_dim2=256,
        dropout=0.3,
        seed=42,
        resume=True,
    )

    assert report["status"] == "complete"
    assert report["test"]["n"] == 6
    assert (tmp_path / "split-0" / "best-model.pt").is_file()
    resumed = train_split(
        rows[:12],
        rows[12:18],
        rows[18:],
        embeddings=embeddings,
        lookup=lookup,
        split_index=0,
        output_dir=tmp_path / "split-0",
        device="cpu",
        epochs=1,
        batch_size=4,
        learning_rate=1e-4,
        weight_decay=0.01,
        hidden_dim1=512,
        hidden_dim2=256,
        dropout=0.3,
        seed=42,
        resume=True,
    )
    assert resumed["test"] == report["test"]


def test_finalizer_updates_release_metadata_without_training_artifacts(tmp_path):
    repository = tmp_path / "repo"
    family = repository / "huggingface" / "models" / "bge-m3-mlp"
    entries = []
    manifest_entries = []
    run_id = "recovery"
    for dataset, language in (("hbs", "hbs"), ("sl", "slovenian")):
        for variant in ("masked", "unmasked"):
            checkpoint = family / language / f"{variant}.pt"
            checkpoint.parent.mkdir(parents=True, exist_ok=True)
            checkpoint.write_bytes(f"{dataset}-{variant}".encode())
            digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
            selection = {
                "dataset": dataset,
                "release_language": language,
                "variant": variant,
                "selected_split": 1,
                "selected_validation_macro_f1": 0.8,
                "release_checkpoint": str(checkpoint),
                "release_checkpoint_sha256": digest,
                "test_across_three_splits": {
                    "f1_macro": {"mean": 0.7},
                    "qwk": {"mean": 0.6},
                },
                "comparison_to_paper": {"material_difference": False},
            }
            selection_path = (
                family
                / "training"
                / "runs"
                / run_id
                / language
                / variant
                / "selection.json"
            )
            selection_path.parent.mkdir(parents=True, exist_ok=True)
            selection_path.write_text(json.dumps(selection), encoding="utf-8")
            entry = {
                "model": "bge-m3-mlp",
                "language": language,
                "mode": variant,
                "available": False,
                "unavailable_reason": "missing",
                "weight_path": f"bge-m3-mlp/{language}/{variant}.pt",
            }
            entries.append(dict(entry))
            manifest_entries.append(dict(entry))
    (family / "availability.json").write_text(
        json.dumps({"model": "bge-m3-mlp", "entries": entries}), encoding="utf-8"
    )
    manifest_path = repository / "huggingface" / "models" / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "expected_slots": 4,
                "available_slots": 0,
                "unavailable_slots": 4,
                "entries": manifest_entries,
            }
        ),
        encoding="utf-8",
    )

    comparison = finalize_release(repository_root=repository, run_id=run_id)

    availability = json.loads((family / "availability.json").read_text())
    manifest = json.loads(manifest_path.read_text())
    assert all(entry["available"] for entry in availability["entries"])
    assert manifest["available_slots"] == 4
    assert manifest["unavailable_slots"] == 0
    assert comparison.is_file()
    assert "training/" not in (family / "README.md").read_text()
