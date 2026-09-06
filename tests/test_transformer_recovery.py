from pathlib import Path

import torch

from aspectbench.inference.hf_bridge import load_release_modules
from aspectbench.training.transformer_recovery import select_release_checkpoint, train_split


ROOT = Path(__file__).resolve().parents[1]


def _report(tmp_path, split, validation, test):
    state = {
        "xlmr.embeddings.word_embeddings.weight": torch.zeros(2, 2),
        "classifier.weight": torch.full((3, 2), float(split)),
        "classifier.bias": torch.zeros(3),
    }
    checkpoint = tmp_path / f"split-{split}.pt"
    torch.save({"model_state_dict": state}, checkpoint)
    return {
        "split_index": split,
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


def test_transformer_release_selection_uses_validation_not_test(tmp_path):
    reports = [
        _report(tmp_path, 0, 0.70, 0.99),
        _report(tmp_path, 1, 0.82, 0.70),
        _report(tmp_path, 2, 0.75, 0.80),
    ]
    selection = select_release_checkpoint(
        reports,
        family="xlmr",
        dataset="hbs",
        family_root=tmp_path / "xlmr",
        difference_threshold=0.03,
    )

    assert selection["selected_split"] == 1
    saved = torch.load(
        tmp_path / "xlmr" / "hbs" / "unmasked.pt",
        map_location="cpu",
        weights_only=True,
    )
    assert torch.all(saved["classifier.weight"] == 1)


class _Tokenizer:
    sep_token = "</s>"
    pad_token_id = 0

    def __init__(self):
        self.texts = []

    def __call__(self, text, **kwargs):
        self.texts.append(text)
        return {
            "input_ids": torch.ones(1, kwargs["max_length"], dtype=torch.long),
            "attention_mask": torch.ones(1, kwargs["max_length"], dtype=torch.long),
        }


def test_han_unmasked_keeps_name_and_removes_literal_tags():
    inference, _ = load_release_modules(ROOT)
    engine = inference.InferenceEngine.__new__(inference.InferenceEngine)
    engine.mode = "unmasked"
    engine.language = "hbs"
    engine.device = torch.device("cpu")
    engine.spec = {"max_sentences": 2, "max_length": 8}
    engine.spacy_nlp = None
    engine.tokenizer = _Tokenizer()
    prepared = [
        inference.prepare_record(
            {
                "article": "Poziv za <aspect>Primer Grupu</aspect> je uspeo.",
                "sentiment": 1,
            },
            "unmasked",
        )
    ]

    result = engine._han_inputs(prepared)

    assert result["input_ids"].shape == (1, 2, 8)
    encoded = engine.tokenizer.texts[0]
    assert "Primer Grupu" in encoded
    assert "<aspect>" not in encoded
    assert encoded.count(inference.HAN_ASPECT_TOKEN) == 1


def test_split_resume_continues_from_saved_epoch(tmp_path, monkeypatch):
    class Engine:
        def __init__(self):
            self.device = torch.device("cpu")
            self.backend = "encoder"
            self.model = torch.nn.Linear(1, 3)

    monkeypatch.setattr(
        "aspectbench.training.transformer_recovery.build_training_engine",
        lambda **kwargs: Engine(),
    )

    def fake_logits(engine, records):
        features = torch.tensor(
            [[float(row["feature"])] for row in records], dtype=torch.float32
        )
        return engine.model(features)

    monkeypatch.setattr(
        "aspectbench.training.transformer_recovery._logits", fake_logits
    )
    rows = [
        {
            "uuid": f"row-{index}",
            "article": f"Tekst <aspect>cilj-{index}</aspect>.",
            "aspect": f"cilj-{index}",
            "sentiment": (-1, 0, 1)[index % 3],
            "feature": index / 10,
        }
        for index in range(12)
    ]
    common = dict(
        train_records=rows[:6],
        validation_records=rows[6:9],
        test_records=rows[9:],
        repository_root=ROOT,
        family="xlmr",
        dataset="hbs",
        split_index=0,
        output_dir=tmp_path / "split-0",
        base_model="fake",
        base_model_root=None,
        revision=None,
        device="cpu",
        batch_size=2,
        accumulation_steps=1,
        learning_rate=1e-2,
        weight_decay=0.0,
        max_length=8,
        max_sentences=2,
        interaction_layers=1,
        interaction_heads=1,
        aggregation_heads=1,
        final_mlp_hidden_dim=2,
        dropout=0.0,
        precision="float32",
        checkpoint_every_steps=1,
        seed=42,
        resume=True,
        require_spacy=False,
    )
    first = train_split(epochs=1, **common)
    (tmp_path / "split-0" / "_SUCCESS.json").unlink()
    resumed = train_split(epochs=2, **common)

    assert first["history"][-1]["epoch"] == 1
    assert [row["epoch"] for row in resumed["history"]] == [1, 2]
    state = torch.load(
        tmp_path / "split-0" / "last-training-state.pt",
        map_location="cpu",
        weights_only=False,
    )
    assert state["epoch_index"] == 2
    assert state["next_batch_index"] == 0
