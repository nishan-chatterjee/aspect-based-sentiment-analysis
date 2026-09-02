from datetime import datetime, timezone
import json

import pytest

from aspectbench.inference import (
    build_ensemble_rows,
    publish_inference_outputs,
    resolve_output_filename,
)


def _row(model, prediction, confidence, probabilities, *, variant="masked"):
    return {
        "record_id": "doc-1",
        "model": model,
        "language": "hbs",
        "variant": variant,
        "article": "Primer teksta.",
        "aspect": "Primer",
        "sentiment": None,
        "prediction": prediction,
        "prediction_name": {-1: "negative", 0: "neutral", 1: "positive"}[prediction],
        "probabilities": probabilities,
        "uncertainty": {"confidence": confidence},
    }


def test_majority_and_confidence_votes_can_differ():
    rows = [
        _row("xlmr", 1, 0.60, {"-1 (negative)": 0.1, "0 (neutral)": 0.3, "1 (positive)": 0.6}),
        _row("longformer", 0, 0.95, {"-1 (negative)": 0.02, "0 (neutral)": 0.95, "1 (positive)": 0.03}),
        _row("mdeberta-v3", 1, 0.70, {"-1 (negative)": 0.1, "0 (neutral)": 0.2, "1 (positive)": 0.7}),
    ]

    result = build_ensemble_rows(rows)[0]

    assert result["majority_vote"]["prediction"] == 1
    assert result["majority_prediction"] == 1
    assert result["majority_vote"]["vote_counts"] == {"-1": 0, "0": 1, "1": 2}
    assert result["confidence_vote"]["prediction"] == 0
    assert result["confidence_prediction"] == 0
    assert result["confidence_vote"]["selected_model"] == "longformer"


def test_majority_tie_uses_mean_probability_then_documented_label_order():
    rows = [
        _row("a", -1, 0.51, {"-1 (negative)": 0.51, "0 (neutral)": 0.0, "1 (positive)": 0.49}),
        _row("b", 1, 0.80, {"-1 (negative)": 0.1, "0 (neutral)": 0.1, "1 (positive)": 0.8}),
    ]

    majority = build_ensemble_rows(rows)[0]["majority_vote"]

    assert majority["prediction"] == 1
    assert majority["tied"] is True


def test_output_names_and_layout(tmp_path):
    moment = datetime(2026, 9, 2, 14, 7, tzinfo=timezone.utc)
    assert resolve_output_filename("timestamp", seed=42, now=moment) == (
        "2026-09-02-1407-predictions.json"
    )
    assert resolve_output_filename("seed", seed=17) == "seed-17-predictions.json"
    assert resolve_output_filename("trial.json", seed=42) == "trial.json"
    with pytest.raises(ValueError):
        resolve_output_filename("../private", seed=42)

    row = _row("xlmr", 1, 0.60, {"1 (positive)": 0.60})
    raw, ensemble = publish_inference_outputs(
        [row],
        dataset="hbs",
        run_id="demo",
        output_root=tmp_path,
        filename="seed",
        seed=17,
    )
    assert raw == tmp_path / "inference" / "hbs" / "demo" / "seed-17-predictions.json"
    assert ensemble.name == "seed-17-predictions-ensemble.json"
    assert json.loads(ensemble.read_text(encoding="utf-8"))[0]["majority_vote"][
        "prediction"
    ] == 1


def test_duplicate_expert_rows_are_rejected():
    row = _row("xlmr", 1, 0.60, {"1 (positive)": 0.60})
    with pytest.raises(ValueError, match="Duplicate expert"):
        build_ensemble_rows([row, row])
