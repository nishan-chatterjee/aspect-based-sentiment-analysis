from __future__ import annotations

import numpy as np

from aspectbench.privacy.experiments import (
    aspect_distribution_report,
    counterfactual_aggregate,
    exact_overlap_report,
    finite_json,
    model_membership_report,
    neutral_context_rows,
    normalize_private_text,
    permuted_aspect_rows,
    pseudonymized_rows,
)


def row(article: str, aspect: str, sentiment: int, uuid: str):
    return {
        "article": f"{article} <aspect>{aspect}</aspect>",
        "aspect": aspect,
        "sentiment": sentiment,
        "uuid": uuid,
    }


def prediction(probabilities):
    labels = ("-1 (negative)", "0 (neutral)", "1 (positive)")
    return {"class_probabilities": dict(zip(labels, probabilities, strict=True))}


def test_exact_overlap_is_hashed_and_aggregate_only():
    train = [row("Alpha", "Entity A", -1, "one"), row("Beta", "Entity B", 0, "two")]
    validation = [row("Gamma", "Entity C", 1, "three")]
    test = [dict(train[0]), row("Delta", "Entity D", 1, "four")]
    report = exact_overlap_report(train, validation, test)
    overlap = report["cross_cohort"]["train_vs_test"]["normalized_article_and_aspect"]
    assert overlap["right_rows_overlapping_left"] == 1
    assert "Alpha" not in str(report)
    assert "Entity A" not in str(report)


def test_aspect_distribution_and_counterfactual_helpers():
    train = [row("A", "Seen", -1, "1"), row("B", "Seen", -1, "2"), row("C", "Other", 1, "3")]
    validation = [row("D", "Seen", -1, "4"), row("E", "New", 0, "5")]
    report = aspect_distribution_report(train, validation, validation)
    assert report["support"]["validation"]["train_seen_fraction"] == 0.5
    assert 0.0 <= report["aspect_only_label_baseline"]["validation"]["macro_f1"] <= 1.0
    pseudonyms = pseudonymized_rows(validation, language="hbs")
    assert all("Sintetička Meta" in item["article"] for item in pseudonyms)
    assert all(item["aspect"] == "Sintetička Meta" for item in pseudonyms)
    permuted = permuted_aspect_rows(validation, seed=1)
    assert len(permuted) == len(validation)
    neutral = neutral_context_rows(["One", "Two"], language="sl")
    assert len(neutral) == 2 and all("<aspect>" in item["article"] for item in neutral)


def test_counterfactual_report_contains_only_aggregates():
    original = [prediction([0.8, 0.1, 0.1]), prediction([0.2, 0.7, 0.1])]
    changed = [prediction([0.4, 0.5, 0.1]), prediction([0.2, 0.6, 0.2])]
    report = counterfactual_aggregate(original, changed)
    assert report["n"] == 2
    assert report["prediction_flip_rate"] == 0.5
    assert report["js_divergence_nats"]["mean"] >= 0.0


def test_membership_attack_selects_leaking_score_direction():
    members = []
    nonmembers = []
    for index in range(40):
        common = {
            "confidence": 0.2 + index / 1000,
            "negative_entropy": -0.9,
            "margin": 0.1,
            "length_log": 4.0,
            "aspect_frequency_log": 1.0,
            "aspect_seen": 1,
            "correct": 1,
            "label": (index % 3) - 1,
        }
        members.append({**common, "gold_log_probability": -2.0})
        nonmembers.append({**common, "gold_log_probability": -0.1})
    report = model_membership_report(
        members, nonmembers, seed=3, bootstrap_samples=20, permutation_samples=20
    )
    attack = report["univariate_attacks"]["gold_log_probability"]
    assert attack["auc"] == 1.0
    assert attack["score_direction"] == "lower_original_score_means_member"


def test_normalization_and_finite_json():
    assert normalize_private_text("  NÁME\nHere ") == "náme here"
    assert finite_json({"bad": float("nan"), "array": np.asarray([1])[0]}) == {
        "bad": None,
        "array": 1,
    }
