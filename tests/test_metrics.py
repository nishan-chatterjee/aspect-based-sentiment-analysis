import pytest

from aspectbench.evaluation.metrics import classification_metrics


def test_perfect_fixed_label_metrics():
    report = classification_metrics([-1, 0, 1], [-1, 0, 1])
    assert report["f1_macro"] == pytest.approx(1.0)
    assert report["qwk"] == pytest.approx(1.0)
    assert report["per_class"]["negative"]["support"] == 1
    assert report["confusion_matrix"] == [[1, 0, 0], [0, 1, 0], [0, 0, 1]]


def test_imbalance_and_absent_class_are_explicit():
    report = classification_metrics([0, 0, 0, 1], [0, 0, 0, 0])
    assert report["imbalance"]["support"] == {"negative": 0, "neutral": 3, "positive": 1}
    assert report["imbalance"]["absent_classes"] == ["negative"]
    assert report["imbalance"]["max_to_min_nonzero_support_ratio"] == pytest.approx(3.0)
    assert report["f1_macro"] < report["f1_weighted"]


def test_invalid_labels_fail_loudly():
    with pytest.raises(ValueError, match="must be one of"):
        classification_metrics([-1, 0, 1], [-1, 0, 2])
