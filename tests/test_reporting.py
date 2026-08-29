import pytest

from aspectbench.evaluation import build_evaluation_report


PREDICTIONS = [
    {"aspect": "Known", "sentiment": -1, "prediction": -1},
    {"aspect": " known ", "sentiment": 0, "prediction": 0},
    {"aspect": "New", "sentiment": -1, "prediction": 0},
    {"aspect": "New", "sentiment": 1, "prediction": 1},
]


def test_aspect_and_seen_unseen_report():
    report = build_evaluation_report(
        PREDICTIONS,
        training_records=[{"aspect": "KNOWN"}],
    )
    assert report["overall"]["n"] == 4
    assert report["by_aspect"]["summary"]["n_aspects"] == 2
    assert set(report["by_aspect"]["aspects"]) == {"known", "new"}
    assert report["seen_unseen"]["seen"]["n"] == 2
    assert report["seen_unseen"]["unseen"]["n"] == 2
    assert report["seen_unseen"]["seen"]["accuracy"] == pytest.approx(1.0)


def test_missing_aspect_fails():
    with pytest.raises(ValueError, match="non-empty"):
        build_evaluation_report([{"aspect": "", "sentiment": 0, "prediction": 0}])
