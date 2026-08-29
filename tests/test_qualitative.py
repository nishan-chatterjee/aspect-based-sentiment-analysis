from aspectbench.analysis import build_error_review


def test_error_review_prioritizes_confident_errors():
    rows = [
        {"record_id": "a", "aspect": "A", "sentiment": 1, "prediction": -1,
         "uncertainty": {"confidence": 0.95}, "article": "one"},
        {"record_id": "b", "aspect": "B", "sentiment": 0, "prediction": 1,
         "uncertainty": {"confidence": 0.60}, "article": "two"},
        {"record_id": "c", "aspect": "C", "sentiment": 1, "prediction": 1,
         "uncertainty": {"confidence": 0.99}, "article": "three"},
    ]
    report = build_error_review(rows)
    assert [row["record_id"] for row in report["records"]] == ["a", "b"]
    assert report["error_transitions"] == {"0->1": 1, "1->-1": 1}
