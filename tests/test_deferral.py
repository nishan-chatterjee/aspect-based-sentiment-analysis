from aspectbench.deferral.query import _select_gated


def row(identifier, confidence):
    return {"record_id": identifier, "uncertainty": {"confidence": confidence}}


def test_gate_selects_lowest_confidence_deterministically():
    rows = [row("high", 0.9), row("low", 0.4), row("middle", 0.7)]
    assert _select_gated(rows, 1 / 3) == {"low"}
    assert _select_gated(rows, 0) == set()
    assert _select_gated(rows, 1) == {"high", "middle", "low"}
