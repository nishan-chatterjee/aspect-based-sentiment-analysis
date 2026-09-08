from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]


def test_generates_three_reproducible_stratified_splits(tmp_path):
    rows = [
        {"article": f"document {index}", "aspect": f"entity {index}", "sentiment": (index % 3) - 1}
        for index in range(60)
    ]
    source = tmp_path / "records.json"
    source.write_text(json.dumps(rows), encoding="utf-8")
    output = tmp_path / "generated"
    command = [
        sys.executable,
        str(ROOT / "scripts/3.7-prepare-training-splits.py"),
        "--dataset", "hbs", "--input", str(source), "--output-dir", str(output),
        "--split-count", "3", "--seeds", "1729", "6174", "8191",
        "--validation-fraction", "0.2",
    ]
    subprocess.run(command, check=True, capture_output=True, text=True)
    manifest = json.loads((output / "prepared-splits.json").read_text())
    assert [entry["seed"] for entry in manifest["entries"]] == [1729, 6174, 8191]
    assert all(entry["generated"] for entry in manifest["entries"])
    assert len({Path(entry["path"]).read_text() for entry in manifest["entries"]}) == 3
    for entry in manifest["entries"]:
        payload = json.loads(Path(entry["path"]).read_text())
        assert len(payload["train"]) == 48
        assert len(payload["val"]) == 12
        assert {row["sentiment"] for row in payload["val"]} == {-1, 0, 1}
