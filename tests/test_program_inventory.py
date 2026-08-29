import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_public_program_grid_has_no_embedded_examples():
    inventory = json.loads(
        (ROOT / "selective-deferral-programs/inventory.json").read_text(encoding="utf-8")
    )["programs"]
    assert len(inventory) == 18
    assert sum(row["release_status"] == "sanitized-post-release" for row in inventory) == 4
    for metadata in inventory:
        path = (
            ROOT
            / "selective-deferral-programs"
            / metadata["model"]
            / metadata["language"]
            / metadata["prompt_variant"]
            / "program.json"
        )
        rendered = path.read_text(encoding="utf-8")
        assert "**Example:**" not in rendered
        payload = json.loads(rendered)
        assert payload["predict"]["demos"] == []
        assert payload["predict"]["traces"] == []
        assert payload["predict"]["train"] == []
