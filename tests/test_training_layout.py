from pathlib import Path

from aspectbench.training.runner import _activate_checkpoint


def test_training_checkpoint_gets_release_style_active_view(tmp_path: Path) -> None:
    checkpoint = tmp_path / "models" / "bertic" / "hbs" / "masked" / "run" / "best-model.pt"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"checkpoint")

    active = _activate_checkpoint(
        checkpoint,
        output_model_root=tmp_path / "models",
        model="bertic",
        language="hbs",
        variant="masked",
    )

    assert active == tmp_path / "models" / "_active" / "slavic-specific" / "hbs" / "masked.pt"
    assert active.read_bytes() == b"checkpoint"
    assert active.with_suffix(".metadata.json").is_file()
