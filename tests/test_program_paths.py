import json
from pathlib import Path

import pytest

from aspectbench.deferral.programs import program_path, validate_program_metadata


def _write_program(root: Path, source: str, run_id: str | None = None) -> Path:
    directory = root / source / "longformer" / "hbs" / "masked"
    if run_id:
        directory /= run_id
    directory.mkdir(parents=True)
    program = directory / "program.json"
    program.write_text("{}", encoding="utf-8")
    (directory / "metadata.json").write_text(
        json.dumps(
            {"model": "longformer", "language": "hbs", "prompt_variant": "masked"}
        ),
        encoding="utf-8",
    )
    return program


def test_precalibrated_and_optimized_programs_are_separate(tmp_path: Path) -> None:
    packaged = _write_program(tmp_path, "precalibrated")
    optimized = _write_program(tmp_path, "optimized", "new-dataset-run")

    assert program_path(
        source="precalibrated", model="longformer", dataset="hbs",
        variant="masked", program_root=tmp_path,
    ) == packaged
    assert program_path(
        source="optimized", model="longformer", dataset="hbs", variant="masked",
        program_root=tmp_path, run_id="new-dataset-run",
    ) == optimized


def test_optimized_program_requires_run_id(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="program-run-id"):
        program_path(
            source="optimized", model="longformer", dataset="hbs",
            variant="masked", program_root=tmp_path,
        )


def test_metadata_mismatch_fails_closed(tmp_path: Path) -> None:
    program = _write_program(tmp_path, "precalibrated")
    with pytest.raises(ValueError, match="metadata mismatch"):
        validate_program_metadata(
            program, model="longformer", dataset="hbs", variant="unmasked"
        )
