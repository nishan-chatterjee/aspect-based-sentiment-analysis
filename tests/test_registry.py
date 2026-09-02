import pytest

from aspectbench.registry import available_models, resolve_model, select_models


def test_language_specific_adapters_are_separate():
    assert resolve_model("slavic-specific", language="hbs").name == "bertic"
    assert resolve_model("slavic-specific", language="slovenian").name == "sloberta"
    assert "bertic" in available_models("hbs")
    assert "sloberta" not in available_models("hbs")


def test_one_few_all_selection():
    assert [spec.name for spec in select_models(["longformer"], language="hbs")] == [
        "longformer"
    ]
    assert [spec.name for spec in select_models(["xlmr", "longformer"], language="sl")] == [
        "xlmr",
        "longformer",
    ]
    assert [
        spec.name for spec in select_models(["xlmr,longformer mdeberta-v3"], language="hbs")
    ] == ["xlmr", "longformer", "mdeberta-v3"]
    assert len(select_models(["all"], language="hbs")) == len(available_models("hbs"))
    with pytest.raises(ValueError, match="alone"):
        select_models(["all", "xlmr"], language="hbs")
