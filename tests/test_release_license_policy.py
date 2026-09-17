"""Keep generated and published model licensing/citation notices consistent."""

import importlib.util
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "huggingface" / "scripts"


def load_policy():
    sys.path.insert(0, str(SCRIPTS))
    try:
        spec = importlib.util.spec_from_file_location(
            "release_license_policy_test", SCRIPTS / "license_policy.py"
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    finally:
        sys.path.pop(0)


def test_curated_model_cards_and_license_files():
    policy = load_policy()
    for name in policy.MODEL_SPECS:
        family = ROOT / "huggingface" / "models" / name
        card = (family / "README.md").read_text()
        assert policy.BIBTEX in card
        assert policy.model_license_notice(name) in card
        assert (family / "LICENSE").read_text() == policy.model_license_notice(name)
        assert "license: cc-by-nc-4.0\n" in card
        assert "> [!NOTE]" in card
        if name == "slavic-specific":
            assert "owners, confirmed on 17 September 2026" in card


def test_generated_cards_preserve_license_and_citation():
    policy = load_policy()
    for name in policy.MODEL_SPECS:
        card = policy.decorate_model_card(
            "---\nlicense: other\n---\n\n# Model\n\nIntroduction.\n\n## Input\n\nExample.\n",
            name,
        )
        assert policy.BIBTEX in card
        assert policy.model_license_notice(name) in card
        assert card.index("## License") < card.index("## Input")


def test_toolkit_code_is_not_relicensed_by_model_notice():
    policy = load_policy()
    assert (ROOT / "huggingface" / "LICENSE").read_text() == (ROOT / "LICENSE").read_text()
    card = (ROOT / "huggingface" / "README.md").read_text()
    assert "license_name: polyform-noncommercial-1.0.0\n" in card
    assert "license_link: LICENSE\n" in card
    assert policy.BIBTEX in card


def test_standard_polyform_text_is_unmodified_and_prior_mit_is_preserved():
    official = (ROOT / "docs/license-options/POLYFORM-NONCOMMERCIAL-1.0.0.txt").read_text()
    license_text = (ROOT / "LICENSE").read_text()
    assert license_text.endswith(official)
    assert "Required Notice: Copyright" in license_text
    assert "regardless of the source of funding" in official
    prior = (ROOT / "licenses/MIT-previous-releases.txt").read_text()
    assert prior.startswith("MIT License\n")
    assert "Copyright (c) 2025 nishan" in prior
    assert prior == (ROOT / "huggingface/licenses/MIT-previous-releases.txt").read_text()


def test_academic_option_is_explicitly_unadopted_custom_draft():
    draft = (ROOT / "docs/license-options/ACADEMIC-SOFTWARE-LICENSE.draft.txt").read_text()
    assert "CUSTOM DRAFT" in draft
    assert "NOT ADOPTED" in draft
    assert "not legal advice" in draft
