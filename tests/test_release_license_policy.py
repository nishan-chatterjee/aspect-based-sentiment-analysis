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
        if name == "slavic-specific":
            assert "license: other\n" in card
            assert "license_link: LICENSE" in card
            assert "No CC BY-NC grant" in card
        else:
            assert "license: cc-by-nc-4.0\n" in card


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
    assert "license: mit\n" in card
    assert policy.BIBTEX in card
