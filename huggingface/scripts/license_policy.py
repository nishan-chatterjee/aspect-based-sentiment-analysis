"""Release licensing and citation notices; never relicense upstream assets."""

from __future__ import annotations

import re

from model_registry import MODEL_SPECS

ARTICLE_URL = "https://www.frontiersin.org/journals/artificial-intelligence/articles/10.3389/frai.2026.1844418/abstract"
NC_URL = "https://creativecommons.org/licenses/by-nc/4.0/"
NC_LEGAL_URL = NC_URL + "legalcode.en"
BIBTEX = """@article{chatterjee2026aspectbench,
  title   = {Evaluating Fine-Tuned, Embedding-Based, and Zero-Shot Models for
             Aspect-Based Sentiment Analysis in South Slavic News},
  author  = {Chatterjee, Nishan and Koloski, Boshko and Doucet, Antoine and
             Pollak, Senja and Purver, Matthew},
  journal = {Frontiers in Artificial Intelligence},
  volume  = {9},
  year    = {2026},
  doi     = {10.3389/frai.2026.1844418},
  url     = {https://www.frontiersin.org/journals/artificial-intelligence/articles/10.3389/frai.2026.1844418/abstract}
}"""

# Public upstream card metadata inspected 2026-09-16. These are not new grants.
UPSTREAM_LICENSES = {
    "FacebookAI/xlm-roberta-base": "MIT",
    "markussagen/xlm-roberta-longformer-base-4096": "Apache-2.0",
    "microsoft/mdeberta-v3-base": "MIT",
    "google/mt5-base": "Apache-2.0",
    "classla/bcms-bertic": "Apache-2.0",
    "EMBEDDIA/sloberta": "CC BY-SA 4.0",
    "BAAI/bge-m3": "MIT",
}


def model_license_notice(model_name: str) -> str:
    scope = "the AspectBench fine-tuned checkpoint contributions / trained heads"
    text = f"""## License

Copyright (c) 2026 the AspectBench model contributors.

{scope[0].upper() + scope[1:]} are licensed under
[Creative Commons Attribution-NonCommercial 4.0 International (CC BY-NC 4.0)]({NC_URL}).
The [official legal code]({NC_LEGAL_URL}) is incorporated by reference.
Noncommercial research, evaluation, adaptation and redistribution are allowed
subject to attribution and the license terms. Commercial use is not granted
under this license; contact the project for separate permission.

Noncommercial describes the purpose of a use, not whether its user is a
university or a company. Academic affiliation does not automatically make a
commercial project noncommercial. The legal code controls, including its
exceptions and limitations. The material is provided as-is without warranties.

This notice does not relicense third-party base weights, tokenizers, code or
configuration assets, and does not revoke any rights previously granted.
AspectBench adapted the listed base models for aspect-based sentiment analysis;
retain their original attribution and license notices when redistributing.
"""
    bases = MODEL_SPECS[model_name]["base_model"]
    if isinstance(bases, str):
        bases = {"HBS and Slovenian": bases}
    for language, repo in bases.items():
        text += f"\n- {language}: [{repo}](https://huggingface.co/{repo}) — upstream {UPSTREAM_LICENSES[repo]}."
    if model_name == "slavic-specific":
        text += """

**Slovenian SloBERTa permission:** the AspectBench fine-tuned SloBERTa
checkpoints are released under CC BY-NC 4.0 with permission from the SloBERTa
owners, confirmed on 17 September 2026. This permission concerns the AspectBench
fine-tuned release; the original EMBEDDIA/SloBERTa release retains its upstream
[CC BY-SA 4.0 license](https://creativecommons.org/licenses/by-sa/4.0/legalcode.en).
"""
    if model_name == "bge-m3-mlp":
        text += "\n\nThe frozen BGE-M3 encoder is downloaded separately and remains upstream MIT; the noncommercial grant covers the trained AspectBench MLP heads.\n"
    return text.strip() + "\n"


def citation_section() -> str:
    return "## Citation\n\nPlease cite the accompanying article when using AspectBench:\n\n```bibtex\n" + BIBTEX + "\n```\n"


def decorate_model_card(card: str, model_name: str) -> str:
    """Add notices to generated cards as well as existing curated cards."""
    metadata = "license: cc-by-nc-4.0"
    card = re.sub(r"^license:.*$", metadata, card, count=1, flags=re.MULTILINE)
    card = re.sub(r"^# (.+)$", r"# \1\n\n" + model_usage_note(), card, count=1, flags=re.MULTILINE)
    # Insert before first body section, immediately after the model introduction.
    index = card.find("\n## ")
    if index == -1:
        index = len(card)
    card = card[:index].rstrip() + "\n\n" + model_license_notice(model_name) + "\n" + card[index:].lstrip()
    return card.rstrip() + "\n\n" + citation_section()


def model_usage_note() -> str:
    return (
        "> [!NOTE]\n"
        "> AspectBench fine-tuned checkpoint contributions / trained heads are CC BY-NC 4.0: "
        "attribution is required, and commercial use requires separate permission. "
        "Noncommercial describes the use's purpose, not its user's affiliation. "
        "Third-party base assets and datasets retain their own terms; see the License section.\n"
    )
