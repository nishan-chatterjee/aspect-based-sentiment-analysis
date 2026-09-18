---
library_name: pytorch
tags:
- aspect-based-sentiment-analysis
- south-slavic
- text-classification
license: cc-by-nc-4.0
---

# AspectBench HAN-XLM-R

> [!NOTE]
> AspectBench fine-tuned checkpoint contributions / trained heads are CC BY-NC 4.0: attribution is required, and commercial use requires separate permission. Noncommercial describes the use's purpose, not its user's affiliation. Third-party base assets and datasets retain their own terms; see the License section.

Model-only checkpoints for HBS and Slovenian document-level aspect-based
sentiment analysis. This repository contains 4/4 language-mode
checkpoint slots. It is used with the shared inference toolkit in
[`nishan-chatterjee/aspect-based-sentiment-analysis`](https://huggingface.co/nishan-chatterjee/aspect-based-sentiment-analysis).

## License

Copyright (c) 2026 the AspectBench model contributors.

The AspectBench fine-tuned checkpoint contributions / trained heads are licensed under
[Creative Commons Attribution-NonCommercial 4.0 International (CC BY-NC 4.0)](https://creativecommons.org/licenses/by-nc/4.0/).
The [official legal code](https://creativecommons.org/licenses/by-nc/4.0/legalcode.en) is incorporated by reference.
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

- HBS and Slovenian: [FacebookAI/xlm-roberta-base](https://huggingface.co/FacebookAI/xlm-roberta-base) — upstream MIT.

## What this repository contains

Each `.pt` file is a tensor-only state dictionary containing the complete
fine-tuned XLM-R sentence encoder plus the learned sentence-position,
cross-sentence Transformer, aspect-query attention, and three-class MLP
parameters. These are full inference checkpoints (about 1.18 GB each), not a
small adapter. Configuration and tokenizer assets are included; training data,
optimizer/scaler state, logs, and row-level predictions are excluded.

HAN-XLM-R is a custom hierarchical architecture rather than a standard
Transformers model class. Use the shared `InferenceEngine` or `aspectbench` CLI
to reconstruct it; `AutoModel.from_pretrained()` alone cannot load the HAN
layers. The toolkit sentence-splits the article, encodes up to 128 sentences ×
96 tokens, and applies the correct masked or unmasked aspect transformation.

## Input format

Every article must mark the target span with literal tags, even when using an
unmasked checkpoint:

```text
Tokom šestonedeljnog testiranja, redakcija je više puta kontaktirala <aspect>Primer Grupu</aspect> zbog nove usluge. Prvi odgovor <aspect>Primer Grupe</aspect> stigao je istog dana, a tehnički tim je zatim otklonio prijavljenu grešku bez dodatnih troškova. U završnom upitniku većina korisnika ocenila je podršku kao jasnu i pouzdanu.
```

- `masked`: the tagged text is replaced with `[ASPECT]`; the model does not see
  the target name.
- `unmasked`: the tags are removed and the model sees the target name.
- Gold `sentiment` is optional: `-1` = negative, `0` = neutral, `1` = positive.
  It is reported in the result but never used to produce the prediction.

## Available checkpoints

| Language | Mode | Status | Best validation Macro-F1 |
|---|---|---|---:|
| hbs | masked | Available | 0.9138 |
| hbs | unmasked | Available (retrained) | 0.9096 |
| slovenian | masked | Available | 0.8108 |
| slovenian | unmasked | Available (retrained) | 0.8103 |

`availability.json` contains the machine-readable selection record. A missing
checkpoint is never replaced with a checkpoint from another mode or language.

## Recovery provenance

The missing unmasked heads were retrained over all three fixed splits and selected only by validation Macro-F1. Optimizer state, logs, and row-level outputs are excluded from this model repository.

| Language | Selected split | Validation Macro-F1 | Mean test Macro-F1 | Mean test QWK |
|---|---:|---:|---:|---:|
| hbs | 1 | 0.9096 | 0.7939 | 0.7669 |
| slovenian | 2 | 0.8103 | 0.7158 | 0.6530 |

## Getting started

Create the portable environment from the toolkit repository:

```bash
conda env create -f environment.yml
conda activate aspectbench
```

`environment.yml` is maintained once in the shared toolkit rather than copied
into every model repository, preventing dependency versions from drifting
between family releases.

Or install the runtime packages in an existing environment:

```bash
python -m pip install -U torch transformers accelerate huggingface-hub sentencepiece numpy spacy sentence-transformers
```

Download the toolkit and this model repository into the expected directory
layout:

```python
from pathlib import Path
from huggingface_hub import snapshot_download

ROOT = Path("huggingface")
snapshot_download(
    repo_id="nishan-chatterjee/aspect-based-sentiment-analysis",
    local_dir=ROOT,
)
snapshot_download(
    repo_id="nishan-chatterjee/aspectbench-han-xlmr",
    local_dir=ROOT / "models" / "han-xlmr",
)
```

The model repository includes the tokenizer and configuration assets required
to reconstruct the architecture. No separate base-model cache is needed.

For a GitHub checkout, the equivalent one-command download and inference path
is:

```bash
python huggingface/scripts/download.py --model han-xlmr
CUDA_VISIBLE_DEVICES=0 aspectbench infer --models han-xlmr --dataset hbs \
  --variant unmasked --input-doc 'Poziv za <aspect>Primer Grupu</aspect> je uspeo.' \
  --mc-passes 8
```

## Python / Jupyter prediction

```python
from pathlib import Path
import sys

ROOT = Path("huggingface").resolve()
sys.path.insert(0, str(ROOT / "scripts"))

from inference import InferenceEngine

engine = InferenceEngine(
    model_name="han-xlmr",
    language="hbs",
    mode="masked",
    model_root=ROOT / "models",
    device="cuda",  # use "cpu" when no GPU is available
)

prediction = engine.predict(
    {
        "article": "Tokom šestonedeljnog testiranja, redakcija je više puta kontaktirala <aspect>Primer Grupu</aspect> zbog nove usluge. Prvi odgovor <aspect>Primer Grupe</aspect> stigao je istog dana, a tehnički tim je zatim otklonio prijavljenu grešku bez dodatnih troškova. U završnom upitniku većina korisnika ocenila je podršku kao jasnu i pouzdanu.",
        "sentiment": 1,
    },
    mc_passes=10,
)
prediction
```

For a real batch, reuse the loaded engine:

```python
records = [
    {"article": "Tokom šestonedeljnog testiranja, redakcija je više puta kontaktirala <aspect>Primer Grupu</aspect> zbog nove usluge. Prvi odgovor <aspect>Primer Grupe</aspect> stigao je istog dana, a tehnički tim je zatim otklonio prijavljenu grešku bez dodatnih troškova. U završnom upitniku većina korisnika ocenila je podršku kao jasnu i pouzdanu.", "sentiment": 1},
    {"article": "Pritužbe na <aspect>Drugi Sistem</aspect> nisu riješene.", "sentiment": -1},
]
predictions = engine.predict_batch(records, batch_size=2, mc_passes=10)
```

## Command-line prediction

Run from the toolkit directory:

```bash
python scripts/predict.py \
  --model-name han-xlmr \
  --language hbs \
  --mode masked \
  --model-root models \
  --device cuda \
  --mc-passes 10 \
  --article 'Tokom šestonedeljnog testiranja, redakcija je više puta kontaktirala <aspect>Primer Grupu</aspect> zbog nove usluge. Prvi odgovor <aspect>Primer Grupe</aspect> stigao je istog dana, a tehnički tim je zatim otklonio prijavljenu grešku bez dodatnih troškova. U završnom upitniku većina korisnika ocenila je podršku kao jasnu i pouzdanu.' \
  --sentiment 1
```

## Output fields

| Field | Meaning |
|---|---|
| `input_article` | Original article, including `<aspect>` tags. |
| `tagged_aspects` | Target strings extracted from the tags. |
| `aspect_used` | Target representation actually supplied to the model. |
| `gold_sentiment` | Optional user-supplied reference label. |
| `predicted_sentiment` | Predicted integer label: `-1`, `0`, or `1`. |
| `predicted_sentiment_name` | Human-readable class name. |
| `class_probabilities` | Probability assigned to every sentiment class. |
| `uncertainty_across_classes` | Entropy, confidence, probability margin, and—when MC dropout is enabled—mutual information and vote statistics. |
| `inference` | Device, MC-dropout flag, and checkpoint path. |

The `.pt` files contain model tensors only. Optimizer, scheduler, and
gradient-scaler state is excluded.

## Citation

Download [citation.bib](citation.bib) for Overleaf and cite with `\cite{chatterjee2026}`.

Please cite the accompanying article when using AspectBench:

```bibtex
@article{chatterjee2026,
  title     = {Evaluating fine-tuned, embedding-based, and zero-shot models for aspect-based sentiment analysis in {South Slavic} news},
  author    = {Chatterjee, Nishan and Koloski, Boshko and Doucet, Antoine and Pollak, Senja and Purver, Matthew},
  journal   = {Frontiers in Artificial Intelligence},
  volume    = {9},
  year      = {2026},
  publisher = {Frontiers Media SA},
  issn      = {2624-8212},
  doi       = {10.3389/frai.2026.1844418},
  url       = {https://www.frontiersin.org/journals/artificial-intelligence/articles/10.3389/frai.2026.1844418}
}
```
