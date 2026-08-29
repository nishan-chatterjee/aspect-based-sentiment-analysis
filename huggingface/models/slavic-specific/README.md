---
library_name: pytorch
tags:
- aspect-based-sentiment-analysis
- south-slavic
- text-classification
license: other
---

# AspectBench BERTić / SloBERTa

Model-only checkpoints for HBS and Slovenian document-level aspect-based
sentiment analysis. This repository contains 4/4 language-mode
checkpoint slots. It is used with the shared inference toolkit in
[`nishan-chatterjee/aspect-based-sentiment-analysis`](https://huggingface.co/nishan-chatterjee/aspect-based-sentiment-analysis).

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
| hbs | masked | Available | 0.9151 |
| hbs | unmasked | Available | 0.9251 |
| slovenian | masked | Available | 0.8545 |
| slovenian | unmasked | Available | 0.8639 |

`availability.json` contains the machine-readable selection record. A missing
checkpoint is never replaced with a checkpoint from another mode or language.

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
    repo_id="nishan-chatterjee/aspectbench-slavic-specific",
    local_dir=ROOT / "models" / "slavic-specific",
)
```

The model repository includes the tokenizer and configuration assets required
to reconstruct the architecture. No separate base-model cache is needed.

## Python / Jupyter prediction

```python
from pathlib import Path
import sys

ROOT = Path("huggingface").resolve()
sys.path.insert(0, str(ROOT / "scripts"))

from inference import InferenceEngine

engine = InferenceEngine(
    model_name="slavic-specific",
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
  --model-name slavic-specific \
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
