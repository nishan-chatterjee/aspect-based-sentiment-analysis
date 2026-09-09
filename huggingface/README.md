---
library_name: pytorch
tags:
- aspect-based-sentiment-analysis
- south-slavic
- text-classification
license: other
---

# AspectBench: reusable document-level ABSA inference

This directory provides one inference interface for seven AspectBench model
families in HBS and Slovenian. It supports single and batched predictions,
masked and unmasked aspect handling, and uncertainty estimates from Monte
Carlo dropout.

- [Model collection overview](https://huggingface.co/collections/nishan-chatterjee/aspect-based-sentiment-analysis-6a9016a6d9cab7b093f122d3)
- [Reusable inference toolkit](https://huggingface.co/nishan-chatterjee/aspect-based-sentiment-analysis)
- [GitHub repository](https://github.com/nishan-chatterjee/aspect-based-sentiment-analysis)

## Input contract

Every article must mark the target using literal `<aspect>...</aspect>` tags:

```text
Tokom šestonedeljnog testiranja, redakcija je više puta kontaktirala
<aspect>Primer Grupu</aspect> zbog nove usluge. Prvi odgovor
<aspect>Primer Grupe</aspect> stigao je istog dana, a tehnički tim je zatim
otklonio prijavljenu grešku bez dodatnih troškova. U završnom upitniku većina
korisnika ocenila je podršku kao jasnu i pouzdanu.
```

Keep these tags in the input for both inference modes:

- `masked`: each tagged span and its paired aspect are replaced by `[ASPECT]`.
  The checkpoint sees the target location but not its name.
- `unmasked`: tags are removed during preprocessing and the checkpoint sees the
  target text. An optional `aspect` field can select a target explicitly;
  otherwise the first tagged span is used.

The optional gold `sentiment` is `-1` (negative), `0` (neutral), or `1`
(positive). It is echoed in the result and is never used to make a prediction.

Batch JSON can be a top-level list or `{"records": [...]}`:

```json
{
  "records": [
    {
      "article": "Kupci so v šesttedenskem preizkusu uporabljali dostavo podjetja <aspect>Modri Gaj</aspect>. Večina paketov je prispela pravočasno, podpora <aspect>Modrega Gaja</aspect> pa je manjkajoči naslov dopolnila še isti dan.",
      "sentiment": 1
    }
  ]
}
```

## Choose a model

Start with **masked XLM-R**. It has the best three-run test Macro-F1 among the
seven released single-model families in both languages: 73.80 for Slovenian
and 82.21 for HBS. Add **masked Longformer** when retaining more of a long
article matters, and **masked HAN-XLM-R** when you want a structurally different
hierarchical expert. The GitHub toolkit can run these three together and return
individual predictions, majority voting, confidence voting, and uncertainty.

### Public model repositories

| Family | Repository |
|---|---|
| XLM-R | [aspectbench-xlmr](https://huggingface.co/nishan-chatterjee/aspectbench-xlmr) |
| HAN-XLM-R | [aspectbench-han-xlmr](https://huggingface.co/nishan-chatterjee/aspectbench-han-xlmr) |
| XLM-R Longformer | [aspectbench-longformer](https://huggingface.co/nishan-chatterjee/aspectbench-longformer) |
| mDeBERTa-v3 | [aspectbench-mdeberta-v3](https://huggingface.co/nishan-chatterjee/aspectbench-mdeberta-v3) |
| mT5 | [aspectbench-mt5](https://huggingface.co/nishan-chatterjee/aspectbench-mt5) |
| BERTić / SloBERTa | [aspectbench-slavic-specific](https://huggingface.co/nishan-chatterjee/aspectbench-slavic-specific) |
| BGE-M3 + MLP | [aspectbench-bge-m3-mlp](https://huggingface.co/nishan-chatterjee/aspectbench-bge-m3-mlp) |

Both HBS and Slovenian model weights are released publicly with project-partner
approval. Model repositories contain inference artifacts only—not training
records, optimizer state, cached embeddings, logs, or row-level predictions.

The following test-set scores are means over three fixed train/validation
splits; `±` is the standard deviation. Precision and recall are macro-averaged,
and all metrics except QWK are percentages. XLM-R `unmasked` is the paper's
**Truncated** strategy. XLM-R `masked` is **Truncated + Masked**: it was
completed after the accepted-manuscript table was assembled and is reported in
the preserved final-results analysis.

### Slovenian released-model results

| Model | Strategy | Accuracy | Precision | Recall | Macro F1 | QWK | Negative F1 | Neutral F1 | Positive F1 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| BGE-M3 + MLP | Unmasked | 90.17 ± 0.21 | 79.00 ± 2.69 | 63.16 ± 1.02 | 68.39 ± 1.03 | .640 ± .012 | 40.98 ± 2.25 | 94.18 ± 0.11 | 70.02 ± 1.14 |
| BGE-M3 + MLP | Masked | 90.39 ± 0.11 | 75.34 ± 1.72 | 62.86 ± 1.07 | 67.21 ± 1.29 | .653 ± .007 | 35.82 ± 3.88 | 94.29 ± 0.06 | 71.53 ± 0.64 |
| **XLM-R (truncated)** | Unmasked | 90.28 ± 0.33 | 73.12 ± 1.55 | 67.58 ± 0.29 | 70.03 ± 0.80 | .659 ± .013 | 43.90 ± 1.67 | 94.20 ± 0.19 | 71.99 ± 1.13 |
| **XLM-R (truncated)** | **Masked** | **91.11 ± 0.15** | **78.97 ± 2.18** | **70.59 ± 1.65** | **73.80 ± 1.69** | **.697 ± .006** | **51.56 ± 5.61** | **94.65 ± 0.09** | **75.18 ± 0.70** |
| HAN-XLM-R | Unmasked | 89.28 ± 0.25 | 71.05 ± 0.71 | 69.29 ± 0.69 | 70.11 ± 0.43 | .640 ± .007 | 46.36 ± 1.92 | 93.53 ± 0.15 | 70.42 ± 0.72 |
| HAN-XLM-R | Masked | 90.37 ± 0.15 | 74.64 ± 1.30 | 69.92 ± 1.39 | 71.57 ± 0.47 | .685 ± .001 | 46.05 ± 1.58 | 94.14 ± 0.11 | 74.51 ± 0.10 |
| XLM-R Longformer | Unmasked | 90.28 ± 0.18 | 73.72 ± 2.23 | 69.58 ± 2.13 | 71.47 ± 2.01 | .663 ± .008 | 48.12 ± 5.78 | 94.19 ± 0.10 | 72.09 ± 0.64 |
| XLM-R Longformer | Masked | 91.13 ± 0.43 | 78.68 ± 2.77 | 69.00 ± 0.55 | 72.50 ± 1.05 | .697 ± .013 | 47.50 ± 1.89 | 94.66 ± 0.26 | 75.35 ± 1.10 |
| mDeBERTa-v3 | Unmasked | 89.83 ± 0.97 | 74.59 ± 2.17 | 68.30 ± 1.19 | 70.71 ± 1.82 | .660 ± .020 | 46.18 ± 3.64 | 93.84 ± 0.65 | 72.11 ± 1.48 |
| mDeBERTa-v3 | Masked | 90.62 ± 0.26 | 77.41 ± 0.52 | 67.75 ± 0.71 | 71.30 ± 0.60 | .678 ± .011 | 45.86 ± 0.67 | 94.36 ± 0.15 | 73.66 ± 1.00 |
| mT5 | Unmasked | 90.25 ± 0.38 | 71.45 ± 1.87 | 65.21 ± 1.55 | 67.75 ± 0.42 | .661 ± .005 | 36.53 ± 1.59 | 94.16 ± 0.30 | 72.57 ± 0.70 |
| mT5 | Masked | 90.87 ± 0.24 | 77.27 ± 1.44 | 64.87 ± 1.91 | 69.37 ± 1.12 | .668 ± .013 | 40.85 ± 3.08 | 94.59 ± 0.14 | 72.69 ± 1.30 |
| SloBERTa | Unmasked | 91.28 ± 0.25 | 75.98 ± 1.33 | 71.17 ± 2.52 | 73.16 ± 1.03 | .700 ± .006 | 49.19 ± 3.57 | 94.77 ± 0.16 | 75.51 ± 0.75 |
| SloBERTa | Masked | 91.62 ± 0.02 | 77.19 ± 0.83 | 70.03 ± 0.12 | 73.03 ± 0.31 | .710 ± .005 | 47.59 ± 1.47 | 94.98 ± 0.02 | 76.52 ± 0.60 |

### HBS released-model results

| Model | Strategy | Accuracy | Precision | Recall | Macro F1 | QWK | Negative F1 | Neutral F1 | Positive F1 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| BGE-M3 + MLP | Unmasked | 83.95 ± 0.12 | 80.98 ± 0.17 | 77.16 ± 0.27 | 78.76 ± 0.25 | .759 ± .002 | 65.63 ± 0.60 | 84.16 ± 0.20 | 86.49 ± 0.01 |
| BGE-M3 + MLP | Masked | 84.05 ± 0.22 | 80.59 ± 0.12 | 77.72 ± 0.18 | 78.93 ± 0.11 | .761 ± .002 | 65.91 ± 0.33 | 84.18 ± 0.37 | 86.70 ± 0.22 |
| **XLM-R (truncated)** | Unmasked | 85.42 ± 0.22 | 84.63 ± 0.83 | 78.32 ± 0.60 | 80.84 ± 0.24 | .789 ± .003 | 69.36 ± 1.30 | 85.76 ± 0.20 | 87.41 ± 0.47 |
| **XLM-R (truncated)** | **Masked** | **86.37 ± 0.15** | **82.20 ± 0.18** | **82.33 ± 0.54** | **82.21 ± 0.35** | **.809 ± .004** | **71.35 ± 0.93** | **86.18 ± 0.08** | **89.10 ± 0.18** |
| HAN-XLM-R | Unmasked | 83.86 ± 0.43 | 80.81 ± 1.22 | 78.57 ± 0.56 | 79.50 ± 0.81 | .764 ± .005 | 68.34 ± 1.99 | 83.74 ± 0.39 | 86.41 ± 0.29 |
| HAN-XLM-R | Masked | 84.62 ± 0.30 | 80.39 ± 0.73 | 80.84 ± 0.47 | 80.53 ± 0.50 | .784 ± .004 | 69.84 ± 1.06 | 84.36 ± 0.35 | 87.39 ± 0.16 |
| XLM-R Longformer | Unmasked | 84.45 ± 0.34 | 83.28 ± 0.60 | 77.22 ± 0.50 | 79.65 ± 0.14 | .770 ± .005 | 67.63 ± 0.69 | 84.86 ± 0.22 | 86.47 ± 0.64 |
| XLM-R Longformer | Masked | 85.87 ± 0.09 | 81.39 ± 0.61 | 81.43 ± 0.19 | 81.37 ± 0.31 | .800 ± .001 | 69.60 ± 1.00 | 85.80 ± 0.22 | 88.69 ± 0.20 |
| mDeBERTa-v3 | Unmasked | 82.93 ± 1.78 | 82.75 ± 0.63 | 75.38 ± 2.28 | 78.08 ± 1.69 | .749 ± .028 | 66.12 ± 1.81 | 83.44 ± 1.01 | 84.67 ± 2.83 |
| mDeBERTa-v3 | Masked | 85.17 ± 0.40 | 83.00 ± 0.73 | 79.91 ± 0.02 | 81.15 ± 0.34 | .780 ± .004 | 70.96 ± 0.32 | 85.08 ± 0.64 | 87.42 ± 0.24 |
| mT5 | Unmasked | 83.98 ± 0.21 | 80.86 ± 0.99 | 76.90 ± 1.48 | 78.37 ± 0.53 | .761 ± .007 | 64.25 ± 1.22 | 83.92 ± 0.19 | 86.94 ± 0.46 |
| mT5 | Masked | 83.89 ± 0.64 | 78.48 ± 0.58 | 80.15 ± 0.79 | 79.07 ± 0.84 | .766 ± .010 | 66.43 ± 1.38 | 83.59 ± 1.03 | 87.19 ± 0.27 |
| BERTić | Unmasked | 86.31 ± 0.58 | 84.49 ± 0.17 | 79.74 ± 1.31 | 81.70 ± 0.86 | .802 ± .010 | 70.03 ± 1.57 | 86.38 ± 0.44 | 88.68 ± 0.66 |
| BERTić | Masked | 85.78 ± 0.38 | 82.69 ± 0.61 | 80.48 ± 0.36 | 81.44 ± 0.17 | .795 ± .004 | 70.30 ± 1.21 | 85.77 ± 0.37 | 88.24 ± 0.57 |


## Getting started

For a new portable environment:

```bash
conda env create -f environment.yml
conda activate aspectbench
```

`environment-full.yml` is the complete export of the development `absa`
environment. On the existing cluster, the current environment remains valid:

```bash
module load Anaconda3/2024.02-1
source /opt/easybuild/software/Anaconda3/2024.02-1/etc/profile.d/conda.sh
conda deactivate 2>/dev/null || true
conda activate absa
hash -r
which python
python -c 'import sys, torch; print(sys.executable); print(torch.__version__)'
```

The `module load` command can replace `PATH` while an older shell still has
`CONDA_PREFIX=.../envs/absa`. In that state the prompt says `(absa)`, `uv` finds
packages in the environment, but `python` is the EasyBuild base interpreter.
Deactivating and reactivating repairs the shell. The supplied validation
launchers avoid this ambiguity entirely by using the environment interpreter
by absolute path. Override it when necessary with:

```bash
export ABSA_PYTHON="$HOME/.conda/envs/absa/bin/python"
```

To use pip instead:

```bash
python -m pip install -r requirements.txt
```

Authenticate and download all available model repositories into the expected
layout:

```bash
hf auth login
python scripts/download.py
```

Each model repository includes its tokenizer and architecture configuration in
`LANGUAGE/base_model/`. Inference therefore works without a separate clone of
the training repository or a populated Transformers cache.

## Directory layout

```text
huggingface/
├── models/
│   └── MODEL_NAME/
│       ├── hbs/
│       │   ├── base_model/
│       │   ├── masked.pt
│       │   └── unmasked.pt
│       ├── slovenian/
│       │   ├── base_model/
│       │   ├── masked.pt
│       │   └── unmasked.pt
│       ├── availability.json
│       └── README.md
├── examples/
├── scripts/
├── environment.yml
├── environment-full.yml
└── requirements.txt
```

Only checkpoint filenames whose original trained artifacts exist are present.
The `.pt` files are model-only PyTorch state dictionaries; optimizer,
scheduler, and gradient-scaler state is excluded.

## Python and Jupyter usage

Download the toolkit and one family directly from a notebook if needed:

```python
from pathlib import Path
from huggingface_hub import snapshot_download

ROOT = Path("huggingface")
snapshot_download(
    repo_id="nishan-chatterjee/aspect-based-sentiment-analysis",
    local_dir=ROOT,
)
snapshot_download(
    repo_id="nishan-chatterjee/aspectbench-mdeberta-v3",
    local_dir=ROOT / "models" / "mdeberta-v3",
)
```

Load once and predict many times:

```python
from pathlib import Path
import sys

ROOT = Path("huggingface").resolve()
sys.path.insert(0, str(ROOT / "scripts"))

from inference import InferenceEngine

engine = InferenceEngine(
    model_name="mdeberta-v3",
    language="slovenian",
    mode="unmasked",
    model_root=ROOT / "models",
    device="cuda",  # or "cpu"
)

single = engine.predict(
    {
        "article": "Kupci so v šesttedenskem preizkusu uporabljali dostavo podjetja <aspect>Modri Gaj</aspect>. Večina paketov je prispela pravočasno, podpora <aspect>Modrega Gaja</aspect> pa je manjkajoči naslov dopolnila še isti dan.",
        "sentiment": 1,
    },
    mc_passes=10,
)

batch = engine.predict_batch(
    [
        {"article": "Kupci so več tednov uporabljali <aspect>Modri Gaj</aspect>. Podpora <aspect>Modrega Gaja</aspect> je vse prijave rešila pravočasno.", "sentiment": 1},
        {"article": "Pritožbe glede <aspect>Drugega sistema</aspect> niso rešene.", "sentiment": -1},
    ],
    batch_size=2,
    mc_passes=10,
)
```

## Single prediction on an interactive SLURM node

```bash
cd huggingface

python scripts/predict.py \
  --model-name xlmr \
  --language hbs \
  --mode masked \
  --model-root models \
  --device cuda \
  --mc-passes 10 \
  --article 'Tokom šestonedeljnog testiranja redakcija je više puta kontaktirala <aspect>Primer Grupu</aspect>. Odgovor <aspect>Primer Grupe</aspect> stigao je istog dana, a prijavljena greška otklonjena je bez dodatnih troškova.' \
  --sentiment 1
```

`--base-model-root` remains available for legacy layouts, but is unnecessary
when the bundled `base_model/` assets are present.

## Ten-record Slovenian batch

```bash
python scripts/predict_batch.py \
  --model-name mdeberta-v3 \
  --language slovenian \
  --mode unmasked \
  --input examples/sl-tagged-synthetic-examples.json \
  --output sl-predictions.json \
  --batch-size 4 \
  --model-root models \
  --device cuda \
  --mc-passes 10
```

The command reports the number of written predictions and the absolute output
path. Use `examples/hbs-tagged-examples.json` for the ten HBS examples. Both
files contain machine-generated paragraphs intended only for quick inference
checks; they are not dataset samples and must not be used for evaluation. The
HBS file contains six broadly Serbo-Croatian, three Croatian, and one Bosnian
example.

## Output fields

| Field | Meaning |
|---|---|
| `model`, `language`, `mode` | Checkpoint selection used for inference. |
| `input_article` | Original article with its `<aspect>` tags. |
| `tagged_aspects` | All targets extracted from the article. |
| `aspect_used` | The target representation supplied to the model; `[ASPECT]` in masked mode. |
| `gold_sentiment` | Optional user-supplied reference label. |
| `predicted_sentiment` | Predicted integer label: `-1`, `0`, or `1`. |
| `predicted_sentiment_name` | `negative`, `neutral`, or `positive`. |
| `class_probabilities` | Probability assigned to each of the three classes. |
| `uncertainty_across_classes` | Predictive entropy, normalized entropy, confidence, and top-two margin. |
| `uncertainty_across_classes.mc_dropout` | With `mc_passes >= 2`: expected entropy, mutual information, agreement, variation ratio, and vote counts. |
| `inference` | Device, MC-dropout flag, and local checkpoint path. |

Set `--mc-passes 0` for deterministic inference. A value of `1` is rejected
because it is neither deterministic inference nor a meaningful MC sample.

## Checkpoint availability

The intended grid is 7 repository families × 2 languages × 2 modes = 28
inference slots. All 28 trained checkpoint files are now available:

| Family | HBS masked | HBS unmasked | SL masked | SL unmasked |
|---|---:|---:|---:|---:|
| XLM-R | yes | yes | yes | yes |
| HAN-XLM-R | yes | yes | yes | yes |
| XLM-R Longformer | yes | yes | yes | yes |
| mDeBERTa-v3 | yes | yes | yes | yes |
| mT5 | yes | yes | yes | yes |
| BERTić / SloBERTa | yes | yes | yes | yes |
| BGE-M3 + MLP | yes | yes | yes | yes |

The formerly absent two XLM-R unmasked heads, two HAN-XLM-R unmasked heads,
and four BGE-M3 MLP heads were retrained across three splits, selected by
validation Macro-F1, packaged with their required base/tokenizer metadata, and
then validated from fresh Hugging Face downloads. No masked or
different-family checkpoint was relabeled as a substitute. The complete
four-GPU release smoke passed 28/28 slots for model loading, single and batch
prediction, probability invariants, and MC-dropout uncertainty. Exact remote
revisions and reports are preserved locally under the ignored
`validation-runs/hf-release-smoke/` directory.

## Validate single and batched inference

The validator loads every available slot, performs one-record and true batched
inference, and checks the output schema and probability invariants:

```bash
bash scripts/run_validation_slurm.sh
```

This selects `$HOME/.conda/envs/absa/bin/python` when the activated shell points
at the wrong interpreter. Every available language/mode slot reports both
`single=completed` and `batch=completed`; missing checkpoint artifacts report
`SKIP`, and real errors report `FAIL` with the failing stage. The default batch
contains all ten examples for the relevant language. `validation-report.json`
retains the complete single result and all batch result dictionaries—including
class probabilities and uncertainty fields—for every tested slot.

### Interactive validation on 1, 2, or 4 GPUs

After opening an interactive allocation, run the validator directly from its
shell. The argument is the number of GPUs to use:

```bash
cd huggingface

# Choose exactly one of these commands.
ABSA_MC_PASSES=10 bash scripts/run_validation_interactive.sh 1
ABSA_MC_PASSES=10 bash scripts/run_validation_interactive.sh 2
ABSA_MC_PASSES=10 bash scripts/run_validation_interactive.sh 4
```

The launcher assigns model families across the requested GPUs; each GPU handles
its assigned families sequentially so several large models are not loaded onto
the same device at once. It respects an existing `CUDA_VISIBLE_DEVICES` from the
interactive allocation. To select devices yourself, the number of comma-separated
IDs must match the positional argument:

```bash
GPU_IDS=0,2 ABSA_MC_PASSES=10 bash scripts/run_validation_interactive.sh 2
```

To check only selected families during a quick diagnostic run:

```bash
ABSA_MODELS=xlmr,longformer ABSA_MC_PASSES=2 \
  bash scripts/run_validation_interactive.sh 2
```

Each available checkpoint is tested with one single prediction and one genuine
ten-record batch prediction. Progress is labelled by GPU and model family.
Per-family reports and logs are saved below a unique timestamped
`validation-runs/` directory, while their complete merged report is written to
`validation-report.json`. The process exits nonzero if any available checkpoint
fails; intentionally absent checkpoints are reported as `SKIP`.

If activation leaves `python` pointing at the EasyBuild base interpreter, the
launcher automatically tries the active Conda prefix and
`$HOME/.conda/envs/absa/bin/python`. An explicit interpreter always wins:

```bash
ABSA_PYTHON="$HOME/.conda/envs/absa/bin/python" \
  bash scripts/run_validation_interactive.sh 2
```

### Queued SLURM validation

To submit the complete test as a one-GPU SLURM job:

```bash
bash scripts/submit_validation_slurm.sh
```

Logs are written to `logs/aspectbench-validate-JOB_ID.{out,err}` and the full
machine-readable result is `validation-report.json`. Override defaults with,
for example, `PARTITION=gpu-a40 TIME_LIMIT=06:00:00 ABSA_MC_PASSES=10`.

Or run it directly:

```bash
python scripts/validate_all.py \
  --model-root models \
  --examples-root examples \
  --device cuda \
  --batch-size 10 \
  --mc-passes 2 \
  --output validation-report.json
```

Add `--require-complete-matrix` when documented missing checkpoint artifacts
should also cause a nonzero exit status.

## Why masked and unmasked weights are separate

A masked checkpoint must remain masked at inference. Exposing target text to a
model trained only with `[ASPECT]` creates a train/inference distribution
mismatch; it does not create an unmasked model. Separate files are therefore
published only for independently trained variants.

The public repositories are linked in the table near the top of this card and
grouped in the AspectBench model collection. `scripts/download.py` restores the
complete repository layout automatically.
