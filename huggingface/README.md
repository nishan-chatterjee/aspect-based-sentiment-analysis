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

The intended grid is 7 families × 2 languages × 2 modes = 28 slots. Twenty
trained checkpoint files are currently recoverable:

| Family | HBS masked | HBS unmasked | SL masked | SL unmasked |
|---|---:|---:|---:|---:|
| XLM-R | yes | unavailable artifact | yes | unavailable artifact |
| HAN-XLM-R | yes | unavailable artifact | yes | unavailable artifact |
| XLM-R Longformer | yes | yes | yes | yes |
| mDeBERTa-v3 | yes | yes | yes | yes |
| mT5 | yes | yes | yes | yes |
| BERTić / SloBERTa | yes | yes | yes | yes |
| BGE-M3 + MLP | unavailable artifact | unavailable artifact | unavailable artifact | unavailable artifact |

The eight unavailable mode-specific slots span six model-language
combinations: XLM-R/HBS, XLM-R/Slovenian, HAN-XLM-R/HBS,
HAN-XLM-R/Slovenian, BGE-M3/HBS, and BGE-M3/Slovenian. The count is eight
rather than six because both masked and unmasked BGE-M3 heads are absent in
each language. These are not tokenizer failures. Validation metrics identify
the selected experiments, but their original checkpoint files are absent:
two XLM-R unmasked checkpoints, two HAN-XLM-R unmasked checkpoints, and four
BGE-M3 MLP heads. Scores and availability reasons are recorded in
`models/manifest.json` and each `availability.json`. They must be
recovered from an archive or retrained before they can be uploaded; no masked
or different-family checkpoint is relabeled as a substitute.

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

## Model repositories

- `nishan-chatterjee/aspectbench-xlmr`
- `nishan-chatterjee/aspectbench-han-xlmr`
- `nishan-chatterjee/aspectbench-longformer`
- `nishan-chatterjee/aspectbench-mdeberta-v3`
- `nishan-chatterjee/aspectbench-mt5`
- `nishan-chatterjee/aspectbench-slavic-specific`
- `nishan-chatterjee/aspectbench-bge-m3-mlp` (metadata only until the missing
  trained MLP heads are recovered or retrained)

They are grouped in the `nishan-chatterjee/aspect-based-sentiment-analysis`
collection. `scripts/download.py` restores all repositories to this directory
structure automatically.
