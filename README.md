# AspectBench

AspectBench is a reusable toolkit for document-level aspect-based sentiment
analysis in HBS and Slovenian news. It provides a shared model registry,
training and inference interfaces, uncertainty and selective-deferral tools,
and evaluation designed for class imbalance and seen/unseen target analysis.

This branch is the clean implementation. The provenance-heavy paper release is
preserved separately; historical filenames are not part of the public API.

## Included workflows

The clean release provides:

- one module per model architecture or language-specific adapter;
- canonical `hbs` and `sl` language identifiers with legacy aliases;
- reusable overall, per-class, per-aspect, and seen/unseen scoring;
- synthetic examples, tests, and an executable evaluation notebook;
- unified one/few/all pretrained inference with MC-dropout uncertainty;
- a one-update fine-tuning smoke that saves and strictly reloads its best
  checkpoint without touching release weights;
- OpenAI-compatible vLLM and llama.cpp serving, DSPy selective inference, and
  resumable MIPROv2 optimization;
- atomic shards, manifests, progress JSON, and file logs under `models/_runs`;
- 18 audited selective-deferral programs (four explicitly sanitized);
- strict ignore rules for real data, checkpoints, record outputs, and private
  DSPy originals;
- the already validated `huggingface/` release, unchanged.

## Install for development

```bash
python -m pip install -e '.[dev]'
pytest
```

On the cluster, activate the existing environment first:

```bash
source /opt/easybuild/software/Anaconda3/2024.02-1/etc/profile.d/conda.sh
conda activate absa
python -m pip install -e . --no-deps
```

## Inspect the model registry

```bash
aspectbench models --models all
aspectbench models --models longformer bertic --language hbs
```

## GPU smoke tests

The full interactive runbook contains copy/paste commands for one, a few, or
all models on `CUDA_VISIBLE_DEVICES=0,1` or `0,1,2,3`:

```bash
CUDA_VISIBLE_DEVICES=0,1 python scripts/1.0-models-inference.py \
  --models longformer --dataset hbs --variant masked \
  --input examples.json --mc-passes 8 --run-id inference-smoke

CUDA_VISIBLE_DEVICES=0 python scripts/3.0-models-finetune.py \
  --models longformer --dataset hbs --variant masked \
  --input labeled-smoke.json --batch-size 1 --run-id update-smoke
```

See [docs/interactive-smoke-tests.md](docs/interactive-smoke-tests.md) for
single-document input, parallel launchers, server health checks, DSPy query and
optimization, progress monitoring, resumption, and cleanup.
The checks already completed during the refactor are recorded in
[docs/validation-report.md](docs/validation-report.md).

## Score predictions

Prediction records use original labels (`-1`, `0`, `1`) and contain `aspect`,
`sentiment`, and `prediction` fields:

```bash
aspectbench score \
  --predictions examples/synthetic/predictions.json \
  --train-data examples/synthetic/train.json \
  --output runs/synthetic-report.json
```

The report includes overall Macro-F1 and quadratic weighted kappa, per-class
precision/recall/F1/support, imbalance diagnostics, Macro-F1 and QWK for every
aspect, macro averages across aspects, and seen/unseen target results. See
`notebooks/evaluation-and-aspect-reporting.ipynb` for an interactive example.

## Data and model storage

- HBS data is obtained through the authenticated CLARIN release.
- Slovenian data is private and must be imported from authorized local storage.
- Released pretrained artifacts live under `huggingface/` or are downloaded
  from the associated Hugging Face collection.
- Newly trained checkpoints go under `models/`, which is ignored by Git.
- Exact DSPy programs derived from restricted records go under
  `selective-deferral-programs/private-originals/`, also ignored by Git.

See `data/README.md` and `selective-deferral-programs/README.md` before adding
artifacts. See `docs/extending-models-and-data.md` for the kebab-case model
extension protocol and custom dataset contract.
