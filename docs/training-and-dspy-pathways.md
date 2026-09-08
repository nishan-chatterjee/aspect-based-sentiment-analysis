# Training distributions and DSPy validation

## What Pathway B trains

`aspectbench train` trains exactly one supplied train/validation distribution.
It saves the best validation-Macro-F1 epoch, an optimizer-bearing resume state,
and MC-dropout predictions for every `--uncertainty-input`. It does not silently
create or loop over three splits.

`scripts/3.8-train-split-grid-four-gpu.sh` is the reproducible grid layer:

- with no `SPLIT_INPUT`, it resolves the existing `data/<language>/*_train_val_{0,1,2}.json` files;
- with one `SPLIT_INPUT`, it creates one or three stratified distributions using seeds `1729`, `6174`, and `8191`;
- `SPLIT_COUNT=1` and one model/variant runs one distribution only;
- `MODELS=all` expands to every language-compatible registry model;
- every model/variant/split has a distinct run directory and resumes from the last completed epoch; a preempted current epoch restarts;
- completed MC-dropout shards are reused, so preemption during uncertainty export does not discard finished shards;
- after all tasks finish, `3.9-finalize-training-grid.py` activates the split with the best validation Macro-F1.

The generic trainer initializes from `MODEL_ROOT` (by default the selected
fine-tuned release checkpoint), so it is intended for continued fine-tuning or
dataset transfer. It is not an exact from-base reproduction of every paper
architecture. The dedicated `3.3` BGE-M3 and `3.5` XLM-R/HAN recovery launchers
are the from-base recovery paths currently supported.

Outputs for a grid named `RUN_ID=my-grid` are:

```text
models/<model>/<language>/<variant>/my-grid/<model>/<variant>/split-<n>/
  best-model.pt
  last-training-state.pt
  training-report.json
  uncertainty/test/{shard-*.json,predictions-with-uncertainty.json,_SUCCESS.json}
models/_runs/training-grid/my-grid/
  _data/prepared-splits.json
  _logs/*.log
  split-selection.json
models/_active/<release-family>/<language>/<variant>.pt
```

### Existing three distributions, one model

```bash
PYTHON_BIN=/Utilisateurs/nchatt01/.conda/envs/absa/bin/python \
GPU_IDS=0,1,2,3 DATASET=hbs MODELS=longformer VARIANT=masked \
RUN_ID=longformer-hbs-masked-3split SPLIT_COUNT=3 \
MC_PASSES=8 EPOCHS=3 BATCH_SIZE=2 \
bash scripts/3.8-train-split-grid-four-gpu.sh -- --gradient-accumulation-steps 16
```

Use `SPLIT_COUNT=1` for only distribution zero. To derive three distributions
from a new labeled file, add `SPLIT_INPUT=/path/to/records.json`. A custom
loader is specified as `SPLIT_LOADER=my_package.my_loader:load`; it receives a
`pathlib.Path` and must return a row list, `{"records": [...]}`, or a
`{"train": [...], "val": [...]}` mapping. Every row needs `article`, `aspect`,
and integer `sentiment` in `{-1,0,1}`.

### Four-A40 optimizer-update smoke matrix

The existing smoke path performs one update, verifies non-zero gradients and a
parameter delta, saves/reloads a checkpoint, and keeps logs under
`models/_runs/`:

```bash
PYTHON_BIN=/Utilisateurs/nchatt01/.conda/envs/absa/bin/python \
bash scripts/run-aspectbench.sh --train --smoke \
  --gpus 0,1,2,3 --dataset hbs --models all --variant both \
  --run-id hbs-training-smoke \
  --input huggingface/examples/hbs-tagged-examples.json --batch-size 1
```

Run the analogous command with `--dataset sl` and
`sl-tagged-synthetic-examples.json`. These are mechanics tests, not performance
estimates.

## Public DSPy program smoke

`2.5-dspy-release-smoke-four-gpu.sh` reserves GPUs 0–2 for fine-tuned PLMs and
GPU 3 for one Gemma 27B Q4 llama.cpp server. It forces `GATE_RATE=1`, so passing
requires an actual DSPy response rather than a PLM-only fallback. The verifier
expects 18 public programs and records explicit skips for the ten absent slots:
mT5 (both languages), BGE-M3-MLP (both languages), and Slovenian mDeBERTa-v3,
each with two prompt variants.

```bash
PYTHON_BIN=/Utilisateurs/nchatt01/.conda/envs/absa/bin/python \
GPU_IDS=0,1,2,3 RUN_ID=dspy-public-program-smoke \
GEMMA_MODEL=models/gemma3-27b-qat/gemma-3-27b-it-q4_0.gguf \
LLAMA_SERVER=./llama.cpp/build/bin/llama-server \
CONTEXT_SIZE=49152 PARALLEL=4 \
bash scripts/2.5-dspy-release-smoke-four-gpu.sh
```

On a 48 GB A40/A6000, begin with 49,152 context tokens and four parallel
sequences. Increase in steps of 12,288 only after observing headroom; decrease
by 12,288 on allocation failure. A single 96 GB H100 can generally start at
589,824 tokens and 48 parallel sequences for the paper-era high-throughput
configuration, but actual KV-cache use depends on build, quantization, and
prompt length.

## Optimization/calibration smoke and reusable output

The functional smoke uses GPU 0 for the PLM and GPU 3 for Gemma, optimizes four
train/four validation records, then reloads the new private program for one
forced-deference query:

```bash
PYTHON_BIN=/Utilisateurs/nchatt01/.conda/envs/absa/bin/python \
GPU_IDS=0,1,2,3 RUN_ID=dspy-optimize-smoke \
bash scripts/4.3-dspy-optimization-smoke-four-gpu.sh
```

New programs are written to
`selective-deferral-programs/optimized/<model>/<dataset>/<prompt-variant>/<run-id>/`.
That tree is ignored because optimization examples may be restricted. Audited,
sanitized programs intended for publication belong under the parallel
`precalibrated/` tree. A smoke result proves load/compile/save/reload/query
mechanics only; real calibration should use disjoint authorized train and
validation records without the `--train-limit`/`--val-limit` smoke caps.
