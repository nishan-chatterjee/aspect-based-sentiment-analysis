# Interactive GPU smoke tests

Run these commands from the repository root after acquiring an interactive
node. Every workflow resumes by default. State lives under `models/_runs/`,
model smoke checkpoints under `models/<model>/<language>/<variant>/smoke/`,
and logs under each run's `_logs/` directory.

## Environment and test article

```bash
source /opt/easybuild/software/Anaconda3/2024.02-1/etc/profile.d/conda.sh
conda activate absa
python -m pip install -e . --no-deps

ARTICLE='Tokom šestonedeljnog testiranja, redakcija je više puta kontaktirala <aspect>Primer Grupu</aspect> zbog nove usluge. Prvi odgovor <aspect>Primer Grupe</aspect> stigao je istog dana, a tehnički tim je zatim otklonio prijavljenu grešku bez dodatnih troškova. U završnom upitniku većina korisnika ocenila je podršku kao jasnu i pouzdanu.'
```

## 1. Pretrained inference and uncertainty

One model on the first visible GPU:

```bash
CUDA_VISIBLE_DEVICES=0,1 python scripts/1.0-models-inference.py \
  --models longformer --dataset hbs --variant masked \
  --input-doc "$ARTICLE" --mc-passes 8 --batch-size 1 \
  --run-id hbs-longformer-input-smoke
```

The result is
`models/_runs/inference/hbs-longformer-input-smoke/predictions.json`. It
contains the class probabilities, predictive entropy, confidence, top-two
margin, and MC-dropout mutual information/agreement.

Use `--input examples.json` for one or many JSON/JSONL records. Run a few
models concurrently on two GPUs:

```bash
bash scripts/1.1-all-inference.sh \
  --gpus 0,1 --models xlmr,longformer,mdeberta-v3 \
  --run-prefix hbs-few-smoke -- \
  --dataset hbs --variant masked --input examples.json \
  --mc-passes 8 --batch-size 4 --shard-size 16
```

Replace the model list with `all` and use `--gpus 0,1,2,3` for the complete
available grid. Missing release checkpoints are reported as skips only in the
all-model selection; an explicitly requested missing checkpoint fails loudly.

## 2. Start Gemma with llama.cpp or vLLM

Test one backend at a time. The launcher starts the server in the background,
writes its PID/log/manifest, and verifies `/v1/models`.

llama.cpp on two visible GPUs (the local llama.cpp build controls how layers
are split):

```bash
CUDA_VISIBLE_DEVICES=0,1 bash scripts/2.0-serve-llm.sh \
  --backend llama-cpp --conda-env vllm \
  --llama-server /path/to/llama.cpp/build/bin/llama-server \
  --model /path/to/gemma-3-27b-it-q4_0.gguf \
  --model-alias gemma27b --split-mode layer \
  --port 8000 --run-id gemma-llama-cpp
```

vLLM tensor-parallel on two GPUs:

```bash
CUDA_VISIBLE_DEVICES=0,1 bash scripts/2.0-serve-llm.sh \
  --backend vllm --conda-env vllm \
  --model /path/to/gemma-3-27b-it-q4_0.gguf \
  --tokenizer /path/to/gemma-3-27b-text-config \
  --model-alias gemma27b --tensor-parallel-size 2 \
  --port 8000 --run-id gemma-vllm
```

Add `--dry-run` to inspect either resolved command without launching it.

## 3. DSPy inference with a packaged program

Activate the DSPy environment while the endpoint remains running:

```bash
conda activate vllm
CUDA_VISIBLE_DEVICES=2 python scripts/2.1-dspy-inference.py \
  --models longformer mdeberta-v3 bertic --primary-model longformer \
  --dataset hbs --variant masked --input examples.json \
  --endpoint-model gemma27b --api-base http://127.0.0.1:8000 \
  --program selective-deferral-programs/longformer/hbs/masked/program.json \
  --gate-rate 1.0 --mc-passes 8 --run-id hbs-dspy-smoke
```

`--gate-rate 1.0` queries every smoke record; production values such as 0.10,
0.20, or 0.30 query the least-confident fraction. Ungated records keep the PLM
prediction. Endpoint failures are recorded as `failed-fallback`, keeping the
primary prediction without losing completed shards. Add `--retry-failed` when
resuming to query only those failed records again; the merged output keeps the
latest version of each record.

The same command accepts `--input-doc "$ARTICLE"` instead of a file. Select a
different packaged or newly optimized JSON with `--program`; that resolved
path is recorded in the run manifest.

## 4. Real one-update fine-tuning smoke

The article needs a gold label for a training update:

```bash
conda activate absa
CUDA_VISIBLE_DEVICES=0 python scripts/3.0-models-finetune.py \
  --models longformer --dataset hbs --variant masked \
  --input-doc "$ARTICLE" --sentiment 1 --batch-size 1 \
  --learning-rate 1e-5 --run-id longformer-weight-update-smoke
```

This performs forward, backward, one AdamW step, save, strict reload, and one
post-reload prediction. Inspect
`models/_runs/training-smoke/longformer-weight-update-smoke/training-smoke-report.json`
and the checkpoint under
`models/longformer/hbs/masked/smoke/longformer-weight-update-smoke/`.

Use the parallel launcher for a few or all models:

```bash
bash scripts/3.1-all-finetuning-smoke.sh \
  --gpus 0,1,2,3 --models all --run-prefix hbs-all-train-smoke -- \
  --dataset hbs --variant masked --input labeled-smoke.json \
  --batch-size 1 --learning-rate 1e-5
```

## 5. Optimize, save, reload, and query a DSPy program

Use small, labeled train/validation files for the calibration smoke. They must
be authorized local files and are never committed.

```bash
conda activate vllm
CUDA_VISIBLE_DEVICES=2 python scripts/4.0-dspy-optimize.py \
  --train-input data/hbs/smoke-train.json \
  --val-input data/hbs/smoke-val.json \
  --models longformer mdeberta-v3 bertic --primary-model longformer \
  --dataset hbs --variant masked \
  --endpoint-model gemma27b --api-base http://127.0.0.1:8000 \
  --auto light --mc-passes 8 --run-id hbs-longformer-opt-smoke
```

The reusable output is
`models/_runs/dspy-optimization/hbs-longformer-opt-smoke/optimized-program.json`.
Pass that exact path to `scripts/2.1-dspy-inference.py --program ...` to verify
serialization and reuse.

## 6. Monitor, resume, and clean up

```bash
python scripts/9.0-progress.py models/_runs/inference/hbs-longformer-input-smoke
tail -f models/_runs/inference/hbs-longformer-input-smoke/_logs/inference.log
tail -f models/_runs/servers/gemma-vllm/_logs/server.log
```

Rerun the same command and `--run-id` to resume. Use `--no-resume` when a
pre-existing run should be treated as an error. Shards are written atomically
before progress advances. When a smoke is no longer needed, remove only its
explicit run directory and matching `models/<model>/.../smoke/<run-id>`
directory. Stop a server with the PID recorded in its run directory, for
example `kill "$(cat models/_runs/servers/gemma-vllm/server.pid)"`.
