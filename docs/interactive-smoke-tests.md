# Interactive GPU runbook

Run from the repository root after acquiring an interactive node. Workflows
resume by default. Run state, atomic shards, progress, and logs live below
`models/_runs/`; newly trained checkpoints live below `models/MODEL/...`.

## 0. Install the command and define a test article

```bash
source /opt/easybuild/software/Anaconda3/2024.02-1/etc/profile.d/conda.sh
conda activate absa
python -m pip install -e . --no-deps
aspectbench models --models all

ARTICLE='Tokom šestonedeljnog testiranja, redakcija je više puta kontaktirala <aspect>Primer Grupu</aspect> zbog nove usluge. Prvi odgovor <aspect>Primer Grupe</aspect> stigao je istog dana, a tehnički tim je zatim otklonio prijavljenu grešku bez dodatnih troškova. U završnom upitniku većina korisnika ocenila je podršku kao jasnu i pouzdanu.'
```

The registry contains `xlmr`, `han-xlmr`, `longformer`, `mdeberta-v3`, `mt5`,
`bertic` (HBS), `sloberta` (Slovenian), and `bge-m3-mlp`. Any shorter model
list below is an example.

## 1. Predict and estimate uncertainty

One model and one document:

```bash
CUDA_VISIBLE_DEVICES=0 aspectbench infer \
  --models longformer --dataset hbs --variant masked \
  --input-doc "$ARTICLE" --mc-passes 8 --batch-size 1 \
  --run-id hbs-longformer-document-smoke
```

One, a few, or all models from a file:

```bash
CUDA_VISIBLE_DEVICES=0,1 bash scripts/run-aspectbench.sh --inference \
  --gpus 0,1 --models 'xlmr,longformer,mdeberta-v3' --dataset hbs \
  --variant best --run-id hbs-few-inference \
  --input data/hbs/hbs_test.json --limit 8 --mc-passes 8

CUDA_VISIBLE_DEVICES=0,1,2,3 bash scripts/run-aspectbench.sh --inference \
  --gpus 0,1,2,3 --models all --dataset sl --variant best \
  --run-id sl-all-inference --input data/sl/slovene_test.json \
  --limit 8 --mc-passes 8
```

Each prediction contains class probabilities, confidence, entropy, top-two
margin, MC expected entropy, mutual information, agreement, and variation
ratio. An explicit unavailable model fails; `all` logs and skips unavailable
released checkpoints.

## 2. Full training and one-update smoke

Full training selects the best validation Macro-F1 checkpoint, stores the last
optimizer state for exact resumption, then runs MC uncertainty on validation
and every `--uncertainty-input NAME=PATH` split.

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 bash scripts/run-aspectbench.sh --train \
  --gpus 0,1,2,3 --models all --dataset hbs --variant best \
  --run-id hbs-full-train --train-input data/hbs/hbs_train_val_0.json \
  --val-input data/hbs/hbs_train_val_0.json \
  --uncertainty-input test=data/hbs/hbs_test.json \
  --epochs 3 --batch-size 4 --mc-passes 10
```

The run-specific best checkpoint remains under `models/MODEL/.../RUN-ID/` and
is also activated under `models/_active/` in the release-style layout. Add
`--model-root models/_active` to later inference, DSPy query, or optimization
commands to use user-trained checkpoints instead of `huggingface/models`.

To prove forward/backward/update/save/reload without retraining:

```bash
CUDA_VISIBLE_DEVICES=0 bash scripts/run-aspectbench.sh --train --smoke \
  --gpus 0 --models longformer --dataset hbs --variant masked \
  --run-id longformer-update-smoke --input-doc "$ARTICLE" \
  --sentiment 1 --batch-size 1 --learning-rate 1e-5
```

## 3. Serve Gemma and Qwen with vLLM

Activate `vllm` in each server terminal. These commands use two GPUs per
server (four total). The default student port is 8000 and teacher port 8001.

```bash
source /opt/easybuild/software/Anaconda3/2024.02-1/etc/profile.d/conda.sh
conda activate vllm

CUDA_VISIBLE_DEVICES=0,1 vllm serve \
  models/gemma3-27b-qat/gemma-3-27b-it-q4_0.gguf \
  --tokenizer models/gemma3-27b-qat --served-model-name gemma27b \
  --host 0.0.0.0 --port 8000 --tensor-parallel-size 2 \
  --max-model-len 12288 --gpu-memory-utilization 0.94
```

```bash
CUDA_VISIBLE_DEVICES=2,3 vllm serve \
  models/qwen2.5-72b/qwen2.5-72b-instruct-q4_k_m.gguf \
  --tokenizer models/qwen2.5-72b --served-model-name qwen72b \
  --host 0.0.0.0 --port 8001 --tensor-parallel-size 2 \
  --max-model-len 12288 --gpu-memory-utilization 0.94 \
  --chat-template backup/camera-ready/configs/chat-templates/qwen2_5_instruct_chat_template.jinja
```

The reusable launcher provides the same health checks, PID, manifest, and log:

```bash
CUDA_VISIBLE_DEVICES=0,1 bash scripts/2.0-serve-llm.sh \
  --backend vllm --model models/gemma3-27b-qat/gemma-3-27b-it-q4_0.gguf \
  --tokenizer models/gemma3-27b-qat --model-alias gemma27b \
  --tensor-parallel-size 2 --max-model-len 12288 --port 8000 \
  --run-id gemma-vllm
```

Actual fit depends on the vLLM version’s GGUF support and tokenizer/config
files. Use the llama.cpp path below if the quantized model does not load.

## 4. Serve one Gemma per GPU with llama.cpp

For a 48 GB A40/A6000, begin at `-c 196608 -np 16`: 12,288 context tokens
per slot. If memory is insufficient, reduce both by the same factor:
`98304/8`, then `49152/4`. For a 96 GB H100, `589824/48` retains the same
per-slot budget. Do not reduce `-c` alone while keeping too many slots.

```bash
# Four independent 48 GB GPU endpoints at ports 18000-18003.
GPU_IDS=0,1,2,3 PORTS=18000,18001,18002,18003 \
LLAMA_SERVER=./llama.cpp/build/bin/llama-server \
MODEL=models/gemma3-27b-qat/gemma-3-27b-it-q4_0.gguf \
CONTEXT_SIZE=196608 PARALLEL=16 \
bash scripts/2.3-launch-gemma-llama-cpp.sh

# One 96 GB H100.
CUDA_VISIBLE_DEVICES=0 bash scripts/2.0-serve-llm.sh \
  --backend llama-cpp --llama-server ./llama.cpp/build/bin/llama-server \
  --model models/gemma3-27b-qat/gemma-3-27b-it-q4_0.gguf \
  --model-alias gemma27b --host 0.0.0.0 --port 8000 \
  --context-size 589824 --parallel 48 --run-id gemma-h100
```

## 5. Query a precalibrated DSPy program

DSPy first reuses the fine-tuned PLM(s) to produce uncertainty, then queries
the program only for the lowest-confidence `GATE_RATE` fraction. PLM
`VARIANT_MODE` and DSPy `PROMPT_VARIANTS` are independent settings.

```bash
source /opt/easybuild/software/Anaconda3/2024.02-1/etc/profile.d/conda.sh
conda activate vllm

PYTHON_BIN=/Utilisateurs/nchatt01/.conda/envs/vllm/bin/python \
DATASET=hbs INPUT=data/hbs/hbs_test.json TASK_FILTER=all \
VARIANT_MODE=best PROMPT_VARIANTS='masked unmasked' GATE_RATE=0.10 \
STUDENT_API_BASES='http://127.0.0.1:18000/v1,http://127.0.0.1:18001/v1,http://127.0.0.1:18002/v1,http://127.0.0.1:18003/v1' \
NUM_WORKERS_PER_ENDPOINTS='12,12,12,12' MAX_PARALLEL=1 \
SKIP_ENDPOINT_CHECK=0 PROGRAM_SOURCE=precalibrated \
bash scripts/2.4-query-dspy-programs.sh
```

Run several gate rates by wrapping the same command:

```bash
for G in 0.10 0.20 0.30; do
  PYTHON_BIN=/Utilisateurs/nchatt01/.conda/envs/vllm/bin/python \
  DATASET=hbs INPUT=data/hbs/hbs_test.json TASK_FILTER=all \
  VARIANT_MODE=best PROMPT_VARIANTS='masked unmasked' GATE_RATE="$G" \
  STUDENT_API_BASES='http://l3icalcul07:18000/v1,http://l3icalcul07:18001/v1,http://l3icalcul07:18002/v1,http://l3icalcul07:18003/v1' \
  NUM_WORKERS_PER_ENDPOINTS='12,12,12,12' \
  bash scripts/2.4-query-dspy-programs.sh
done
```

For one record, use the direct CLI and replace `--input` with
`--input-doc "$ARTICLE"`. A packaged program is resolved automatically:

```bash
aspectbench defer-query --models longformer mdeberta-v3 bertic \
  --primary-model longformer --dataset hbs --variant unmasked \
  --prompt-variant masked --input-doc "$ARTICLE" \
  --endpoint-model gemma27b --api-base http://127.0.0.1:8000/v1 \
  --program-source precalibrated --gate-rate 1.0 --mc-passes 8 \
  --run-id hbs-dspy-document-smoke
```

## 6. Optimize and reuse a new DSPy program

Optimization uses labeled training/validation uncertainty only. It never
selects prompts on the held-out test split. Student Gemma handles task calls;
teacher Qwen proposes instructions. New programs are written to the ignored
tree `selective-deferral-programs/optimized/MODEL/DATASET/PROMPT/RUN-ID/`.

```bash
PYTHON_BIN=/Utilisateurs/nchatt01/.conda/envs/vllm/bin/python \
STUDENT_API_BASE=http://127.0.0.1:8000/v1 \
TEACHER_API_BASE=http://127.0.0.1:8001/v1 \
STUDENT_MODEL=gemma27b TEACHER_MODEL=qwen72b \
DATASET=hbs TRAIN_INPUT=data/hbs/hbs_train_val_0.json \
VAL_INPUT=data/hbs/hbs_train_val_0.json TASK_FILTER=longformer \
VARIANT_MODE=best PROMPT_VARIANTS='masked unmasked' \
MAX_PARALLEL=1 SKIP_ENDPOINT_CHECK=0 AUTO=light \
bash scripts/4.2-optimize-dspy-programs.sh
```

Reuse a generated program by setting its exact run ID:

```bash
PYTHON_BIN=/Utilisateurs/nchatt01/.conda/envs/vllm/bin/python \
DATASET=hbs INPUT=data/hbs/hbs_test.json TASK_FILTER=longformer \
VARIANT_MODE=best PROMPT_VARIANTS=masked PROGRAM_SOURCE=optimized \
PROGRAM_RUN_ID=dspy-optimize-hbs-longformer-unmasked-masked \
STUDENT_API_BASE=http://127.0.0.1:8000/v1 GATE_RATE=0.10 \
bash scripts/2.4-query-dspy-programs.sh
```

## 7. Train the four missing BGE-M3 + MLP heads on four GPUs

This recovery launcher assigns HBS masked/unmasked and Slovenian
masked/unmasked to GPUs 0–3. Each task caches embeddings once and trains all
three fixed split files. Keep the same `RUN_ID` when resuming.

```bash
source /opt/easybuild/software/Anaconda3/2024.02-1/etc/profile.d/conda.sh
conda activate absa
cd /Utilisateurs/nchatt01/GitHub/aspect-based-sentiment-analysis

PYTHON_BIN=/Utilisateurs/nchatt01/.conda/envs/absa/bin/python \
GPU_IDS=0,1,2,3 RUN_ID=bge-m3-paper-recovery \
EMBEDDING_BATCH_SIZE=8 EMBEDDING_PRECISION=float32 \
bash scripts/3.3-train-bge-m3-mlp-four-gpu.sh
```

In a second terminal, monitor without interrupting the jobs:

```bash
watch -n 5 nvidia-smi
tail -F huggingface/models/bge-m3-mlp/training/runs/bge-m3-paper-recovery/_logs/*.log
```

If embedding OOMs, rerun the same command with
`EMBEDDING_BATCH_SIZE=4` (then `2` if necessary). Finished embedding shards
and epochs are retained. The BGE revision is pinned and recorded in the cache
manifest. After all four processes succeed, the launcher
updates local release metadata and writes:

```text
huggingface/models/bge-m3-mlp/training/runs/bge-m3-paper-recovery/comparison-to-paper.json
```

Validate all four promoted heads before considering upload:

```bash
python huggingface/scripts/validate_all.py \
  --model bge-m3-mlp --model-root huggingface/models \
  --examples-root huggingface/examples --device cuda --batch-size 10 \
  --mc-passes 2 --require-complete-matrix \
  --output huggingface/models/bge-m3-mlp/training/runs/bge-m3-paper-recovery/inference-validation.json

# This only prints the intended private upload; it does not upload.
python huggingface/scripts/upload.py --root huggingface --model bge-m3-mlp
```

Do not add `--execute` until the comparison and inference-validation reports
have been reviewed.

## 8. Monitor, resume, and remove smoke artifacts

```bash
aspectbench progress models/_runs/inference/hbs-longformer-document-smoke
tail -f models/_runs/inference/hbs-longformer-document-smoke/_logs/inference.log
tail -f models/_runs/servers/gemma-vllm/_logs/server.log
find models/_runs -name progress.json -print
```

Rerun the same command and `--run-id` to resume. `--no-resume` fails if a run
already exists. Stop a managed server with the PID in its run directory, for
example `kill "$(cat models/_runs/servers/gemma-vllm/server.pid)"`. Remove only
the explicit smoke run and matching `models/MODEL/LANGUAGE/VARIANT/.../RUN-ID`
checkpoint directory after inspection; training and inference shards are
otherwise reusable.
