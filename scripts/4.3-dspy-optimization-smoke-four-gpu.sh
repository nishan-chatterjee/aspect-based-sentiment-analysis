#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; cd "$ROOT"
PYTHON_BIN="${PYTHON_BIN:-/Utilisateurs/nchatt01/.conda/envs/absa/bin/python}"
GPU_IDS="${GPU_IDS:-0,1,2,3}"; IFS=',' read -r -a gpus <<< "$GPU_IDS"
[[ ${#gpus[@]} -eq 4 ]] || { echo "GPU_IDS must list exactly four GPUs" >&2; exit 2; }
RUN_ID="${RUN_ID:-dspy-optimization-smoke}"
MODEL="${MODEL:-xlmr}"; DATASET="${DATASET:-hbs}"; VARIANT="${VARIANT:-masked}"; PROMPT_VARIANT="${PROMPT_VARIANT:-masked}"
prefix="$([[ "$DATASET" == hbs ]] && echo hbs || echo slovene)"
TRAIN_INPUT="${TRAIN_INPUT:-data/$DATASET/${prefix}_train_val_0.json}"; VAL_INPUT="${VAL_INPUT:-$TRAIN_INPUT}"
example_name="$([[ "$DATASET" == hbs ]] && echo hbs-tagged-examples || echo sl-tagged-synthetic-examples)"
TEST_INPUT="${TEST_INPUT:-huggingface/examples/$example_name.json}"
GEMMA_MODEL="${GEMMA_MODEL:-models/gemma3-27b-qat/gemma-3-27b-it-q4_0.gguf}"
LLAMA_SERVER="${LLAMA_SERVER:-./llama.cpp/build/bin/llama-server}"; PORT="${PORT:-18000}"
CONTEXT_SIZE="${CONTEXT_SIZE:-49152}"; PARALLEL="${PARALLEL:-4}"; START_SERVER="${START_SERVER:-1}"; STOP_SERVER="${STOP_SERVER:-1}"
API_BASE="${API_BASE:-http://127.0.0.1:$PORT/v1}"; server_pid=""
cleanup() { [[ "$START_SERVER" == 1 && "$STOP_SERVER" == 1 && -n "$server_pid" ]] && kill "$server_pid" 2>/dev/null || true; }
trap cleanup EXIT INT TERM
if [[ "$START_SERVER" == 1 ]]; then
  CUDA_VISIBLE_DEVICES="${gpus[3]}" bash scripts/2.0-serve-llm.sh --backend llama-cpp --model "$GEMMA_MODEL" --model-alias gemma27b --host 127.0.0.1 --port "$PORT" --conda-env vllm --llama-server "$LLAMA_SERVER" --context-size "$CONTEXT_SIZE" --parallel "$PARALLEL" --run-id "$RUN_ID-gemma"
  server_pid="$(<"models/_runs/servers/$RUN_ID-gemma/server.pid")"
fi
echo "Optimizing a private reusable program; PLM uses GPU ${gpus[0]}, Gemma uses GPU ${gpus[3]}."
CUDA_VISIBLE_DEVICES="${gpus[0]}" "$PYTHON_BIN" scripts/4.0-dspy-optimize.py \
  --train-input "$TRAIN_INPUT" --val-input "$VAL_INPUT" --train-limit 4 --val-limit 4 \
  --models "$MODEL" --primary-model "$MODEL" --dataset "$DATASET" --variant "$VARIANT" \
  --prompt-variant "$PROMPT_VARIANT" --endpoint-model gemma27b --api-base "$API_BASE" \
  --teacher-endpoint-model gemma27b --teacher-api-base "$API_BASE" --auto light \
  --mc-passes 2 --batch-size 1 --shard-size 4 --device cuda --run-id "$RUN_ID" --resume
CUDA_VISIBLE_DEVICES="${gpus[0]}" "$PYTHON_BIN" scripts/2.1-dspy-inference.py \
  --input "$TEST_INPUT" --limit 1 --models "$MODEL" --primary-model "$MODEL" \
  --dataset "$DATASET" --variant "$VARIANT" --prompt-variant "$PROMPT_VARIANT" \
  --endpoint-model gemma27b --api-base "$API_BASE" --program-source optimized \
  --program-run-id "$RUN_ID" --gate-rate 1 --mc-passes 2 --batch-size 1 --shard-size 1 \
  --device cuda --retry-failed --run-id "$RUN_ID-query"
"$PYTHON_BIN" -c 'import json,sys; p=json.load(open(sys.argv[1])); assert p and p[0].get("dspy_status") == "complete" and p[0].get("deferred") is True; print("optimization/query smoke passed")' "models/_runs/dspy-inference/$RUN_ID-query/predictions.json"
echo "Program: selective-deferral-programs/optimized/$MODEL/$DATASET/$PROMPT_VARIANT/$RUN_ID/program.json"
