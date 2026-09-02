#!/usr/bin/env bash
set -euo pipefail

# One llama.cpp server per visible GPU. Override these environment variables
# for a different node, model, binary, or context budget.
GPU_IDS="${GPU_IDS:-0,1,2,3}"
PORTS="${PORTS:-18000,18001,18002,18003}"
MODEL="${MODEL:-models/gemma3-27b-qat/gemma-3-27b-it-q4_0.gguf}"
MODEL_ALIAS="${MODEL_ALIAS:-gemma27b}"
LLAMA_SERVER="${LLAMA_SERVER:-./llama.cpp/build/bin/llama-server}"
CONTEXT_SIZE="${CONTEXT_SIZE:-196608}"
PARALLEL="${PARALLEL:-16}"
RUN_PREFIX="${RUN_PREFIX:-gemma27b-llama-cpp}"

IFS=',' read -r -a gpu_array <<< "$GPU_IDS"
IFS=',' read -r -a port_array <<< "$PORTS"
if (( ${#gpu_array[@]} != ${#port_array[@]} )); then
  echo "GPU_IDS and PORTS must contain the same number of entries." >&2
  exit 2
fi
if (( CONTEXT_SIZE != PARALLEL * 12288 )); then
  echo "CONTEXT_SIZE must equal PARALLEL * 12288 for the documented per-slot budget." >&2
  echo "Examples: 196608/16, 98304/8, 49152/4, 589824/48." >&2
  exit 2
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
pids=()
for index in "${!gpu_array[@]}"; do
  gpu="${gpu_array[$index]}"
  port="${port_array[$index]}"
  echo "Starting $MODEL_ALIAS on GPU $gpu, port $port"
  CUDA_VISIBLE_DEVICES="$gpu" bash "$SCRIPT_DIR/2.0-serve-llm.sh" \
    --backend llama-cpp --conda-env vllm --llama-server "$LLAMA_SERVER" \
    --model "$MODEL" --model-alias "$MODEL_ALIAS" --host 0.0.0.0 \
    --port "$port" --context-size "$CONTEXT_SIZE" --parallel "$PARALLEL" \
    --run-id "$RUN_PREFIX-$port" &
  pids+=("$!")
done
for pid in "${pids[@]}"; do wait "$pid"; done
echo "All endpoints passed health checks; server PIDs are under models/_runs/servers/."
