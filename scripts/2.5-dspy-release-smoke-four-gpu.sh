#!/usr/bin/env bash
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; cd "$ROOT"
PYTHON_BIN="${PYTHON_BIN:-/Utilisateurs/nchatt01/.conda/envs/absa/bin/python}"
GPU_IDS="${GPU_IDS:-0,1,2,3}"
RUN_ID="${RUN_ID:-dspy-release-smoke}"
GEMMA_MODEL="${GEMMA_MODEL:-models/gemma3-27b-qat/gemma-3-27b-it-q4_0.gguf}"
LLAMA_SERVER="${LLAMA_SERVER:-./llama.cpp/build/bin/llama-server}"
PORT="${PORT:-18000}"
CONTEXT_SIZE="${CONTEXT_SIZE:-49152}"
PARALLEL="${PARALLEL:-4}"
MC_PASSES="${MC_PASSES:-2}"
START_SERVER="${START_SERVER:-1}"
STOP_SERVER="${STOP_SERVER:-1}"
API_BASE="${API_BASE:-http://127.0.0.1:$PORT/v1}"
DRY_RUN="${DRY_RUN:-0}"
IFS=',' read -r -a gpus <<< "$GPU_IDS"
[[ ${#gpus[@]} -eq 4 ]] || { echo "GPU_IDS must list exactly four GPUs" >&2; exit 2; }
plm_gpus=("${gpus[0]}" "${gpus[1]}" "${gpus[2]}"); server_gpu="${gpus[3]}"
smoke_root="models/_runs/dspy-release-smoke/$RUN_ID"; mkdir -p "$smoke_root/_logs"
server_run="$RUN_ID-gemma"
server_pid=""
cleanup() {
  if [[ "$START_SERVER" == 1 && "$STOP_SERVER" == 1 && -n "$server_pid" ]]; then kill "$server_pid" 2>/dev/null || true; fi
}
trap cleanup EXIT INT TERM
if [[ "$START_SERVER" == 1 ]]; then
  launch=(bash scripts/2.0-serve-llm.sh --backend llama-cpp --model "$GEMMA_MODEL" --model-alias gemma27b --host 127.0.0.1 --port "$PORT" --conda-env vllm --llama-server "$LLAMA_SERVER" --context-size "$CONTEXT_SIZE" --parallel "$PARALLEL" --run-id "$server_run")
  [[ "$DRY_RUN" == 1 ]] && launch+=(--dry-run)
  CUDA_VISIBLE_DEVICES="$server_gpu" "${launch[@]}"
  [[ "$DRY_RUN" == 1 ]] || server_pid="$(<"models/_runs/servers/$server_run/server.pid")"
else
  curl -fsS "${API_BASE%/v1}/v1/models" >/dev/null || { echo "Existing endpoint is unavailable: $API_BASE" >&2; exit 1; }
fi

models_hbs=(xlmr han-xlmr longformer mdeberta-v3 mt5 bertic bge-m3-mlp)
models_sl=(xlmr han-xlmr longformer mdeberta-v3 mt5 sloberta bge-m3-mlp)
best_variant() { case "$1:$2" in hbs:xlmr|hbs:han-xlmr) echo masked;; hbs:*) echo unmasked;; sl:mt5|sl:sloberta) echo unmasked;; sl:*) echo masked;; esac; }
pids=(); labels=(); task=0; status=0
for dataset in hbs sl; do
  [[ "$dataset" == hbs ]] && models=("${models_hbs[@]}") || models=("${models_sl[@]}")
  [[ "$dataset" == hbs ]] && input="huggingface/examples/hbs-tagged-examples.json" || input="huggingface/examples/sl-tagged-synthetic-examples.json"
  for model in "${models[@]}"; do
    plm_variant="$(best_variant "$dataset" "$model")"
    for prompt_variant in masked unmasked; do
      program="selective-deferral-programs/precalibrated/$model/$dataset/$prompt_variant/program.json"
      label="$dataset-$model-$plm_variant-$prompt_variant"
      if [[ ! -s "$program" ]]; then echo "[SKIP $label] no public precalibrated program"; continue; fi
      run="$RUN_ID-$label"; log="$smoke_root/_logs/$label.log"; gpu="${plm_gpus[$((task % 3))]}"; task=$((task + 1))
      command=("$PYTHON_BIN" scripts/2.1-dspy-inference.py --input "$input" --limit 1 --models "$model" --primary-model "$model" --dataset "$dataset" --variant "$plm_variant" --prompt-variant "$prompt_variant" --endpoint-model gemma27b --api-base "$API_BASE" --program-source precalibrated --gate-rate 1.0 --mc-passes "$MC_PASSES" --batch-size 1 --shard-size 1 --device cuda --retry-failed --run-id "$run")
      echo "[RUN $label] PLM GPU $gpu -> $log"
      if [[ "$DRY_RUN" == 1 ]]; then printf 'CUDA_VISIBLE_DEVICES=%q ' "$gpu"; printf '%q ' "${command[@]}"; printf '\n'; continue; fi
      CUDA_VISIBLE_DEVICES="$gpu" "${command[@]}" >"$log" 2>&1 & pids+=("$!"); labels+=("$label")
      if (( ${#pids[@]} >= 3 )); then wait "${pids[0]}" || status=1; pids=("${pids[@]:1}"); labels=("${labels[@]:1}"); fi
    done
  done
done
for index in "${!pids[@]}"; do wait "${pids[$index]}" || status=1; done
[[ "$DRY_RUN" == 1 ]] && { echo "Dry run complete"; exit 0; }
"$PYTHON_BIN" scripts/2.6-verify-dspy-release-smoke.py --run-id "$RUN_ID" --output "$smoke_root/summary.json" || status=1
(( status == 0 )) || { echo "DSPy smoke failed; inspect $smoke_root" >&2; exit 1; }
echo "DSPy release smoke passed: $smoke_root/summary.json"
