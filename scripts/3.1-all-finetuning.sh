#!/usr/bin/env bash
set -euo pipefail

GPUS=""
MODELS="all"
RUN_PREFIX=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --gpus) GPUS="$2"; shift 2 ;;
    --models) MODELS="$2"; shift 2 ;;
    --run-prefix) RUN_PREFIX="$2"; shift 2 ;;
    --) shift; break ;;
    -h|--help)
      echo "Usage: $0 --gpus 0,1 --models all|xlmr,longformer --run-prefix ID -- [training flags]"
      exit 0 ;;
    *) echo "Unknown launcher flag $1" >&2; exit 2 ;;
  esac
done
[[ -n "$GPUS" && -n "$RUN_PREFIX" ]] || {
  echo "--gpus and --run-prefix are required" >&2; exit 2;
}
DATASET=""
previous=""
for argument in "$@"; do
  [[ "$previous" == "--dataset" ]] && DATASET="$argument"
  previous="$argument"
done
ALL_REQUESTED="0"
if [[ "$MODELS" == "all" ]]; then
  ALL_REQUESTED="1"
  if [[ "$DATASET" == "hbs" ]]; then
    MODELS="xlmr,han-xlmr,longformer,mdeberta-v3,mt5,bertic,bge-m3-mlp"
  else
    MODELS="xlmr,han-xlmr,longformer,mdeberta-v3,mt5,sloberta,bge-m3-mlp"
  fi
fi
MODELS="${MODELS// /,}"
IFS=',' read -r -a gpu_array <<< "$GPUS"
IFS=',' read -r -a model_array <<< "$MODELS"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
mkdir -p models/_runs/_launcher-logs
pids=()
for index in "${!model_array[@]}"; do
  model="${model_array[$index]}"
  [[ -n "$model" ]] || continue
  gpu="${gpu_array[$((index % ${#gpu_array[@]}))]}"
  log="models/_runs/_launcher-logs/${RUN_PREFIX}-${model}.log"
  availability_flags=()
  [[ "$ALL_REQUESTED" == "1" ]] && availability_flags+=(--skip-unavailable)
  echo "[$model] CUDA device $gpu; log $log"
  CUDA_VISIBLE_DEVICES="$gpu" python "$SCRIPT_DIR/3.0-models-finetune.py" \
    --models "$model" --run-id "${RUN_PREFIX}-${model}" \
    "${availability_flags[@]}" "$@" >"$log" 2>&1 &
  pids+=("$!")
  if (( ${#pids[@]} >= ${#gpu_array[@]} )); then
    wait "${pids[0]}"
    pids=("${pids[@]:1}")
  fi
done
for pid in "${pids[@]}"; do wait "$pid"; done
