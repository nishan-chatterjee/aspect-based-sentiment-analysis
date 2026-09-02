#!/usr/bin/env bash
set -euo pipefail

ACTION=""
SMOKE="0"
MODELS="all"
GPUS="0"
DATASET=""
VARIANT="best"
RUN_ID=""
OUTPUT_ROOT="outputs"
FILENAME="predictions"
SEED="42"
PYTHON_BIN="${PYTHON_BIN:-python}"
FORWARDED=()

usage() {
  cat <<'EOF'
Usage:
  bash scripts/run-aspectbench.sh --inference [options]
  bash scripts/run-aspectbench.sh --train [--smoke] [options]

Orchestration options:
  --models all|xlmr,longformer,...|"xlmr longformer ..."
  --gpus 0,1,2,3
  --dataset hbs|sl
  --variant best|masked|unmasked|both
  --run-id NAME
  --output-root DIR
  --filename predictions|timestamp|seed|STEM
  --seed INTEGER

Inference schedules model/variant tasks independently, then combines them into
detailed, majority-vote, and confidence-vote files. All remaining flags are
forwarded to the numbered inference or training CLI.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --inference) ACTION="inference"; shift ;;
    --train) ACTION="train"; shift ;;
    --smoke) SMOKE="1"; shift ;;
    --models) MODELS="$2"; shift 2 ;;
    --gpus) GPUS="$2"; shift 2 ;;
    --dataset) DATASET="$2"; FORWARDED+=(--dataset "$2"); shift 2 ;;
    --variant) VARIANT="$2"; shift 2 ;;
    --run-id) RUN_ID="$2"; shift 2 ;;
    --output-root) OUTPUT_ROOT="$2"; shift 2 ;;
    --filename) FILENAME="$2"; shift 2 ;;
    --seed) SEED="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) FORWARDED+=("$1"); shift ;;
  esac
done
[[ "$ACTION" == "inference" || "$ACTION" == "train" ]] || { usage; exit 2; }
[[ "$DATASET" == "hbs" || "$DATASET" == "sl" ]] || {
  echo "--dataset must be hbs or sl" >&2; exit 2;
}
[[ "$VARIANT" =~ ^(best|masked|unmasked|both)$ ]] || {
  echo "--variant must be best, masked, unmasked, or both" >&2; exit 2;
}
[[ -n "$RUN_ID" ]] || { echo "--run-id is required" >&2; exit 2; }

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
IFS=',' read -r -a model_array <<< "$MODELS"
IFS=',' read -r -a gpu_array <<< "$GPUS"

best_variant() {
  case "$DATASET:$1" in
    hbs:xlmr|hbs:han-xlmr) echo masked ;;
    hbs:longformer|hbs:mdeberta-v3|hbs:mt5|hbs:bertic|hbs:bge-m3-mlp) echo unmasked ;;
    sl:xlmr|sl:han-xlmr|sl:longformer|sl:mdeberta-v3|sl:bge-m3-mlp) echo masked ;;
    sl:mt5|sl:sloberta) echo unmasked ;;
    *) echo "No best variant mapping for $DATASET/$1" >&2; return 2 ;;
  esac
}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [[ "$ACTION" == "inference" ]]; then
  ENTRY="$SCRIPT_DIR/1.0-models-inference.py"
elif [[ "$SMOKE" == "1" ]]; then
  ENTRY="$SCRIPT_DIR/3.2-models-finetune-smoke.py"
else
  ENTRY="$SCRIPT_DIR/3.0-models-finetune.py"
fi
mkdir -p models/_runs/_launcher-logs

task_models=()
task_variants=()
for model in "${model_array[@]}"; do
  [[ -n "$model" ]] || continue
  if [[ "$VARIANT" == "both" ]]; then
    task_models+=("$model" "$model")
    task_variants+=(masked unmasked)
  elif [[ "$VARIANT" == "best" ]]; then
    task_models+=("$model")
    task_variants+=("$(best_variant "$model")")
  else
    task_models+=("$model")
    task_variants+=("$VARIANT")
  fi
done

pids=()
prediction_paths=()
for index in "${!task_models[@]}"; do
  model="${task_models[$index]}"
  prompt_variant="${task_variants[$index]}"
  gpu="${gpu_array[$((index % ${#gpu_array[@]}))]}"
  task_run_id="${RUN_ID}-${model}-${prompt_variant}"
  log="models/_runs/_launcher-logs/${task_run_id}.log"
  availability_flags=()
  [[ "$ALL_REQUESTED" == "1" ]] && availability_flags+=(--skip-unavailable)
  publish_flags=()
  if [[ "$ACTION" == "inference" ]]; then
    publish_flags+=(--no-publish)
    prediction_paths+=("models/_runs/inference/${task_run_id}/predictions.json")
  fi
  echo "[$ACTION/$model/$prompt_variant] GPU $gpu; log $log"
  CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON_BIN" "$ENTRY" \
    --models "$model" --variant "$prompt_variant" --run-id "$task_run_id" \
    --seed "$SEED" "${availability_flags[@]}" "${publish_flags[@]}" \
    "${FORWARDED[@]}" >"$log" 2>&1 &
  pids+=("$!")
  if (( ${#pids[@]} >= ${#gpu_array[@]} )); then
    wait "${pids[0]}"
    pids=("${pids[@]:1}")
  fi
done
for pid in "${pids[@]}"; do wait "$pid"; done
if [[ "$ACTION" == "inference" ]]; then
  "$PYTHON_BIN" "$SCRIPT_DIR/1.2-aggregate-inference.py" \
    --predictions "${prediction_paths[@]}" --dataset "$DATASET" \
    --run-id "$RUN_ID" --output-root "$OUTPUT_ROOT" \
    --filename "$FILENAME" --seed "$SEED"
fi
echo "All requested tasks completed. Logs: models/_runs/_launcher-logs/"
