#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python}"
DATASET="${DATASET:?Set DATASET=hbs or DATASET=sl}"
TRAIN_INPUT="${TRAIN_INPUT:?Set TRAIN_INPUT to authorized calibration training JSON}"
VAL_INPUT="${VAL_INPUT:?Set VAL_INPUT to authorized validation JSON}"
TASK_FILTER="${TASK_FILTER:-all}"
VARIANT_MODE="${VARIANT_MODE:-best}"
PROMPT_VARIANTS="${PROMPT_VARIANTS:-masked unmasked}"
STUDENT_API_BASE="${STUDENT_API_BASE:-http://127.0.0.1:8000/v1}"
TEACHER_API_BASE="${TEACHER_API_BASE:-http://127.0.0.1:8001/v1}"
STUDENT_MODEL="${STUDENT_MODEL:-gemma27b}"
TEACHER_MODEL="${TEACHER_MODEL:-qwen72b}"
MAX_PARALLEL="${MAX_PARALLEL:-1}"
SKIP_ENDPOINT_CHECK="${SKIP_ENDPOINT_CHECK:-0}"
RUN_PREFIX="${RUN_PREFIX:-dspy-optimize}"
MC_PASSES="${MC_PASSES:-8}"
AUTO="${AUTO:-light}"

if [[ "$SKIP_ENDPOINT_CHECK" != "1" ]]; then
  for base in "$STUDENT_API_BASE" "$TEACHER_API_BASE"; do
    base="${base%/v1}"
    curl -fsS "$base/v1/models" >/dev/null || {
      echo "Endpoint check failed: $base/v1/models" >&2; exit 1;
    }
  done
fi
if [[ "$TASK_FILTER" == "all" ]]; then
  if [[ "$DATASET" == "hbs" ]]; then
    primary_models=(xlmr han-xlmr longformer mdeberta-v3 mt5 bertic bge-m3-mlp)
  elif [[ "$DATASET" == "sl" ]]; then
    primary_models=(xlmr han-xlmr longformer mdeberta-v3 mt5 sloberta bge-m3-mlp)
  else
    echo "DATASET must be hbs or sl." >&2; exit 2
  fi
else
  normalized="${TASK_FILTER//,/ }"; read -r -a primary_models <<< "$normalized"
fi
best_variant() {
  case "$DATASET:$1" in
    hbs:xlmr|hbs:han-xlmr) echo masked ;;
    hbs:longformer|hbs:mdeberta-v3|hbs:mt5|hbs:bertic|hbs:bge-m3-mlp) echo unmasked ;;
    sl:xlmr|sl:han-xlmr|sl:longformer|sl:mdeberta-v3|sl:bge-m3-mlp) echo masked ;;
    sl:mt5|sl:sloberta) echo unmasked ;;
    *) return 2 ;;
  esac
}

mkdir -p models/_runs/_launcher-logs
pids=()
for primary in "${primary_models[@]}"; do
  plm_variant="$VARIANT_MODE"
  [[ "$plm_variant" == "best" ]] && plm_variant="$(best_variant "$primary")"
  for prompt_variant in $PROMPT_VARIANTS; do
    run_id="$RUN_PREFIX-$DATASET-$primary-$plm_variant-$prompt_variant"
    log="models/_runs/_launcher-logs/$run_id.log"
    "$PYTHON_BIN" scripts/4.0-dspy-optimize.py \
      --train-input "$TRAIN_INPUT" --val-input "$VAL_INPUT" --models all \
      --primary-model "$primary" --dataset "$DATASET" --variant "$plm_variant" \
      --prompt-variant "$prompt_variant" --endpoint-model "$STUDENT_MODEL" \
      --api-base "$STUDENT_API_BASE" --teacher-endpoint-model "$TEACHER_MODEL" \
      --teacher-api-base "$TEACHER_API_BASE" --mc-passes "$MC_PASSES" \
      --auto "$AUTO" --run-id "$run_id" >"$log" 2>&1 &
    pids+=("$!")
    if (( ${#pids[@]} >= MAX_PARALLEL )); then
      wait "${pids[0]}"; pids=("${pids[@]:1}")
    fi
  done
done
for pid in "${pids[@]}"; do wait "$pid"; done
