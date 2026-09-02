#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export MODE="${MODE:-calibrate}"
export OUTPUT_ROOT="${OUTPUT_ROOT:-reviews/uncertainty/llm-selective-deferral}"
export TASK_FILTER="${TASK_FILTER:-all}"
export PROMPT_VARIANTS="${PROMPT_VARIANTS:-masked unmasked}"
export AUTORUNS="${AUTORUNS:-medium}"
export PRIMARY_EXPERT="${PRIMARY_EXPERT:-longformer_masked}"
export MAX_PARALLEL="${MAX_PARALLEL:-1}"
export TRAIN_SIZE="${TRAIN_SIZE:-300}"
export VAL_SIZE="${VAL_SIZE:-300}"
export BALANCED_SAMPLING="${BALANCED_SAMPLING:-1}"
export PREFER_BALANCED_SPLITS="${PREFER_BALANCED_SPLITS:-1}"
export CALIBRATION_SAMPLING="${CALIBRATION_SAMPLING:-low_confidence_stratified}"
export UNCERTAIN_POOL_RATE="${UNCERTAIN_POOL_RATE:-0.10}"
export DATA_AWARE_PROPOSER="${DATA_AWARE_PROPOSER:-1}"
export FEWSHOT_AWARE_PROPOSER="${FEWSHOT_AWARE_PROPOSER:-1}"
export HAN_EXPERT="${HAN_EXPERT:-legacy_han_xlmr_masked}"

run_language() {
  local language="$1"
  local experts="$2"
  echo "=== Optimizing selective deferral for primary=longformer_masked language=${language} ==="
  TASK_FILTER="$language" PRIMARY_EXPERT=longformer_masked EXPERTS="$experts" bash "${SCRIPT_DIR}/llm-selective-deferral.sh"
}

SLOVENIAN_EXPERTS="${SLOVENIAN_EXPERTS:-${HAN_EXPERT} longformer_masked slavic_specific_masked}"
SERBIAN_EXPERTS="${SERBIAN_EXPERTS:-${HAN_EXPERT} longformer_masked mdeberta_masked slavic_specific_masked}"

run_language slovenian "$SLOVENIAN_EXPERTS"
run_language serbian "$SERBIAN_EXPERTS"
