#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "${SCRIPT_DIR}/.." && pwd)}"
TASK_FILE="${TASK_FILE:-${PROJECT_ROOT}/hpc-tasks/tasks_all.tsv}"
TASK_ID="${1:-${SLURM_ARRAY_TASK_ID:-}}"

if [[ -z "$TASK_ID" ]]; then
  echo "Usage: $0 TASK_ID" >&2
  echo "Or set SLURM_ARRAY_TASK_ID." >&2
  exit 2
fi

if [[ ! -f "$TASK_FILE" ]]; then
  echo "Task file not found: $TASK_FILE" >&2
  echo "Create it with: python hpc-tasks/make_task_matrix.py --task-set all" >&2
  exit 2
fi

row="$(awk -F '\t' -v id="$TASK_ID" 'NR > 1 && $1 == id {print; found=1} END {if (!found) exit 1}' "$TASK_FILE")" || {
  echo "Task id $TASK_ID not found in $TASK_FILE" >&2
  exit 2
}

IFS=$'\t' read -r task_id approach language variant run_index <<< "$row"

cd "$PROJECT_ROOT"

PYTHON_BIN="${PYTHON_BIN:-python}"
DATA_DIR="${DATA_DIR:-data}"
MODEL_ROOT="${MODEL_ROOT:-models}"
OUTPUT_ROOT="${OUTPUT_ROOT:-reviews}"
SEED="${SEED:-42}"
NUM_WORKERS="${NUM_WORKERS:-2}"
ENCODER_NO_AMP="${ENCODER_NO_AMP:-1}"
DRY_RUN="${DRY_RUN:-0}"

run_command() {
  printf 'Command:'
  printf ' %q' "$@"
  printf '\n'
  if [[ "$DRY_RUN" != "1" ]]; then
    "$@"
  fi
}

# Keep batch-level progress bars from turning redirected logs into long walls.
# Set TQDM_DISABLE=0 before calling this script if you want live tqdm bars.
export TQDM_DISABLE="${TQDM_DISABLE:-1}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export PYTHONUNBUFFERED="${PYTHONUNBUFFERED:-1}"

mask_args=()
if [[ "$variant" == "masked" ]]; then
  mask_args+=(--mask_aspect)
fi

echo "Task ${task_id}: approach=${approach} language=${language} variant=${variant} run_index=${run_index}"
echo "Python: ${PYTHON_BIN}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
echo "TQDM_DISABLE=${TQDM_DISABLE}"
echo "Started at $(date)"

if [[ "$approach" == "mt5" ]]; then
  batch_size="${MT5_BATCH_SIZE:-4}"
  grad_accum="${MT5_GRAD_ACCUM_STEPS:-8}"
  epochs="${MT5_EPOCHS:-10}"
  lr="${MT5_LR:-3e-5}"
  max_source_len="${MT5_MAX_SOURCE_LEN:-512}"
  mt5_no_amp="${MT5_NO_AMP:-1}"
  amp_args=()
  if [[ "$mt5_no_amp" == "1" ]]; then
    amp_args+=(--no_amp)
  fi

  run_command "$PYTHON_BIN" scripts/2-experts/8.2\ additional_mt5_baseline.py \
    --split "$language" \
    --data_dir "$DATA_DIR" \
    --model_root "$MODEL_ROOT" \
    --output_root "$OUTPUT_ROOT" \
    --epochs "$epochs" \
    --batch_size "$batch_size" \
    --grad_accum_steps "$grad_accum" \
    --lr "$lr" \
    --max_source_len "$max_source_len" \
    --num_workers "$NUM_WORKERS" \
      --seed "$SEED" \
      --run_indices "$run_index" \
      --skip_completed \
      --gradient_checkpointing \
      "${amp_args[@]}" \
      "${mask_args[@]}"
else
  if [[ "$approach" == "longformer" ]]; then
    batch_size="${LONGFORMER_BATCH_SIZE:-2}"
    grad_accum="${LONGFORMER_GRAD_ACCUM_STEPS:-16}"
    epochs="${LONGFORMER_EPOCHS:-10}"
    lr="${LONGFORMER_LR:-1e-5}"
    max_len="${LONGFORMER_MAX_LEN:-4096}"
    extra_args=(--gradient_checkpointing)
  else
    batch_size="${ENCODER_BATCH_SIZE:-16}"
    grad_accum="${ENCODER_GRAD_ACCUM_STEPS:-2}"
    epochs="${ENCODER_EPOCHS:-10}"
    lr="${ENCODER_LR:-2e-5}"
    max_len="${ENCODER_MAX_LEN:-512}"
    extra_args=()
  fi
  amp_args=()
  if [[ "$approach" != "longformer" && "$ENCODER_NO_AMP" == "1" ]]; then
    amp_args+=(--no_amp)
  fi

  run_command "$PYTHON_BIN" scripts/2-experts/8.1\ additional_encoder_baselines.py \
    --approach "$approach" \
    --split "$language" \
    --data_dir "$DATA_DIR" \
    --model_root "$MODEL_ROOT" \
    --output_root "$OUTPUT_ROOT" \
    --epochs "$epochs" \
    --batch_size "$batch_size" \
    --grad_accum_steps "$grad_accum" \
    --lr "$lr" \
    --max_len "$max_len" \
    --num_workers "$NUM_WORKERS" \
    --seed "$SEED" \
    --run_indices "$run_index" \
    --skip_completed \
    "${extra_args[@]}" \
    "${amp_args[@]}" \
    "${mask_args[@]}"
fi

echo "Finished at $(date)"
