#!/bin/bash
set -euo pipefail

# Queue-style launcher for an interactive allocation.
#
# It schedules one split-run per GPU slot:
#   approach x language x masking variant x run_index
#
# Completed split-runs are skipped conservatively: both best_model_i.pt and
# training_metrics_i.json must exist, and the metrics file must contain the
# requested number of train/eval epochs.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "${SCRIPT_DIR}/.." && pwd)}"
cd "$PROJECT_ROOT"

if [[ -n "${PYTHON_BIN:-}" ]]; then
  :
else
  module load Anaconda3 2>/dev/null || true
  if [[ -f /opt/easybuild/software/Anaconda3/2024.02-1/etc/profile.d/conda.sh ]]; then
    source /opt/easybuild/software/Anaconda3/2024.02-1/etc/profile.d/conda.sh
  fi
  if [[ -n "${CONDA_ENV:-}" && -x "${HOME}/.conda/envs/${CONDA_ENV}/bin/python" ]]; then
    PYTHON_BIN="${HOME}/.conda/envs/${CONDA_ENV}/bin/python"
  elif [[ -n "${CONDA_ENV:-}" ]] && command -v conda >/dev/null 2>&1; then
    conda activate "$CONDA_ENV"
    PYTHON_BIN="$(command -v python)"
  elif [[ -n "${CONDA_PREFIX:-}" && -x "${CONDA_PREFIX}/bin/python" ]]; then
    PYTHON_BIN="${CONDA_PREFIX}/bin/python"
  elif command -v conda >/dev/null 2>&1; then
    conda activate absa 2>/dev/null || true
    PYTHON_BIN="$(command -v python3 || command -v python)"
  else
    PYTHON_BIN="$(command -v python3 || command -v python)"
  fi
fi

if [[ -z "${PYTHON_BIN:-}" || ! -x "$PYTHON_BIN" ]]; then
  echo "Could not resolve a Python executable. Set PYTHON_BIN or CONDA_ENV=absa." >&2
  exit 2
fi

GPU_IDS="${GPU_IDS:-0,1,2,3}"
TASK_SET="${TASK_SET:-all}"
TASK_FILE="${TASK_FILE:-reviews/_queue/tasks_${TASK_SET}.tsv}"
OUTPUT_ROOT="${OUTPUT_ROOT:-reviews}"
DATA_DIR="${DATA_DIR:-data}"
MODEL_ROOT="${MODEL_ROOT:-models}"
LOG_DIR="${LOG_DIR:-${OUTPUT_ROOT}/_logs}"
QUEUE_DIR="$(dirname "$TASK_FILE")"

SEED="${SEED:-42}"
NUM_WORKERS="${NUM_WORKERS:-2}"
MERGE_AFTER="${MERGE_AFTER:-1}"

LONGFORMER_EPOCHS="${LONGFORMER_EPOCHS:-10}"
LONGFORMER_BATCH_SIZE="${LONGFORMER_BATCH_SIZE:-2}"
LONGFORMER_GRAD_ACCUM_STEPS="${LONGFORMER_GRAD_ACCUM_STEPS:-16}"
LONGFORMER_LR="${LONGFORMER_LR:-1e-5}"
LONGFORMER_MAX_LEN="${LONGFORMER_MAX_LEN:-4096}"

ENCODER_EPOCHS="${ENCODER_EPOCHS:-10}"
ENCODER_BATCH_SIZE="${ENCODER_BATCH_SIZE:-16}"
ENCODER_GRAD_ACCUM_STEPS="${ENCODER_GRAD_ACCUM_STEPS:-2}"
ENCODER_LR="${ENCODER_LR:-2e-5}"
ENCODER_MAX_LEN="${ENCODER_MAX_LEN:-512}"
ENCODER_NO_AMP="${ENCODER_NO_AMP:-1}"

MT5_EPOCHS="${MT5_EPOCHS:-10}"
MT5_BATCH_SIZE="${MT5_BATCH_SIZE:-4}"
MT5_GRAD_ACCUM_STEPS="${MT5_GRAD_ACCUM_STEPS:-8}"
MT5_LR="${MT5_LR:-3e-5}"
MT5_MAX_SOURCE_LEN="${MT5_MAX_SOURCE_LEN:-512}"
MT5_MAX_TARGET_LEN="${MT5_MAX_TARGET_LEN:-8}"
MT5_NO_AMP="${MT5_NO_AMP:-1}"

# Compact logs by default. Set TQDM_DISABLE=0 for live batch bars.
export TQDM_DISABLE="${TQDM_DISABLE:-1}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export PYTHONUNBUFFERED="${PYTHONUNBUFFERED:-1}"

mkdir -p "$LOG_DIR" "$QUEUE_DIR"
IFS=',' read -r -a GPUS <<< "$GPU_IDS"
if [[ "${#GPUS[@]}" -eq 0 ]]; then
  echo "GPU_IDS is empty." >&2
  exit 2
fi

"$PYTHON_BIN" hpc-tasks/make_task_matrix.py --task-set "$TASK_SET" --output "$TASK_FILE"

epochs_for() {
  case "$1" in
    longformer) echo "$LONGFORMER_EPOCHS" ;;
    mt5) echo "$MT5_EPOCHS" ;;
    mdeberta|slavic_specific) echo "$ENCODER_EPOCHS" ;;
    *) echo "$ENCODER_EPOCHS" ;;
  esac
}

is_complete() {
  local approach="$1"
  local language="$2"
  local variant="$3"
  local run_index="$4"
  local epochs="$5"
  "$PYTHON_BIN" -c '
import json, pathlib, sys
root, approach, variant, language, run_index, epochs = sys.argv[1:]
run_index = int(run_index)
epochs = int(epochs)
d = pathlib.Path(root) / approach / variant / language
checkpoint = d / f"best_model_{run_index}.pt"
metrics_path = d / f"training_metrics_{run_index}.json"
if not checkpoint.exists() or not metrics_path.exists():
    raise SystemExit(1)
try:
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
except Exception:
    raise SystemExit(1)
if len(metrics.get("train_metrics", [])) < epochs:
    raise SystemExit(1)
if len(metrics.get("eval_metrics", [])) < epochs:
    raise SystemExit(1)
raise SystemExit(0)
' "$OUTPUT_ROOT" "$approach" "$variant" "$language" "$run_index" "$epochs"
}

run_task_on_gpu() {
  local gpu="$1"
  local task_id="$2"
  local approach="$3"
  local language="$4"
  local variant="$5"
  local run_index="$6"
  local log_file="${LOG_DIR}/queue_${task_id}_${approach}_${language}_${variant}_run${run_index}.log"

  (
    export CUDA_VISIBLE_DEVICES="$gpu"
    echo "Task ${task_id}: ${approach} ${language} ${variant} run ${run_index}"
    echo "GPU: ${gpu}"
    echo "Python: ${PYTHON_BIN}"
    echo "Started at $(date)"
    echo "TQDM_DISABLE=${TQDM_DISABLE}"

    common_args=(
      --split "$language"
      --data_dir "$DATA_DIR"
      --model_root "$MODEL_ROOT"
      --output_root "$OUTPUT_ROOT"
      --num_workers "$NUM_WORKERS"
      --seed "$SEED"
      --run_indices "$run_index"
      --skip_completed
    )

    mask_args=()
    if [[ "$variant" == "masked" ]]; then
      mask_args+=(--mask_aspect)
    fi

    if [[ "$approach" == "mt5" ]]; then
      amp_args=()
      if [[ "$MT5_NO_AMP" == "1" ]]; then amp_args+=(--no_amp); fi
      "$PYTHON_BIN" reviews/8.2\ additional_mt5_baseline.py \
        "${common_args[@]}" \
        --epochs "$MT5_EPOCHS" \
        --batch_size "$MT5_BATCH_SIZE" \
        --grad_accum_steps "$MT5_GRAD_ACCUM_STEPS" \
        --lr "$MT5_LR" \
        --max_source_len "$MT5_MAX_SOURCE_LEN" \
        --max_target_len "$MT5_MAX_TARGET_LEN" \
        --gradient_checkpointing \
        "${amp_args[@]}" \
        "${mask_args[@]}"
    else
      if [[ "$approach" == "longformer" ]]; then
        epochs="$LONGFORMER_EPOCHS"
        batch="$LONGFORMER_BATCH_SIZE"
        accum="$LONGFORMER_GRAD_ACCUM_STEPS"
        lr="$LONGFORMER_LR"
        max_len="$LONGFORMER_MAX_LEN"
        extra=(--gradient_checkpointing)
      else
        epochs="$ENCODER_EPOCHS"
        batch="$ENCODER_BATCH_SIZE"
        accum="$ENCODER_GRAD_ACCUM_STEPS"
        lr="$ENCODER_LR"
        max_len="$ENCODER_MAX_LEN"
        extra=()
      fi
      amp_args=()
      if [[ "$approach" != "longformer" && "$ENCODER_NO_AMP" == "1" ]]; then
        amp_args+=(--no_amp)
      fi
      "$PYTHON_BIN" reviews/8.1\ additional_encoder_baselines.py \
        --approach "$approach" \
        "${common_args[@]}" \
        --epochs "$epochs" \
        --batch_size "$batch" \
        --grad_accum_steps "$accum" \
        --lr "$lr" \
        --max_len "$max_len" \
        "${extra[@]}" \
        "${amp_args[@]}" \
        "${mask_args[@]}"
    fi
    echo "Finished at $(date)"
  ) > "$log_file" 2>&1 &

  LAST_PID="$!"
}

wait_for_slot() {
  local slot
  while true; do
    for slot in "${!GPUS[@]}"; do
      if [[ -z "${PIDS[$slot]:-}" ]]; then
        FREE_SLOT="$slot"
        return 0
      fi
      if ! kill -0 "${PIDS[$slot]}" 2>/dev/null; then
        if wait "${PIDS[$slot]}"; then
          echo "Completed ${NAMES[$slot]} on GPU ${GPUS[$slot]}."
        else
          echo "FAILED ${NAMES[$slot]} on GPU ${GPUS[$slot]}; see ${LOGS[$slot]}" >&2
          FAILURES=$((FAILURES + 1))
        fi
        PIDS[$slot]=""
        NAMES[$slot]=""
        LOGS[$slot]=""
        FREE_SLOT="$slot"
        return 0
      fi
    done
    sleep 20
  done
}

echo "Starting interactive GPU queue at $(date)"
echo "Task set: ${TASK_SET}"
echo "Task file: ${TASK_FILE}"
echo "GPU_IDS=${GPU_IDS}"
echo "Python: ${PYTHON_BIN}"
echo "Output root: ${OUTPUT_ROOT}"
echo "TQDM_DISABLE=${TQDM_DISABLE}"

declare -a PIDS
declare -a NAMES
declare -a LOGS
FAILURES=0
SKIPPED=0
STARTED=0

for i in "${!GPUS[@]}"; do
  PIDS[$i]=""
  NAMES[$i]=""
  LOGS[$i]=""
done

while IFS=$'\t' read -r task_id approach language variant run_index; do
  if [[ "$task_id" == "task_id" ]]; then
    continue
  fi
  run_index="${run_index//$'\r'/}"
  expected_epochs="$(epochs_for "$approach")"
  if is_complete "$approach" "$language" "$variant" "$run_index" "$expected_epochs"; then
    echo "Skipping complete task ${task_id}: ${approach} ${language} ${variant} run ${run_index}"
    SKIPPED=$((SKIPPED + 1))
    continue
  fi

  wait_for_slot
  slot="$FREE_SLOT"
  gpu="${GPUS[$slot]}"
  log_file="${LOG_DIR}/queue_${task_id}_${approach}_${language}_${variant}_run${run_index}.log"
  run_task_on_gpu "$gpu" "$task_id" "$approach" "$language" "$variant" "$run_index"
  pid="$LAST_PID"
  PIDS[$slot]="$pid"
  NAMES[$slot]="${task_id}:${approach}_${language}_${variant}_run${run_index}"
  LOGS[$slot]="$log_file"
  STARTED=$((STARTED + 1))
  echo "Started ${NAMES[$slot]} on GPU ${gpu}; log: ${log_file}"
done < "$TASK_FILE"

for slot in "${!GPUS[@]}"; do
  if [[ -n "${PIDS[$slot]:-}" ]]; then
    if wait "${PIDS[$slot]}"; then
      echo "Completed ${NAMES[$slot]} on GPU ${GPUS[$slot]}."
    else
      echo "FAILED ${NAMES[$slot]} on GPU ${GPUS[$slot]}; see ${LOGS[$slot]}" >&2
      FAILURES=$((FAILURES + 1))
    fi
  fi
done

echo "Queue finished at $(date)"
echo "Started: ${STARTED}; skipped complete: ${SKIPPED}; failures: ${FAILURES}"

if [[ "$MERGE_AFTER" == "1" ]]; then
  echo "Merging available test predictions into notebook-compatible summaries..."
  "$PYTHON_BIN" hpc-tasks/merge_results.py --output-root "$OUTPUT_ROOT"
fi

if [[ "$FAILURES" -gt 0 ]]; then
  exit 1
fi
