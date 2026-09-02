#!/bin/bash
set -euo pipefail

# Replacement for:
#   ./reviews/sh-job-longformer.sh && ./reviews/sh-job-mdeberta.sh && ...
#
# This runs a local GPU work queue inside an interactive allocation. Each task is
# one split-run: approach x language x masking variant x run index. This keeps
# GPUs busy when Slovenian finishes earlier than Serbian.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="${ABSA_RELEASE_ROOT:-$(cd "${SCRIPT_DIR}/../.." && pwd)}"
cd "$ROOT_DIR"

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
  echo "Could not locate Python. Set CONDA_ENV=absa or PYTHON_BIN=/path/to/python." >&2
  exit 2
fi

GPU_IDS="${GPU_IDS:-0,1,2,3}"
TASK_SET="${TASK_SET:-all}"
OUTPUT_ROOT="${OUTPUT_ROOT:-reviews}"
DATA_DIR="${DATA_DIR:-data}"
MODEL_ROOT="${MODEL_ROOT:-models}"
LOG_DIR="${LOG_DIR:-${OUTPUT_ROOT}/_logs}"
SEED="${SEED:-42}"
NUM_WORKERS="${NUM_WORKERS:-2}"
MERGE_AFTER="${MERGE_AFTER:-1}"

LONGFORMER_EPOCHS="${LONGFORMER_EPOCHS:-10}"
LONGFORMER_BATCH_SIZE="${LONGFORMER_BATCH_SIZE:-2}"
LONGFORMER_GRAD_ACCUM_STEPS="${LONGFORMER_GRAD_ACCUM_STEPS:-16}"
LONGFORMER_LR="${LONGFORMER_LR:-1e-5}"
LONGFORMER_MAX_LEN="${LONGFORMER_MAX_LEN:-4096}"
LONGFORMER_NO_AMP="${LONGFORMER_NO_AMP:-0}"

MDEBERTA_EPOCHS="${MDEBERTA_EPOCHS:-10}"
MDEBERTA_BATCH_SIZE="${MDEBERTA_BATCH_SIZE:-8}"
MDEBERTA_GRAD_ACCUM_STEPS="${MDEBERTA_GRAD_ACCUM_STEPS:-4}"
MDEBERTA_LR="${MDEBERTA_LR:-1e-5}"
MDEBERTA_MAX_LEN="${MDEBERTA_MAX_LEN:-512}"
MDEBERTA_NO_AMP="${MDEBERTA_NO_AMP:-1}"

SLAVIC_EPOCHS="${SLAVIC_EPOCHS:-10}"
SLAVIC_BATCH_SIZE="${SLAVIC_BATCH_SIZE:-16}"
SLAVIC_GRAD_ACCUM_STEPS="${SLAVIC_GRAD_ACCUM_STEPS:-2}"
SLAVIC_LR="${SLAVIC_LR:-2e-5}"
SLAVIC_MAX_LEN="${SLAVIC_MAX_LEN:-512}"
SLAVIC_NO_AMP="${SLAVIC_NO_AMP:-1}"

MT5_EPOCHS="${MT5_EPOCHS:-10}"
MT5_BATCH_SIZE="${MT5_BATCH_SIZE:-4}"
MT5_GRAD_ACCUM_STEPS="${MT5_GRAD_ACCUM_STEPS:-8}"
MT5_LR="${MT5_LR:-3e-5}"
MT5_MAX_SOURCE_LEN="${MT5_MAX_SOURCE_LEN:-512}"
MT5_MAX_TARGET_LEN="${MT5_MAX_TARGET_LEN:-8}"
MT5_NO_AMP="${MT5_NO_AMP:-1}"

export TQDM_DISABLE="${TQDM_DISABLE:-1}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export PYTHONUNBUFFERED="${PYTHONUNBUFFERED:-1}"

mkdir -p "$LOG_DIR"
IFS=',' read -r -a GPUS <<< "$GPU_IDS"
if [[ "${#GPUS[@]}" -eq 0 ]]; then
  echo "GPU_IDS is empty." >&2
  exit 2
fi

include_approach() {
  case "$TASK_SET" in
    all) return 0 ;;
    core) [[ "$1" == "longformer" || "$1" == "mdeberta" || "$1" == "mt5" ]] ;;
    longformer) [[ "$1" == "longformer" ]] ;;
    mdeberta) [[ "$1" == "mdeberta" ]] ;;
    mt5) [[ "$1" == "mt5" ]] ;;
    slavic|slavic_specific) [[ "$1" == "slavic_specific" ]] ;;
    *) echo "Unknown TASK_SET=${TASK_SET}" >&2; exit 2 ;;
  esac
}

epochs_for() {
  case "$1" in
    longformer) echo "$LONGFORMER_EPOCHS" ;;
    mdeberta) echo "$MDEBERTA_EPOCHS" ;;
    mt5) echo "$MT5_EPOCHS" ;;
    slavic_specific) echo "$SLAVIC_EPOCHS" ;;
  esac
}

is_complete() {
  local approach="$1"
  local language="$2"
  local variant="$3"
  local run_index="$4"
  local expected_epochs="$5"
  "$PYTHON_BIN" -c '
import json, pathlib, sys
root, approach, variant, language, run_index, expected_epochs = sys.argv[1:]
run_index = int(run_index)
expected_epochs = int(expected_epochs)
d = pathlib.Path(root) / approach / variant / language
checkpoint = d / f"best_model_{run_index}.pt"
metrics_path = d / f"training_metrics_{run_index}.json"
if not checkpoint.exists() or not metrics_path.exists():
    raise SystemExit(1)
try:
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
except Exception:
    raise SystemExit(1)
if len(metrics.get("train_metrics", [])) < expected_epochs:
    raise SystemExit(1)
if len(metrics.get("eval_metrics", [])) < expected_epochs:
    raise SystemExit(1)
raise SystemExit(0)
' "$OUTPUT_ROOT" "$approach" "$variant" "$language" "$run_index" "$expected_epochs"
}

run_task() {
  local gpu="$1"
  local task_id="$2"
  local approach="$3"
  local language="$4"
  local variant="$5"
  local run_index="$6"
  local log_file="${LOG_DIR}/allqueue_${task_id}_${approach}_${language}_${variant}_run${run_index}.log"

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
      cmd=(
        "$PYTHON_BIN" "scripts/2-experts/8.2 additional_mt5_baseline.py"
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
      )
      printf 'Command:'
      printf ' %q' "${cmd[@]}"
      printf '\n'
      "${cmd[@]}"
    else
      case "$approach" in
        longformer)
          epochs="$LONGFORMER_EPOCHS"
          batch="$LONGFORMER_BATCH_SIZE"
          accum="$LONGFORMER_GRAD_ACCUM_STEPS"
          lr="$LONGFORMER_LR"
          max_len="$LONGFORMER_MAX_LEN"
          no_amp="$LONGFORMER_NO_AMP"
          extra=(--gradient_checkpointing)
          ;;
        mdeberta)
          epochs="$MDEBERTA_EPOCHS"
          batch="$MDEBERTA_BATCH_SIZE"
          accum="$MDEBERTA_GRAD_ACCUM_STEPS"
          lr="$MDEBERTA_LR"
          max_len="$MDEBERTA_MAX_LEN"
          no_amp="$MDEBERTA_NO_AMP"
          extra=()
          ;;
        slavic_specific)
          epochs="$SLAVIC_EPOCHS"
          batch="$SLAVIC_BATCH_SIZE"
          accum="$SLAVIC_GRAD_ACCUM_STEPS"
          lr="$SLAVIC_LR"
          max_len="$SLAVIC_MAX_LEN"
          no_amp="$SLAVIC_NO_AMP"
          extra=()
          ;;
      esac
      amp_args=()
      if [[ "$no_amp" == "1" ]]; then amp_args+=(--no_amp); fi
      cmd=(
        "$PYTHON_BIN" "scripts/2-experts/8.1 additional_encoder_baselines.py"
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
      )
      printf 'Command:'
      printf ' %q' "${cmd[@]}"
      printf '\n'
      "${cmd[@]}"
    fi
    echo "Finished at $(date)"
  ) > "$log_file" 2>&1 &

  LAST_PID="$!"
  LAST_LOG="$log_file"
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

echo "Starting all-model GPU queue at $(date)"
echo "TASK_SET=${TASK_SET}"
echo "GPU_IDS=${GPU_IDS}"
echo "Python: ${PYTHON_BIN}"
echo "Output root: ${OUTPUT_ROOT}"
echo "TQDM_DISABLE=${TQDM_DISABLE}"
echo "mDeBERTa full precision: MDEBERTA_NO_AMP=${MDEBERTA_NO_AMP}"
echo "Slavic full precision: SLAVIC_NO_AMP=${SLAVIC_NO_AMP}"
echo "mT5 full precision: MT5_NO_AMP=${MT5_NO_AMP}"

declare -a PIDS
declare -a NAMES
declare -a LOGS
for slot in "${!GPUS[@]}"; do
  PIDS[$slot]=""
  NAMES[$slot]=""
  LOGS[$slot]=""
done

FAILURES=0
SKIPPED=0
STARTED=0
TASK_ID=0

for approach in longformer mdeberta mt5 slavic_specific; do
  if ! include_approach "$approach"; then
    TASK_ID=$((TASK_ID + 12))
    continue
  fi
  for language in slovenian serbian; do
    for variant in unmasked masked; do
      for run_index in 0 1 2; do
        expected_epochs="$(epochs_for "$approach")"
        if is_complete "$approach" "$language" "$variant" "$run_index" "$expected_epochs"; then
          echo "Skipping complete task ${TASK_ID}: ${approach} ${language} ${variant} run ${run_index}"
          SKIPPED=$((SKIPPED + 1))
          TASK_ID=$((TASK_ID + 1))
          continue
        fi

        wait_for_slot
        slot="$FREE_SLOT"
        gpu="${GPUS[$slot]}"
        run_task "$gpu" "$TASK_ID" "$approach" "$language" "$variant" "$run_index"
        PIDS[$slot]="$LAST_PID"
        NAMES[$slot]="${TASK_ID}:${approach}_${language}_${variant}_run${run_index}"
        LOGS[$slot]="$LAST_LOG"
        STARTED=$((STARTED + 1))
        echo "Started ${NAMES[$slot]} on GPU ${gpu}; log: ${LAST_LOG}"
        TASK_ID=$((TASK_ID + 1))
      done
    done
  done
done

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

if [[ "$MERGE_AFTER" == "1" && -f hpc-tasks/merge_results.py ]]; then
  echo "Merging available test predictions into notebook-compatible summaries..."
  "$PYTHON_BIN" hpc-tasks/merge_results.py --output-root "$OUTPUT_ROOT"
fi

if [[ "$FAILURES" -gt 0 ]]; then
  exit 1
fi
