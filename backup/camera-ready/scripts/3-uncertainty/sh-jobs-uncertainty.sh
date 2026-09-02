#!/bin/bash
set -euo pipefail

# Queue Option-A uncertainty jobs across the GPUs in GPU_IDS.
#
# Example:
#   PYTHON_BIN=/path/to/absa/bin/python \
#   GPU_IDS=0,1 \
#   bash reviews/sh-jobs-uncertainty.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="${ABSA_RELEASE_ROOT:-$(cd "${SCRIPT_DIR}/../.." && pwd)}"
cd "$ROOT_DIR"

if [[ -z "${PYTHON_BIN:-}" ]]; then
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
  echo "Could not locate Python. Set PYTHON_BIN=/path/to/python or CONDA_ENV=absa." >&2
  exit 2
fi

GPU_IDS="${GPU_IDS:-0,1}"
DATA_DIR="${DATA_DIR:-data}"
MODEL_ROOT="${MODEL_ROOT:-models}"
OUTPUT_ROOT="${OUTPUT_ROOT:-reviews/uncertainty}"
LOG_DIR="${LOG_DIR:-${OUTPUT_ROOT}/_logs}"
NUM_WORKERS="${NUM_WORKERS:-2}"
NUM_MC_SAMPLES="${NUM_MC_SAMPLES:-10}"
SEED="${SEED:-42}"
SKIP_COMPLETED="${SKIP_COMPLETED:-1}"
TQDM_DISABLE="${TQDM_DISABLE:-1}"
NO_AMP="${NO_AMP:-0}"
POLL_SECONDS="${POLL_SECONDS:-20}"
TASK_FILTER="${TASK_FILTER:-all}"
PROGRESS_EVERY="${PROGRESS_EVERY:-25}"

HAN_BATCH_SIZE="${HAN_BATCH_SIZE:-64}"
LONGFORMER_BATCH_SIZE="${LONGFORMER_BATCH_SIZE:-256}"
MDEBERTA_BATCH_SIZE="${MDEBERTA_BATCH_SIZE:-512}"
SLAVIC_BATCH_SIZE="${SLAVIC_BATCH_SIZE:-512}"

CHECKPOINT_LIMIT="${CHECKPOINT_LIMIT:-}"
LIMIT_ITEMS="${LIMIT_ITEMS:-}"
SPLITS="${SPLITS:-test train_val_0 train_val_1 train_val_2}"

export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export PYTHONUNBUFFERED="${PYTHONUNBUFFERED:-1}"
export TQDM_DISABLE

mkdir -p "$LOG_DIR"
IFS=',' read -r -a GPUS <<< "$GPU_IDS"
if [[ "${#GPUS[@]}" -eq 0 ]]; then
  echo "GPU_IDS is empty." >&2
  exit 2
fi

batch_size_for() {
  case "$1" in
    han_xlmr_masked) echo "$HAN_BATCH_SIZE" ;;
    longformer_masked) echo "$LONGFORMER_BATCH_SIZE" ;;
    mdeberta_masked) echo "$MDEBERTA_BATCH_SIZE" ;;
    slavic_specific_masked) echo "$SLAVIC_BATCH_SIZE" ;;
    *) echo "Unknown expert: $1" >&2; exit 2 ;;
  esac
}

include_task() {
  local expert="$1"
  local language="$2"
  case "$TASK_FILTER" in
    all) return 0 ;;
    slovenian|serbian) [[ "$language" == "$TASK_FILTER" ]] ;;
    han|han_xlmr|han_xlmr_masked) [[ "$expert" == "han_xlmr_masked" ]] ;;
    longformer|longformer_masked) [[ "$expert" == "longformer_masked" ]] ;;
    mdeberta|mdeberta_masked) [[ "$expert" == "mdeberta_masked" ]] ;;
    slavic|slavic_specific|slavic_specific_masked) [[ "$expert" == "slavic_specific_masked" ]] ;;
    *:*) [[ "${expert}:${language}" == "$TASK_FILTER" ]] ;;
    *) echo "Unknown TASK_FILTER=${TASK_FILTER}" >&2; exit 2 ;;
  esac
}

expected_marker() {
  local expert="$1"
  local language="$2"
  echo "${OUTPUT_ROOT}/${expert}/${language}/_SUCCESS.json"
}

is_complete() {
  local expert="$1"
  local language="$2"
  local marker
  marker="$(expected_marker "$expert" "$language")"
  "$PYTHON_BIN" - "$marker" "$NUM_MC_SAMPLES" "$SPLITS" <<'PY'
import json
import pathlib
import sys

marker = pathlib.Path(sys.argv[1])
expected_mc = int(sys.argv[2])
expected_splits = sys.argv[3].split()
if not marker.exists():
    raise SystemExit(1)
try:
    data = json.loads(marker.read_text(encoding="utf-8"))
except Exception:
    raise SystemExit(1)
if int(data.get("num_mc_samples_per_checkpoint", -1)) != expected_mc:
    raise SystemExit(1)
seen = {row.get("split") for row in data.get("splits", [])}
if not set(expected_splits).issubset(seen):
    raise SystemExit(1)
for row in data.get("splits", []):
    if row.get("split") in expected_splits:
        output_path = pathlib.Path(row.get("output_path", ""))
        if not output_path.exists():
            raise SystemExit(1)
raise SystemExit(0)
PY
}

run_task() {
  local gpu="$1"
  local task_id="$2"
  local expert="$3"
  local language="$4"
  local batch_size
  local log_file
  batch_size="$(batch_size_for "$expert")"
  log_file="${LOG_DIR}/uncertainty_${task_id}_${expert}_${language}.log"

  (
    export CUDA_VISIBLE_DEVICES="$gpu"
    echo "Task ${task_id}: ${expert} ${language}"
    echo "GPU: ${gpu}"
    echo "Python: ${PYTHON_BIN}"
    echo "Started at $(date)"
    echo "Batch size: ${batch_size}"
    echo "NUM_MC_SAMPLES=${NUM_MC_SAMPLES}"
    echo "SPLITS=${SPLITS}"

    cmd=(
      "$PYTHON_BIN" "scripts/3-uncertainty/predict_uncertainty_experts.py"
      --expert "$expert"
      --language "$language"
      --data_dir "$DATA_DIR"
      --model_root "$MODEL_ROOT"
      --output_root "$OUTPUT_ROOT"
      --num_workers "$NUM_WORKERS"
      --num_mc_samples "$NUM_MC_SAMPLES"
      --batch_size "$batch_size"
      --seed "$SEED"
      --progress_every "$PROGRESS_EVERY"
      --splits $SPLITS
    )

    if [[ "$SKIP_COMPLETED" == "1" ]]; then
      cmd+=(--skip_completed)
    fi
    if [[ "$NO_AMP" == "1" ]]; then
      cmd+=(--no_amp)
    fi
    if [[ -n "$CHECKPOINT_LIMIT" ]]; then
      cmd+=(--checkpoint_limit "$CHECKPOINT_LIMIT")
    fi
    if [[ -n "$LIMIT_ITEMS" ]]; then
      cmd+=(--limit_items "$LIMIT_ITEMS")
    fi

    printf 'Command:'
    printf ' %q' "${cmd[@]}"
    printf '\n'
    "${cmd[@]}"
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
    sleep "$POLL_SECONDS"
  done
}

TASKS=(
  "han_xlmr_masked slovenian"
  "han_xlmr_masked serbian"
  "longformer_masked slovenian"
  "slavic_specific_masked slovenian"
  "longformer_masked serbian"
  "mdeberta_masked serbian"
  "slavic_specific_masked serbian"
)

echo "Starting uncertainty GPU queue at $(date)"
echo "GPU_IDS=${GPU_IDS}"
echo "Python: ${PYTHON_BIN}"
echo "Output root: ${OUTPUT_ROOT}"
echo "Logs: ${LOG_DIR}"
echo "TASK_FILTER=${TASK_FILTER}"
echo "PROGRESS_EVERY=${PROGRESS_EVERY}"
echo "Tasks: ${#TASKS[@]}"

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

for task in "${TASKS[@]}"; do
  read -r expert language <<< "$task"
  if ! include_task "$expert" "$language"; then
    TASK_ID=$((TASK_ID + 1))
    continue
  fi
  if [[ "$SKIP_COMPLETED" == "1" ]] && is_complete "$expert" "$language"; then
    echo "Skipping complete task ${TASK_ID}: ${expert} ${language}"
    SKIPPED=$((SKIPPED + 1))
    TASK_ID=$((TASK_ID + 1))
    continue
  fi

  wait_for_slot
  slot="$FREE_SLOT"
  gpu="${GPUS[$slot]}"
  run_task "$gpu" "$TASK_ID" "$expert" "$language"
  PIDS[$slot]="$LAST_PID"
  NAMES[$slot]="${TASK_ID}:${expert}_${language}"
  LOGS[$slot]="$LAST_LOG"
  STARTED=$((STARTED + 1))
  echo "Started ${NAMES[$slot]} on GPU ${gpu}; log: ${LAST_LOG}"
  TASK_ID=$((TASK_ID + 1))
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

if [[ "$FAILURES" -gt 0 ]]; then
  exit 1
fi
