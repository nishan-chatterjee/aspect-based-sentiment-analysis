#!/bin/bash
set -euo pipefail

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
  elif [[ -n "${CONDA_PREFIX:-}" && -x "${CONDA_PREFIX}/bin/python" ]]; then
    :
  elif command -v conda >/dev/null 2>&1; then
    conda activate absa 2>/dev/null || conda activate vllm 2>/dev/null || true
  fi
  PYTHON_BIN="${PYTHON_BIN:-$(command -v python3 || command -v python)}"
fi

if [[ -z "${PYTHON_BIN:-}" ]]; then
  echo "Could not locate a Python executable." >&2
  exit 1
fi

export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export TQDM_DISABLE="${TQDM_DISABLE:-1}"

APPROACH="mt5"
PY_SCRIPT="scripts/2-experts/8.2 additional_mt5_baseline.py"
DATA_DIR="${DATA_DIR:-data}"
MODEL_ROOT="${MODEL_ROOT:-models}"
OUTPUT_ROOT="${OUTPUT_ROOT:-reviews}"
LOG_DIR="${LOG_DIR:-${OUTPUT_ROOT}/_logs}"
GPU_IDS="${GPU_IDS:-0,1,2,3}"
RUN_INDICES="${RUN_INDICES:-0 1 2}"

EPOCHS="${EPOCHS:-10}"
BATCH_SIZE="${BATCH_SIZE:-4}"
GRAD_ACCUM_STEPS="${GRAD_ACCUM_STEPS:-8}"
LR="${LR:-3e-5}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0.01}"
MAX_SOURCE_LEN="${MAX_SOURCE_LEN:-512}"
MAX_TARGET_LEN="${MAX_TARGET_LEN:-8}"
NUM_WORKERS="${NUM_WORKERS:-2}"
SEED="${SEED:-42}"
NO_AMP="${NO_AMP:-1}"

mkdir -p "$LOG_DIR"
IFS=',' read -r -a GPUS <<< "$GPU_IDS"
read -r -a RUNS <<< "$RUN_INDICES"

if [[ "${#GPUS[@]}" -eq 0 ]]; then
  echo "GPU_IDS is empty." >&2
  exit 1
fi

gpu_for() {
  local idx="$1"
  echo "${GPUS[$((idx % ${#GPUS[@]}))]}"
}

run_config() {
  local idx="$1"
  local lang="$2"
  local variant="$3"
  local gpu
  gpu="$(gpu_for "$idx")"
  local log_file="${LOG_DIR}/${APPROACH}_${lang}_${variant}.log"

  (
    export CUDA_VISIBLE_DEVICES="$gpu"
    cmd=(
      "$PYTHON_BIN" "$PY_SCRIPT"
      --split "$lang"
      --data_dir "$DATA_DIR"
      --model_root "$MODEL_ROOT"
      --output_root "$OUTPUT_ROOT"
      --epochs "$EPOCHS"
      --batch_size "$BATCH_SIZE"
      --grad_accum_steps "$GRAD_ACCUM_STEPS"
      --lr "$LR"
      --weight_decay "$WEIGHT_DECAY"
      --max_source_len "$MAX_SOURCE_LEN"
      --max_target_len "$MAX_TARGET_LEN"
      --num_workers "$NUM_WORKERS"
      --seed "$SEED"
      --run_indices "${RUNS[@]}"
      --skip_completed
      --gradient_checkpointing
    )
    if [[ "$variant" == "masked" ]]; then
      cmd+=(--mask_aspect)
    fi
    if [[ "$NO_AMP" == "1" ]]; then
      cmd+=(--no_amp)
    fi
    if [[ "${TEST_ONLY:-0}" == "1" ]]; then
      cmd+=(--test_only)
    fi

    echo "GPU: $gpu"
    echo "Command: ${cmd[*]}"
    "${cmd[@]}"
  ) > "$log_file" 2>&1 &

  pids+=("$!")
  names+=("${lang}_${variant}")
  logs+=("$log_file")
  echo "Started ${APPROACH} ${lang} ${variant} on GPU ${gpu}; log: ${log_file}"
}

echo "Starting mT5 additional comparison baselines at $(date)"
echo "GPU_IDS=${GPU_IDS}"
echo "Output root: ${OUTPUT_ROOT}"

pids=()
names=()
logs=()

run_config 0 "slovenian" "unmasked"
run_config 1 "slovenian" "masked"
run_config 2 "serbian" "unmasked"
run_config 3 "serbian" "masked"

status=0
for i in "${!pids[@]}"; do
  if wait "${pids[$i]}"; then
    echo "Finished ${names[$i]} successfully. Log: ${logs[$i]}"
  else
    echo "FAILED ${names[$i]}. Log: ${logs[$i]}" >&2
    status=1
  fi
done

echo "mT5 additional comparison baselines finished at $(date)"
exit "$status"
