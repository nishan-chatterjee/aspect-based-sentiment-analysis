#!/bin/bash
set -euo pipefail

# Queue minimum-viable-set masked encoder experiments across local GPUs.
#
# Example:
#   PYTHON_BIN=/path/to/absa/bin/python \
#   GPU_IDS=0,1,2,3 \
#   NUM_MC_SAMPLES=10 \
#   TQDM_DISABLE=1 \
#   PROGRESS_EVERY=25 \
#   bash scripts/6-mvs/sh-jobs-minimum-viable-set.sh

DRY_RUN="${DRY_RUN:-0}"
while [[ $# -gt 0 ]]; do
  case "$1" in
    --dry-run)
      DRY_RUN=1
      shift
      ;;
    --help|-h)
      cat <<'EOF'
Usage: bash scripts/6-mvs/sh-jobs-minimum-viable-set.sh [--dry-run]

Environment:
  LANGUAGES=slovenian,serbian,sr,sh,hr,bs,sr_latin,sr_cyrillic
  PERCENTAGES=1,2,5,10,20,30,40,50,75,100
  GPU_IDS=0,1,2,3
  RUN_INDICES="0 1 2"
  RUN_MC_DROPOUT=1

The language and script subsets read their split files from additional-tasks/data.
Completed task directories are detected before launch, so rerunning the same
command resumes incomplete work without replacing complete outputs.
EOF
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      exit 2
      ;;
  esac
done

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

GPU_IDS="${GPU_IDS:-0,1,2,3}"
APPROACHES="${APPROACHES:-slavic_specific}"
LANGUAGES="${LANGUAGES:-slovenian,serbian}"
PERCENTAGES="${PERCENTAGES:-1,2,5,10,20,30,40,50,75,100}"
RUN_INDICES="${RUN_INDICES:-0 1 2}"

DATA_DIR="${DATA_DIR:-data}"
MODEL_ROOT="${MODEL_ROOT:-models}"
OUTPUT_ROOT="${OUTPUT_ROOT:-results/minimum-viable-set/reviews}"
UNCERTAINTY_ROOT="${UNCERTAINTY_ROOT:-results/minimum-viable-set/uncertainty}"
LOG_DIR="${LOG_DIR:-results/minimum-viable-set/_logs}"
NUM_WORKERS="${NUM_WORKERS:-2}"
SEED="${SEED:-42}"
SUBSET_SEED="${SUBSET_SEED:-${SEED}}"
SKIP_COMPLETED="${SKIP_COMPLETED:-1}"
RUN_MC_DROPOUT="${RUN_MC_DROPOUT:-1}"
NUM_MC_SAMPLES="${NUM_MC_SAMPLES:-10}"
SPLITS="${SPLITS:-test train_val_0 train_val_1 train_val_2}"
POLL_SECONDS="${POLL_SECONDS:-20}"
PROGRESS_EVERY="${PROGRESS_EVERY:-25}"
CHECKPOINT_LIMIT="${CHECKPOINT_LIMIT:-}"
LIMIT_ITEMS="${LIMIT_ITEMS:-}"

EPOCHS="${EPOCHS:-30}"
MIN_EPOCHS="${MIN_EPOCHS:-4}"
EARLY_STOP_PATIENCE="${EARLY_STOP_PATIENCE:-3}"
EARLY_STOP_METRIC="${EARLY_STOP_METRIC:-f1_macro}"

LONGFORMER_BATCH_SIZE="${LONGFORMER_BATCH_SIZE:-2}"
LONGFORMER_GRAD_ACCUM_STEPS="${LONGFORMER_GRAD_ACCUM_STEPS:-16}"
LONGFORMER_LR="${LONGFORMER_LR:-1e-5}"
LONGFORMER_MAX_LEN="${LONGFORMER_MAX_LEN:-4096}"
LONGFORMER_NO_AMP="${LONGFORMER_NO_AMP:-0}"

MDEBERTA_BATCH_SIZE="${MDEBERTA_BATCH_SIZE:-8}"
MDEBERTA_GRAD_ACCUM_STEPS="${MDEBERTA_GRAD_ACCUM_STEPS:-4}"
MDEBERTA_LR="${MDEBERTA_LR:-1e-5}"
MDEBERTA_MAX_LEN="${MDEBERTA_MAX_LEN:-512}"
MDEBERTA_NO_AMP="${MDEBERTA_NO_AMP:-1}"

SLAVIC_BATCH_SIZE="${SLAVIC_BATCH_SIZE:-16}"
SLAVIC_GRAD_ACCUM_STEPS="${SLAVIC_GRAD_ACCUM_STEPS:-2}"
SLAVIC_LR="${SLAVIC_LR:-2e-5}"
SLAVIC_MAX_LEN="${SLAVIC_MAX_LEN:-512}"
SLAVIC_NO_AMP="${SLAVIC_NO_AMP:-1}"

MC_LONGFORMER_BATCH_SIZE="${MC_LONGFORMER_BATCH_SIZE:-256}"
MC_MDEBERTA_BATCH_SIZE="${MC_MDEBERTA_BATCH_SIZE:-512}"
MC_SLAVIC_BATCH_SIZE="${MC_SLAVIC_BATCH_SIZE:-512}"

export TQDM_DISABLE="${TQDM_DISABLE:-1}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export PYTHONUNBUFFERED="${PYTHONUNBUFFERED:-1}"

mkdir -p "$LOG_DIR"
IFS=',' read -r -a GPUS <<< "$GPU_IDS"
if [[ "${#GPUS[@]}" -eq 0 ]]; then
  echo "GPU_IDS is empty." >&2
  exit 2
fi

IFS=',' read -r -a APPROACH_LIST <<< "$APPROACHES"
IFS=',' read -r -a LANGUAGE_LIST <<< "$LANGUAGES"
IFS=',' read -r -a PERCENTAGE_LIST <<< "$PERCENTAGES"

percent_tag() {
  "$PYTHON_BIN" - "$1" <<'PY'
import sys
p = float(sys.argv[1])
if p.is_integer():
    print("pct_%03d" % int(p))
else:
    text = ("%.4f" % p).rstrip("0").rstrip(".").replace(".", "p")
    print("pct_" + text)
PY
}

train_params_for() {
  case "$1" in
    longformer)
      TRAIN_BATCH_SIZE="$LONGFORMER_BATCH_SIZE"
      TRAIN_GRAD_ACCUM_STEPS="$LONGFORMER_GRAD_ACCUM_STEPS"
      TRAIN_LR="$LONGFORMER_LR"
      TRAIN_MAX_LEN="$LONGFORMER_MAX_LEN"
      TRAIN_NO_AMP="$LONGFORMER_NO_AMP"
      TRAIN_EXTRA=(--gradient_checkpointing)
      ;;
    mdeberta)
      TRAIN_BATCH_SIZE="$MDEBERTA_BATCH_SIZE"
      TRAIN_GRAD_ACCUM_STEPS="$MDEBERTA_GRAD_ACCUM_STEPS"
      TRAIN_LR="$MDEBERTA_LR"
      TRAIN_MAX_LEN="$MDEBERTA_MAX_LEN"
      TRAIN_NO_AMP="$MDEBERTA_NO_AMP"
      TRAIN_EXTRA=()
      ;;
    slavic_specific)
      TRAIN_BATCH_SIZE="$SLAVIC_BATCH_SIZE"
      TRAIN_GRAD_ACCUM_STEPS="$SLAVIC_GRAD_ACCUM_STEPS"
      TRAIN_LR="$SLAVIC_LR"
      TRAIN_MAX_LEN="$SLAVIC_MAX_LEN"
      TRAIN_NO_AMP="$SLAVIC_NO_AMP"
      TRAIN_EXTRA=()
      ;;
    *) echo "Unknown approach: $1" >&2; exit 2 ;;
  esac
}

mc_batch_size_for() {
  case "$1" in
    longformer) echo "$MC_LONGFORMER_BATCH_SIZE" ;;
    mdeberta) echo "$MC_MDEBERTA_BATCH_SIZE" ;;
    slavic_specific) echo "$MC_SLAVIC_BATCH_SIZE" ;;
    *) echo "Unknown approach: $1" >&2; exit 2 ;;
  esac
}

subset_data_path() {
  local language="$1"
  local kind="$2"
  local run_index="${3:-}"
  case "$language" in
    sr|sh|hr|bs|sr_latin|sr_cyrillic)
      if [[ "$kind" == "test" ]]; then
        printf 'additional-tasks/data/%s_test.json' "$language"
      else
        printf 'additional-tasks/data/%s_train_val_%s.json' "$language" "$run_index"
      fi
      ;;
    *)
      if [[ "$kind" == "test" ]]; then
        printf '%s/%s_test_complete.json' "$DATA_DIR" "$language"
      else
        printf '%s/%s_train_val_complete_%s.json' "$DATA_DIR" "$language" "$run_index"
      fi
      ;;
  esac
}

validate_language_data() {
  local language="$1"
  local run_index path
  path="$(subset_data_path "$language" test)"
  if [[ ! -f "$path" ]]; then
    echo "Missing test data for ${language}: ${path}" >&2
    return 1
  fi
  for run_index in $RUN_INDICES; do
    path="$(subset_data_path "$language" train_val "$run_index")"
    if [[ ! -f "$path" ]]; then
      echo "Missing train/val data for ${language}, run ${run_index}: ${path}" >&2
      return 1
    fi
  done
}

is_training_complete() {
  local approach="$1"
  local language="$2"
  local pct="$3"
  local tag run_index dir
  tag="$(percent_tag "$pct")"
  dir="${OUTPUT_ROOT}/${approach}/masked/${language}/${tag}"
  [[ -f "${dir}/_SUCCESS.json" && -f "${dir}/test_metrics_summary.json" ]] || return 1
  for run_index in $RUN_INDICES; do
    [[ -f "${dir}/best_model_${run_index}.pt" ]] || return 1
    [[ -f "${dir}/training_metrics_${run_index}.json" ]] || return 1
    [[ -f "${dir}/test_predictions_${run_index}.json" ]] || return 1
  done
}

is_mc_complete() {
  local approach="$1"
  local language="$2"
  local pct="$3"
  local tag dir
  if [[ "$RUN_MC_DROPOUT" != "1" ]]; then
    return 0
  fi
  tag="$(percent_tag "$pct")"
  dir="${UNCERTAINTY_ROOT}/${approach}_masked/${language}/${tag}"
  [[ -f "${dir}/_SUCCESS.json" ]]
}

is_task_complete() {
  is_training_complete "$1" "$2" "$3" && is_mc_complete "$1" "$2" "$3"
}

for language in "${LANGUAGE_LIST[@]}"; do
  validate_language_data "$language"
done

run_task() {
  local gpu="$1"
  local task_id="$2"
  local approach="$3"
  local language="$4"
  local pct="$5"
  local tag
  local log_file
  tag="$(percent_tag "$pct")"
  log_file="${LOG_DIR}/mvs_${task_id}_${approach}_${language}_${tag}.log"

  (
    export CUDA_VISIBLE_DEVICES="$gpu"
    echo "Task ${task_id}: ${approach} ${language} ${pct}% (${tag})"
    echo "GPU: ${gpu}"
    echo "Python: ${PYTHON_BIN}"
    echo "Started at $(date)"
    echo "RUN_INDICES=${RUN_INDICES}"

    train_params_for "$approach"
    train_cmd=(
      "$PYTHON_BIN" "scripts/6-mvs/train_progressive_encoder.py"
      --approach "$approach"
      --split "$language"
      --percent "$pct"
      --data_dir "$DATA_DIR"
      --model_root "$MODEL_ROOT"
      --output_root "$OUTPUT_ROOT"
      --epochs "$EPOCHS"
      --min_epochs "$MIN_EPOCHS"
      --early_stop_patience "$EARLY_STOP_PATIENCE"
      --early_stop_metric "$EARLY_STOP_METRIC"
      --batch_size "$TRAIN_BATCH_SIZE"
      --grad_accum_steps "$TRAIN_GRAD_ACCUM_STEPS"
      --lr "$TRAIN_LR"
      --max_len "$TRAIN_MAX_LEN"
      --num_workers "$NUM_WORKERS"
      --seed "$SEED"
      --subset_seed "$SUBSET_SEED"
      --run_indices $RUN_INDICES
      --mask_aspect
      "${TRAIN_EXTRA[@]}"
    )
    if [[ "$SKIP_COMPLETED" == "1" ]]; then
      train_cmd+=(--skip_completed)
    fi
    if [[ "$TRAIN_NO_AMP" == "1" ]]; then
      train_cmd+=(--no_amp)
    fi

    printf 'Training command:'
    printf ' %q' "${train_cmd[@]}"
    printf '\n'
    "${train_cmd[@]}"

    if [[ "$RUN_MC_DROPOUT" == "1" ]]; then
      checkpoint_dir="${OUTPUT_ROOT}/${approach}/masked/${language}/${tag}"
      uncertainty_dir="${UNCERTAINTY_ROOT}/${approach}_masked/${language}/${tag}"
      mc_batch_size="$(mc_batch_size_for "$approach")"
      mc_cmd=(
        "$PYTHON_BIN" "scripts/6-mvs/mc_dropout_encoder.py"
        --approach "$approach"
        --language "$language"
        --percentage_tag "$tag"
        --checkpoint_dir "$checkpoint_dir"
        --output_dir "$uncertainty_dir"
        --data_dir "$DATA_DIR"
        --model_root "$MODEL_ROOT"
        --num_workers "$NUM_WORKERS"
        --num_mc_samples "$NUM_MC_SAMPLES"
        --batch_size "$mc_batch_size"
        --seed "$SEED"
        --progress_every "$PROGRESS_EVERY"
        --splits $SPLITS
      )
      if [[ "$SKIP_COMPLETED" == "1" ]]; then
        mc_cmd+=(--skip_completed)
      fi
      if [[ "$TRAIN_NO_AMP" == "1" ]]; then
        mc_cmd+=(--no_amp)
      fi
      if [[ -n "$CHECKPOINT_LIMIT" ]]; then
        mc_cmd+=(--checkpoint_limit "$CHECKPOINT_LIMIT")
      fi
      if [[ -n "$LIMIT_ITEMS" ]]; then
        mc_cmd+=(--limit_items "$LIMIT_ITEMS")
      fi

      printf 'MC command:'
      printf ' %q' "${mc_cmd[@]}"
      printf '\n'
      "${mc_cmd[@]}"
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
    sleep "$POLL_SECONDS"
  done
}

TOTAL_TASKS=$(( ${#APPROACH_LIST[@]} * ${#LANGUAGE_LIST[@]} * ${#PERCENTAGE_LIST[@]} ))
TOTAL_RUNS=$(( TOTAL_TASKS * $(wc -w <<< "$RUN_INDICES") ))
if [[ "$DRY_RUN" == "1" ]]; then
  echo "DRY RUN: no training or MC dropout will be launched."
else
  echo "Starting minimum-viable-set GPU queue at $(date)"
fi
echo "GPU_IDS=${GPU_IDS}"
echo "Approaches=${APPROACHES}"
echo "Languages=${LANGUAGES}"
echo "Percentages=${PERCENTAGES}"
echo "Python: ${PYTHON_BIN}"
echo "Output root: ${OUTPUT_ROOT}"
echo "Uncertainty root: ${UNCERTAINTY_ROOT}"
echo "Logs: ${LOG_DIR}"
echo "Early stopping: metric=${EARLY_STOP_METRIC}, patience=${EARLY_STOP_PATIENCE}, min_epochs=${MIN_EPOCHS}, max_epochs=${EPOCHS}"
echo "Task directories: ${TOTAL_TASKS}; run-level trainings: ${TOTAL_RUNS}"

if [[ "$DRY_RUN" == "1" ]]; then
  complete=0
  pending=0
  task_id=0
  for approach in "${APPROACH_LIST[@]}"; do
    for pct in "${PERCENTAGE_LIST[@]}"; do
      for language in "${LANGUAGE_LIST[@]}"; do
        tag="$(percent_tag "$pct")"
        if is_task_complete "$approach" "$language" "$pct"; then
          status="complete"
          complete=$((complete + 1))
        elif is_training_complete "$approach" "$language" "$pct"; then
          status="training complete; MC pending"
          pending=$((pending + 1))
        else
          status="pending training"
          pending=$((pending + 1))
        fi
        printf '%03d  %-16s %-12s %-8s  %s\n' "$task_id" "$approach" "$language" "$tag" "$status"
        task_id=$((task_id + 1))
      done
    done
  done
  echo "Dry-run summary: complete=${complete}; pending=${pending}."
  exit 0
fi

declare -a PIDS
declare -a NAMES
declare -a LOGS
for slot in "${!GPUS[@]}"; do
  PIDS[$slot]=""
  NAMES[$slot]=""
  LOGS[$slot]=""
done

FAILURES=0
STARTED=0
SKIPPED_COMPLETE=0
TASK_ID=0

for approach in "${APPROACH_LIST[@]}"; do
  for pct in "${PERCENTAGE_LIST[@]}"; do
    for language in "${LANGUAGE_LIST[@]}"; do
      if is_task_complete "$approach" "$language" "$pct"; then
        echo "Skipping complete ${TASK_ID}:${approach}_${language}_$(percent_tag "$pct")."
        SKIPPED_COMPLETE=$((SKIPPED_COMPLETE + 1))
        TASK_ID=$((TASK_ID + 1))
        continue
      fi
      wait_for_slot
      slot="$FREE_SLOT"
      gpu="${GPUS[$slot]}"
      run_task "$gpu" "$TASK_ID" "$approach" "$language" "$pct"
      PIDS[$slot]="$LAST_PID"
      NAMES[$slot]="${TASK_ID}:${approach}_${language}_$(percent_tag "$pct")"
      LOGS[$slot]="$LAST_LOG"
      STARTED=$((STARTED + 1))
      echo "Started ${NAMES[$slot]} on GPU ${gpu}; log: ${LAST_LOG}"
      TASK_ID=$((TASK_ID + 1))
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
echo "Started: ${STARTED}; skipped complete: ${SKIPPED_COMPLETE}; failures: ${FAILURES}"

if [[ "$FAILURES" -gt 0 ]]; then
  exit 1
fi
