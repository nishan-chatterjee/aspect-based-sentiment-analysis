#!/bin/bash
set -euo pipefail

# Run cheap multi-expert learning-to-defer experiments from completed
# uncertainty outputs.
#
# Example:
#   PYTHON_BIN=/path/to/absa/bin/python \
#   bash reviews/sh-jobs-multi-expert-defer.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="${ABSA_RELEASE_ROOT:-$(cd "${SCRIPT_DIR}/../.." && pwd)}"
cd "$ROOT_DIR"

if [[ -z "${PYTHON_BIN:-}" ]]; then
  if [[ -n "${CONDA_PREFIX:-}" && -x "${CONDA_PREFIX}/bin/python" ]]; then
    PYTHON_BIN="${CONDA_PREFIX}/bin/python"
  else
    PYTHON_BIN="$(command -v python3 || command -v python)"
  fi
fi

if [[ -z "${PYTHON_BIN:-}" || ! -x "$PYTHON_BIN" ]]; then
  echo "Could not locate Python. Set PYTHON_BIN=/path/to/python." >&2
  exit 2
fi

UNCERTAINTY_ROOT="${UNCERTAINTY_ROOT:-reviews/uncertainty}"
OUTPUT_ROOT="${OUTPUT_ROOT:-reviews/uncertainty/multi-expert-defer}"
LOG_DIR="${LOG_DIR:-${OUTPUT_ROOT}/_logs}"
TASK_FILTER="${TASK_FILTER:-all}"
SETTINGS="${SETTINGS:-masked}"
SPLIT_INDEX="${SPLIT_INDEX:-0}"
METRIC="${METRIC:-f1_macro}"
MAX_PARALLEL="${MAX_PARALLEL:-2}"
SEED="${SEED:-42}"
FORCE="${FORCE:-0}"

MAX_TRAIN_ITEMS="${MAX_TRAIN_ITEMS:-}"
MAX_VAL_ITEMS="${MAX_VAL_ITEMS:-}"
LIMIT_TEST_ITEMS="${LIMIT_TEST_ITEMS:-}"
DEFER_RATES="${DEFER_RATES:-0,0.05,0.10,0.15,0.20,0.25,0.30,0.40,0.50}"
RANDOM_BASELINE_REPEATS="${RANDOM_BASELINE_REPEATS:-20}"

SLOVENIAN_EXPERTS="${SLOVENIAN_EXPERTS:-longformer_masked slavic_specific_masked han_xlmr_masked}"
SERBIAN_EXPERTS="${SERBIAN_EXPERTS:-longformer_masked slavic_specific_masked mdeberta_masked}"

export PYTHONUNBUFFERED="${PYTHONUNBUFFERED:-1}"
mkdir -p "$LOG_DIR"

include_task() {
  local language="$1"
  local setting="$2"
  case "$TASK_FILTER" in
    all) return 0 ;;
    slovenian|serbian) [[ "$language" == "$TASK_FILTER" ]] ;;
    masked|unmasked) [[ "$setting" == "$TASK_FILTER" ]] ;;
    *:*) [[ "${language}:${setting}" == "$TASK_FILTER" ]] ;;
    *) echo "Unknown TASK_FILTER=${TASK_FILTER}" >&2; exit 2 ;;
  esac
}

wait_for_slot() {
  while true; do
    local running
    running="$(jobs -rp | wc -l)"
    if [[ "$running" -lt "$MAX_PARALLEL" ]]; then
      return 0
    fi
    sleep 5
  done
}

experts_for() {
  local language="$1"
  case "$language" in
    slovenian) echo "$SLOVENIAN_EXPERTS" ;;
    serbian) echo "$SERBIAN_EXPERTS" ;;
    *) echo "Unknown language: $language" >&2; exit 2 ;;
  esac
}

run_task() {
  local task_id="$1"
  local language="$2"
  local setting="$3"
  local experts
  local log_file
  experts="$(experts_for "$language")"
  log_file="${LOG_DIR}/multi_expert_defer_${task_id}_${language}_${setting}.log"

  (
    echo "Task ${task_id}: ${language} ${setting}"
    echo "Python: ${PYTHON_BIN}"
    echo "Experts: ${experts}"
    echo "Started at $(date)"

    cmd=(
      "$PYTHON_BIN" "scripts/4-deferral/multi_expert_defer.py"
      --language "$language"
      --setting "$setting"
      --uncertainty-root "$UNCERTAINTY_ROOT"
      --output-root "$OUTPUT_ROOT"
      --split-index "$SPLIT_INDEX"
      --metric "$METRIC"
      --defer-rates "$DEFER_RATES"
      --random-baseline-repeats "$RANDOM_BASELINE_REPEATS"
      --seed "$SEED"
      --experts $experts
    )

    if [[ -n "$MAX_TRAIN_ITEMS" ]]; then cmd+=(--max-train-items "$MAX_TRAIN_ITEMS"); fi
    if [[ -n "$MAX_VAL_ITEMS" ]]; then cmd+=(--max-val-items "$MAX_VAL_ITEMS"); fi
    if [[ -n "$LIMIT_TEST_ITEMS" ]]; then cmd+=(--limit-test-items "$LIMIT_TEST_ITEMS"); fi
    if [[ "$FORCE" == "1" ]]; then cmd+=(--force); fi

    printf 'Command:'
    printf ' %q' "${cmd[@]}"
    printf '\n'
    "${cmd[@]}"
    echo "Finished at $(date)"
  ) > "$log_file" 2>&1 &
  echo "Started task ${task_id}: ${language} ${setting}; log: ${log_file}"
}

echo "Starting multi-expert defer queue at $(date)"
echo "Python: ${PYTHON_BIN}"
echo "Uncertainty root: ${UNCERTAINTY_ROOT}"
echo "Output root: ${OUTPUT_ROOT}"
echo "Settings: ${SETTINGS}"
echo "Task filter: ${TASK_FILTER}"
echo "Metric: ${METRIC}"

TASK_ID=0
for setting in $SETTINGS; do
  for language in slovenian serbian; do
    if ! include_task "$language" "$setting"; then
      TASK_ID=$((TASK_ID + 1))
      continue
    fi
    wait_for_slot
    run_task "$TASK_ID" "$language" "$setting"
    TASK_ID=$((TASK_ID + 1))
  done
done

FAILURES=0
for job in $(jobs -rp); do
  if ! wait "$job"; then
    FAILURES=$((FAILURES + 1))
  fi
done

echo "Multi-expert defer queue finished at $(date); failures=${FAILURES}"
if [[ "$FAILURES" -gt 0 ]]; then
  exit 1
fi
