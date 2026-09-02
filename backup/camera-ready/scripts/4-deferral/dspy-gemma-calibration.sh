#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="${ABSA_RELEASE_ROOT:-$(cd "${SCRIPT_DIR}/../.." && pwd)}"
cd "$ROOT_DIR"

PYTHON_BIN="${PYTHON_BIN:-$(command -v python)}"
if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "Python not found or not executable: $PYTHON_BIN" >&2
  exit 2
fi

OUTPUT_ROOT="${OUTPUT_ROOT:-reviews/uncertainty/llm-dspy-calibration-cot}"
UNCERTAINTY_ROOT="${UNCERTAINTY_ROOT:-reviews/uncertainty}"
LOG_DIR="${LOG_DIR:-${OUTPUT_ROOT}/_logs_calibration}"
TASK_FILTER="${TASK_FILTER:-all}"
AUTORUNS="${AUTORUNS:-medium heavy}"
PROMPT_VARIANTS="${PROMPT_VARIANTS:-unmasked masked}"
PROMPT_STYLE="${PROMPT_STYLE:-current}"
MAX_PARALLEL="${MAX_PARALLEL:-2}"
MAX_RETRIES="${MAX_RETRIES:-1}"
RETRY_DELAY_SECONDS="${RETRY_DELAY_SECONDS:-60}"

STUDENT_API_BASE="${STUDENT_API_BASE:-http://127.0.0.1:8001/v1}"
TEACHER_API_BASE="${TEACHER_API_BASE:-http://127.0.0.1:8000/v1}"
STUDENT_MODEL="${STUDENT_MODEL:-gemma27b}"
TEACHER_MODEL="${TEACHER_MODEL:-qwen72b}"
TEACHER_LABEL="${TEACHER_LABEL:-qwen-2.5-72b}"
OPENAI_API_KEY="${OPENAI_API_KEY:-EMPTY}"

MAX_TOKENS="${MAX_TOKENS:-512}"
STUDENT_MAX_TOKENS="${STUDENT_MAX_TOKENS:-$MAX_TOKENS}"
TEACHER_MAX_TOKENS="${TEACHER_MAX_TOKENS:-$MAX_TOKENS}"
TEMPERATURE="${TEMPERATURE:-1.0}"
TOP_P="${TOP_P:-0.95}"
MIPROV2_TEMP="${MIPROV2_TEMP:-1.0}"
DSPY_NUM_THREADS="${DSPY_NUM_THREADS:-8}"
MAX_ERRORS="${MAX_ERRORS:-50}"
SEED="${SEED:-42}"
MAX_ARTICLE_CHARS="${MAX_ARTICLE_CHARS:-6000}"
NUM_QUERIES="${NUM_QUERIES:-1}"
EVAL_WORKERS="${EVAL_WORKERS:-4}"
TRAIN_SIZE="${TRAIN_SIZE:-}"
VAL_SIZE="${VAL_SIZE:-}"
FORCE="${FORCE:-0}"
SKIP_CALIBRATION_EVAL="${SKIP_CALIBRATION_EVAL:-0}"
SKIP_ENDPOINT_CHECK="${SKIP_ENDPOINT_CHECK:-0}"
PROGRAM_AWARE_PROPOSER="${PROGRAM_AWARE_PROPOSER:-1}"
DATA_AWARE_PROPOSER="${DATA_AWARE_PROPOSER:-0}"
TIP_AWARE_PROPOSER="${TIP_AWARE_PROPOSER:-1}"
FEWSHOT_AWARE_PROPOSER="${FEWSHOT_AWARE_PROPOSER:-0}"
VIEW_DATA_BATCH_SIZE="${VIEW_DATA_BATCH_SIZE:-3}"

export OPENAI_API_KEY
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export LITELLM_LOCAL_MODEL_COST_MAP="${LITELLM_LOCAL_MODEL_COST_MAP:-true}"
export LITELLM_LOG="${LITELLM_LOG:-ERROR}"
export DSPY_CACHEDIR="${DSPY_CACHEDIR:-${OUTPUT_ROOT}/.dspy_cache}"

mkdir -p "$LOG_DIR" "$DSPY_CACHEDIR"

include_task() {
  local expert="$1"
  local language="$2"
  case "$TASK_FILTER" in
    all) return 0 ;;
    slovenian|serbian) [[ "$language" == "$TASK_FILTER" ]] ;;
    *:*) [[ "${expert}:${language}" == "$TASK_FILTER" ]] ;;
    *) [[ "$expert" == "$TASK_FILTER" ]] ;;
  esac
}

wait_for_slot() {
  while true; do
    local next_pids=()
    local pid
    if [[ "${#ACTIVE_PIDS[@]}" -gt 0 ]]; then
      for pid in "${ACTIVE_PIDS[@]}"; do
        if kill -0 "$pid" 2>/dev/null; then
          next_pids+=("$pid")
        else
          if ! wait "$pid"; then
            FAILURES=$((FAILURES + 1))
          fi
        fi
      done
    fi
    ACTIVE_PIDS=("${next_pids[@]}")
    if [[ "${#ACTIVE_PIDS[@]}" -lt "$MAX_PARALLEL" ]]; then
      return 0
    fi
    sleep 10
  done
}

run_with_retries() {
  local description="$1"
  shift
  local cmd=("$@")
  local attempt
  for attempt in $(seq 1 "$MAX_RETRIES"); do
    echo "------------------------------------------------------------------------"
    echo "Attempt ${attempt}/${MAX_RETRIES}: ${description}"
    printf 'Command:'
    printf ' %q' "${cmd[@]}"
    printf '\n'
    echo "------------------------------------------------------------------------"
    if "${cmd[@]}"; then
      echo "Completed: ${description}"
      return 0
    fi
    if [[ "$attempt" -lt "$MAX_RETRIES" ]]; then
      echo "Retrying in ${RETRY_DELAY_SECONDS}s: ${description}"
      sleep "$RETRY_DELAY_SECONDS"
    fi
  done
  echo "FAILED after ${MAX_RETRIES} attempts: ${description}" >&2
  return 1
}

run_task() {
  local task_id="$1"
  local expert="$2"
  local language="$3"
  local prompt_variant="$4"
  local autorun="$5"
  local log_file="${LOG_DIR}/calibration_${task_id}_${expert}_${language}_${prompt_variant}_${autorun}.log"
  (
    echo "Task ${task_id}: ${expert} ${language} ${prompt_variant} ${autorun}"
    echo "Started at $(date)"
    cmd=(
      "$PYTHON_BIN" "scripts/4-deferral/dspy-gemma-calibration.py"
      --expert "$expert"
      --language "$language"
      --prompt-variant "$prompt_variant"
      --prompt-style "$PROMPT_STYLE"
      --autorun "$autorun"
      --uncertainty-root "$UNCERTAINTY_ROOT"
      --output-root "$OUTPUT_ROOT"
      --student-model "$STUDENT_MODEL"
      --teacher-model "$TEACHER_MODEL"
      --student-api-base "$STUDENT_API_BASE"
      --teacher-api-base "$TEACHER_API_BASE"
      --api-key "$OPENAI_API_KEY"
      --teacher-label "$TEACHER_LABEL"
      --temperature "$TEMPERATURE"
      --top-p "$TOP_P"
      --max-tokens "$MAX_TOKENS"
      --student-max-tokens "$STUDENT_MAX_TOKENS"
      --teacher-max-tokens "$TEACHER_MAX_TOKENS"
      --miprov2-temp "$MIPROV2_TEMP"
      --dspy-num-threads "$DSPY_NUM_THREADS"
      --max-errors "$MAX_ERRORS"
      --seed "$SEED"
      --max-article-chars "$MAX_ARTICLE_CHARS"
      --num-queries "$NUM_QUERIES"
      --eval-workers "$EVAL_WORKERS"
      --view-data-batch-size "$VIEW_DATA_BATCH_SIZE"
    )
    if [[ -n "$TRAIN_SIZE" ]]; then cmd+=(--train-size "$TRAIN_SIZE"); fi
    if [[ -n "$VAL_SIZE" ]]; then cmd+=(--val-size "$VAL_SIZE"); fi
    if [[ "$PROGRAM_AWARE_PROPOSER" != "1" ]]; then cmd+=(--disable-program-aware-proposer); fi
    if [[ "$DATA_AWARE_PROPOSER" != "1" ]]; then cmd+=(--disable-data-aware-proposer); fi
    if [[ "$TIP_AWARE_PROPOSER" != "1" ]]; then cmd+=(--disable-tip-aware-proposer); fi
    if [[ "$FEWSHOT_AWARE_PROPOSER" != "1" ]]; then cmd+=(--disable-fewshot-aware-proposer); fi
    if [[ "$FORCE" == "1" ]]; then cmd+=(--force); fi
    if [[ "$SKIP_CALIBRATION_EVAL" == "1" ]]; then cmd+=(--skip-calibration-eval); fi
    if [[ "$SKIP_ENDPOINT_CHECK" == "1" ]]; then cmd+=(--skip-endpoint-check); fi
    run_with_retries "${expert}/${language}/${prompt_variant}/${autorun}" "${cmd[@]}"
    echo "Finished at $(date)"
  ) > "$log_file" 2>&1 &
  ACTIVE_PIDS+=("$!")
  echo "Started task ${task_id}: ${expert} ${language} ${prompt_variant} ${autorun}; log: ${log_file}"
}

TASKS=(
  "han_xlmr_masked slovenian"
  "longformer_masked slovenian"
  "slavic_specific_masked slovenian"
  "longformer_masked serbian"
  "mdeberta_masked serbian"
  "slavic_specific_masked serbian"
)

echo "Starting DSPy calibration queue at $(date)"
echo "Python: ${PYTHON_BIN}"
echo "Output root: ${OUTPUT_ROOT}"
echo "Task filter: ${TASK_FILTER}"
echo "Autoruns: ${AUTORUNS}"
echo "Prompt variants: ${PROMPT_VARIANTS}"
echo "Prompt style: ${PROMPT_STYLE}"
echo "Max parallel: ${MAX_PARALLEL}"
echo "Student: ${STUDENT_MODEL} @ ${STUDENT_API_BASE}"
echo "Teacher: ${TEACHER_MODEL} @ ${TEACHER_API_BASE}"
echo "Max tokens label/student/teacher: ${MAX_TOKENS}/${STUDENT_MAX_TOKENS}/${TEACHER_MAX_TOKENS}"
echo "Max article chars: ${MAX_ARTICLE_CHARS}"
echo "MIPRO proposers program/data/tip/fewshot: ${PROGRAM_AWARE_PROPOSER}/${DATA_AWARE_PROPOSER}/${TIP_AWARE_PROPOSER}/${FEWSHOT_AWARE_PROPOSER}"

TASK_ID=0
FAILURES=0
ACTIVE_PIDS=()
for task in "${TASKS[@]}"; do
  read -r expert language <<< "$task"
  if ! include_task "$expert" "$language"; then
    TASK_ID=$((TASK_ID + 4))
    continue
  fi
  for prompt_variant in $PROMPT_VARIANTS; do
    for autorun in $AUTORUNS; do
      wait_for_slot
      run_task "$TASK_ID" "$expert" "$language" "$prompt_variant" "$autorun"
      TASK_ID=$((TASK_ID + 1))
    done
  done
done

if [[ "${#ACTIVE_PIDS[@]}" -gt 0 ]]; then
  for job in "${ACTIVE_PIDS[@]}"; do
    if ! wait "$job"; then
      FAILURES=$((FAILURES + 1))
    fi
  done
fi

echo "Calibration queue finished at $(date); failures=${FAILURES}"
if [[ "$FAILURES" -gt 0 ]]; then
  exit 1
fi
