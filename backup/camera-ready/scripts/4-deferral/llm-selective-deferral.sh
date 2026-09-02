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

MODE="${MODE:-calibrate}" # calibrate | query
OUTPUT_ROOT="${OUTPUT_ROOT:-reviews/uncertainty/llm-selective-deferral}"
UNCERTAINTY_ROOT="${UNCERTAINTY_ROOT:-reviews/uncertainty}"
LOG_DIR="${LOG_DIR:-${OUTPUT_ROOT}/_logs_${MODE}}"
TASK_FILTER="${TASK_FILTER:-all}" # all | slovenian | serbian | language:variant
PROMPT_VARIANTS="${PROMPT_VARIANTS:-masked unmasked}"
AUTORUNS="${AUTORUNS:-medium}"
MAX_PARALLEL="${MAX_PARALLEL:-1}"
MAX_RETRIES="${MAX_RETRIES:-1}"
RETRY_DELAY_SECONDS="${RETRY_DELAY_SECONDS:-60}"

STUDENT_API_BASE="${STUDENT_API_BASE:-http://127.0.0.1:8000/v1}"
TEACHER_API_BASE="${TEACHER_API_BASE:-http://127.0.0.1:8001/v1}"
STUDENT_API_BASES="${STUDENT_API_BASES:-$STUDENT_API_BASE}"
STUDENT_MODEL="${STUDENT_MODEL:-gemma27b}"
TEACHER_MODEL="${TEACHER_MODEL:-qwen72b}"
TEACHER_LABEL="${TEACHER_LABEL:-qwen-2.5-72b}"
ENDPOINT_TYPE="${ENDPOINT_TYPE:-chat}"
STUDENT_ENDPOINT_TYPE="${STUDENT_ENDPOINT_TYPE:-$ENDPOINT_TYPE}"
TEACHER_ENDPOINT_TYPE="${TEACHER_ENDPOINT_TYPE:-$ENDPOINT_TYPE}"
OPENAI_API_KEY="${OPENAI_API_KEY:-EMPTY}"

MAX_TOKENS="${MAX_TOKENS:-1024}"
STUDENT_MAX_TOKENS="${STUDENT_MAX_TOKENS:-$MAX_TOKENS}"
TEACHER_MAX_TOKENS="${TEACHER_MAX_TOKENS:-$MAX_TOKENS}"
REQUEST_LOGPROBS="${REQUEST_LOGPROBS:-1}"
TOP_LOGPROBS="${TOP_LOGPROBS:-5}"
TEMPERATURE="${TEMPERATURE:-1.0}"
TOP_P="${TOP_P:-0.95}"
TOP_K="${TOP_K:-}"
MIPROV2_TEMP="${MIPROV2_TEMP:-1.0}"
DSPY_NUM_THREADS="${DSPY_NUM_THREADS:-8}"
MAX_ERRORS="${MAX_ERRORS:-50}"
SEED="${SEED:-42}"
MAX_ARTICLE_CHARS="${MAX_ARTICLE_CHARS:-10000}"
EVAL_WORKERS="${EVAL_WORKERS:-4}"
NUM_WORKERS_PER_ENDPOINT="${NUM_WORKERS_PER_ENDPOINT:-6}"
NUM_WORKERS_PER_ENDPOINTS="${NUM_WORKERS_PER_ENDPOINTS:-}"
TRAIN_SIZE="${TRAIN_SIZE:-}"
VAL_SIZE="${VAL_SIZE:-}"
LIMIT_ITEMS="${LIMIT_ITEMS:-}"
SAMPLE_ITEMS="${SAMPLE_ITEMS:-}"
PRIMARY_EXPERT="${PRIMARY_EXPERT:-}"
EXPERTS="${EXPERTS:-}"
FORCE="${FORCE:-0}"
DRY_RUN="${DRY_RUN:-0}"
SKIP_ENDPOINT_CHECK="${SKIP_ENDPOINT_CHECK:-0}"
PROGRAM_AWARE_PROPOSER="${PROGRAM_AWARE_PROPOSER:-1}"
DATA_AWARE_PROPOSER="${DATA_AWARE_PROPOSER:-1}"
TIP_AWARE_PROPOSER="${TIP_AWARE_PROPOSER:-1}"
FEWSHOT_AWARE_PROPOSER="${FEWSHOT_AWARE_PROPOSER:-1}"
VIEW_DATA_BATCH_SIZE="${VIEW_DATA_BATCH_SIZE:-3}"
HARD_CANDIDATE_MULTIPLIER="${HARD_CANDIDATE_MULTIPLIER:-3.0}"
MIN_CONFIDENCE="${MIN_CONFIDENCE:-0.80}"
BALANCED_SAMPLING="${BALANCED_SAMPLING:-1}"
PREFER_BALANCED_SPLITS="${PREFER_BALANCED_SPLITS:-1}"
HARD_GATED_QUERY="${HARD_GATED_QUERY:-1}"
CALIBRATION_SAMPLING="${CALIBRATION_SAMPLING:-low_confidence_stratified}"
UNCERTAIN_POOL_RATE="${UNCERTAIN_POOL_RATE:-0.10}"
GATE_RATE="${GATE_RATE:-}"

export OPENAI_API_KEY
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export LITELLM_LOCAL_MODEL_COST_MAP="${LITELLM_LOCAL_MODEL_COST_MAP:-true}"
export LITELLM_LOG="${LITELLM_LOG:-ERROR}"
export DSPY_CACHEDIR="${DSPY_CACHEDIR:-${OUTPUT_ROOT}/.dspy_cache}"

mkdir -p "$LOG_DIR" "$DSPY_CACHEDIR"

if [[ -z "$NUM_WORKERS_PER_ENDPOINTS" && "$NUM_WORKERS_PER_ENDPOINT" == *,* ]]; then
  NUM_WORKERS_PER_ENDPOINTS="$NUM_WORKERS_PER_ENDPOINT"
  NUM_WORKERS_PER_ENDPOINT="6"
  echo "Detected comma-separated NUM_WORKERS_PER_ENDPOINT; using NUM_WORKERS_PER_ENDPOINTS=${NUM_WORKERS_PER_ENDPOINTS}"
fi

include_task() {
  local language="$1"
  local prompt_variant="$2"
  case "$TASK_FILTER" in
    all) return 0 ;;
    slovenian|serbian) [[ "$language" == "$TASK_FILTER" ]] ;;
    *:*) [[ "${language}:${prompt_variant}" == "$TASK_FILTER" ]] ;;
    *) return 1 ;;
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

primary_for_language() {
  local language="$1"
  if [[ -n "$PRIMARY_EXPERT" ]]; then
    echo "$PRIMARY_EXPERT"
  else
    echo "legacy_han_xlmr_masked"
  fi
}

run_task() {
  local task_id="$1"
  local language="$2"
  local prompt_variant="$3"
  local autorun="$4"
  local primary
  primary="$(primary_for_language "$language")"
  local log_file="${LOG_DIR}/${MODE}_${task_id}_${primary}_${language}_${prompt_variant}_${autorun}.log"
  (
    echo "Task ${task_id}: ${MODE} ${language} ${prompt_variant} ${autorun}"
    echo "Started at $(date)"
    cmd=(
      "$PYTHON_BIN" "scripts/4-deferral/llm-selective-deferral.py" "$MODE"
      --language "$language"
      --prompt-variant "$prompt_variant"
      --autorun "$autorun"
      --primary-expert "$primary"
      --uncertainty-root "$UNCERTAINTY_ROOT"
      --output-root "$OUTPUT_ROOT"
      --student-model "$STUDENT_MODEL"
      --teacher-model "$TEACHER_MODEL"
      --student-api-base "$STUDENT_API_BASE"
      --teacher-api-base "$TEACHER_API_BASE"
      --student-endpoint-type "$STUDENT_ENDPOINT_TYPE"
      --teacher-endpoint-type "$TEACHER_ENDPOINT_TYPE"
      --api-key "$OPENAI_API_KEY"
      --teacher-label "$TEACHER_LABEL"
      --temperature "$TEMPERATURE"
      --top-p "$TOP_P"
      --max-tokens "$MAX_TOKENS"
      --student-max-tokens "$STUDENT_MAX_TOKENS"
      --teacher-max-tokens "$TEACHER_MAX_TOKENS"
      --top-logprobs "$TOP_LOGPROBS"
      --miprov2-temp "$MIPROV2_TEMP"
      --dspy-num-threads "$DSPY_NUM_THREADS"
      --max-errors "$MAX_ERRORS"
      --seed "$SEED"
      --max-article-chars "$MAX_ARTICLE_CHARS"
      --eval-workers "$EVAL_WORKERS"
      --num-workers-per-endpoint "$NUM_WORKERS_PER_ENDPOINT"
      --hard-candidate-multiplier "$HARD_CANDIDATE_MULTIPLIER"
      --min-confidence "$MIN_CONFIDENCE"
      --calibration-sampling "$CALIBRATION_SAMPLING"
      --uncertain-pool-rate "$UNCERTAIN_POOL_RATE"
    )
    if [[ "$BALANCED_SAMPLING" == "1" ]]; then
      cmd+=(--balanced-sampling)
    else
      cmd+=(--no-balanced-sampling)
    fi
    if [[ "$PREFER_BALANCED_SPLITS" == "1" ]]; then
      cmd+=(--prefer-balanced-splits)
    else
      cmd+=(--no-prefer-balanced-splits)
    fi
    if [[ "$HARD_GATED_QUERY" == "1" ]]; then
      cmd+=(--hard-gated-query)
    else
      cmd+=(--no-hard-gated-query)
    fi
    if [[ "$MODE" == "query" ]]; then
      cmd+=(--student-api-bases "$STUDENT_API_BASES")
    fi
    if [[ "$REQUEST_LOGPROBS" == "1" ]]; then
      cmd+=(--request-logprobs)
    else
      cmd+=(--no-request-logprobs)
    fi
    if [[ -n "$EXPERTS" ]]; then
      # shellcheck disable=SC2206
      expert_args=($EXPERTS)
      cmd+=(--experts "${expert_args[@]}")
    fi
    if [[ -n "$TRAIN_SIZE" ]]; then cmd+=(--train-size "$TRAIN_SIZE"); fi
    if [[ -n "$VAL_SIZE" ]]; then cmd+=(--val-size "$VAL_SIZE"); fi
    if [[ -n "$TOP_K" ]]; then cmd+=(--top-k "$TOP_K"); fi
    if [[ -n "$LIMIT_ITEMS" ]]; then cmd+=(--limit-items "$LIMIT_ITEMS"); fi
    if [[ -n "$SAMPLE_ITEMS" ]]; then cmd+=(--sample-items "$SAMPLE_ITEMS"); fi
    if [[ -n "$GATE_RATE" ]]; then cmd+=(--gate-rate "$GATE_RATE"); fi
    if [[ -n "$NUM_WORKERS_PER_ENDPOINTS" ]]; then cmd+=(--num-workers-per-endpoints "$NUM_WORKERS_PER_ENDPOINTS"); fi
    if [[ "$PROGRAM_AWARE_PROPOSER" != "1" ]]; then cmd+=(--disable-program-aware-proposer); fi
    if [[ "$DATA_AWARE_PROPOSER" != "1" ]]; then cmd+=(--disable-data-aware-proposer); fi
    if [[ "$TIP_AWARE_PROPOSER" != "1" ]]; then cmd+=(--disable-tip-aware-proposer); fi
    if [[ "$FEWSHOT_AWARE_PROPOSER" != "1" ]]; then cmd+=(--disable-fewshot-aware-proposer); fi
    if [[ "$FORCE" == "1" ]]; then cmd+=(--force); fi
    if [[ "$DRY_RUN" == "1" ]]; then cmd+=(--dry-run); fi
    if [[ "$SKIP_ENDPOINT_CHECK" == "1" ]]; then cmd+=(--skip-endpoint-check); fi
    run_with_retries "${MODE}/${language}/${prompt_variant}/${autorun}" "${cmd[@]}"
    echo "Finished at $(date)"
  ) > "$log_file" 2>&1 &
  local pid="$!"
  if [[ "$DRY_RUN" == "1" ]]; then
    if ! wait "$pid"; then
      FAILURES=$((FAILURES + 1))
    fi
    echo "Dry-run task ${task_id}: ${MODE} ${language} ${prompt_variant} ${autorun}; log: ${log_file}"
    sed -n '/Preflight/,$p' "$log_file"
  else
    ACTIVE_PIDS+=("$pid")
    echo "Started task ${task_id}: ${MODE} ${language} ${prompt_variant} ${autorun}; log: ${log_file}"
  fi
}

if [[ "$MODE" != "calibrate" && "$MODE" != "query" ]]; then
  echo "MODE must be calibrate or query, got: $MODE" >&2
  exit 2
fi

echo "Starting LLM selective deferral ${MODE} queue at $(date)"
echo "Python: ${PYTHON_BIN}"
echo "Output root: ${OUTPUT_ROOT}"
echo "Task filter: ${TASK_FILTER}"
echo "Prompt variants: ${PROMPT_VARIANTS}"
echo "Autoruns: ${AUTORUNS}"
echo "Student: ${STUDENT_MODEL} @ ${STUDENT_API_BASE}"
echo "Teacher: ${TEACHER_MODEL} @ ${TEACHER_API_BASE}"
echo "Endpoint types student/teacher: ${STUDENT_ENDPOINT_TYPE}/${TEACHER_ENDPOINT_TYPE}"
echo "Query student endpoints: ${STUDENT_API_BASES}"
echo "Workers per endpoint: ${NUM_WORKERS_PER_ENDPOINTS:-${NUM_WORKERS_PER_ENDPOINT}}"
echo "Request logprobs/top: ${REQUEST_LOGPROBS}/${TOP_LOGPROBS}"
echo "Sampling: temperature=${TEMPERATURE} top_p=${TOP_P} top_k=${TOP_K:-unset}"
echo "Balanced sampling / prefer balanced splits: ${BALANCED_SAMPLING}/${PREFER_BALANCED_SPLITS}"
echo "Hard-gated query: ${HARD_GATED_QUERY}"
echo "Calibration sampling / uncertain pool rate: ${CALIBRATION_SAMPLING}/${UNCERTAIN_POOL_RATE}"
echo "Query gate rate: ${GATE_RATE:-threshold/disagreement}"
echo "Max parallel: ${MAX_PARALLEL}"
echo "Dry run: ${DRY_RUN}"

TASK_ID=0
FAILURES=0
ACTIVE_PIDS=()
for language in slovenian serbian; do
  for prompt_variant in $PROMPT_VARIANTS; do
    if ! include_task "$language" "$prompt_variant"; then
      TASK_ID=$((TASK_ID + 1))
      continue
    fi
    for autorun in $AUTORUNS; do
      wait_for_slot
      run_task "$TASK_ID" "$language" "$prompt_variant" "$autorun"
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

echo "LLM selective deferral ${MODE} queue finished at $(date); failures=${FAILURES}"
if [[ "$FAILURES" -ne 0 ]]; then
  exit 1
fi
