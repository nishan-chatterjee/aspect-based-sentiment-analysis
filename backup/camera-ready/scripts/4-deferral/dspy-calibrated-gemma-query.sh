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
LOG_DIR="${LOG_DIR:-${OUTPUT_ROOT}/_logs_test_query}"
TASK_FILTER="${TASK_FILTER:-all}"
AUTORUNS="${AUTORUNS:-medium heavy}"
PROMPT_VARIANTS="${PROMPT_VARIANTS:-unmasked masked}"
PROMPT_STYLE="${PROMPT_STYLE:-current}"
MAX_PARALLEL="${MAX_PARALLEL:-4}"

STUDENT_MODEL="${STUDENT_MODEL:-gemma27b}"
STUDENT_API_BASES="${STUDENT_API_BASES:-http://127.0.0.1:8000/v1,http://127.0.0.1:8001/v1,http://127.0.0.1:8010/v1,http://127.0.0.1:8011/v1}"
OPENAI_API_KEY="${OPENAI_API_KEY:-EMPTY}"
TEACHER_LABEL="${TEACHER_LABEL:-qwen-2.5-72b}"

MAX_TOKENS="${MAX_TOKENS:-1024}"
TEMPERATURE="${TEMPERATURE:-1.0}"
TOP_P="${TOP_P:-0.95}"
MIPROV2_TEMP="${MIPROV2_TEMP:-1.0}"
MAX_ARTICLE_CHARS="${MAX_ARTICLE_CHARS:-10000}"
NUM_QUERIES="${NUM_QUERIES:-1}"
NUM_WORKERS="${NUM_WORKERS:-8}"
LIMIT_ITEMS="${LIMIT_ITEMS:-}"
SAMPLE_ITEMS="${SAMPLE_ITEMS:-}"
SEED="${SEED:-42}"
FORCE="${FORCE:-0}"
SKIP_ENDPOINT_CHECK="${SKIP_ENDPOINT_CHECK:-0}"

export OPENAI_API_KEY
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export LITELLM_LOCAL_MODEL_COST_MAP="${LITELLM_LOCAL_MODEL_COST_MAP:-true}"
export LITELLM_LOG="${LITELLM_LOG:-ERROR}"
export DSPY_CACHEDIR="${DSPY_CACHEDIR:-${OUTPUT_ROOT}/.dspy_cache}"

mkdir -p "$LOG_DIR" "$DSPY_CACHEDIR"
IFS=',' read -r -a API_BASE_ARRAY <<< "$STUDENT_API_BASES"
if [[ "${#API_BASE_ARRAY[@]}" -eq 0 ]]; then
  echo "STUDENT_API_BASES is empty." >&2
  exit 2
fi

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

run_task() {
  local task_id="$1"
  local expert="$2"
  local language="$3"
  local prompt_variant="$4"
  local autorun="$5"
  local api_index=$((task_id % ${#API_BASE_ARRAY[@]}))
  local api_base="${API_BASE_ARRAY[$api_index]}"
  local log_file="${LOG_DIR}/testquery_${task_id}_${expert}_${language}_${prompt_variant}_${autorun}.log"
  (
    echo "Task ${task_id}: ${expert} ${language} ${prompt_variant} ${autorun}"
    echo "API base: ${api_base}"
    echo "Started at $(date)"
    cmd=(
      "$PYTHON_BIN" "scripts/4-deferral/dspy-calibrated-gemma-query.py"
      --expert "$expert"
      --language "$language"
      --prompt-variant "$prompt_variant"
      --prompt-style "$PROMPT_STYLE"
      --autorun "$autorun"
      --uncertainty-root "$UNCERTAINTY_ROOT"
      --output-root "$OUTPUT_ROOT"
      --student-model "$STUDENT_MODEL"
      --student-api-base "$api_base"
      --api-key "$OPENAI_API_KEY"
      --teacher-label "$TEACHER_LABEL"
      --temperature "$TEMPERATURE"
      --top-p "$TOP_P"
      --max-tokens "$MAX_TOKENS"
      --miprov2-temp "$MIPROV2_TEMP"
      --max-article-chars "$MAX_ARTICLE_CHARS"
      --num-queries "$NUM_QUERIES"
      --num-workers "$NUM_WORKERS"
      --seed "$SEED"
    )
    if [[ -n "$LIMIT_ITEMS" ]]; then cmd+=(--limit-items "$LIMIT_ITEMS"); fi
    if [[ -n "$SAMPLE_ITEMS" ]]; then cmd+=(--sample-items "$SAMPLE_ITEMS"); fi
    if [[ "$FORCE" == "1" ]]; then cmd+=(--force); fi
    if [[ "$SKIP_ENDPOINT_CHECK" == "1" ]]; then cmd+=(--skip-endpoint-check); fi
    printf 'Command:'
    printf ' %q' "${cmd[@]}"
    printf '\n'
    "${cmd[@]}"
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

echo "Starting DSPy calibrated Gemma test-query queue at $(date)"
echo "Python: ${PYTHON_BIN}"
echo "Output root: ${OUTPUT_ROOT}"
echo "Task filter: ${TASK_FILTER}"
echo "Autoruns: ${AUTORUNS}"
echo "Prompt variants: ${PROMPT_VARIANTS}"
echo "Prompt style: ${PROMPT_STYLE}"
echo "Max parallel: ${MAX_PARALLEL}"
echo "Student: ${STUDENT_MODEL} @ ${STUDENT_API_BASES}"

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

echo "Test-query queue finished at $(date); failures=${FAILURES}"
if [[ "$FAILURES" -gt 0 ]]; then
  exit 1
fi
