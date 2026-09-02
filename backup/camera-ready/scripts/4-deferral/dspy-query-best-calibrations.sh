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
SELECTION_JSON="${SELECTION_JSON:-${OUTPUT_ROOT}/best_calibrations.json}"
SELECTION_TSV="${SELECTION_TSV:-${OUTPUT_ROOT}/best_calibrations.tsv}"
LOG_DIR="${LOG_DIR:-${OUTPUT_ROOT}/_logs_best_test_query}"

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
MAX_PARALLEL="${MAX_PARALLEL:-4}"
LIMIT_ITEMS="${LIMIT_ITEMS:-}"
SAMPLE_ITEMS="${SAMPLE_ITEMS:-}"
SEED="${SEED:-42}"
FORCE="${FORCE:-0}"
SKIP_ENDPOINT_CHECK="${SKIP_ENDPOINT_CHECK:-0}"
AUTORUNS="${AUTORUNS:-medium heavy}"
PROMPT_VARIANTS="${PROMPT_VARIANTS:-unmasked masked}"
PROMPT_STYLE="${PROMPT_STYLE:-current}"

export OPENAI_API_KEY
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export LITELLM_LOCAL_MODEL_COST_MAP="${LITELLM_LOCAL_MODEL_COST_MAP:-true}"
export LITELLM_LOG="${LITELLM_LOG:-ERROR}"
export DSPY_CACHEDIR="${DSPY_CACHEDIR:-${OUTPUT_ROOT}/.dspy_cache}"

mkdir -p "$DSPY_CACHEDIR" "$LOG_DIR"

echo "Selecting best calibration per expert/language..."
"$PYTHON_BIN" scripts/4-deferral/dspy-select-best-calibrations.py \
  --output-root "$OUTPUT_ROOT" \
  --max-tokens "$MAX_TOKENS" \
  --teacher-label "$TEACHER_LABEL" \
  --miprov2-temp "$MIPROV2_TEMP" \
  --selection-json "$SELECTION_JSON" \
  --selection-tsv "$SELECTION_TSV" \
  --autoruns $AUTORUNS \
  --prompt-variants $PROMPT_VARIANTS \
  --allow-missing

cmd=(
  "$PYTHON_BIN" "scripts/4-deferral/dspy-query-best-calibrations.py"
  --selection-json "$SELECTION_JSON"
  --python-bin "$PYTHON_BIN"
  --output-root "$OUTPUT_ROOT"
  --uncertainty-root "$UNCERTAINTY_ROOT"
  --log-dir "$LOG_DIR"
  --student-model "$STUDENT_MODEL"
  --student-api-bases "$STUDENT_API_BASES"
  --prompt-style "$PROMPT_STYLE"
  --api-key "$OPENAI_API_KEY"
  --teacher-label "$TEACHER_LABEL"
  --temperature "$TEMPERATURE"
  --top-p "$TOP_P"
  --max-tokens "$MAX_TOKENS"
  --miprov2-temp "$MIPROV2_TEMP"
  --max-article-chars "$MAX_ARTICLE_CHARS"
  --num-queries "$NUM_QUERIES"
  --num-workers "$NUM_WORKERS"
  --max-parallel "$MAX_PARALLEL"
  --seed "$SEED"
)
if [[ -n "$LIMIT_ITEMS" ]]; then cmd+=(--limit-items "$LIMIT_ITEMS"); fi
if [[ -n "$SAMPLE_ITEMS" ]]; then cmd+=(--sample-items "$SAMPLE_ITEMS"); fi
if [[ "$FORCE" == "1" ]]; then cmd+=(--force); fi
if [[ "$SKIP_ENDPOINT_CHECK" == "1" ]]; then cmd+=(--skip-endpoint-check); fi

echo "Running selected calibrated Gemma test queries..."
printf 'Command:'
printf ' %q' "${cmd[@]}"
printf '\n'
"${cmd[@]}"
