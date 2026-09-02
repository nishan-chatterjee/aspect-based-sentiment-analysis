#!/usr/bin/env bash
set -euo pipefail

# Environment-style matrix launcher compatible with the paper-era workflow.
PYTHON_BIN="${PYTHON_BIN:-python}"
DATASET="${DATASET:?Set DATASET=hbs or DATASET=sl}"
INPUT="${INPUT:?Set INPUT to a JSON/JSONL record file}"
TASK_FILTER="${TASK_FILTER:-all}"
VARIANT_MODE="${VARIANT_MODE:-best}"
PROMPT_VARIANTS="${PROMPT_VARIANTS:-masked unmasked}"
STUDENT_API_BASES="${STUDENT_API_BASES:-${STUDENT_API_BASE:-http://127.0.0.1:8000/v1}}"
NUM_WORKERS_PER_ENDPOINTS="${NUM_WORKERS_PER_ENDPOINTS:-1}"
ENDPOINT_MODEL="${ENDPOINT_MODEL:-gemma27b}"
GATE_RATE="${GATE_RATE:-0.10}"
MAX_PARALLEL="${MAX_PARALLEL:-1}"
SKIP_ENDPOINT_CHECK="${SKIP_ENDPOINT_CHECK:-0}"
PROGRAM_SOURCE="${PROGRAM_SOURCE:-precalibrated}"
PROGRAM_RUN_ID="${PROGRAM_RUN_ID:-}"
RUN_PREFIX="${RUN_PREFIX:-dspy-query}"
MC_PASSES="${MC_PASSES:-8}"

if [[ "$DATASET" != "hbs" && "$DATASET" != "sl" ]]; then
  echo "DATASET must be hbs or sl." >&2; exit 2
fi
if [[ "$VARIANT_MODE" != "best" && "$VARIANT_MODE" != "masked" && "$VARIANT_MODE" != "unmasked" ]]; then
  echo "VARIANT_MODE must be best, masked, or unmasked." >&2; exit 2
fi
if [[ "$SKIP_ENDPOINT_CHECK" != "1" ]]; then
  IFS=',' read -r -a endpoint_array <<< "$STUDENT_API_BASES"
  for base in "${endpoint_array[@]}"; do
    base="${base%/v1}"
    curl -fsS "$base/v1/models" >/dev/null || {
      echo "Endpoint check failed: $base/v1/models" >&2; exit 1;
    }
  done
fi

if [[ "$TASK_FILTER" == "all" ]]; then
  mapfile -t primary_models < <(
    find "selective-deferral-programs/precalibrated" -mindepth 1 -maxdepth 1 -type d -printf '%f\n' |
      while read -r model; do
        [[ -d "selective-deferral-programs/precalibrated/$model/$DATASET" ]] && echo "$model"
      done | sort
  )
else
  normalized="${TASK_FILTER//,/ }"
  read -r -a primary_models <<< "$normalized"
fi

best_variant() {
  case "$DATASET:$1" in
    hbs:xlmr|hbs:han-xlmr) echo masked ;;
    hbs:longformer|hbs:mdeberta-v3|hbs:mt5|hbs:bertic|hbs:bge-m3-mlp) echo unmasked ;;
    sl:xlmr|sl:han-xlmr|sl:longformer|sl:mdeberta-v3|sl:bge-m3-mlp) echo masked ;;
    sl:mt5|sl:sloberta) echo unmasked ;;
    *) echo "Unknown best variant for $DATASET/$1" >&2; return 2 ;;
  esac
}

mkdir -p models/_runs/_launcher-logs
pids=()
for primary in "${primary_models[@]}"; do
  plm_variant="$VARIANT_MODE"
  [[ "$plm_variant" == "best" ]] && plm_variant="$(best_variant "$primary")"
  for prompt_variant in $PROMPT_VARIANTS; do
    program_flags=(--program-source "$PROGRAM_SOURCE")
    [[ -n "$PROGRAM_RUN_ID" ]] && program_flags+=(--program-run-id "$PROGRAM_RUN_ID")
    run_id="$RUN_PREFIX-$DATASET-$primary-$plm_variant-$prompt_variant-gate-$GATE_RATE"
    log="models/_runs/_launcher-logs/$run_id.log"
    echo "[$primary/$plm_variant/$prompt_variant] $log"
    "$PYTHON_BIN" scripts/2.1-dspy-inference.py \
      --input "$INPUT" --models all --primary-model "$primary" \
      --dataset "$DATASET" --variant "$plm_variant" --prompt-variant "$prompt_variant" \
      --endpoint-model "$ENDPOINT_MODEL" --api-bases "$STUDENT_API_BASES" \
      --num-workers-per-endpoint "$NUM_WORKERS_PER_ENDPOINTS" \
      --gate-rate "$GATE_RATE" --mc-passes "$MC_PASSES" --run-id "$run_id" \
      "${program_flags[@]}" >"$log" 2>&1 &
    pids+=("$!")
    if (( ${#pids[@]} >= MAX_PARALLEL )); then
      wait "${pids[0]}"; pids=("${pids[@]:1}")
    fi
  done
done
for pid in "${pids[@]}"; do wait "$pid"; done
