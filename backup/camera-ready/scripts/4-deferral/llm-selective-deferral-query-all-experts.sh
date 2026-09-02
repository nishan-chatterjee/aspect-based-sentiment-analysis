#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export MODE="${MODE:-query}"
export OUTPUT_ROOT="${OUTPUT_ROOT:-reviews/uncertainty/llm-selective-deferral}"
export PROMPT_VARIANTS="${PROMPT_VARIANTS:-masked unmasked}"
export AUTORUNS="${AUTORUNS:-medium}"
export MAX_PARALLEL="${MAX_PARALLEL:-1}"
export HARD_GATED_QUERY="${HARD_GATED_QUERY:-1}"
export GATE_RATE="${GATE_RATE:-0.10}"
export BALANCED_SAMPLING="${BALANCED_SAMPLING:-1}"
export PREFER_BALANCED_SPLITS="${PREFER_BALANCED_SPLITS:-1}"
export HAN_EXPERT="${HAN_EXPERT:-legacy_han_xlmr_masked}"
export PRIMARY_FILTER="${PRIMARY_FILTER:-}"

primary_enabled() {
  local language="$1"
  local primary="$2"
  if [[ -z "$PRIMARY_FILTER" ]]; then
    return 0
  fi
  local item
  for item in $PRIMARY_FILTER; do
    if [[ "$item" == "$primary" || "$item" == "${language}:${primary}" ]]; then
      return 0
    fi
  done
  return 1
}

run_primary() {
  local language="$1"
  local primary="$2"
  local experts="$3"
  if ! primary_enabled "$language" "$primary"; then
    echo "=== Skipping selective deferral query for primary=${primary} language=${language} due to PRIMARY_FILTER=${PRIMARY_FILTER} ==="
    return 0
  fi
  echo "=== Querying selective deferral for primary=${primary} language=${language} ==="
  TASK_FILTER="$language" PRIMARY_EXPERT="$primary" EXPERTS="$experts" bash "${SCRIPT_DIR}/llm-selective-deferral.sh"
}

SLOVENIAN_EXPERTS="${SLOVENIAN_EXPERTS:-${HAN_EXPERT} longformer_masked slavic_specific_masked}"
SERBIAN_EXPERTS="${SERBIAN_EXPERTS:-${HAN_EXPERT} longformer_masked mdeberta_masked slavic_specific_masked}"

run_primary slovenian "$HAN_EXPERT" "$SLOVENIAN_EXPERTS"
run_primary slovenian longformer_masked "$SLOVENIAN_EXPERTS"
run_primary slovenian slavic_specific_masked "$SLOVENIAN_EXPERTS"

run_primary serbian "$HAN_EXPERT" "$SERBIAN_EXPERTS"
run_primary serbian longformer_masked "$SERBIAN_EXPERTS"
run_primary serbian mdeberta_masked "$SERBIAN_EXPERTS"
run_primary serbian slavic_specific_masked "$SERBIAN_EXPERTS"
