#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export MODE="${MODE:-query}"
export OUTPUT_ROOT="${OUTPUT_ROOT:-reviews/uncertainty/llm-selective-deferral}"
export TASK_FILTER="${TASK_FILTER:-all}"
export PROMPT_VARIANTS="${PROMPT_VARIANTS:-masked unmasked}"
export AUTORUNS="${AUTORUNS:-medium}"
export PRIMARY_EXPERT="${PRIMARY_EXPERT:-legacy_han_xlmr_masked}"
export MAX_PARALLEL="${MAX_PARALLEL:-1}"
export BALANCED_SAMPLING="${BALANCED_SAMPLING:-1}"
export PREFER_BALANCED_SPLITS="${PREFER_BALANCED_SPLITS:-1}"

bash "${SCRIPT_DIR}/llm-selective-deferral.sh"
