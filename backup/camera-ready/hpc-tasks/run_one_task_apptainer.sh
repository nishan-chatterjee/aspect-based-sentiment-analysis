#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "${SCRIPT_DIR}/.." && pwd)}"
SIF_PATH="${SIF_PATH:-${PROJECT_ROOT}/hpc-tasks/absa-comparisons.sif}"
TASK_ID="${1:-${SLURM_ARRAY_TASK_ID:-}}"

if [[ -z "$TASK_ID" ]]; then
  echo "Usage: $0 TASK_ID" >&2
  exit 2
fi

if [[ ! -f "$SIF_PATH" ]]; then
  echo "Apptainer image not found: $SIF_PATH" >&2
  echo "Build it with: hpc-tasks/build_apptainer.sh" >&2
  exit 2
fi

apptainer exec --nv \
  --bind "${PROJECT_ROOT}:/workspace" \
  "$SIF_PATH" \
  bash /workspace/hpc-tasks/run_one_task.sh "$TASK_ID"
