#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "${SCRIPT_DIR}/.." && pwd)}"
cd "$PROJECT_ROOT"

TASK_SET="${TASK_SET:-all}"
TASK_FILE="${TASK_FILE:-hpc-tasks/tasks_${TASK_SET}.tsv}"
MAX_PARALLEL="${MAX_PARALLEL:-4}"

python hpc-tasks/make_task_matrix.py --task-set "$TASK_SET" --output "$TASK_FILE"
num_tasks="$(($(wc -l < "$TASK_FILE") - 1))"
last_task="$((num_tasks - 1))"

mkdir -p reviews/_slurm reviews/_logs

echo "Submitting ${num_tasks} tasks from ${TASK_FILE}"
echo "Array: 0-${last_task}%${MAX_PARALLEL}"
echo "USE_APPTAINER=${USE_APPTAINER:-1}"

sbatch \
  --array="0-${last_task}%${MAX_PARALLEL}" \
  --export=ALL,PROJECT_ROOT="$PROJECT_ROOT",TASK_FILE="$PROJECT_ROOT/$TASK_FILE" \
  hpc-tasks/slurm/absa-array.sbatch
