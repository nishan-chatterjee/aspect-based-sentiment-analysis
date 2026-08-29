#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
HF_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
PROJECT_ROOT="$(cd "${HF_ROOT}/.." && pwd)"
PARTITION="${PARTITION:-gpu-a40}"
TIME_LIMIT="${TIME_LIMIT:-04:00:00}"
ABSA_PYTHON="${ABSA_PYTHON:-${HOME}/.conda/envs/absa/bin/python}"

mkdir -p "${HF_ROOT}/logs"
cd "$PROJECT_ROOT"

echo "Submitting single + batched validation for every AspectBench slot"
echo "Python: ${ABSA_PYTHON}"
echo "Partition: ${PARTITION}"
echo "Time limit: ${TIME_LIMIT}"

sbatch \
  --partition="$PARTITION" \
  --time="$TIME_LIMIT" \
  --export=ALL,PROJECT_ROOT="$PROJECT_ROOT",HF_ROOT="$HF_ROOT",ABSA_PYTHON="$ABSA_PYTHON" \
  "${HF_ROOT}/scripts/validate_all_models_slurm.sh"
