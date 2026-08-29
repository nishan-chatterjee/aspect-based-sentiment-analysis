#!/usr/bin/env bash
#SBATCH --partition=gpu-a40
#SBATCH --time=04:00:00
#SBATCH --job-name=aspectbench-validate
#SBATCH --output=huggingface/logs/%x-%j.out
#SBATCH --error=huggingface/logs/%x-%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32GB
#SBATCH --gres=gpu:1

set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-${SLURM_SUBMIT_DIR}}"
HF_ROOT="${HF_ROOT:-${PROJECT_ROOT}/huggingface}"
export ABSA_PYTHON="${ABSA_PYTHON:-${HOME}/.conda/envs/absa/bin/python}"
export ABSA_MC_PASSES="${ABSA_MC_PASSES:-2}"
export ABSA_BATCH_SIZE="${ABSA_BATCH_SIZE:-10}"
export ABSA_DEVICE="${ABSA_DEVICE:-cuda}"

cd "$PROJECT_ROOT"

module load Anaconda3/2024.02-1
source /opt/easybuild/software/Anaconda3/2024.02-1/etc/profile.d/conda.sh

echo "Job: ${SLURM_JOB_ID}"
echo "Node: $(hostname)"
echo "Project: ${PROJECT_ROOT}"
echo "Python requested: ${ABSA_PYTHON}"
echo "MC passes: ${ABSA_MC_PASSES}"
echo "Batch size: ${ABSA_BATCH_SIZE}"

set +e
srun bash "${HF_ROOT}/scripts/run_validation_slurm.sh" "$HF_ROOT"
validation_status=$?
set -e

report="${HF_ROOT}/validation-report.json"
if [[ -f "$report" && -x "$ABSA_PYTHON" ]]; then
  "$ABSA_PYTHON" -c 'import json, pathlib, sys; p=json.loads(pathlib.Path(sys.argv[1]).read_text()); print(f"FINAL REPORT: passed={p.get('"'"'passed'"'"', 0)} failed={p.get('"'"'failed'"'"', 0)} skipped_slots={p.get('"'"'skipped'"'"', p.get('"'"'unavailable'"'"', 0))} missing_model_languages={p.get('"'"'missing_model_language_combinations'"'"', '"'"'n/a'"'"')} path={sys.argv[1]}")' "$report"
fi

if [[ "$validation_status" -eq 0 ]]; then
  echo "VALIDATION JOB COMPLETED"
else
  echo "VALIDATION JOB FAILED (exit ${validation_status})" >&2
fi
exit "$validation_status"
