#!/bin/bash
#SBATCH --partition=gpu-a40
#SBATCH --time=24:00:00
#SBATCH --job-name=absa-review
#SBATCH --output=reviews/_slurm/%x-%A_%a.out
#SBATCH --error=reviews/_slurm/%x-%A_%a.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48GB
#SBATCH --gres=gpu:1

set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$SLURM_SUBMIT_DIR}"
cd "$PROJECT_ROOT"
mkdir -p reviews/_slurm reviews/_logs

echo "SLURM job id: ${SLURM_JOB_ID}"
echo "SLURM array task id: ${SLURM_ARRAY_TASK_ID}"
echo "Node: $(hostname)"
echo "Project root: ${PROJECT_ROOT}"

if [[ "${USE_APPTAINER:-1}" == "1" ]]; then
  bash hpc-tasks/run_one_task_apptainer.sh "$SLURM_ARRAY_TASK_ID"
else
  module load Anaconda3
  source /opt/easybuild/software/Anaconda3/2024.02-1/etc/profile.d/conda.sh
  conda activate "${CONDA_ENV:-absa}"
  PYTHON_BIN="$(which python)" bash hpc-tasks/run_one_task.sh "$SLURM_ARRAY_TASK_ID"
fi
