#!/usr/bin/env bash
set -euo pipefail

# Run inside an interactive SLURM allocation. This deliberately selects a
# Python executable by path instead of trusting a possibly stale activated
# prompt after `module load Anaconda3` has rewritten PATH.
HF_ROOT="${1:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
MC_PASSES="${ABSA_MC_PASSES:-2}"
BATCH_SIZE="${ABSA_BATCH_SIZE:-10}"
DEVICE="${ABSA_DEVICE:-cuda}"

python_has_torch() {
  [[ -n "$1" && -x "$1" ]] && "$1" -c 'import torch' >/dev/null 2>&1
}

PYTHON_BIN="${ABSA_PYTHON:-}"
if ! python_has_torch "$PYTHON_BIN"; then
  PYTHON_BIN="${CONDA_PREFIX:-}/bin/python"
fi
if ! python_has_torch "$PYTHON_BIN"; then
  PYTHON_BIN="${HOME}/.conda/envs/absa/bin/python"
fi
if ! python_has_torch "$PYTHON_BIN"; then
  PYTHON_BIN="$(command -v python 2>/dev/null || true)"
fi
if ! python_has_torch "$PYTHON_BIN"; then
  echo "ERROR: no Python interpreter with PyTorch was found." >&2
  echo "Set ABSA_PYTHON=/absolute/path/to/the/absa/environment/bin/python." >&2
  exit 2
fi

export PATH="$(dirname "$PYTHON_BIN"):${PATH}"
hash -r

echo "Python: ${PYTHON_BIN}"
"$PYTHON_BIN" -c 'import sys, torch; print(f"Executable: {sys.executable}"); print(f"PyTorch: {torch.__version__}"); print(f"CUDA available: {torch.cuda.is_available()}"); print(f"CUDA runtime: {torch.version.cuda}")'

"$PYTHON_BIN" "${HF_ROOT}/scripts/validate_all.py" \
  --model-root "${HF_ROOT}/models" \
  --examples-root "${HF_ROOT}/examples" \
  --device "${DEVICE}" \
  --batch-size "${BATCH_SIZE}" \
  --mc-passes "${MC_PASSES}" \
  --output "${HF_ROOT}/validation-report.json"
