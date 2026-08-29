#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   bash scripts/run_validation_interactive.sh 1
#   bash scripts/run_validation_interactive.sh 2
#   bash scripts/run_validation_interactive.sh 4
# Optional: GPU_IDS=0,2 ABSA_MODELS=xlmr,longformer ... 2

HF_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
N_GPUS="${1:-1}"
MC_PASSES="${ABSA_MC_PASSES:-2}"
BATCH_SIZE="${ABSA_BATCH_SIZE:-10}"
DEVICE="${ABSA_DEVICE:-cuda}"
MODEL_CSV="${ABSA_MODELS:-mt5,longformer,mdeberta-v3,han-xlmr,xlmr,slavic-specific,bge-m3-mlp}"

if ! [[ "$N_GPUS" =~ ^[1-9][0-9]*$ ]]; then
  echo "ERROR: N_GPUS must be a positive integer (normally 1, 2, or 4)." >&2
  exit 2
fi

python_has_torch() {
  [[ -n "$1" && -x "$1" ]] && "$1" -c 'import torch' >/dev/null 2>&1
}

PYTHON_BIN="${ABSA_PYTHON:-}"
if ! python_has_torch "$PYTHON_BIN"; then PYTHON_BIN="${CONDA_PREFIX:-}/bin/python"; fi
if ! python_has_torch "$PYTHON_BIN"; then PYTHON_BIN="${HOME}/.conda/envs/absa/bin/python"; fi
if ! python_has_torch "$PYTHON_BIN"; then PYTHON_BIN="$(command -v python 2>/dev/null || true)"; fi
if ! python_has_torch "$PYTHON_BIN"; then
  echo "ERROR: no Python interpreter with PyTorch was found." >&2
  echo "Set ABSA_PYTHON=/absolute/path/to/the/absa/environment/bin/python." >&2
  exit 2
fi

IFS=',' read -r -a MODELS <<< "$MODEL_CSV"
if [[ "${#MODELS[@]}" -eq 0 ]]; then
  echo "ERROR: ABSA_MODELS selected no model families." >&2
  exit 2
fi
for model in "${MODELS[@]}"; do
  case "$model" in
    xlmr|han-xlmr|longformer|mdeberta-v3|mt5|slavic-specific|bge-m3-mlp) ;;
    *) echo "ERROR: unknown model in ABSA_MODELS: ${model}" >&2; exit 2 ;;
  esac
done

if [[ -n "${GPU_IDS:-}" ]]; then
  IFS=',' read -r -a GPUS <<< "$GPU_IDS"
  if [[ "${#GPUS[@]}" -ne "$N_GPUS" ]]; then
    echo "ERROR: GPU_IDS contains ${#GPUS[@]} IDs but N_GPUS=${N_GPUS}." >&2
    exit 2
  fi
elif [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  IFS=',' read -r -a VISIBLE_GPUS <<< "$CUDA_VISIBLE_DEVICES"
  if [[ "${#VISIBLE_GPUS[@]}" -lt "$N_GPUS" ]]; then
    echo "ERROR: CUDA_VISIBLE_DEVICES exposes ${#VISIBLE_GPUS[@]} GPUs but N_GPUS=${N_GPUS}." >&2
    exit 2
  fi
  GPUS=("${VISIBLE_GPUS[@]:0:N_GPUS}")
else
  GPUS=()
  for ((index=0; index<N_GPUS; index++)); do GPUS+=("$index"); done
fi

if [[ "$DEVICE" == cuda* ]]; then
  visible_count="$($PYTHON_BIN -c 'import torch; print(torch.cuda.device_count())')"
  if [[ "$visible_count" -lt "$N_GPUS" ]]; then
    echo "ERROR: PyTorch sees ${visible_count} GPUs but N_GPUS=${N_GPUS}." >&2
    echo "Run this inside an interactive allocation containing at least ${N_GPUS} GPUs." >&2
    exit 2
  fi
fi

run_id="$(date -u +%Y%m%dT%H%M%SZ)-$$"
run_root="${HF_ROOT}/validation-runs/${run_id}"
report_root="${run_root}/reports"
log_root="${run_root}/logs"
mkdir -p "$report_root" "$log_root"

echo "Interactive AspectBench validation"
echo "Python: ${PYTHON_BIN}"
echo "Workers / GPUs: ${N_GPUS} (${GPUS[*]})"
echo "Models: ${MODELS[*]}"
echo "MC passes: ${MC_PASSES}"
echo "Batch size: ${BATCH_SIZE}"
echo "Run directory: ${run_root}"
"$PYTHON_BIN" -c 'import sys, torch; print(f"Executable: {sys.executable}"); print(f"PyTorch: {torch.__version__}"); print(f"CUDA available: {torch.cuda.is_available()}"); print(f"CUDA devices visible before worker isolation: {torch.cuda.device_count()}")'

worker_pids=()
for ((worker=0; worker<N_GPUS; worker++)); do
  (
    worker_status=0
    gpu_id="${GPUS[$worker]}"
    for ((task=worker; task<${#MODELS[@]}; task+=N_GPUS)); do
      model="${MODELS[$task]}"
      report="${report_root}/${model}.json"
      log="${log_root}/${model}.log"
      echo "[GPU ${gpu_id}] START ${model}"
      set +e
      CUDA_VISIBLE_DEVICES="$gpu_id" "$PYTHON_BIN" "${HF_ROOT}/scripts/validate_all.py" \
        --model "$model" \
        --model-root "${HF_ROOT}/models" \
        --examples-root "${HF_ROOT}/examples" \
        --device "$DEVICE" \
        --batch-size "$BATCH_SIZE" \
        --mc-passes "$MC_PASSES" \
        --output "$report" 2>&1 | sed "s/^/[GPU ${gpu_id} ${model}] /" | tee "$log"
      command_status=${PIPESTATUS[0]}
      set -e
      if [[ "$command_status" -eq 0 ]]; then
        echo "[GPU ${gpu_id}] COMPLETE ${model}"
      else
        echo "[GPU ${gpu_id}] FAILED ${model} (exit ${command_status})" >&2
        worker_status=1
      fi
    done
    exit "$worker_status"
  ) &
  worker_pids+=("$!")
done

worker_failures=0
for pid in "${worker_pids[@]}"; do
  if ! wait "$pid"; then worker_failures=1; fi
done

set +e
"$PYTHON_BIN" "${HF_ROOT}/scripts/merge_validation_reports.py" \
  --input-dir "$report_root" \
  --expected-reports "${#MODELS[@]}" \
  --output "${HF_ROOT}/validation-report.json"
merge_status=$?
set -e

echo "Per-family reports and logs: ${run_root}"
if [[ "$worker_failures" -ne 0 || "$merge_status" -ne 0 ]]; then
  echo "INTERACTIVE VALIDATION FAILED" >&2
  exit 1
fi
echo "INTERACTIVE VALIDATION COMPLETED"
